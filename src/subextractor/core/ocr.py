"""画面硬字幕 OCR 提取（设计大纲 §1.2，M6 可选扩展）。

流程：ffmpeg 抽帧（画面底部字幕区域）→ PaddleOCR 逐帧识别 → 相邻帧文本
相似度合并成 cue → 输出 Segment 列表，接入既有断句/翻译/写出管线。

OCR 是可选扩展：
- 依赖：``pip install -e .[ocr]``（paddleocr + paddlepaddle），未安装时报指引；
- 模型：``subextractor fetch-model --kind ocr --lang ja`` 联网一次预置到
  ``models/ocr/paddle``（含 ``det/``、``rec/`` 与 ``lang.txt``），之后完全离线。
"""

from __future__ import annotations

import difflib
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..models import registry
from ..utils.logging import get_logger
from . import media
from .errors import PipelineCancelled
from .subtitle import Segment

logger = get_logger(__name__)

# 项目语言码 → PaddleOCR lang 参数
OCR_LANGS: dict[str, str] = {
    "ja": "japan",
    "ko": "korean",
    "en": "en",
    "fr": "fr",
    "zh": "ch",
}

LANG_FILE = "lang.txt"


@dataclass(slots=True)
class OcrConfig:
    """硬字幕提取参数。"""

    lang: str = "auto"  # OCR 源语；auto = 读模型目录里的 lang.txt
    interval: float = 0.5  # 抽帧间隔（秒）
    roi_bottom: float = 0.35  # 字幕区域：画面底部占比（0.35 = 底部 35%）
    min_duration: float = 0.5  # 字幕最短持续（秒），短于此时长丢弃
    similarity: float = 0.8  # 相邻帧文本相似度阈值（判定同一条字幕）
    max_frames: int | None = None  # 调试用：最多处理的帧数


def ocr_available() -> bool:
    """OCR 扩展依赖（paddleocr）是否已安装。"""
    try:
        import paddleocr  # noqa: F401
    except ImportError:
        return False
    return True


def _require_ocr() -> None:
    if not ocr_available():
        raise RuntimeError(
            "OCR 扩展未安装：pip install -e .[ocr]（paddleocr + paddlepaddle）\n"
            "模型预置：subextractor fetch-model --kind ocr --lang <ja|en|fr|ko|zh>"
        )


def _resolve_lang(config_lang: str, model_dir: Path) -> str:
    """确定 PaddleOCR 语言参数：显式指定优先，否则读 fetch 时写入的 lang.txt。"""
    lang = config_lang
    if lang == "auto":
        lang_file = model_dir / LANG_FILE
        if lang_file.is_file():
            lang = lang_file.read_text(encoding="utf-8").strip()
    if lang not in OCR_LANGS:
        supported = " / ".join(sorted(OCR_LANGS))
        raise ValueError(f"OCR 语言 {lang!r} 不支持（可选：{supported}）；请用 --src 指定")
    return OCR_LANGS[lang]


class OcrEngine:
    """PaddleOCR 封装：本地推理模型、逐帧识别。"""

    def __init__(self, model_dir: str | Path | None = None, lang: str = "auto") -> None:
        self.model_dir = Path(model_dir) if model_dir else None
        self.lang = lang
        self._ocr = None

    def load(self) -> None:
        _require_ocr()
        if self._ocr is not None:
            return
        from paddleocr import PaddleOCR

        model_dir = self.model_dir or registry.resolve_ocr_model()
        lang = _resolve_lang(self.lang, model_dir)
        logger.info("OCR 引擎 paddleocr｜lang=%s｜模型 %s", lang, model_dir)
        self._ocr = PaddleOCR(
            lang=lang,
            text_detection_model_dir=str(model_dir / "det"),
            text_recognition_model_dir=str(model_dir / "rec"),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,  # paddlepaddle 3.3 CPU 的 MKLDNN/PIR 存在转换缺陷，禁用后走普通算子
        )

    def recognize(self, image: Path) -> tuple[str, float]:
        """识别一帧，返回（多行文本, 平均置信度）；无文本返回 ("", 0.0)。"""
        self.load()
        results = self._ocr.predict(str(image))
        texts: list[str] = []
        scores: list[float] = []
        for result in results or []:
            texts.extend(t for t in result["rec_texts"] if t.strip())
            scores.extend(float(s) for s in result["rec_scores"])
        if not texts:
            return "", 0.0
        return "\n".join(texts), sum(scores) / max(len(scores), 1)

    def close(self) -> None:
        self._ocr = None


def _normalize(text: str) -> str:
    """相似度比较前的归一化：去空白、小写（拉丁语大小写抖动不影响判定）。"""
    return "".join(text.split()).lower()


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def merge_frames(
    frames: list[tuple[float, str, float]],
    *,
    interval: float,
    min_duration: float = 0.5,
    similarity: float = 0.8,
) -> list[Segment]:
    """把逐帧识别结果合并成字幕 cue。

    ``frames`` 为 ``(帧起始时间, 文本, 置信度)`` 列表（空文本 = 无字幕帧）。
    相邻帧归一化文本相似度 ≥ ``similarity`` 视为同一条字幕；cue 文本取
    置信度最高的一帧；持续短于 ``min_duration`` 的 cue 丢弃（闪烁噪点）。
    """
    cues: list[Segment] = []
    cur_norm: str | None = None
    cur_start = 0.0
    cur_end = 0.0
    best_text = ""
    best_score = -1.0

    def flush() -> None:
        nonlocal cur_norm, best_score
        if cur_norm is not None and cur_end - cur_start >= min_duration:
            cues.append(Segment(0, best_text, cur_start, cur_end))
        cur_norm = None
        best_score = -1.0

    for t, text, score in frames:
        norm = _normalize(text)
        if not norm:
            flush()
            continue
        if cur_norm is None:
            cur_norm, cur_start, cur_end = norm, t, t + interval
            best_text, best_score = text, score
        elif _similar(norm, cur_norm) >= similarity:
            cur_end = t + interval
            if score > best_score:
                best_text, best_score = text, score
        else:
            flush()
            cur_norm, cur_start, cur_end = norm, t, t + interval
            best_text, best_score = text, score
    flush()
    return cues


def extract_subtitles(
    video_path: str | Path,
    config: OcrConfig | None = None,
    *,
    engine: OcrEngine | None = None,
    on_progress: Callable[[str, float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[list[Segment], str]:
    """从视频画面提取硬字幕，返回（Segment 列表, 源语代码）。"""
    cfg = config or OcrConfig()
    source = Path(video_path)
    info = media.probe(source)
    if not info.has_video:
        raise ValueError(f"文件不含画面，无法做硬字幕提取：{info.path}")

    eng = engine or OcrEngine(lang=cfg.lang)
    own_engine = engine is None
    source_lang = cfg.lang
    try:
        model_dir = eng.model_dir or registry.resolve_ocr_model()
        source_lang = _resolve_lang(cfg.lang, model_dir)
        # 反向取项目语言码（OCR_LANGS 的键）
        source_lang = next(k for k, v in OCR_LANGS.items() if v == source_lang)
        eng.load()

        with tempfile.TemporaryDirectory(prefix="subextractor-ocr-") as tmp:
            frames_dir = Path(tmp) / "frames"
            frames_dir.mkdir()
            roi = cfg.roi_bottom
            vf = f"fps=1/{cfg.interval:g},crop=iw:ih*{roi:g}:0:ih*{1 - roi:g}"
            ffmpeg = media.ffmpeg_path()
            if not ffmpeg:
                raise RuntimeError("未找到 ffmpeg")
            if on_progress:
                on_progress("抽取字幕帧", 0.0)
            subprocess.run(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(source), "-vf", vf, "-q:v", "3",
                    str(frames_dir / "f%06d.jpg"),
                ],
                capture_output=True,
                check=True,
            )

            images = sorted(frames_dir.glob("f*.jpg"))
            if cfg.max_frames is not None:
                images = images[: cfg.max_frames]
            total = len(images)
            logger.info("抽帧完成：%d 帧（间隔 %gs，底部 %.0f%%）", total, cfg.interval, roi * 100)
            if total == 0:
                return [], source_lang

            frames: list[tuple[float, str, float]] = []
            for index, image in enumerate(images):
                if should_cancel and should_cancel():
                    raise PipelineCancelled("任务已取消")
                text, score = eng.recognize(image)
                frames.append((index * cfg.interval, text, score))
                if on_progress and (index % 10 == 0 or index == total - 1):
                    on_progress("识别画面字幕", (index + 1) / total)

        segments = merge_frames(
            frames,
            interval=cfg.interval,
            min_duration=cfg.min_duration,
            similarity=cfg.similarity,
        )
        logger.info("OCR 合并完成：%d 帧 → %d 条字幕", len(frames), len(segments))
        return segments, source_lang
    finally:
        if own_engine:
            eng.close()
