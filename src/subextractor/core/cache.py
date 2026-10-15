"""ASR 断点缓存（M5）。

把「抽音轨 + 识别」这一最耗时阶段的原始结果落盘：重跑时若输入文件指纹
与识别配置都没变，则直接读缓存跳过识别；断句与翻译每次重跑（秒级，无
缓存价值）。典型场景：批量跑到一半被 kill，重启后已完成的文件秒过。

布局：``<cache_dir>/<文件名>-<指纹8位>/``
    meta.json  输入文件指纹（name / size / mtime_ns）与识别配置
    asr.json   原始 ASR 分段（含词级时间戳，断句前）

指纹取「文件名 + 大小 + mtime_ns」的 SHA-1，而非完整路径：同名同内容的
文件放在不同目录可以共享缓存；文件一旦被改动即失效。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..utils.logging import get_logger
from .asr import AsrResult
from .subtitle import Segment, Word

logger = get_logger(__name__)

CACHE_DIR_NAME = ".subextractor-cache"
_META_FILE = "meta.json"
_DATA_FILE = "asr.json"


@dataclass(slots=True)
class AsrCacheEntry:
    """一次缓存命中：原始 ASR 结果 + 实际生效的档位。"""

    result: AsrResult
    resolved_size: str | None


def default_cache_dir(output_dir: Path) -> Path:
    """批处理/单文件共用的默认缓存位置：``<输出目录>/.subextractor-cache``。"""
    return Path(output_dir) / CACHE_DIR_NAME


def _fingerprint(input_path: Path) -> str:
    st = input_path.stat()
    payload = f"{input_path.name}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def _entry_dir(cache_dir: Path, input_path: Path) -> Path:
    return Path(cache_dir) / f"{input_path.stem}-{_fingerprint(input_path)}"


def load_asr_cache(
    cache_dir: Path | str,
    input_path: Path,
    asr_model: str,
    source_lang: str,
) -> AsrCacheEntry | None:
    """按指纹查找缓存；目录缺失、指纹不符或识别配置变更均返回 None。"""
    entry = _entry_dir(Path(cache_dir), input_path)
    meta_path = entry / _META_FILE
    data_path = entry / _DATA_FILE
    if not (meta_path.is_file() and data_path.is_file()):
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data = json.loads(data_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("缓存读取失败，将重新识别：%s（%s）", entry, exc)
        return None

    # 缓存以「配置的档位/源语」为匹配键；auto 档在同设备上解析结果稳定，
    # 换设备后如需强制重算可用 --no-resume。
    if meta.get("asr_model") != asr_model or meta.get("source_lang") != source_lang:
        return None

    try:
        segments = [
            Segment(
                index=int(item["index"]),
                text=str(item["text"]),
                start=float(item["start"]),
                end=float(item["end"]),
                words=[
                    Word(text=str(w["text"]), start=float(w["start"]), end=float(w["end"]))
                    for w in item.get("words", [])
                ],
            )
            for item in data["segments"]
        ]
        result = AsrResult(
            segments=segments,
            language=str(data["language"]),
            language_probability=float(data["language_probability"]),
            duration=data.get("duration"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("缓存数据损坏，将重新识别：%s（%s）", entry, exc)
        return None

    logger.info("命中 ASR 缓存：%s（%d 条原始分段）", entry.name, len(segments))
    return AsrCacheEntry(result=result, resolved_size=meta.get("resolved_size"))


def save_asr_cache(
    cache_dir: Path | str,
    input_path: Path,
    result: AsrResult,
    *,
    asr_model: str,
    source_lang: str,
    resolved_size: str | None = None,
) -> Path:
    """把原始 ASR 结果写入缓存目录，返回缓存条目路径。"""
    entry = _entry_dir(Path(cache_dir), input_path)
    entry.mkdir(parents=True, exist_ok=True)

    st = input_path.stat()
    meta = {
        "input_name": input_path.name,
        "input_size": st.st_size,
        "input_mtime_ns": st.st_mtime_ns,
        "asr_model": asr_model,
        "source_lang": source_lang,
        "resolved_size": resolved_size,
        "engine": "faster-whisper",
    }
    data = {
        "language": result.language,
        "language_probability": result.language_probability,
        "duration": result.duration,
        "segments": [
            {
                "index": seg.index,
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "words": [
                    {"text": w.text, "start": w.start, "end": w.end} for w in seg.words
                ],
            }
            for seg in result.segments
        ],
    }

    (entry / _META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (entry / _DATA_FILE).write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("已写入 ASR 缓存：%s（%d 条原始分段）", entry.name, len(result.segments))
    return entry


def clear_asr_cache(cache_dir: Path | str) -> int:
    """清空缓存目录，返回删除的条目数（供 --no-resume 场景或手动清理）。"""
    root = Path(cache_dir)
    if not root.is_dir():
        return 0
    removed = 0
    for entry in root.iterdir():
        if entry.is_dir() and (entry / _META_FILE).is_file():
            for child in entry.iterdir():
                child.unlink(missing_ok=True)
            entry.rmdir()
            removed += 1
    return removed
