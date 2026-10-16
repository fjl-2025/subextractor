"""端到端编排（设计大纲 §5）。

管线顺序（M3 起）：
    抽音轨 → 源语识别 → 断句切分 → 翻译 → 写出

断句必须插在识别与翻译之间，译文才能与最终 cue 一一对应；
``name.orig.srt`` 反映的也是切分后的最终分段（而非原始 ASR 分段）。

M5 起支持两类复用：
- ASR 断点缓存（``cache_dir``）：输入指纹与识别配置未变时跳过抽音轨+识别；
- 共享引擎（``asr_engine`` / ``translator``）：批处理顺序模式下跨文件复用
  模型，外部传入的对象由调用方负责关闭，内部创建的在退出前自动关闭。
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import SUPPORTED_TARGET_LANGS, get_language_params
from ..models import registry
from ..utils.logging import get_logger
from . import asr, cache, glossary, media, ocr, postcheck, segment, writer
from .errors import PipelineCancelled
from .subtitle import SubtitleDoc
from .translate import Translator, create_translator

logger = get_logger(__name__)


@dataclass(slots=True)
class PipelineConfig:
    input_path: Path
    source_lang: str = "auto"
    target_lang: str | None = None  # None 表示只出源语字幕
    asr_model: str = registry.AUTO
    device: str = "auto"
    compute_type: str = registry.AUTO
    translator: str = "nllb"
    output_dir: Path | None = None
    emit_ass: bool = False
    glossary_path: Path | None = None  # 术语表（TSV），对译文做后处理替换
    cache_dir: Path | None = None  # ASR 断点缓存目录；None 表示不启用
    ocr_config: ocr.OcrConfig | None = None  # 非空时走画面硬字幕 OCR，跳过音轨+ASR

    def output_path_for(self, suffix: str) -> Path:
        """输出路径：默认与输入文件同目录、同名词。"""
        base = self.output_dir or self.input_path.parent
        return base / f"{self.input_path.stem}{suffix}"


@dataclass(slots=True)
class PipelineResult:
    doc: SubtitleDoc
    outputs: list[Path] = field(default_factory=list)
    asr_model: str | None = None  # 自动档位实际解析出的档位
    issues: list[postcheck.Issue] = field(default_factory=list)  # postcheck 报告，供 GUI 展示
    # 与视频同名的双语成品（name.srt / name.ass）：播放器打开视频时自动外挂
    main_output: Path | None = None


ProgressCallback = Callable[[str, float], None]
"""回调签名：``(阶段名, 0.0~1.0 进度)``，供 GUI 显示。"""

CancelChecker = Callable[[], bool]
"""取消检查：返回 True 时管线在最近的检查点抛 ``PipelineCancelled``。"""


def run(
    config: PipelineConfig,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelChecker | None = None,
    asr_engine: asr.AsrEngine | None = None,
    translator: Translator | None = None,
) -> PipelineResult:
    """执行管线：抽音轨 → 识别 → 断句 →（可选）翻译 → 写字幕文件。

    阶段边界与识别/翻译循环内都会检查 ``should_cancel``；取消时抛
    ``PipelineCancelled``，临时音轨由调用处的临时目录上下文自动清理。

    ``asr_engine`` / ``translator`` 供批处理复用已加载的模型；未传时内部
    自建（翻译引擎用完即关，外部传入的由调用方负责释放）。
    """

    def notify(stage: str, progress: float) -> None:
        if on_progress:
            on_progress(stage, progress)

    def check_cancelled() -> None:
        if should_cancel and should_cancel():
            raise PipelineCancelled("任务已取消")

    if config.target_lang and config.target_lang not in SUPPORTED_TARGET_LANGS:
        supported = " / ".join(SUPPORTED_TARGET_LANGS)
        raise ValueError(f"目标语 {config.target_lang!r} 不在支持范围：{supported}")

    source = Path(config.input_path)
    if not source.is_file():
        raise FileNotFoundError(f"输入文件不存在：{source}")

    if config.ocr_config is not None:
        # 硬字幕路径：不做音轨抽取与语音识别，也不写 ASR 缓存
        notify("画面字幕提取", 0.0)
        segments, language = ocr.extract_subtitles(
            source,
            config.ocr_config,
            on_progress=lambda stage, progress: notify(stage, 0.05 + 0.75 * progress),
            should_cancel=should_cancel,
        )
        doc = SubtitleDoc(
            source_lang=language,
            target_lang=config.target_lang,
            segments=segments,
        )
        doc.renumber()
        resolved_size = None
        logger.info("画面字幕提取完成：源语 %s，共 %d 条", language, len(doc.segments))
    else:
        if config.cache_dir is not None:
            cached = cache.load_asr_cache(
                config.cache_dir, source, config.asr_model, config.source_lang
            )
        else:
            cached = None

        if cached is not None:
            result = cached.result
            resolved_size = cached.resolved_size
            notify("读取缓存", 0.1)
        else:
            # 音轨是中间产物，用临时目录承载，识别结束即清理（取消时同样清理）
            with tempfile.TemporaryDirectory(prefix="subextractor-") as tmp:
                notify("抽取音轨", 0.0)
                check_cancelled()
                audio_path = media.extract_audio(source, Path(tmp) / "audio.wav")
                logger.info("音轨已抽取：%s", audio_path)

                check_cancelled()
                notify("识别语音", 0.05)
                engine = asr_engine or asr.AsrEngine(
                    asr.AsrConfig(
                        model_size=config.asr_model,
                        source_lang=config.source_lang,
                        device=config.device,
                        compute_type=config.compute_type,
                    )
                )
                result = engine.transcribe(
                    audio_path,
                    on_progress=lambda progress: notify("识别语音", 0.05 + 0.75 * progress),
                    should_cancel=should_cancel,
                )
            resolved_size = engine.resolved_size

            if config.cache_dir is not None:
                try:
                    cache.save_asr_cache(
                        config.cache_dir,
                        source,
                        result,
                        asr_model=config.asr_model,
                        source_lang=config.source_lang,
                        resolved_size=resolved_size,
                    )
                except OSError as exc:  # 缓存只是加速手段，失败不阻断输出
                    logger.warning("ASR 缓存写入失败（不影响本次输出）：%s", exc)

        doc = SubtitleDoc(
            source_lang=result.language,
            target_lang=config.target_lang,
            segments=result.segments,
        )
        doc.renumber()
        logger.info(
            "识别完成：源语 %s（置信度 %.2f），共 %d 条 ASR 分段",
            result.language,
            result.language_probability,
            len(doc.segments),
        )

    # 断句：按源语参数切分 + 规整时间轴（此后的 cue 即最终分段）
    check_cancelled()
    notify("断句切分", 0.82)
    src_params = segment.resolve_params(doc.source_lang)
    raw_count = len(doc.segments)
    doc.segments = segment.segment_document(doc.segments, src_params)
    logger.info("断句完成：%d 条 ASR 分段 → %d 条 cue", raw_count, len(doc.segments))

    issues = postcheck.check(doc, src_params)
    if issues:
        errors = sum(1 for issue in issues if issue.level == "error")
        logger.warning("输出校验：%d 项提示（含 %d 项错误）", len(issues), errors)
        for issue in issues[:5]:
            logger.warning("  [%s] #%d %s", issue.level, issue.index, issue.message)

    translated = False
    if config.target_lang:
        if config.target_lang == doc.source_lang:
            logger.warning("源语与目标语相同（%s），跳过翻译", config.target_lang)
        else:
            check_cancelled()
            notify("翻译文本", 0.86)
            own_translator = translator is None
            engine_tr = translator or create_translator(
                config.translator,
                device=config.device,
                compute_type=config.compute_type,
            )
            try:
                translations = engine_tr.translate(
                    [seg.text for seg in doc.segments],
                    doc.source_lang,
                    config.target_lang,
                    should_cancel=should_cancel,
                )
            finally:
                if own_translator:
                    engine_tr.close()
            if len(translations) != len(doc.segments):
                raise RuntimeError("译文条数与字幕条数不一致，翻译结果不可用")

            if config.glossary_path is not None:
                entries = glossary.load(config.glossary_path)
                if entries:
                    translations = glossary.apply_all(
                        translations, entries, config.target_lang
                    )
                    logger.info("术语表已应用：%d 条规则（%s）", len(entries), config.glossary_path)

            dst_params = get_language_params(config.target_lang)
            for seg, text in zip(doc.segments, translations):
                seg.translation = segment.wrap_text(text or seg.text, dst_params)
            translated = True
            logger.info(
                "翻译完成：%s → %s，共 %d 条", doc.source_lang, config.target_lang, len(translations)
            )
            notify("翻译文本", 0.96)

    check_cancelled()
    notify("写入字幕", 0.97)
    outputs = [writer.write_srt(doc, config.output_path_for(".orig.srt"))]
    main_output: Path | None = None
    if translated:
        lang = config.target_lang
        outputs.append(writer.write_srt(doc, config.output_path_for(f".{lang}.srt"), mode="target"))
        # 双语成品（外+中）：与视频同名的 name.srt，播放器打开视频时自动外挂
        main_path = config.output_path_for(".srt")
        backed_up = writer.backup_existing(main_path)
        if backed_up is not None:
            logger.info("检测到同名外挂字幕，已备份片源字幕：%s", backed_up.name)
        main_output = writer.write_srt(doc, main_path, mode="bilingual")
        outputs.append(main_output)
    if config.emit_ass:
        if translated:
            # 双语 ASS 同样同名输出（name.ass），需要分样式时由播放器优先加载
            ass_path = config.output_path_for(".ass")
            ass_backup = writer.backup_existing(ass_path)
            if ass_backup is not None:
                logger.info("检测到同名 ASS 字幕，已备份片源字幕：%s", ass_backup.name)
            outputs.append(writer.write_ass(doc, ass_path, mode="bilingual"))
        else:
            outputs.append(writer.write_ass(doc, config.output_path_for(".orig.ass"), mode="source"))
    notify("完成", 1.0)

    return PipelineResult(
        doc=doc, outputs=outputs, asr_model=resolved_size, issues=issues, main_output=main_output
    )
