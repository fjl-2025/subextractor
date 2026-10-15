"""目录批处理与任务队列（M5）。

职责：
- ``collect_media_files``：把输入路径（文件/目录）展开成待处理媒体清单；
- ``run_batch``：按顺序或有限并发执行一批管线任务，逐项收集成败，单文件
  失败不中断批次，取消在文件边界生效。

顺序模式（concurrency=1，默认）下，若各任务的识别/翻译配置一致，则整批
复用同一组 ASR / 翻译引擎——模型只加载一次，20 个文件可省去逐个重载的
数分钟开销；引擎对象惰性加载，全部命中断点缓存时不会真正载入模型。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..utils.logging import get_logger
from . import asr, pipeline
from .errors import PipelineCancelled
from .subtitle import SubtitleDoc
from .translate import Translator, create_translator

logger = get_logger(__name__)

SUPPORTED_MEDIA_EXTS: frozenset[str] = frozenset(
    {
        ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".flv", ".wmv",
        ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",
    }
)

Runner = Callable[..., pipeline.PipelineResult]
"""与 :func:`pipeline.run` 同签名的可替换执行器（测试可注入 stub）。"""


@dataclass(slots=True)
class BatchItemResult:
    """单个文件的处理结果。"""

    input_path: Path
    outputs: list[Path] = field(default_factory=list)
    cues: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class BatchResult:
    """整批结果：成功/失败清单 + 是否被取消。"""

    succeeded: list[BatchItemResult] = field(default_factory=list)
    failed: list[BatchItemResult] = field(default_factory=list)
    cancelled: bool = False

    @property
    def total(self) -> int:
        return len(self.succeeded) + len(self.failed)


ItemStarted = Callable[[int, int, Path], None]
"""``(序号 1 起, 总数, 输入路径)``。"""

ItemProgress = Callable[[int, int, str, float], None]
"""``(序号, 总数, 阶段名, 0.0~1.0 进度)``；并发模式下在 worker 线程回调。"""

ItemDone = Callable[[BatchItemResult, int, int, SubtitleDoc | None], None]
"""``(单项结果, 序号, 总数, 成功项的字幕文档)``；并发模式下在 worker 线程回调。"""


def collect_media_files(root: Path) -> list[Path]:
    """把输入路径展开成媒体清单：文件原样返回；目录按扩展名过滤、按文件名排序。

    目录里没有可识别的媒体文件时抛 ``ValueError``，路径不存在抛 ``FileNotFoundError``。
    """
    source = Path(root)
    if source.is_file():
        return [source]
    if source.is_dir():
        files = [
            p
            for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_MEDIA_EXTS
        ]
        if not files:
            raise ValueError(f"目录中没有可处理的媒体文件：{source}")
        return sorted(files, key=lambda p: p.name.lower())
    raise FileNotFoundError(f"输入路径不存在：{source}")


def _asr_settings(config: pipeline.PipelineConfig) -> tuple[str, ...]:
    """影响共享引擎复用与否的识别/翻译配置项。"""
    return (
        config.asr_model,
        config.source_lang,
        config.device,
        config.compute_type,
        config.translator,
    )


def _make_item(
    config: pipeline.PipelineConfig,
    result: pipeline.PipelineResult | None,
    exc: Exception | None,
) -> tuple[BatchItemResult, SubtitleDoc | None]:
    if exc is not None:
        message = str(exc).strip() or exc.__class__.__name__
        return BatchItemResult(input_path=config.input_path, error=message), None
    assert result is not None
    item = BatchItemResult(
        input_path=config.input_path, outputs=result.outputs, cues=len(result.doc.segments)
    )
    return item, result.doc


def run_batch(
    configs: list[pipeline.PipelineConfig],
    *,
    concurrency: int = 1,
    should_cancel: Callable[[], bool] | None = None,
    on_item_started: ItemStarted | None = None,
    on_item_progress: ItemProgress | None = None,
    on_item_done: ItemDone | None = None,
    runner: Runner | None = None,
    asr_engine: asr.AsrEngine | None = None,
    translator: Translator | None = None,
) -> BatchResult:
    """执行一批管线任务，逐项收集成败，单文件异常不中断整批。

    - 顺序模式（``concurrency <= 1``）：识别配置一致的任务共享同一组引擎；
      取消在文件边界生效（当前文件完成或中止后不再开始下一个）。
    - 并发模式（``concurrency > 1``）：线程池并发，各任务自建引擎；
      不支持共享引擎（传入即抛 ``ValueError``）。进度/完成回调会在 worker
      线程触发，调用方需自行保证线程安全（Qt 信号为队列连接，天然安全）。
    """
    if not configs:
        raise ValueError("批处理任务列表为空")
    if concurrency > 1 and (asr_engine is not None or translator is not None):
        raise ValueError("并发模式下不支持共享引擎，请去掉 asr_engine / translator 参数")

    run_one: Runner = runner or pipeline.run
    total = len(configs)
    result = BatchResult()

    def progress_for(index: int) -> Callable[[str, float], None] | None:
        if on_item_progress is None:
            return None

        def callback(stage: str, progress: float) -> None:
            on_item_progress(index, total, stage, progress)

        return callback

    if concurrency <= 1:
        shareable = len({_asr_settings(c) for c in configs}) == 1
        shared_asr = asr_engine
        shared_tr = translator
        own_translator = False
        if shareable and (shared_asr is None or shared_tr is None):
            first = configs[0]
            if shared_asr is None:
                shared_asr = asr.AsrEngine(
                    asr.AsrConfig(
                        model_size=first.asr_model,
                        source_lang=first.source_lang,
                        device=first.device,
                        compute_type=first.compute_type,
                    )
                )
            if shared_tr is None:
                shared_tr = create_translator(
                    first.translator, device=first.device, compute_type=first.compute_type
                )
                own_translator = True

        try:
            for index, config in enumerate(configs, start=1):
                if should_cancel and should_cancel():
                    result.cancelled = True
                    break
                if on_item_started:
                    on_item_started(index, total, config.input_path)
                try:
                    pr = run_one(
                        config,
                        on_progress=progress_for(index),
                        should_cancel=should_cancel,
                        asr_engine=shared_asr,
                        translator=shared_tr,
                    )
                except PipelineCancelled:
                    result.cancelled = True
                    break
                except Exception as exc:  # noqa: BLE001 —— 单项失败收集后继续
                    logger.warning("批处理单项失败：%s（%s）", config.input_path.name, exc)
                    item, doc = _make_item(config, None, exc)
                    result.failed.append(item)
                else:
                    item, doc = _make_item(config, pr, None)
                    result.succeeded.append(item)
                if on_item_done:
                    on_item_done(item, index, total, doc)
        finally:
            if own_translator and shared_tr is not None:
                shared_tr.close()
        return result

    def task(index: int, config: pipeline.PipelineConfig) -> pipeline.PipelineResult:
        if on_item_started:
            on_item_started(index, total, config.input_path)
        return run_one(
            config,
            on_progress=progress_for(index),
            should_cancel=should_cancel,
            asr_engine=None,
            translator=None,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(task, index, config): (index, config)
            for index, config in enumerate(configs, start=1)
        }
        for future in as_completed(futures):
            index, config = futures[future]
            try:
                pr = future.result()
            except PipelineCancelled:  # 取消后未启动/中止的任务都归入 cancelled
                result.cancelled = True
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("批处理单项失败：%s（%s）", config.input_path.name, exc)
                item, doc = _make_item(config, None, exc)
                result.failed.append(item)
            else:
                item, doc = _make_item(config, pr, None)
                result.succeeded.append(item)
            if on_item_done:
                on_item_done(item, index, total, doc)

    return result
