"""端到端编排（设计大纲 §5）。

流程：文件 → 抽音轨 → 源语识别 → 二次断句 → 翻译 → 合并 → 写盘。
M1 起逐步填充；M0 仅定义配置与结果模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .subtitle import SubtitleDoc


@dataclass(slots=True)
class PipelineConfig:
    input_path: Path
    source_lang: str = "auto"
    target_lang: str | None = None  # None 表示只出源语字幕
    asr_model: str = "small"
    translator: str = "nllb"
    output_dir: Path | None = None
    emit_ass: bool = False


@dataclass(slots=True)
class PipelineResult:
    doc: SubtitleDoc
    outputs: list[Path] = field(default_factory=list)


ProgressCallback = Callable[[str, float], None]
"""回调签名：``(阶段名, 0.0~1.0 进度)``，供 GUI 显示。"""


def run(config: PipelineConfig, on_progress: ProgressCallback | None = None) -> PipelineResult:
    """执行完整管线。M1 实现。"""
    raise NotImplementedError("M1：管线编排待实现")
