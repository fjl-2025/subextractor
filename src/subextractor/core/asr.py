"""源语识别：faster-whisper 封装（设计大纲 §4）。

M1 实现。设计要点：
- 支持模型档位（tiny / base / small / medium / large-v3），int8 量化以适配 CPU。
- 源语 auto 或强制指定（强制指定可规避中日/日英混淆）。
- 输出词级时间戳，供 segment 做二次切分。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .subtitle import Segment


@dataclass(slots=True)
class AsrConfig:
    model_size: str = "small"
    source_lang: str = "auto"  # auto 或 en / fr / ja / ko / ...
    word_timestamps: bool = True
    device: str = "auto"  # auto / cpu / cuda
    compute_type: str = "int8"


class AsrEngine:
    """faster-whisper 封装，M1 实现。"""

    def __init__(self, config: AsrConfig | None = None, model_dir: Path | None = None) -> None:
        self.config = config or AsrConfig()
        self.model_dir = model_dir
        self._model = None

    def load(self) -> None:
        raise NotImplementedError("M1：ASR 模型加载待实现")

    def transcribe(self, audio_path: str | Path) -> list[Segment]:
        raise NotImplementedError("M1：源语识别待实现")
