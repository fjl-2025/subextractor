"""源语识别：faster-whisper 封装（设计大纲 §4）。

设计要点：
- 模型档位 tiny / base / small / medium / large-v3；默认 auto 按算力自动选档
  （无 CUDA 优先 small，有 CUDA 优先 large-v3），量化类型同步自动（CPU=int8 / CUDA=float16）。
- 源语 auto 或强制指定（强制指定可规避中日/日英混淆）。
- 输出词级时间戳，供 segment 做二次切分（M3）。
- 模型只从本地目录加载（``local_files_only=True``），保证运行时离线。
- 音频自行解码为 ndarray 后传入，绕开 faster-whisper 内部的 PyAV 解码路径。
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..models import registry
from ..utils.logging import get_logger
from .errors import PipelineCancelled
from .subtitle import Segment, Word

logger = get_logger(__name__)

TranscribeProgress = Callable[[float], None]
"""识别进度回调：``(0.0~1.0)``。"""


@dataclass(slots=True)
class AsrConfig:
    model_size: str = registry.AUTO  # auto 或 tiny / base / small / medium / large-v3
    source_lang: str = "auto"  # auto 或 en / fr / ja / ko / ...
    word_timestamps: bool = True
    device: str = "auto"  # auto / cpu / cuda
    compute_type: str = registry.AUTO  # auto 或 int8 / float16 / int8_float16
    beam_size: int = 5
    vad_filter: bool = True


@dataclass(slots=True)
class AsrResult:
    segments: list[Segment]
    language: str
    language_probability: float
    duration: float | None = None


def load_audio(audio_path: str | Path, sample_rate: int = 16_000) -> np.ndarray:
    """把 16k 单声道 PCM wav 读成 float32 数组（值域 -1~1）。

    绕过 faster-whisper 自身的 PyAV 解码：其 1.2.1 版本调用
    ``av.open(..., metadata_errors="ignore")``，而 PyAV 19.0.1 的 ``av.open``
    没有该参数，会直接抛 TypeError。传入 ndarray 时 faster-whisper
    不会走 decode_audio 分支，因此与 PyAV 版本解耦。
    """
    path = Path(audio_path)
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != sample_rate or handle.getnchannels() != 1:
            raise ValueError(
                f"需要 {sample_rate}Hz 单声道 PCM wav，实际为 "
                f"{handle.getframerate()}Hz / {handle.getnchannels()} 声道：{path}"
            )
        frames = handle.readframes(handle.getnframes())

    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


class AsrEngine:
    """faster-whisper 封装；模型只从本地目录加载，保证离线。"""

    def __init__(self, config: AsrConfig | None = None, model_path: Path | None = None) -> None:
        self.config = config or AsrConfig()
        self.model_path = Path(model_path) if model_path else None
        self._model = None
        self.resolved_size: str | None = None

    def load(self) -> None:
        """加载本地模型（惰性，只执行一次）。

        ``model_size`` / ``compute_type`` 为 auto 时按设备算力自动选择。
        """
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        size = self.config.model_size
        if size == registry.AUTO:
            size = registry.resolve_auto_tier(self.config.device)
        compute_type = self.config.compute_type
        if compute_type == registry.AUTO:
            compute_type = registry.resolve_compute_type(self.config.device)

        path = self.model_path or registry.resolve_asr_model(size)
        logger.info(
            "ASR 档位 %s｜device=%s｜compute_type=%s", size, self.config.device, compute_type
        )
        self._model = WhisperModel(
            str(path),
            device=self.config.device,
            compute_type=compute_type,
            local_files_only=True,
        )
        self.resolved_size = size

    def transcribe(
        self,
        audio_path: str | Path,
        on_progress: TranscribeProgress | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AsrResult:
        """识别音频，返回带时间轴的分段（可选词级时间戳）。

        ``should_cancel`` 每产出一个分段检查一次，为真时抛 ``PipelineCancelled``。
        """
        self.load()

        audio = load_audio(audio_path)
        language = None if self.config.source_lang == "auto" else self.config.source_lang
        raw_segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=self.config.beam_size,
            word_timestamps=self.config.word_timestamps,
            vad_filter=self.config.vad_filter,
        )

        total = info.duration or 0.0
        segments: list[Segment] = []
        for raw in raw_segments:
            if should_cancel and should_cancel():
                raise PipelineCancelled("识别已取消")
            text = raw.text.strip()
            if not text:
                continue

            words = [
                Word(text=word.word.strip(), start=word.start, end=word.end)
                for word in (raw.words or [])
                if word.start is not None and word.end is not None
            ]
            segments.append(
                Segment(
                    index=len(segments) + 1,
                    text=text,
                    start=raw.start,
                    end=raw.end,
                    words=words,
                )
            )

            if on_progress and total > 0:
                on_progress(min(raw.end / total, 1.0))

        if on_progress:
            on_progress(1.0)

        return AsrResult(
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
        )
