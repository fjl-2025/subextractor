"""音轨抽取与媒体探测（设计大纲 §5）。

FFmpeg 二进制优先取 imageio-ffmpeg 内置版本，避免要求用户自行安装。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

AUDIO_SAMPLE_RATE = 16_000
AUDIO_CHANNELS = 1


def ffmpeg_path() -> str | None:
    """返回可用的 ffmpeg 路径；找不到时返回 None。"""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def probe(path: str | Path) -> dict[str, object]:
    """探测媒体信息（时长、音轨等）。M1 实现。"""
    raise NotImplementedError("M1：媒体探测待实现")


def extract_audio(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> Path:
    """抽取音轨为 16k 单声道 wav。M1 实现。"""
    raise NotImplementedError("M1：音轨抽取待实现")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)
