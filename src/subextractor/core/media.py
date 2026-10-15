"""音轨抽取与媒体探测（设计大纲 §5）。

FFmpeg 二进制优先取 imageio-ffmpeg 内置版本，避免要求用户自行安装。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

AUDIO_SAMPLE_RATE = 16_000
AUDIO_CHANNELS = 1

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


@dataclass(slots=True)
class MediaInfo:
    path: Path
    duration: float | None
    has_audio: bool
    has_video: bool


def ffmpeg_path() -> str | None:
    """返回可用的 ffmpeg 路径；找不到时返回 None。"""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _require_ffmpeg() -> str:
    path = ffmpeg_path()
    if not path:
        raise RuntimeError(
            "未找到 ffmpeg：请 pip install imageio-ffmpeg，或把 ffmpeg 放入 PATH"
        )
    return path


def probe(path: str | Path) -> MediaInfo:
    """探测时长与音/视频流（解析 ffmpeg 的 stderr 输出）。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"输入文件不存在：{source}")

    ffmpeg = _require_ffmpeg()
    # 未指定输出文件时 ffmpeg 以非 0 退出码结束，这是预期行为，故 check=False
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(source)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    stderr = result.stderr or ""

    duration: float | None = None
    if match := _DURATION_RE.search(stderr):
        hours, minutes, seconds = match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    info = MediaInfo(
        path=source,
        duration=duration,
        has_audio="Audio:" in stderr,
        has_video="Video:" in stderr,
    )
    if not info.has_audio and not info.has_video:
        raise RuntimeError(f"无法解析媒体文件（未识别到音视频流）：{source}")
    return info


def extract_audio(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> Path:
    """抽取音轨为 16k 单声道 PCM wav，返回 wav 路径。

    ``output_path`` 为空时写入新建的临时目录，由调用方负责清理。
    """
    info = probe(input_path)
    if not info.has_audio:
        raise ValueError(f"文件不含音轨，无法识别语音：{info.path}")

    if output_path is None:
        target = Path(tempfile.mkdtemp(prefix="subextractor-")) / "audio.wav"
    else:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

    _run(
        [
            _require_ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(info.path),
            "-vn",
            "-ac",
            str(AUDIO_CHANNELS),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(target),
        ]
    )
    if not target.is_file():
        raise RuntimeError(f"音轨抽取失败，未生成文件：{target}")
    return target


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)
