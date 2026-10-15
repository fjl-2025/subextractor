"""离线模型发现与校验（设计大纲 §10）。

约定：
- 根目录：环境变量 ``SUBEXTRACTOR_MODELS_DIR`` 优先，否则仓库根 ``models/``。
- 子目录：``asr/``、``mt/``、``vad/``。
- 启动时只做本地扫描，缺失时给出放置指引，绝不联网回退。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ENV_MODELS_DIR = "SUBEXTRACTOR_MODELS_DIR"
CHECKSUMS_FILE = "checksums.json"
_DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "models"


@dataclass(slots=True)
class ModelSpec:
    id: str
    kind: str  # asr / mt / vad
    dir_name: str
    description: str
    required: bool = True


EXPECTED_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "faster-whisper-small",
        "asr",
        "asr/faster-whisper-small",
        "Whisper small：多语种源语识别（默认档）",
    ),
    ModelSpec(
        "faster-whisper-large-v3",
        "asr",
        "asr/faster-whisper-large-v3",
        "Whisper large-v3：高质量档（需 GPU，可选）",
        required=False,
    ),
    ModelSpec(
        "nllb-200-distilled-600M-ct2-int8",
        "mt",
        "mt/nllb-200-distilled-600M-ct2-int8",
        "NLLB-200 600M int8：en/fr/ja/ko 翻译主力",
    ),
    ModelSpec(
        "argos-langs",
        "mt",
        "mt/argos",
        "Argos 轻量翻译包：备选引擎（可选）",
        required=False,
    ),
    ModelSpec(
        "silero-vad",
        "vad",
        "vad/silero-vad",
        "Silero VAD：静音切分与长音频分片",
    ),
)


def models_root() -> Path:
    """模型根目录，支持环境变量覆盖。"""
    override = os.environ.get(ENV_MODELS_DIR)
    return Path(override) if override else _DEFAULT_ROOT


def expected_models() -> tuple[ModelSpec, ...]:
    return EXPECTED_MODELS


def spec_dir(spec: ModelSpec) -> Path:
    return models_root() / spec.dir_name


def is_available(spec: ModelSpec) -> bool:
    """目录存在且含有效内容（排除 .gitkeep）即视为已放置。"""
    path = spec_dir(spec)
    if not path.is_dir():
        return False
    return any(child for child in path.iterdir() if child.name != ".gitkeep")


def missing_models(include_optional: bool = False) -> list[ModelSpec]:
    return [
        spec
        for spec in EXPECTED_MODELS
        if (include_optional or spec.required) and not is_available(spec)
    ]


def load_checksums() -> dict[str, str]:
    """读取 checksums.json（不存在则返回空表）。"""
    path = models_root() / CHECKSUMS_FILE
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("models") or {})
