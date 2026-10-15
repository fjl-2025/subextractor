"""离线模型发现与校验（设计大纲 §10）。

约定：
- 根目录：环境变量 ``SUBEXTRACTOR_MODELS_DIR`` 优先，否则仓库根 ``models/``。
- 子目录：``asr/``、``mt/``、``vad/``。
- 运行时只做本地扫描，缺失时报错并给出放置指引，绝不联网回退。
- 首次获取模型用 ``subextractor fetch-model``（需联网，仅一次）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ENV_MODELS_DIR = "SUBEXTRACTOR_MODELS_DIR"
CHECKSUMS_FILE = "checksums.json"
_DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "models"

# faster-whisper 支持的常用档位（多语种档，不含 .en 专用档）
ASR_SIZES: tuple[str, ...] = ("tiny", "base", "small", "medium", "large-v3")

# 自动档位策略：按算力从「本地已装」的档位里挑一个
AUTO = "auto"
GPU_TIER_ORDER: tuple[str, ...] = ("large-v3", "medium", "small", "base", "tiny")
CPU_TIER_ORDER: tuple[str, ...] = ("small", "medium", "base", "tiny")

# 翻译模型（M2）：NLLB-200-distilled-600M 的 CTranslate2 int8 转换版，
# 已含 CT2 权重与 tokenizer 文件，可直接离线加载。
MT_NLLB_DIR = "mt/nllb-200-distilled-600M-ct2-int8"
NLLB_REPO_ID = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
NLLB_FILES: tuple[str, ...] = (
    "config.json",
    "model.bin",
    "sentencepiece.bpe.model",
    "shared_vocabulary.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class MissingModelError(RuntimeError):
    """本地未找到所需的离线模型。"""


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


def has_content(path: Path) -> bool:
    """目录存在且含有效文件（排除 .gitkeep 占位）即视为已放置。"""
    if not path.is_dir():
        return False
    return any(child.name != ".gitkeep" for child in path.iterdir())


def is_available(spec: ModelSpec) -> bool:
    return has_content(spec_dir(spec))


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


def asr_model_dir(size: str) -> Path:
    """faster-whisper 模型的本地目录：``models/asr/faster-whisper-<size>``。"""
    return models_root() / "asr" / f"faster-whisper-{size}"


def resolve_asr_model(size: str) -> Path:
    """返回已放置的本地 ASR 模型目录；缺失时抛 MissingModelError 并给出指引。"""
    path = asr_model_dir(size)
    if not has_content(path):
        raise MissingModelError(
            f"未找到本地 ASR 模型 faster-whisper-{size}。\n"
            f"期望目录：{path}\n"
            f"首次获取（需联网一次）：subextractor fetch-model --size {size}"
        )
    return path


def nllb_model_dir() -> Path:
    """NLLB 翻译模型的本地目录：``models/mt/nllb-200-distilled-600M-ct2-int8``。"""
    return models_root() / MT_NLLB_DIR


def resolve_nllb_model() -> Path:
    """返回已放置的本地 NLLB 模型目录；缺失时抛 MissingModelError 并给出指引。"""
    path = nllb_model_dir()
    if not has_content(path):
        raise MissingModelError(
            f"未找到本地翻译模型 nllb-200-distilled-600M-ct2-int8。\n"
            f"期望目录：{path}\n"
            f"首次获取（需联网一次）：subextractor fetch-model --kind mt"
        )
    return path


def cuda_device_count() -> int:
    """检测可用 CUDA 设备数；无 GPU 版运行库时返回 0。"""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def installed_asr_sizes() -> list[str]:
    """本地已放置的 ASR 档位（按 ASR_SIZES 顺序）。"""
    return [size for size in ASR_SIZES if has_content(asr_model_dir(size))]


def resolve_auto_tier(
    device: str = AUTO,
    *,
    cuda_count: int | None = None,
    installed: list[str] | None = None,
) -> str:
    """按设备算力与本地已装档位挑一个 ASR 档位。

    - ``device=cuda``，或 ``device=auto`` 且检测到 CUDA → GPU 顺序（large-v3 优先）
    - 否则 → CPU 顺序（small 优先，排除 large-v3）

    排除 large-v3 是实测结论：CPU 上约 1.8x 实时，不适合日常使用。
    """
    count = cuda_device_count() if cuda_count is None else cuda_count
    available = installed_asr_sizes() if installed is None else list(installed)
    on_gpu = device == "cuda" or (device == AUTO and count > 0)
    order = GPU_TIER_ORDER if on_gpu else CPU_TIER_ORDER

    for size in order:
        if size in available:
            return size

    hint = "large-v3" if on_gpu else "small"
    raise MissingModelError(
        "本地没有任何 ASR 模型，自动档位无法选择。\n"
        f"首次获取（需联网一次）：subextractor fetch-model --size {hint}"
    )


def resolve_compute_type(device: str = AUTO, *, cuda_count: int | None = None) -> str:
    """自动量化类型：CUDA 用 float16，CPU 用 int8。"""
    count = cuda_device_count() if cuda_count is None else cuda_count
    on_gpu = device == "cuda" or (device == AUTO and count > 0)
    return "float16" if on_gpu else "int8"
