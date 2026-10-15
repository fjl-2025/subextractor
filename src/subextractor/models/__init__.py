"""模型层：离线模型发现与校验（设计大纲 §10）。"""

from .registry import (
    ModelSpec,
    expected_models,
    is_available,
    load_checksums,
    missing_models,
    models_root,
    spec_dir,
)

__all__ = [
    "ModelSpec",
    "expected_models",
    "is_available",
    "load_checksums",
    "missing_models",
    "models_root",
    "spec_dir",
]
