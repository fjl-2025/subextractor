"""配置层：默认设置与语言断句参数表（设计大纲 §3 / §8）。"""

from .settings import (
    SUPPORTED_TARGET_LANGS,
    LanguageParams,
    get_language_params,
    load_languages,
)

__all__ = [
    "SUPPORTED_TARGET_LANGS",
    "LanguageParams",
    "get_language_params",
    "load_languages",
]
