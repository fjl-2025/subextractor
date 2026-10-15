"""加载配置与语言参数。

语言断句参数存放于 ``config/languages.yaml``，阈值按语言区分——
日语/韩语无空格分词，与 en/fr 的拉丁断行规则完全不同（见设计大纲 §3）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
LANGUAGES_FILE = CONFIG_DIR / "languages.yaml"

# 目标语范围（源语由 ASR 自动检测，不受此限制）
SUPPORTED_TARGET_LANGS: tuple[str, ...] = ("en", "fr", "ja", "ko")


@dataclass(slots=True)
class LanguageParams:
    """单语种的字幕断句与排版参数。"""

    lang: str
    max_chars_per_line: int
    max_lines: int
    min_duration: float
    max_duration: float
    cps_target: float
    no_line_break_before: tuple[str, ...] = ()


def load_languages(path: Path | None = None) -> dict[str, LanguageParams]:
    """读取 languages.yaml，缺省值由 ``defaults`` 段补齐。"""
    source = Path(path) if path else LANGUAGES_FILE
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    defaults = raw.get("defaults") or {}

    result: dict[str, LanguageParams] = {}
    for lang, cfg in (raw.get("languages") or {}).items():
        cfg = cfg or {}
        result[lang] = LanguageParams(
            lang=lang,
            max_chars_per_line=int(cfg["max_chars_per_line"]),
            max_lines=int(cfg.get("max_lines", defaults.get("max_lines", 2))),
            min_duration=float(cfg.get("min_duration", defaults.get("min_duration", 0.8))),
            max_duration=float(cfg.get("max_duration", defaults.get("max_duration", 7.0))),
            cps_target=float(cfg.get("cps_target", defaults.get("cps_target", 17))),
            no_line_break_before=tuple(cfg.get("no_line_break_before") or ()),
        )
    return result


def get_language_params(lang: str) -> LanguageParams:
    langs = load_languages()
    if lang not in langs:
        supported = ", ".join(sorted(langs)) or "（无）"
        raise KeyError(f"未配置的语言 {lang!r}，当前支持：{supported}")
    return langs[lang]
