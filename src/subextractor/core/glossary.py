"""术语表：对译文做后处理替换，不改写翻译引擎（设计大纲 §12 风险对策）。

文件格式（UTF-8，``#`` 开头为注释，空行忽略）::

    # 源词 <TAB> 目标词
    API\t应用程序接口
    変換\t转换

匹配规则按目标语文字体系区分：
- 拉丁语（en/fr）：按词边界匹配、大小写不敏感（``API`` 命中 ``api``，但
  不命中 ``EMAIL`` 里的 ``AI``）；替换结果统一用表中写法。
- 东亚语（ja/ko/zh）：无词边界概念，直接子串替换。

同名条目中较长的源词优先替换，避免短词先替换破坏长词。
"""

from __future__ import annotations

import re
from pathlib import Path

from ..config import get_language_params


def load(path: str | Path) -> list[tuple[str, str]]:
    """读取 TSV 术语表，返回按源词长度降序的 ``(源词, 目标词)`` 列表。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"术语表不存在：{source}")

    entries: list[tuple[str, str]] = []
    for lineno, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(f"术语表第 {lineno} 行格式错误（应为「源词<TAB>目标词」）：{raw!r}")
        entries.append((parts[0].strip(), parts[1].strip()))
    entries.sort(key=lambda item: len(item[0]), reverse=True)
    return entries


def _target_script(dst_lang: str) -> str:
    try:
        return get_language_params(dst_lang).script
    except KeyError:
        return "latin"


def apply(text: str, entries: list[tuple[str, str]], dst_lang: str) -> str:
    """按目标语文字体系对译文应用术语替换。"""
    if not entries or not text:
        return text
    if _target_script(dst_lang) == "ea":
        for src, dst in entries:
            text = text.replace(src, dst)
        return text
    for src, dst in entries:
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text, flags=re.IGNORECASE)
    return text


def apply_all(texts: list[str], entries: list[tuple[str, str]], dst_lang: str) -> list[str]:
    """批量应用（与 ``translate`` 的输出列表对齐）。"""
    return [apply(text, entries, dst_lang) for text in texts]
