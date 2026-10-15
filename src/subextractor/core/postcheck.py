"""输出质量校验（设计大纲 §8）。

M0 已实现基础项（行数 / 行长 / 时长 / 重叠）；
CPS 与语言专项规则（日/韩助词断行）归入 M3。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import LanguageParams
from .subtitle import SubtitleDoc

OVERLAP_EPSILON = 1e-6


@dataclass(slots=True)
class Issue:
    level: str  # "error" / "warn"
    index: int
    message: str


def check(doc: SubtitleDoc, params: LanguageParams) -> list[Issue]:
    """返回问题清单，空列表表示通过。"""
    issues: list[Issue] = []

    for seg in doc.segments:
        if seg.duration < params.min_duration:
            issues.append(Issue("warn", seg.index, f"时长过短 {seg.duration:.2f}s"))
        elif seg.duration > params.max_duration:
            issues.append(Issue("warn", seg.index, f"时长过长 {seg.duration:.2f}s"))

        lines = seg.text.splitlines() or [seg.text]
        if len(lines) > params.max_lines:
            issues.append(Issue("warn", seg.index, f"行数超限 {len(lines)} > {params.max_lines}"))
        for line in lines:
            if len(line) > params.max_chars_per_line:
                issues.append(
                    Issue("warn", seg.index, f"行超长 {len(line)} > {params.max_chars_per_line}")
                )

    for prev, curr in zip(doc.segments, doc.segments[1:]):
        if curr.start < prev.end - OVERLAP_EPSILON:
            issues.append(Issue("error", curr.index, f"与上一条重叠（上一条结束于 {prev.end:.2f}s）"))

    return issues
