"""输出质量校验（设计大纲 §8）。

覆盖项：行数 / 行长 / 时长 / 重叠 / CPS / 助词行首（日韩专项）。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import LanguageParams
from . import segment
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

        # CPS：每秒字符数（去掉换行后的源语文本）
        if seg.duration > 0:
            chars = len(seg.text.replace("\n", ""))
            cps = chars / seg.duration
            if cps > params.cps_target:
                issues.append(
                    Issue("warn", seg.index, f"CPS 超标 {cps:.1f} > {params.cps_target:g}")
                )

        # 助词/助动词不得作为新行开头（日/韩专项；与折行一致按整单元精确匹配）
        if params.no_line_break_before:
            for line in lines[1:]:
                head = segment.first_unit(line, params.lang)
                if head in params.no_line_break_before:
                    issues.append(Issue("warn", seg.index, f"行首助词 {head!r}"))

    for prev, curr in zip(doc.segments, doc.segments[1:]):
        if curr.start < prev.end - OVERLAP_EPSILON:
            issues.append(Issue("error", curr.index, f"与上一条重叠（上一条结束于 {prev.end:.2f}s）"))

    return issues
