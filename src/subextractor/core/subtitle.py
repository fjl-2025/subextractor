"""字幕数据模型（设计大纲 §7）。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Word:
    """词级时间戳，用于 M3 的精确二次切分。"""

    text: str
    start: float
    end: float


@dataclass(slots=True)
class Segment:
    """一条字幕（一个 cue）。"""

    index: int
    text: str
    start: float
    end: float
    translation: str | None = None
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class SubtitleDoc:
    """一份字幕文档。"""

    source_lang: str
    target_lang: str | None = None
    segments: list[Segment] = field(default_factory=list)

    def renumber(self) -> None:
        """按当前顺序重排编号（切分/合并后调用）。"""
        for i, seg in enumerate(self.segments, start=1):
            seg.index = i
