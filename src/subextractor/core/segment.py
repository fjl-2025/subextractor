"""按语言参数二次断句 + 时间轴规整（设计大纲 §8）。

M3 实现。M0 仅提供入口与规则常量，参数来源见 config/languages.yaml。
"""

from __future__ import annotations

from ..config import LanguageParams
from .subtitle import Segment

MIN_CUE_DURATION = 0.8
MAX_CUE_DURATION = 7.0
MIN_CUE_GAP = 0.0  # 相邻 cue 允许的最小间隔（0 表示允许紧邻但不得重叠）


def split_segments(segments: list[Segment], params: LanguageParams) -> list[Segment]:
    """把过长 cue 按语言字符预算拆分为多条。M3 实现。"""
    raise NotImplementedError("M3：断句切分待实现")


def normalize_timeline(segments: list[Segment], params: LanguageParams) -> list[Segment]:
    """修剪重叠、合并过短 cue、压缩过长 cue。M3 实现。"""
    raise NotImplementedError("M3：时间轴规整待实现")
