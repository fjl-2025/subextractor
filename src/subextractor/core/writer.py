"""字幕文件写入（设计大纲 §9）。

输出约定：
- ``name.orig.srt``           源语字幕
- ``name.<lang>.srt``         目标语字幕
- ``name.bilingual.<lang>.srt`` 双语（第 1 行源语、第 2 行译文）

M0 先实现 SRT；M4 起附带基础 ASS（单一样式，用于 GUI「同时输出 ASS」）；
VTT 与东亚/拉丁分样式仍归入 M6。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .subtitle import SubtitleDoc

DEFAULT_ENCODING = "utf-8-sig"

# source：仅源语；target：仅译文；bilingual：源语行 + 译文行
WriteMode = Literal["source", "target", "bilingual"]


def format_timestamp_srt(seconds: float) -> str:
    """秒 → ``HH:MM:SS,mmm``。"""
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def to_srt_text(doc: SubtitleDoc, *, mode: WriteMode = "source") -> str:
    """生成 SRT 文本。

    - ``source``    ：每条只写源语
    - ``target``    ：每条只写译文（缺译文时回退源语）
    - ``bilingual`` ：每条写两行，第一行源语、第二行译文
    """
    blocks: list[str] = []
    for seq, seg in enumerate(doc.segments, start=1):
        lines = [str(seq), f"{format_timestamp_srt(seg.start)} --> {format_timestamp_srt(seg.end)}"]
        if mode == "target":
            lines.append(seg.translation or seg.text)
        else:
            lines.append(seg.text)
            if mode == "bilingual" and seg.translation:
                lines.append(seg.translation)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_srt(
    doc: SubtitleDoc,
    path: str | Path,
    *,
    mode: WriteMode = "source",
    encoding: str = DEFAULT_ENCODING,
) -> Path:
    """写出 SRT 文件并返回路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_srt_text(doc, mode=mode), encoding=encoding)
    return target


def format_timestamp_ass(seconds: float) -> str:
    """秒 → ``H:MM:SS.cc``（ASS 时间戳，厘秒精度）。"""
    total_cs = int(round(max(seconds, 0.0) * 100))
    hours, rem = divmod(total_cs, 360_000)
    minutes, rem = divmod(rem, 6_000)
    secs, cs = divmod(rem, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{cs:02d}"


# 基础样式：720p、底部居中、白字黑边；字体按 Windows 自用场景选微软雅黑
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,44,&H00FFFFFF,&H000000FF,&H00101010,&H64000000,0,0,0,0,100,100,0,0,1,2,1,2,60,60,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def to_ass_text(doc: SubtitleDoc, *, mode: WriteMode = "source") -> str:
    """生成 ASS 文本，事件行与 SRT 同约定（source / target / bilingual）。"""
    events: list[str] = []
    for seq, seg in enumerate(doc.segments, start=1):
        if mode == "target":
            text = seg.translation or seg.text
        else:
            text = seg.text
            if mode == "bilingual" and seg.translation:
                text += f"\\N{seg.translation}"
        text = text.replace("\n", "\\N")
        events.append(
            f"Dialogue: 0,{format_timestamp_ass(seg.start)},{format_timestamp_ass(seg.end)},"
            f"Default,,0,0,0,,{text}"
        )
    return ASS_HEADER + "\n".join(events) + ("\n" if events else "")


def write_ass(
    doc: SubtitleDoc,
    path: str | Path,
    *,
    mode: WriteMode = "source",
    encoding: str = DEFAULT_ENCODING,
) -> Path:
    """写出 ASS 文件并返回路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_ass_text(doc, mode=mode), encoding=encoding)
    return target
