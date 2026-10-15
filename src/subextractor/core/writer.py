"""字幕文件写入（设计大纲 §9）。

输出约定：
- ``name.orig.srt``           源语字幕
- ``name.<lang>.srt``         目标语字幕
- ``name.bilingual.<lang>.srt`` 双语（第 1 行源语、第 2 行译文）

M0 先实现 SRT；VTT / ASS 归入 M6。
"""

from __future__ import annotations

from pathlib import Path

from .subtitle import SubtitleDoc

DEFAULT_ENCODING = "utf-8-sig"


def format_timestamp_srt(seconds: float) -> str:
    """秒 → ``HH:MM:SS,mmm``。"""
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def to_srt_text(doc: SubtitleDoc, *, bilingual: bool = False) -> str:
    """生成 SRT 文本。``bilingual`` 为真时每条字幕追加译文行。"""
    blocks: list[str] = []
    for seq, seg in enumerate(doc.segments, start=1):
        lines = [str(seq), f"{format_timestamp_srt(seg.start)} --> {format_timestamp_srt(seg.end)}"]
        lines.append(seg.text)
        if bilingual and seg.translation:
            lines.append(seg.translation)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_srt(
    doc: SubtitleDoc,
    path: str | Path,
    *,
    bilingual: bool = False,
    encoding: str = DEFAULT_ENCODING,
) -> Path:
    """写出 SRT 文件并返回路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_srt_text(doc, bilingual=bilingual), encoding=encoding)
    return target
