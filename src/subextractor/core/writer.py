"""字幕文件写入（设计大纲 §9）。

输出约定：
- ``name.orig.srt``           源语字幕
- ``name.<lang>.srt``         目标语字幕（如 name.zh.srt）
- ``name.srt``                双语成品（第 1 行源语、第 2 行译文）；
                              与视频同名，播放器打开视频时自动作为外挂字幕加载
- ``name.ass``                双语 ASS 成品（勾选输出时）：源语/译文分样式（M6 起
                              按东亚/拉丁文字体系分别设定字体与字号，译文用
                              ``{\rDst}`` 行内覆写）
- 仅源语模式（不翻译）下 ASS 输出为 ``name.orig.ass``

M0 先实现 SRT；M4 起附带基础 ASS；M6 起 ASS 支持东亚/拉丁分样式；
VTT 仍未实现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ..config import get_language_params
from .subtitle import SubtitleDoc

DEFAULT_ENCODING = "utf-8-sig"

# source：仅源语；target：仅译文；bilingual：源语行 + 译文行
WriteMode = Literal["source", "target", "bilingual"]


def backup_path_for(target: Path) -> Path:
    """片源自带字幕的备份名：``name.srt`` → ``name.source.srt``。"""
    return target.with_name(f"{target.stem}.source{target.suffix}")


def backup_existing(target: Path) -> Path | None:
    """写同名成品前保护已有的片源字幕。

    目标文件已存在且备份尚不存在时，把旧文件**移动**为 ``name.source.<ext>``，
    返回备份路径；备份已存在（通常是上次重跑，旧的已被本工具成品占用）时不动，
    返回 ``None``，避免用本工具的产物覆盖最初那份片源字幕。
    """
    if not target.exists():
        return None
    backup = backup_path_for(target)
    if backup.exists():
        return None
    target.rename(backup)
    return backup


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


# 文字体系 → (字体, 源语字号, 译文字号)。东亚文字用 CJK 字体；
# 译文行字号略小并用淡黄区分（双语字幕常见约定）。
_SCRIPT_STYLES: dict[str, tuple[str, int, int]] = {
    "ea": ("Microsoft YaHei", 44, 38),
    "latin": ("Arial", 40, 34),
}

# ASS 颜色为 &HAABBGGRR：源语纯白、译文淡黄
_SRC_COLOR = "&H00FFFFFF"
_DST_COLOR = "&H0000FFFF"

_ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"""

_ASS_EVENTS_HEAD = """
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _script_of(lang: str | None) -> str:
    """语言 → 文字体系（ea / latin）；未配置的语言回退 latin。"""
    if not lang:
        return "latin"
    try:
        return get_language_params(lang).script
    except KeyError:
        return "latin"


def _style_line(name: str, font: str, size: int, color: str) -> str:
    return (
        f"Style: {name},{font},{size},{color},&H000000FF,&H00101010,&H64000000,"
        "0,0,0,0,100,100,0,0,1,2,1,2,60,60,40,1"
    )


def _ass_header(src_script: str, dst_script: str) -> str:
    src = _SCRIPT_STYLES.get(src_script, _SCRIPT_STYLES["latin"])
    dst = _SCRIPT_STYLES.get(dst_script, _SCRIPT_STYLES["latin"])
    styles = [
        _style_line("Src", src[0], src[1], _SRC_COLOR),
        _style_line("Dst", dst[0], dst[2], _DST_COLOR),
    ]
    return _ASS_HEAD + "\n".join(styles) + "\n" + _ASS_EVENTS_HEAD


def to_ass_text(doc: SubtitleDoc, *, mode: WriteMode = "source") -> str:
    """生成 ASS 文本，事件行与 SRT 同约定（source / target / bilingual）。

    双语模式：源语行用 ``Src`` 样式，译文行以 ``{\\rDst}`` 行内切换到
    ``Dst`` 样式；两种样式的字体与字号按各自语言的文字体系（东亚/拉丁）
    分别设定。
    """
    src_script = _script_of(doc.source_lang)
    dst_script = _script_of(doc.target_lang)
    events: list[str] = []
    for seq, seg in enumerate(doc.segments, start=1):
        style = "Src"
        if mode == "target":
            style = "Dst"
            text = seg.translation or seg.text
        else:
            text = seg.text
            if mode == "bilingual" and seg.translation:
                text += f"\\N{{\\rDst}}{seg.translation}"
        text = text.replace("\n", "\\N")
        events.append(
            f"Dialogue: 0,{format_timestamp_ass(seg.start)},{format_timestamp_ass(seg.end)},"
            f"{style},,0,0,0,,{text}"
        )
    return _ass_header(src_script, dst_script) + "\n".join(events) + ("\n" if events else "")


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
