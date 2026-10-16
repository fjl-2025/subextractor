"""按语言参数二次断句 + 时间轴规整（设计大纲 §8）。

设计要点（M3）：
- 切分只落在「词边界」，依据 ASR 提供的词级时间戳（``Segment.words``）。
- 断句信号按语言区分：
  * ja / ko：日文与韩文无标点（Whisper 日文尤其不加标点），以**词间停顿**
    （相邻词的间隙）为主信号；不强行给日文补句号（日语字幕惯例本就省略句号）。
  * en / fr：以**句读标点**为主信号，停顿为辅。
- 约束来自 ``config/languages.yaml`` 的 :class:`LanguageParams`：
  单 cue ≤ ``max_lines`` 行、每行 ≤ ``max_chars_per_line``、时长在
  ``min_duration`` ~ ``max_duration`` 之间。
- 双向：超预算的拆、过短的合并；相邻 cue 不重叠（修剪至前一条的 end）。
- 拆出的子 cue 继承并重排 ``words``；``words`` 缺失时按字符比例估时回退。
- 日文专项：Whisper 的日语词级时间戳实为**逐字符级**，词边界不可用；改用
  janome 形态素分析近似「文节」边界（见 :func:`_ja_units`），再据此切分/折行。
"""

from __future__ import annotations

import math

from ..config import LanguageParams, load_languages
from ..utils.logging import get_logger
from .subtitle import Segment, Word

logger = get_logger(__name__)

# 词间停顿阈值（秒）：达到即视为一个可断的停顿
PAUSE_SHORT = 0.25
PAUSE_LONG = 0.50
# 句读标点：强断点（句末）与弱断点（句中）
STRONG_BREAK_CHARS = "。！？.!?"
WEAK_BREAK_CHARS = "，、；：,;:"

# 附着到前一文节的词性（助詞/助動詞/接尾辞/記号），形成「文节」近似
_JA_ATTACH_POS = frozenset({"助詞", "助動詞", "接尾辞", "記号"})
_JA_TOKENIZER: object | None = None
_JA_TOKENIZER_FAILED = False

# 未在 languages.yaml 中配置的语种（源语自动识别为其他语种时）使用的通用参数
DEFAULT_PARAMS = LanguageParams(
    lang="und",
    max_chars_per_line=42,
    max_lines=2,
    min_duration=0.8,
    max_duration=7.0,
    cps_target=17.0,
)


def resolve_params(lang: str) -> LanguageParams:
    """取语言的断句参数；未配置的语种回退到通用默认值。"""
    try:
        return load_languages()[lang]
    except KeyError:
        return LanguageParams(
            lang=lang,
            max_chars_per_line=DEFAULT_PARAMS.max_chars_per_line,
            max_lines=DEFAULT_PARAMS.max_lines,
            min_duration=DEFAULT_PARAMS.min_duration,
            max_duration=DEFAULT_PARAMS.max_duration,
            cps_target=DEFAULT_PARAMS.cps_target,
            no_line_break_before=(),
        )


# 中文译文标点归一化表：NLLB 输出中文时用半角标点，需按简体字幕规范转全角
_CJK_PUNCT_MAP = {",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "："}


def normalize_cjk_punctuation(text: str) -> str:
    """把中文译文里的半角标点转成全角（``，。？！；：``）。

    保护两类例外，避免破坏原意：
    - 数字内部：``3.5`` / ``1,000`` / ``10:30``
    - 英文缩写：``U.S.A`` / ``e.g.``
    """
    out: list[str] = []
    for index, char in enumerate(text):
        repl = _CJK_PUNCT_MAP.get(char)
        if repl is None:
            out.append(char)
            continue
        prev = text[index - 1] if index else ""
        nxt = text[index + 1] if index + 1 < len(text) else ""
        both_digit = prev.isdigit() and nxt.isdigit()
        if both_digit and prev.isascii() and nxt.isascii():
            out.append(char)  # 数字内部：小数点 / 千分位 / 时刻
            continue
        if char == "." and prev.isascii() and prev.isalpha() and nxt.isascii() and nxt.isalpha():
            out.append(char)  # 英文缩写（仅 ASCII 字母之间，汉字 isalpha 为真需排除）
            continue
        out.append(repl)
    return "".join(out)


def segment_document(segments: list[Segment], params: LanguageParams) -> list[Segment]:
    """断句主入口：切分 → 规整 → 重排编号。"""
    out = split_segments(segments, params)
    out = normalize_timeline(out, params)
    for index, seg in enumerate(out, start=1):
        seg.index = index
    return out


def split_segments(segments: list[Segment], params: LanguageParams) -> list[Segment]:
    """把每条 ASR 分段按语言字符预算与时长上限拆成若干条 cue。"""
    out: list[Segment] = []
    for seg in segments:
        if seg.words:
            out.extend(_split_with_words(seg, params))
        else:
            out.extend(_split_without_words(seg, params))
    return out


def normalize_timeline(segments: list[Segment], params: LanguageParams) -> list[Segment]:
    """丢弃重复 ASR 片段、修剪重叠、合并过短 cue、压缩过长 cue。"""
    segs = _drop_duplicate_asr_segments(segments)
    segs = _trim_overlaps(segs)
    segs = _merge_short(segs, params)
    segs = _split_too_long(segs, params)
    return _trim_overlaps(segs)


# --------------------------------------------------------------------------- #
# 文本处理
# --------------------------------------------------------------------------- #


def wrap_text(text: str, params: LanguageParams) -> str:
    """把一条 cue 的文本折成 ≤ ``max_lines`` 行、每行 ≤ ``max_chars_per_line``。

    换行点优先落在标点之后，其次是空格（拉丁语系）或文节边界（日文），
    并避免行首单元为禁用助词（精确匹配整单元，副词「もう」/名词「もの」
    不会被助词「も」误禁）。候选窗口带 +2 容差（与 :func:`_merge_short`
    的合并容差一致），选点先比溢出字符数、再比与居中点的距离，保证超预算
    1~2 字时仍能落在文节边界上，而不是被钳到词中间。
    有词级信息时应改用 :func:`_wrap_words`，可避免把词拆开。
    """
    cleaned = _clean(text, params.lang)
    limit = params.max_chars_per_line
    if len(cleaned) <= limit:
        return cleaned

    total = len(cleaned)
    tolerance = 2
    lo = max(1, total - limit - tolerance)
    hi = min(limit + tolerance, total - 1)
    target = total / 2

    def key(pos: int) -> tuple[int, float]:
        """先比两行溢出总量（少溢出优先），再比与居中点的距离。"""
        overflow = max(0, pos - limit) + max(0, total - pos - limit)
        return (overflow, abs(pos - target))

    # 日/中无空格分词，按字符模式折行；文节边界仅日文可用（Janome）
    char_mode = params.lang in ("ja", "zh")
    # 日文行首单元 = pos 到下一个文节边界；中文/非日文 = 首个空格分词
    ja_boundaries: set[int] = {end for _, end in _ja_units(cleaned)} if params.lang == "ja" else set()

    def head_after(pos: int) -> str:
        if params.lang == "zh":
            return cleaned[pos]  # 中文以单字为行首单元（与 first_unit 一致）
        if char_mode:
            nxt = min((b for b in ja_boundaries if b > pos), default=total)
            return cleaned[pos:nxt]
        parts = cleaned[pos:].split()
        return parts[0] if parts else cleaned[pos:]

    best: int | None = None

    def consider(pos: int) -> None:
        nonlocal best
        if _is_forbidden_head(head_after(pos), params):
            return
        if best is None or key(pos) < key(best):
            best = pos

    # 第一优先：标点之后（日语无空格，靠标点断句）
    for pos in range(lo, hi + 1):
        if cleaned[pos - 1] in STRONG_BREAK_CHARS or cleaned[pos - 1] in WEAK_BREAK_CHARS:
            consider(pos)

    # 第二优先：空格（拉丁语系）或文节边界（日文，避免折在词/助词中间）
    if best is None:
        for pos in range(lo, hi + 1):
            if char_mode:
                if ja_boundaries and pos not in ja_boundaries:
                    continue
            else:
                if cleaned[pos - 1] != " " and cleaned[pos] != " ":
                    continue
            consider(pos)

    # 第三优先（日/中字符模式且无可用边界）：任意字符位置
    if best is None and char_mode:
        for pos in range(lo, hi + 1):
            consider(pos)

    if best is None:
        # 容差带内全被禁用规则拦掉时，退回严格窗口右端（硬约束优先）
        best = min(limit, total - 1)

    return cleaned[:best].strip() + "\n" + cleaned[best:].strip()


def _wrap_words(words: list, params: LanguageParams, sep: str) -> str:
    """按词边界折行：优先落在词与词之间，避免把一个词拆到两行（日文尤为关键）。"""
    text = _join_words(words, sep)
    limit = params.max_chars_per_line
    if len(text) <= limit:
        return text

    total = len(text)
    tolerance = 2
    lo = max(1, total - limit - tolerance)
    hi = min(limit + tolerance, total - 1)
    target = total / 2

    # 每个词结束处在拼接文本中的字符位置（即候选断点）
    positions: list[int] = []
    cursor = 0
    for i, word in enumerate(words):
        cursor += len(word.text) + (0 if i == 0 else len(sep))
        positions.append(cursor)

    def key(pos: int) -> tuple[int, float]:
        overflow = max(0, pos - limit) + max(0, total - pos - limit)
        return (overflow, abs(pos - target))

    best: int | None = None
    for i, pos in enumerate(positions):
        if pos < lo or pos > hi:
            continue
        nxt_unit = words[i + 1].text if i + 1 < len(words) else text[pos:].lstrip()
        if params.lang == "zh":
            nxt_unit = nxt_unit[:1]  # 中文行首单元为单字
        if _is_forbidden_head(nxt_unit.lstrip(), params):
            continue
        if best is None or key(pos) < key(best):
            best = pos

    if best is None:  # 没有合适的词边界，退回字符折行
        return wrap_text(text, params)
    return text[:best].strip() + "\n" + text[best:].strip()


def _clean(text: str, lang: str) -> str:
    """规整空白：日文去掉全部空白，其余语种把连续空白压成单空格。"""
    if lang == "ja":
        return "".join(text.split())
    return " ".join(text.split())


def _detect_sep(words: list, lang: str) -> str:
    """推断词与词之间的拼接分隔符。

    日/中无空格；其余语种若词文本已自带空白（Whisper 拉丁语系词带前导空格）
    则不再补空格，避免出现双空格。
    """
    if lang in ("ja", "zh"):
        return ""
    if any(word.text[:1].isspace() for word in words if word.text):
        return ""
    return " "


def _join_words(words: list, sep: str) -> str:
    return sep.join(word.text for word in words).strip()


def _is_forbidden_head(head: str, params: LanguageParams) -> bool:
    """行首单元是否为禁用助词（精确匹配整个单元）。

    ``head`` 应是完整的词/文节（如「もう」「もの」），只有它**恰好**是
    禁用 token（如助词「も」）才算违规，避免前缀匹配误伤。
    """
    if not params.no_line_break_before:
        return False
    return head in params.no_line_break_before


def first_unit(text: str, lang: str) -> str:
    """返回文本开头的首个切分单元：ja 为首个文节，zh 为首个汉字，
    其余语种为首个空格分词。

    供 postcheck 对折行后的行做与折行一致的行首助词校验。
    """
    cleaned = _clean(text, lang)
    if not cleaned:
        return ""
    if lang == "ja":
        spans = _ja_units(cleaned)
        if spans:
            return cleaned[: spans[0][1]]
        return cleaned
    if lang == "zh":
        return cleaned[0]
    parts = cleaned.split()
    return parts[0] if parts else cleaned


# --------------------------------------------------------------------------- #
# 词边界切分
# --------------------------------------------------------------------------- #


def _ja_tokenizer() -> object | None:
    """惰性加载 janome 分词器；不可用时返回 ``None``（只警告一次）。"""
    global _JA_TOKENIZER, _JA_TOKENIZER_FAILED
    if _JA_TOKENIZER is None and not _JA_TOKENIZER_FAILED:
        try:
            from janome.tokenizer import Tokenizer

            _JA_TOKENIZER = Tokenizer()
        except Exception:  # noqa: BLE001 - 依赖缺失时降级，不中断管线
            logger.warning("janome 不可用，日文断行退回字符级折行")
            _JA_TOKENIZER_FAILED = True
            return None
    return _JA_TOKENIZER


def _ja_units(text: str) -> list[tuple[int, int]]:
    """把日文文本切成「文节」近似的字符区间 ``[(start, end), ...]``。

    规则：自立词（名詞/動詞/形容詞/副詞…）各成一节，助詞/助動詞/接尾辞/記号
    附着到前一节。分词结果与原文长度不一致时返回空列表（调用方回退）。
    """
    tokenizer = _ja_tokenizer()
    if tokenizer is None or not text:
        return []

    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokenizer.tokenize(text):  # type: ignore[attr-defined]
        pos = token.part_of_speech.split(",")[0]
        start = cursor
        cursor += len(token.surface)
        if spans and pos in _JA_ATTACH_POS:
            spans[-1] = (spans[-1][0], cursor)
        else:
            spans.append((start, cursor))

    if cursor != len(text):
        return []
    return spans


def _regroup_ja_words(seg: Segment) -> list[Word] | None:
    """把 Whisper 的逐字符词单元重组为文节单元；无法重组时返回 ``None``。

    文节的时间取其所覆盖字符区间的首/末字符时间。
    """
    words = seg.words
    text = _clean(seg.text, "ja")
    if not words or not text or "".join(word.text for word in words) != text:
        return None

    spans = _ja_units(text)
    if not spans:
        return None

    # 每个字符位置所属词的时间区间
    times: list[tuple[float, float]] = []
    for word in words:
        for _ in word.text:
            times.append((word.start, word.end))
    if len(times) != len(text):
        return None

    units: list[Word] = []
    for start, end in spans:
        surface = text[start:end]
        if not surface:
            continue
        units.append(Word(surface, times[start][0], times[end - 1][1]))
    return units or None


def _units_for(seg: Segment, params: LanguageParams) -> tuple[list[Word], str]:
    """取切分单元与拼接分隔符：日文优先重组为文节单元，否则用原词单元。"""
    if params.lang == "ja":
        regrouped = _regroup_ja_words(seg)
        if regrouped is not None:
            return regrouped, ""
    return list(seg.words), _detect_sep(seg.words, params.lang)


def _split_with_words(seg: Segment, params: LanguageParams) -> list[Segment]:
    units, sep = _units_for(seg, params)
    budget = params.max_chars_per_line * params.max_lines
    groups = _pack_words(units, params, budget, sep)

    out: list[Segment] = []
    for start, end in groups:
        chunk = list(units[start:end])
        out.append(
            Segment(
                index=0,
                text=_wrap_words(chunk, params, sep),
                start=chunk[0].start,
                end=chunk[-1].end,
                words=chunk,
            )
        )
    return out


def _pack_words(
    words: list, params: LanguageParams, budget: int, sep: str
) -> list[tuple[int, int]]:
    """贪心装词：每个 cue 不超过字符预算与时长上限，断点优先标点/停顿。"""
    count = len(words)
    joint_sep = len(sep)
    groups: list[tuple[int, int]] = []
    start = 0
    while start < count:
        end = start
        length = 0
        while end < count:
            extra = len(words[end].text) + (0 if end == start else joint_sep)
            overflow_chars = length + extra > budget
            overflow_time = words[end].end - words[start].start > params.max_duration
            if end > start and (overflow_chars or overflow_time):
                break
            length += extra
            end += 1

        if end >= count:
            groups.append((start, count))
            break

        best_i, best_score = start + 1, None
        for i in range(start + 1, end + 1):
            score = _break_score(words, i, params)
            if best_score is None or score >= best_score:
                best_score, best_i = score, i
        groups.append((start, best_i))
        start = best_i
    return groups


def _break_score(words: list, i: int, params: LanguageParams) -> float:
    """在 ``words[i-1] | words[i]`` 处断开的适配度。"""
    prev, nxt = words[i - 1], words[i]
    score = 0.0
    tail = prev.text[-1] if prev.text else ""
    if tail in STRONG_BREAK_CHARS:
        score += 3.0
    elif tail in WEAK_BREAK_CHARS:
        score += 1.0

    gap = nxt.start - prev.end
    if gap >= PAUSE_LONG:
        score += 2.0
    elif gap >= PAUSE_SHORT:
        score += 1.0

    if _is_forbidden_head(nxt.text, params):
        score -= 4.0
    return score


def _split_without_words(seg: Segment, params: LanguageParams) -> list[Segment]:
    """无词级时间戳的回退：按字符预算切文本，时间按字符数比例分配。"""
    text = _clean(seg.text, params.lang)
    if not text:
        return [seg]

    budget = params.max_chars_per_line * params.max_lines
    chunks = _chunk_text(text, params, budget)
    total = sum(len(chunk) for chunk in chunks) or 1
    duration = max(seg.end - seg.start, 0.0)

    out: list[Segment] = []
    cursor = seg.start
    for chunk in chunks:
        span = duration * len(chunk) / total
        out.append(Segment(0, wrap_text(chunk, params), cursor, cursor + span))
        cursor += span
    out[-1].end = seg.end  # 收尾对齐，避免累计误差
    return out


def _chunk_text(text: str, params: LanguageParams, budget: int) -> list[str]:
    """把长文本按预算切成若干段，断点优先标点、其次空格、最后硬切。"""
    pieces: list[str] = []
    rest = text
    while len(rest) > budget:
        window = rest[:budget]
        cut = None
        for i in range(len(window) - 1, 0, -1):
            if window[i] in STRONG_BREAK_CHARS:
                cut = i + 1
                break
        if cut is None:
            for i in range(len(window) - 1, 0, -1):
                if window[i] in WEAK_BREAK_CHARS:
                    cut = i + 1
                    break
        if cut is None and params.lang != "ja":
            space = window.rfind(" ")
            if space > 0:
                cut = space + 1
        if cut is None:
            cut = budget
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        pieces.append(rest)
    return pieces


# --------------------------------------------------------------------------- #
# 时间轴规整
# --------------------------------------------------------------------------- #


def _drop_duplicate_asr_segments(segments: list[Segment]) -> list[Segment]:
    """丢弃 ASR 产生的整句重复片段。

    合成语音/静音边界上，Whisper 偶尔会把同一句识别为两条相邻分段，
    后一条通常只有零点几秒。文本去空白后一致即判为重复，保留时长较长的那条。
    """
    out: list[Segment] = []
    for seg in segments:
        if out:
            prev = out[-1]
            if "".join(seg.text.split()) == "".join(prev.text.split()):
                if seg.duration > prev.duration:
                    out[-1] = seg  # 保留更长的那条
                continue
        out.append(seg)
    return out


def _trim_overlaps(segments: list[Segment]) -> list[Segment]:
    """相邻 cue 不重叠：把后一条的起点修剪到前一条的 end。"""
    out: list[Segment] = []
    for seg in segments:
        if out and seg.start < out[-1].end:
            seg.start = out[-1].end
        if seg.end - seg.start <= 0:
            continue  # 时间被完全吞掉（与上一条完全重叠），丢弃
        out.append(seg)
    return out


def _merge_short(segments: list[Segment], params: LanguageParams) -> list[Segment]:
    """把时长过短的 cue 并入相邻 cue（受时长与字符预算约束，预算允许 +2 容差）。"""
    budget = params.max_chars_per_line * params.max_lines + 2  # 短碎片合并容差
    segs = list(segments)
    while len(segs) > 1:
        merged = False
        for i, seg in enumerate(segs):
            if seg.duration >= params.min_duration:
                continue
            for j in (i - 1, i + 1):
                if j < 0 or j >= len(segs):
                    continue
                first, second = (segs[j], seg) if j < i else (seg, segs[j])
                words = (first.words + second.words) if (first.words and second.words) else []
                sep = _detect_sep(words, params.lang)
                combined = _join_words(words, sep) if words else _clean(first.text, params.lang) + (
                    " " if params.lang != "ja" else ""
                ) + _clean(second.text, params.lang)
                if len(combined) > budget:
                    continue
                new_start = min(first.start, second.start)
                new_end = max(first.end, second.end)
                if new_end - new_start > params.max_duration:
                    continue

                first.start = new_start
                first.end = new_end
                first.words = words
                first.text = _wrap_words(words, params, sep) if words else wrap_text(combined, params)
                del segs[max(i, j)]  # 删除靠后的那条
                merged = True
                break
            if merged:
                break
        if not merged:
            break
    return segs


def _split_too_long(segments: list[Segment], params: LanguageParams) -> list[Segment]:
    """把仍超过时长上限的 cue 按时间等分（文本按字符比例切）。"""
    out: list[Segment] = []
    for seg in segments:
        duration = seg.end - seg.start
        if duration <= params.max_duration:
            out.append(seg)
            continue

        parts_count = int(math.ceil(duration / params.max_duration))
        text = _clean(seg.text, params.lang)
        parts = _chunk_text(text, params, max(1, math.ceil(len(text) / parts_count)))
        if len(parts) < 2:
            out.append(seg)
            continue

        total = sum(len(part) for part in parts) or 1
        cursor = seg.start
        for idx, part in enumerate(parts):
            span = duration * len(part) / total
            end = seg.end if idx == len(parts) - 1 else cursor + span
            out.append(Segment(0, wrap_text(part, params), cursor, end))
            cursor = end
    return out
