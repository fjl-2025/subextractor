"""M0 冒烟测试：版本、语言参数、SRT 写入、质量校验、CLI doctor。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from subextractor import __version__
from subextractor.config import SUPPORTED_TARGET_LANGS, LanguageParams, load_languages
from subextractor.core import postcheck, segment, translate, writer
from subextractor.core.subtitle import Segment, SubtitleDoc, Word
from subextractor.models import registry

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_version() -> None:
    assert __version__ == "0.0.1"


def test_languages_cover_four_targets() -> None:
    langs = load_languages()
    assert set(SUPPORTED_TARGET_LANGS) <= set(langs)
    assert langs["ja"].max_chars_per_line == 13
    assert "は" in langs["ja"].no_line_break_before
    assert langs["ko"].max_chars_per_line == 16
    assert langs["en"].no_line_break_before == ()


def test_write_bilingual_srt(tmp_path: Path) -> None:
    doc = SubtitleDoc(source_lang="en", target_lang="ja")
    doc.segments.append(Segment(1, "Hello world", 0.0, 1.5, translation="こんにちは世界"))
    out = writer.write_srt(doc, tmp_path / "a.bilingual.ja.srt", mode="bilingual")

    text = out.read_text(encoding="utf-8-sig")
    assert "00:00:00,000 --> 00:00:01,500" in text
    assert "Hello world" in text
    assert "こんにちは世界" in text


def test_write_target_only_srt(tmp_path: Path) -> None:
    doc = SubtitleDoc(source_lang="en", target_lang="ja")
    doc.segments.append(Segment(1, "Hello world", 0.0, 1.5, translation="こんにちは世界"))
    out = writer.write_srt(doc, tmp_path / "a.ja.srt", mode="target")

    text = out.read_text(encoding="utf-8-sig")
    assert "こんにちは世界" in text
    assert "Hello world" not in text


def test_nllb_language_code_mapping() -> None:
    assert translate.to_nllb_code("en") == "eng_Latn"
    assert translate.to_nllb_code("ja") == "jpn_Jpan"
    assert translate.to_nllb_code("ko") == "kor_Hang"
    assert translate.to_nllb_code("eng_Latn") == "eng_Latn"  # 已是 FLORES 代码
    with pytest.raises(ValueError):
        translate.to_nllb_code("xx")


def test_missing_nllb_model_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(registry.ENV_MODELS_DIR, str(tmp_path))
    with pytest.raises(registry.MissingModelError):
        registry.resolve_nllb_model()


def test_postcheck_flags_overlap() -> None:
    doc = SubtitleDoc(source_lang="en")
    doc.segments.append(Segment(1, "a", 0.0, 2.0))
    doc.segments.append(Segment(2, "b", 1.0, 3.0))

    issues = postcheck.check(doc, load_languages()["en"])
    assert any(issue.level == "error" for issue in issues)


def test_cli_doctor_runs() -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "subextractor", "doctor"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "subextractor" in result.stdout
    assert "模型状态" in result.stdout


def test_auto_tier_on_cpu_prefers_small_over_large() -> None:
    tier = registry.resolve_auto_tier("cpu", cuda_count=0, installed=["small", "large-v3"])
    assert tier == "small"


def test_auto_tier_on_gpu_prefers_large_v3() -> None:
    tier = registry.resolve_auto_tier("auto", cuda_count=1, installed=["small", "large-v3"])
    assert tier == "large-v3"


def test_auto_tier_without_installed_model_raises() -> None:
    with pytest.raises(registry.MissingModelError):
        registry.resolve_auto_tier("cpu", cuda_count=0, installed=[])


def test_compute_type_follows_device() -> None:
    assert registry.resolve_compute_type("cpu", cuda_count=0) == "int8"
    assert registry.resolve_compute_type("auto", cuda_count=1) == "float16"


def _make_segment(tokens: list[str], duration_each: float, gap: float = 0.0) -> Segment:
    """按固定时长合成一条带词级时间戳的分段，便于断句单测。"""
    words = []
    cursor = 0.0
    for token in tokens:
        words.append(Word(token, cursor, cursor + duration_each))
        cursor += duration_each + gap
    return Segment(1, "", words[0].start, words[-1].end, words=words)


def test_segment_splits_long_ja_cue_by_char_budget() -> None:
    params = load_languages()["ja"]
    tokens = [
        "こんにちは", "みなさん", "ようこそ", "字幕", "抽出", "ツール", "の", "デモ", "へ",
        "これは", "日本語", "の", "音声", "認識", "テスト", "です",
    ]
    seg = _make_segment(tokens, duration_each=0.5)
    seg.text = "".join(tokens)

    out = segment.segment_document([seg], params)
    budget = params.max_chars_per_line * params.max_lines
    assert len(out) >= 2
    for cue in out:
        assert len("".join(cue.text.split())) <= budget
        assert all(len(line) <= params.max_chars_per_line for line in cue.text.splitlines())
    for earlier, later in zip(out, out[1:]):
        assert later.start >= earlier.end - 1e-6


def test_segment_prefers_punctuation_break_in_english() -> None:
    params = LanguageParams(
        lang="en", max_chars_per_line=20, max_lines=2, min_duration=0.8,
        max_duration=7.0, cps_target=17.0,
    )
    tokens = ["Hello.", "This", "is", "a", "test", "of", "the", "system", "and", "more", "words", "here"]
    seg = _make_segment(tokens, duration_each=1.0)
    seg.text = " ".join(tokens)

    out = segment.segment_document([seg], params)
    # 首个强标点在句首，优先在 "Hello." 之后断开，而非填满字符预算
    assert out[0].text == "Hello."


def test_segment_merges_short_fragment() -> None:
    params = load_languages()["en"]
    first = _make_segment(["Hello", "there"], duration_each=1.0)
    first.index, first.text = 1, "Hello there"
    second = Segment(2, "ok", 2.0, 2.3, words=[Word("ok", 2.0, 2.3)])

    out = segment.segment_document([first, second], params)
    assert len(out) == 1
    assert "ok" in out[0].text


def test_normalize_trims_overlap() -> None:
    params = load_languages()["en"]
    first = _make_segment(["one", "two"], duration_each=1.0)
    first.index, first.text = 1, "one two"
    second = _make_segment(["three"], duration_each=1.5)
    second.index, second.text, second.start, second.end = 2, "three", 1.5, 3.0

    out = segment.normalize_timeline([first, second], params)
    assert out[1].start >= out[0].end - 1e-6


def test_normalize_drops_duplicate_asr_segment() -> None:
    """ASR 整句重复片段应被丢弃，保留时长较长的那条。"""
    params = load_languages()["ko"]
    first = Segment(1, "이 길을 곧장 가면 오른쪽에 병원이 있습니다.", 111.72, 114.78)
    second = Segment(2, "이 길을 곧장 가면 오른쪽에 병원이 있습니다.", 115.16, 115.56)

    out = segment.normalize_timeline([first, second], params)
    assert len(out) == 1
    assert out[0].duration == pytest.approx(3.06, abs=0.01)


def test_merge_short_allows_small_budget_tolerance() -> None:
    """短碎片合并允许 +2 字符容差，避免 1 字之差卡住。"""
    params = load_languages()["ja"]
    first = Segment(1, "明日の朝までにこの\nレポートを書き終えなければ", 110.68, 114.50)
    second = Segment(2, "なりません", 114.50, 114.94)

    out = segment.normalize_timeline([first, second], params)
    assert len(out) == 1
    assert "なりません" in out[0].text


def test_postcheck_flags_cps() -> None:
    params = load_languages()["en"]
    doc = SubtitleDoc(source_lang="en")
    doc.segments.append(Segment(1, "x" * 100, 0.0, 1.0))

    issues = postcheck.check(doc, params)
    assert any("CPS" in issue.message for issue in issues)
