"""M0 冒烟测试：版本、语言参数、SRT 写入、质量校验、CLI doctor。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from subextractor import __version__
from subextractor.config import SUPPORTED_TARGET_LANGS, LanguageParams, load_languages
from subextractor.core import batch, cache, glossary, ocr, postcheck, segment, translate, writer
from subextractor.core.asr import AsrResult
from subextractor.core.pipeline import PipelineConfig, PipelineResult, run
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


def test_ass_bilingual_uses_script_specific_styles(tmp_path: Path) -> None:
    """双语 ASS：东亚源语用 CJK 字体，拉丁译文用西文字体，行内切换 Dst 样式。"""
    doc = SubtitleDoc(source_lang="ja", target_lang="en")
    doc.segments.append(
        Segment(1, "こんにちは", 0.0, 1.5, translation="Hello there")
    )
    out = writer.write_ass(doc, tmp_path / "a.bilingual.en.ass", mode="bilingual")

    text = out.read_text(encoding="utf-8-sig")
    assert "Style: Src,Microsoft YaHei,44" in text  # 东亚源语样式
    assert "Style: Dst,Arial,34" in text  # 拉丁译文样式（字号略小）
    assert "\\N{\\rDst}Hello there" in text  # 译文行行内切换样式


def test_ass_latin_pair_uses_latin_styles(tmp_path: Path) -> None:
    doc = SubtitleDoc(source_lang="en", target_lang="fr")
    doc.segments.append(Segment(1, "Hello", 0.0, 1.0, translation="Bonjour"))
    out = writer.write_ass(doc, tmp_path / "a.ass", mode="bilingual")

    text = out.read_text(encoding="utf-8-sig")
    assert "Style: Src,Arial,40" in text
    assert "Style: Dst,Arial,34" in text
    assert "Microsoft YaHei" not in text


def test_ass_unknown_language_falls_back_to_latin(tmp_path: Path) -> None:
    doc = SubtitleDoc(source_lang="sw", target_lang="en")
    doc.segments.append(Segment(1, "Habari", 0.0, 1.0, translation="Hello"))
    out = writer.write_ass(doc, tmp_path / "a.ass", mode="bilingual")

    text = out.read_text(encoding="utf-8-sig")
    assert "Style: Src,Arial,40" in text


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


# ---------------------------------------------------------------- M5 批处理与断点续跑


def _asr_result() -> AsrResult:
    return AsrResult(
        segments=[Segment(1, "hello", 0.0, 1.0, words=[Word("hello", 0.0, 1.0)])],
        language="en",
        language_probability=0.9,
        duration=1.0,
    )


def test_asr_cache_roundtrip_and_invalidation(tmp_path: Path) -> None:
    src = tmp_path / "movie.mp4"
    src.write_bytes(b"0" * 128)
    cache_dir = tmp_path / "cache"

    cache.save_asr_cache(
        cache_dir, src, _asr_result(), asr_model="small", source_lang="auto", resolved_size="small"
    )

    entry = cache.load_asr_cache(cache_dir, src, "small", "auto")
    assert entry is not None
    assert entry.resolved_size == "small"
    assert entry.result.language == "en"
    assert entry.result.segments[0].text == "hello"
    assert entry.result.segments[0].words[0].text == "hello"
    assert entry.result.segments[0].words[0].end == pytest.approx(1.0)

    # 识别配置变更 → 缓存失效
    assert cache.load_asr_cache(cache_dir, src, "large-v3", "auto") is None
    assert cache.load_asr_cache(cache_dir, src, "small", "ja") is None

    # 文件被改动（mtime 变化）→ 缓存失效
    os.utime(src, ns=(0, 1_000_000_000))
    assert cache.load_asr_cache(cache_dir, src, "small", "auto") is None


def test_collect_media_files_filters_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "b.mp4").write_bytes(b"x")
    (tmp_path / "a.mkv").write_bytes(b"x")
    (tmp_path / "A.MP4").write_bytes(b"x")
    (tmp_path / "note.txt").write_bytes(b"x")

    files = batch.collect_media_files(tmp_path)
    assert [p.name for p in files] == ["a.mkv", "A.MP4", "b.mp4"]

    single = tmp_path / "b.mp4"
    assert batch.collect_media_files(single) == [single]

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        batch.collect_media_files(empty)
    with pytest.raises(FileNotFoundError):
        batch.collect_media_files(tmp_path / "missing")


def _stub_pipeline_result() -> PipelineResult:
    doc = SubtitleDoc(source_lang="en")
    doc.segments.append(Segment(1, "hi", 0.0, 1.0))
    return PipelineResult(doc=doc, outputs=[], asr_model="stub")


def _batch_configs(tmp_path: Path, count: int) -> list[PipelineConfig]:
    configs = []
    for i in range(count):
        path = tmp_path / f"v{i}.mp4"
        path.write_bytes(b"x")
        configs.append(PipelineConfig(input_path=path))
    return configs


def test_run_batch_collects_failures(tmp_path: Path) -> None:
    configs = _batch_configs(tmp_path, 3)
    done: list[tuple[batch.BatchItemResult, int, int]] = []

    def runner(config, on_progress=None, should_cancel=None, asr_engine=None, translator=None):
        if config.input_path.name == "v1.mp4":
            raise RuntimeError("boom")
        return _stub_pipeline_result()

    result = batch.run_batch(
        configs, runner=runner, on_item_done=lambda item, i, n, doc: done.append((item, i, n))
    )

    assert len(result.succeeded) == 2
    assert len(result.failed) == 1
    assert result.failed[0].input_path.name == "v1.mp4"
    assert result.failed[0].error == "boom"
    assert not result.cancelled
    assert len(done) == 3


def test_run_batch_shares_engines_when_sequential(tmp_path: Path) -> None:
    configs = _batch_configs(tmp_path, 2)
    engines: list[tuple[object, object]] = []

    def runner(config, on_progress=None, should_cancel=None, asr_engine=None, translator=None):
        engines.append((asr_engine, translator))
        return _stub_pipeline_result()

    batch.run_batch(configs, runner=runner)

    assert engines[0][0] is not None and engines[0][0] is engines[1][0]
    assert engines[0][1] is not None and engines[0][1] is engines[1][1]


def test_run_batch_cancelled_before_start(tmp_path: Path) -> None:
    configs = _batch_configs(tmp_path, 1)
    calls: list[PipelineConfig] = []

    def runner(config, **kwargs):
        calls.append(config)
        return _stub_pipeline_result()

    result = batch.run_batch(configs, runner=runner, should_cancel=lambda: True)

    assert result.cancelled
    assert not calls
    assert not result.succeeded and not result.failed


def test_pipeline_uses_asr_cache(tmp_path: Path) -> None:
    """缓存命中时管线跳过抽音轨+识别，直接产出最终字幕。"""
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")
    cache_dir = tmp_path / "cache"
    cache.save_asr_cache(
        cache_dir,
        src,
        AsrResult(
            segments=[Segment(1, "hello world this is a test", 0.0, 2.0)],
            language="en",
            language_probability=0.9,
            duration=2.0,
        ),
        asr_model="small",
        source_lang="auto",
        resolved_size="small",
    )

    result = run(PipelineConfig(input_path=src, asr_model="small", cache_dir=cache_dir))

    assert result.asr_model == "small"
    assert result.outputs, "应至少产出 name.orig.srt"
    text = result.outputs[0].read_text(encoding="utf-8-sig")
    assert "hello" in text


def test_models_root_frozen_default_and_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """打包版默认模型根为 exe 同级 models/；models-path.txt 可覆盖。"""
    monkeypatch.setattr(registry, "is_frozen", lambda: True)
    exe_dir = tmp_path / "SubExtractor"
    exe_dir.mkdir()
    monkeypatch.setattr(registry.sys, "executable", str(exe_dir / "SubExtractor.exe"))
    monkeypatch.delenv(registry.ENV_MODELS_DIR, raising=False)

    assert registry.models_root() == exe_dir / "models"

    custom = tmp_path / "my-models"
    custom.mkdir()
    registry.save_models_root_override(custom)
    try:
        assert registry.models_root() == custom.resolve()
    finally:
        registry.override_file().unlink(missing_ok=True)


def test_glossary_load_parses_tsv_and_sorts_by_length(tmp_path: Path) -> None:
    g = tmp_path / "g.tsv"
    g.write_text("# 注释行\n\nAPI\t应用程序接口\nOpenAI API\t开放接口\n", encoding="utf-8")

    entries = glossary.load(g)

    assert entries == [("OpenAI API", "开放接口"), ("API", "应用程序接口")]


def test_glossary_load_rejects_bad_lines(tmp_path: Path) -> None:
    g = tmp_path / "g.tsv"
    g.write_text("只有一列\n", encoding="utf-8")
    with pytest.raises(ValueError):
        glossary.load(g)
    with pytest.raises(FileNotFoundError):
        glossary.load(tmp_path / "missing.tsv")


def test_glossary_apply_latin_word_boundary_case_insensitive() -> None:
    entries = [("AI", "人工智能"), ("the cloud", "云端")]
    text = "EMAIL and AI are not the cloud, ai."

    out = glossary.apply(text, entries, "en")

    assert "EMAIL" in out  # 词边界：不命中 EMAIL 中的 AI
    assert "人工智能 are not 云端, 人工智能." in out  # 大小写不敏感，统一用表中写法


def test_glossary_apply_east_asian_substring() -> None:
    entries = [("変換速度", "转换速度"), ("変換", "转换")]

    assert glossary.apply("変換速度を上げる", entries, "ja") == "转换速度を上げる"


class _FakeTranslator:
    name = "fake"

    def translate(self, texts, src, dst, **kwargs):  # noqa: ANN001, ANN202
        return [f"[{dst}] {text}" for text in texts]

    def close(self) -> None:
        pass


def test_pipeline_applies_glossary_after_translation(tmp_path: Path) -> None:
    """术语表在翻译后、写出前生效（译文被替换，源语行不受影响）。"""
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")
    cache_dir = tmp_path / "cache"
    cache.save_asr_cache(
        cache_dir,
        src,
        AsrResult(
            segments=[Segment(1, "this AI test", 0.0, 2.0)],
            language="en",
            language_probability=0.9,
            duration=2.0,
        ),
        asr_model="small",
        source_lang="auto",
        resolved_size="small",
    )
    g = tmp_path / "glossary.fr.tsv"
    g.write_text("AI\tIA\n", encoding="utf-8")

    result = run(
        PipelineConfig(
            input_path=src,
            target_lang="fr",
            asr_model="small",
            glossary_path=g,
            cache_dir=cache_dir,
        ),
        translator=_FakeTranslator(),
    )

    target = next(p for p in result.outputs if p.name.endswith(".fr.srt"))
    text = target.read_text(encoding="utf-8-sig")
    assert "[fr] this IA test" in text
    orig = next(p for p in result.outputs if p.name.endswith(".orig.srt"))
    assert "this AI test" in orig.read_text(encoding="utf-8-sig")


def test_ocr_merge_frames_groups_similar_and_drops_blink() -> None:
    """相似帧合并为一条 cue（文本取高置信帧）；孤立短字幕帧被丢弃。"""
    frames = [
        (0.0, "こんにちは", 0.90),
        (0.5, "こんにちは", 0.95),  # 同一条，文本取置信度高的
        (1.0, "こんにちわ", 0.80),  # 1 字符差异，仍视为同一条
        (1.5, "", 0.0),
        (2.0, "世界へようこそ", 0.99),
        (2.5, "世界へようこそ", 0.97),
        (3.0, "", 0.0),
        (3.5, "一帧闪现", 0.9),  # 仅持续 0.5s 的孤立帧
        (4.0, "", 0.0),
    ]

    cues = ocr.merge_frames(frames, interval=0.5, min_duration=0.8)

    assert len(cues) == 2
    assert cues[0].text == "こんにちは" and cues[0].start == 0.0 and cues[0].end == 1.5
    assert cues[1].text == "世界へようこそ" and cues[1].start == 2.0 and cues[1].end == 3.0


def test_ocr_merge_frames_case_insensitive_for_latin() -> None:
    frames = [(0.0, "Hello World", 0.9), (0.5, "hello world", 0.8)]

    cues = ocr.merge_frames(frames, interval=0.5, min_duration=0.5)

    assert len(cues) == 1
    assert cues[0].text == "Hello World"  # 高置信帧的原文（大小写保留）
    assert cues[0].end == 1.0


def test_ocr_lang_resolution() -> None:
    assert ocr._resolve_lang("ja", Path("x")) == "japan"
    assert ocr._resolve_lang("en", Path("x")) == "en"
    with pytest.raises(ValueError):
        ocr._resolve_lang("de", Path("x"))


def test_ocr_lang_from_model_dir_lang_file(tmp_path: Path) -> None:
    (tmp_path / ocr.LANG_FILE).write_text("ko\n", encoding="utf-8")
    assert ocr._resolve_lang("auto", tmp_path) == "korean"


def test_ocr_missing_model_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(registry.ENV_MODELS_DIR, str(tmp_path))
    with pytest.raises(registry.MissingModelError):
        registry.resolve_ocr_model()
    (registry.ocr_model_dir() / "det").mkdir(parents=True)
    (registry.ocr_model_dir() / "det" / "inference.pdmodel").write_bytes(b"x")
    (registry.ocr_model_dir() / "rec").mkdir(parents=True)
    (registry.ocr_model_dir() / "rec" / "inference.pdmodel").write_bytes(b"x")
    assert registry.resolve_ocr_model() == registry.ocr_model_dir()
