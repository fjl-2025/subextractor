"""M0 冒烟测试：版本、语言参数、SRT 写入、质量校验、CLI doctor。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from subextractor import __version__
from subextractor.config import SUPPORTED_TARGET_LANGS, load_languages
from subextractor.core import postcheck, writer
from subextractor.core.subtitle import Segment, SubtitleDoc

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
    out = writer.write_srt(doc, tmp_path / "a.bilingual.ja.srt", bilingual=True)

    text = out.read_text(encoding="utf-8-sig")
    assert "00:00:00,000 --> 00:00:01,500" in text
    assert "Hello world" in text
    assert "こんにちは世界" in text


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
