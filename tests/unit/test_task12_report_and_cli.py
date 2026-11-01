"""Task 12: format_report text and the scripts/eval.py command line."""
from __future__ import annotations

import re

import pytest

from rs12_helpers import CLASSES, ROOT, haz, line_with, make_db, rate_tokens, run_cli, seg, write_labels

A = ("A", "00:00", "10:00")


def report_for(db, labels_dir, **kw):
    from backend.evaluation import evaluate, format_report, load_labels

    return format_report(evaluate(str(db.path), load_labels(labels_dir), **kw))


def assert_clean_text(text):
    assert isinstance(text, str) and text.strip()
    assert "Traceback" not in text
    assert "None" not in text
    assert not re.search(r"(?i)\bnan\b|\binf\b|ZeroDivision", text)


@pytest.fixture()
def labelled(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz("A", "01:00", "01:05", "pothole", "2", "H1"),
                              haz("A", "05:00", "05:05", "pothole", "2", "H2"), seg("B", "20:00", "21:00")])
    db.candidate("d1", "pothole", 62.0)
    db.candidate("d1", "debris in lane", 1230.0)
    h1 = db.hazard("pothole", "approved")
    h2 = db.hazard("pothole", "rejected")
    h3 = db.hazard("debris in lane", "new")
    db.candidate("d1", "pothole", 400.0, hazard_id=h1)
    db.candidate("d1", "pothole", 410.0, hazard_id=h2)
    db.candidate("d1", "debris in lane", 420.0, hazard_id=h3)
    return db, d


# ------------------------------------------------------------------ text

def test_report_without_any_labels_says_so_and_never_crashes(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.candidate("d1", "pothole", 100.0)
    text = report_for(db, tmp_path / "no_labels")
    assert_clean_text(text)
    assert "recall: not measured (no labels)" in text
    assert "n/a" in text
    for c in CLASSES:
        assert c in text


def test_report_for_an_empty_database(tmp_path):
    db = make_db(tmp_path)
    text = report_for(db, tmp_path / "labels")
    assert_clean_text(text)
    assert "recall: not measured (no labels)" in text


def test_report_with_only_example_rows_says_not_measured(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg("A", "00:00", "02:30", "EXAMPLE ROW: x"),
                              haz("A", "01:12", "01:16", "debris in lane", "1", "H1", "EXAMPLE ROW: y")])
    text = report_for(db, d)
    assert "recall: not measured (no labels)" in text
    assert_clean_text(text)


def test_report_with_labels_has_tables_totals_and_no_not_measured_line(labelled):
    db, d = labelled
    text = report_for(db, d)
    assert_clean_text(text)
    assert "recall: not measured (no labels)" not in text
    low = text.lower()
    for word in ("recall", "precision", "labeled", "matched", "approved", "rejected"):
        assert word in low, word
    for c in CLASSES:
        assert c in text
    assert "false alarm" in low
    assert "unreviewed" in low or "not reviewed" in low or "new" in low


def test_classes_without_labeled_hazards_print_na_for_recall(labelled):
    db, d = labelled
    text = report_for(db, d)
    # every class row of the label table that has no labelled hazards shows n/a, never 0 for the recall
    na_rows = [ln for ln in text.splitlines() if "n/a" in ln and any(c in ln for c in CLASSES[1:])]
    assert na_rows, "classes with no labelled hazards must show n/a"
    assert any("damaged guardrail" in ln and "n/a" in ln for ln in text.splitlines())


def test_label_rates_appear_in_the_pothole_row(labelled):
    db, d = labelled
    text = report_for(db, d)
    rows = [ln for ln in text.splitlines() if "pothole" in ln]
    assert rows
    # recall = 1 of 2 labelled potholes matched = 0.5 (printed as 0.50, 0.5, 50% ...)
    assert any(any(abs(t - 0.5) < 0.006 for t in rate_tokens(ln)) for ln in rows), rows


def test_review_precision_appears(labelled):
    db, d = labelled
    text = report_for(db, d)
    rows = [ln for ln in text.splitlines() if "pothole" in ln]
    # review: 1 approved, 1 rejected -> 0.5; the same row/section prints the counts
    assert any(abs(t - 0.5) < 0.006 for ln in rows for t in rate_tokens(ln))


def test_false_alarm_rate_is_printed(labelled):
    db, d = labelled
    text = report_for(db, d)
    ln = line_with(text, "false alarm")
    assert ln is not None
    # one false alarm in a 60 s clean segment = 60 per hour
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", ln)]
    assert any(abs(x - 60.0) < 0.01 for x in nums) or "60" in ln, ln


def test_no_clean_segments_prints_na_or_omits_the_rate(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz("A", "01:00", "01:05", "pothole")])
    text = report_for(db, d)
    assert_clean_text(text)


def test_report_names_skipped_drives(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.drive("skipme", "IMG_2.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz("A", "01:00", "01:05", "pothole")])
    assert "skipme" in report_for(db, d)


def test_report_is_ascii_safe_for_a_windows_console(labelled):
    db, d = labelled
    text = report_for(db, d)
    text.encode("cp1252")


def test_format_report_is_deterministic(labelled):
    db, d = labelled
    assert report_for(db, d) == report_for(db, d)


# ------------------------------------------------------------------ command line

def cli_db(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.drive("d2", "IMG_2.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz("A", "01:00", "01:05", "pothole", "2", "H1")])
    db.candidate("d1", "pothole", 62.0)
    db.candidate("d1", "pothole", 67.5)         # only within a tolerance of 2.5 s or more
    db.candidate("d2", "debris in lane", 100.0)
    h = db.hazard("pothole", "approved")
    db.candidate("d1", "pothole", 300.0, hazard_id=h)
    return db, d


def test_help_lists_the_options():
    cp = run_cli(["--help"])
    assert cp.returncode == 0
    for opt in ("--labels", "--db", "--tolerance", "--drive"):
        assert opt in cp.stdout


def test_cli_with_labels_prints_the_report_and_exits_zero(tmp_path):
    db, d = cli_db(tmp_path)
    cp = run_cli(["--labels", d, "--db", db.path])
    assert cp.returncode == 0, cp.stderr
    assert_clean_text(cp.stdout)
    for c in CLASSES:
        assert c in cp.stdout
    assert "recall: not measured (no labels)" not in cp.stdout
    assert cp.stderr.strip() == "" or "Traceback" not in cp.stderr


@pytest.mark.parametrize("kind", ["missing", "empty", "examples"])
def test_cli_without_usable_labels_exits_zero_and_says_so(tmp_path, kind):
    db, _ = cli_db(tmp_path)
    d = tmp_path / "lab2"
    if kind == "empty":
        d.mkdir()
    if kind == "examples":
        write_labels(d, "IMG_1", [seg("A", "00:00", "02:30", "EXAMPLE ROW: replace"),
                                  haz("A", "01:12", "01:16", "pothole", "1", "H1", "EXAMPLE ROW: replace")])
    cp = run_cli(["--labels", d, "--db", db.path])
    assert cp.returncode == 0, cp.stderr
    assert "recall: not measured (no labels)" in cp.stdout
    assert "n/a" in cp.stdout
    assert "Traceback" not in cp.stdout + cp.stderr


def test_cli_default_labels_dir_is_labels_in_the_repo(tmp_path):
    db, _ = cli_db(tmp_path)
    cp = run_cli(["--db", db.path], cwd=ROOT)
    assert cp.returncode == 0, cp.stderr
    assert_clean_text(cp.stdout)


def test_cli_uses_roadscout_db_when_db_is_not_given(tmp_path):
    db, d = cli_db(tmp_path)
    cp = run_cli(["--labels", d], env_over={"ROADSCOUT_DB": str(db.path)})
    assert cp.returncode == 0, cp.stderr
    assert "pothole" in cp.stdout


def test_cli_tolerance_changes_the_matching(tmp_path):
    db, d = cli_db(tmp_path)
    narrow = run_cli(["--labels", d, "--db", db.path, "--tolerance", "2"])
    wide = run_cli(["--labels", d, "--db", db.path, "--tolerance", "3"])
    assert narrow.returncode == 0 and wide.returncode == 0
    assert narrow.stdout != wide.stdout
    default = run_cli(["--labels", d, "--db", db.path])
    assert default.stdout == narrow.stdout, "the default tolerance is 2.0 s"


def test_cli_drive_filter(tmp_path):
    db, d = cli_db(tmp_path)
    cp = run_cli(["--labels", d, "--db", db.path, "--drive", "d2"])
    assert cp.returncode == 0, cp.stderr
    full = run_cli(["--labels", d, "--db", db.path])
    assert cp.stdout != full.stdout


def test_cli_unknown_drive_fails_and_names_it(tmp_path):
    db, d = cli_db(tmp_path)
    cp = run_cli(["--labels", d, "--db", db.path, "--drive", "ghost-drive"])
    assert cp.returncode != 0
    assert "ghost-drive" in cp.stderr
    assert "Traceback" not in cp.stderr


def test_cli_missing_db_fails_and_creates_nothing(tmp_path):
    missing = tmp_path / "gone.sqlite"
    cp = run_cli(["--labels", tmp_path / "labels", "--db", missing])
    assert cp.returncode != 0
    assert not missing.exists()
    assert "Traceback" not in cp.stderr


def test_cli_bad_label_file_fails_naming_file_and_line(tmp_path):
    db, _ = cli_db(tmp_path)
    d = tmp_path / "badlabels"
    write_labels(d, "IMG_9001", [seg("A", "00:00", "01:00"), haz("A", "00:10", "00:12", "sinkhole")])
    cp = run_cli(["--labels", d, "--db", db.path])
    assert cp.returncode != 0
    assert "IMG_9001" in cp.stderr
    assert re.search(r"(?i)(line\s*3\b|:3\b)", cp.stderr), cp.stderr
    assert "Traceback" not in cp.stderr


def test_cli_invalid_tolerance_is_a_usage_error(tmp_path):
    db, d = cli_db(tmp_path)
    cp = run_cli(["--labels", d, "--db", db.path, "--tolerance", "soon"])
    assert cp.returncode != 0


def test_cli_never_writes_the_database(tmp_path):
    db, d = cli_db(tmp_path)
    before = db.fingerprint()
    for extra in ([], ["--drive", "d1"], ["--tolerance", "5"]):
        cp = run_cli(["--labels", d, "--db", db.path, *extra])
        assert cp.returncode == 0, cp.stderr
    assert db.fingerprint() == before
    assert not list(tmp_path.glob("*.sqlite-wal"))


def test_cli_needs_no_key_and_no_provider(tmp_path):
    db, d = cli_db(tmp_path)
    cp = run_cli(["--labels", d, "--db", db.path],
                 env_over={"META_API_KEY": "", "ROADSCOUT_PROVIDER": "live"})
    assert cp.returncode == 0, cp.stderr
    assert "META_API_KEY" not in cp.stdout + cp.stderr


def test_cli_output_matches_format_report(tmp_path):
    from backend.evaluation import evaluate, format_report, load_labels

    db, d = cli_db(tmp_path)
    expected = format_report(evaluate(str(db.path), load_labels(d)))
    cp = run_cli(["--labels", d, "--db", db.path])
    assert cp.returncode == 0
    assert cp.stdout.strip() == expected.strip()
