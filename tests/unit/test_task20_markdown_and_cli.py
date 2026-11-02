"""Task 20: render_markdown and scripts/eval_v1.py."""
from __future__ import annotations

import re

import pytest

from rs20_helpers import (CLASSES, ROOT, evaluate_v1, haz, make_db, run_cli_v1, seg, write_cost_log,
                          write_labels)

A = ("A", "00:00", "10:00")


def render(db, labels_dir, **kw):
    from backend.eval_v1 import render_markdown

    return render_markdown(evaluate_v1(db, labels_dir, **kw))


def assert_clean(text):
    assert isinstance(text, str) and text.strip()
    assert "Traceback" not in text
    assert "None" not in text
    assert not re.search(r"(?i)\bnan\b|\binf\b|ZeroDivision", text)


@pytest.fixture()
def full(tmp_path):
    """Labels, candidates (accepted, rejected, unverified), reviewed hazards, two drives with timing, a cost log."""
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV", duration=60.0, created="2026-10-20T12:00:00+00:00",
             finished="2026-10-20T12:02:00+00:00")
    db.drive("d2", "IMG_2.MOV", duration=120.0, created="2026-10-20T13:00:00+00:00",
             finished="2026-10-20T13:01:00+00:00")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz("A", "01:00", "01:05", "pothole", "2", "H1"), seg("B", "12:00", "13:00")])
    write_labels(d, "IMG_2", [seg(*A), haz("A", "01:00", "01:05", "pothole", "2", "H1")])
    row = db.hazard("pothole", "approved")
    db.candidate("d1", "pothole", 62.0, hazard_id=row)
    db.candidate("d2", "pothole", 62.0, hazard_id=row)
    db.candidate("d1", "pothole", 400.0, accepted=0)
    db.candidate("d1", "debris in lane", 730.0, hazard_id=db.hazard("debris in lane", "rejected"))
    db.candidate("d2", "pothole", 450.0, verdict=False)
    log = write_cost_log(tmp_path / "cost.jsonl", [0.5, 0.25, 0.25])      # 1.00 USD / 3 min
    return db, d, log


# ------------------------------------------------------------------ markdown content

def test_markdown_is_a_report_with_headings_and_tables(full):
    db, d, log = full
    text = render(db, d, cost_log=str(log))
    assert_clean(text)
    heads = [l for l in text.splitlines() if l.startswith("#")]
    assert len(heads) >= 3
    assert re.search(r"(?m)^\|\s*-{2,}", text) or re.search(r"(?m)^\|[\s:|-]+\|$", text), "a markdown table separator"


def test_tables_have_consistent_columns(full):
    db, d, log = full
    text = render(db, d, cost_log=str(log))
    block = []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            block.append(line)
            continue
        if block:
            counts = {l.count("|") - l.count("\\|") for l in block}
            assert len(counts) == 1, "ragged table:\n" + "\n".join(block)
            assert len(block) >= 3, "header, separator and at least one row"
            block = []


def test_mentions_both_arms_every_class_and_overall(full):
    db, d, log = full
    text = render(db, d, cost_log=str(log))
    assert "SAM only" in text
    assert "SAM + Muse Spark" in text
    for c in CLASSES:
        assert c in text
    assert "overall" in text.lower()
    low = text.lower()
    for word in ("recall", "precision", "false alarm", "removed"):
        assert word in low, word


def test_ablation_numbers_are_in_the_tables(full):
    db, d, log = full
    text = render(db, d, cost_log=str(log))
    pothole_rows = [l for l in text.splitlines() if l.startswith("|") and "pothole" in l]
    assert pothole_rows
    # pothole: 2 labeled, 2 matched -> recall 1.00 in both arms; precision 2/4 (SAM only) vs 2/2 (+ Muse Spark)
    joined = " ".join(pothole_rows)
    assert re.search(r"1(\.0+)?\b|100(\.0+)?%", joined)
    assert re.search(r"0\.5\b|0\.50|50(\.0+)?%", joined)


def test_cost_and_timing_lines(full):
    db, d, log = full
    text = render(db, d, cost_log=str(log))
    low = text.lower()
    assert "cost" in low and "minute" in low
    assert re.search(r"0\.33", text), "1.00 USD over 3 video minutes = 0.33 per minute"
    assert "processing" in low or "time" in low
    assert re.search(r"\b60(\.0+)?\b", text), "180 s of processing over 3 minutes = 60 s per minute"


def test_dedup_section_is_present(full):
    db, d, log = full
    text = render(db, d, cost_log=str(log)).lower()
    assert "dedup" in text
    assert "merged" in text and "split" in text


def test_review_section_is_present(full):
    db, d, log = full
    text = render(db, d, cost_log=str(log)).lower()
    assert "approved" in text and "rejected" in text and "review" in text


def test_no_labels_says_recall_is_not_measured(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.candidate("d1", "pothole", 62.0)
    text = render(db, tmp_path / "no_labels")
    assert_clean(text)
    assert "recall not measured" in text
    assert re.search(r"labeled hazards: 0 -> recall not measured", text)
    assert "n/a" in text
    for c in CLASSES:
        assert c in text


def test_classes_without_labels_show_na_recall_even_when_others_have_labels(full):
    db, d, log = full
    text = render(db, d, cost_log=str(log))
    deb = [l for l in text.splitlines() if l.startswith("|") and "debris in lane" in l]
    assert deb and all("n/a" in l for l in deb[:2]), "a class with no labeled hazards shows n/a, never 0%"
    guard = [l for l in text.splitlines() if l.startswith("|") and "damaged guardrail" in l]
    assert guard and all("n/a" in l for l in guard)


def test_no_cost_log_prints_na(full):
    db, d, log = full
    text = render(db, d)
    assert_clean(text)
    low = text.lower()
    assert "cost" in low
    cost_lines = [l for l in text.splitlines() if "cost" in l.lower()]
    assert any("n/a" in l for l in cost_lines)


def test_empty_database_report(tmp_path):
    text = render(make_db(tmp_path), tmp_path / "labels")
    assert_clean(text)
    assert "recall not measured" in text


def test_unfinished_drives_have_no_invented_timing(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV", duration=60.0)
    text = render(db, tmp_path / "labels")
    assert_clean(text)
    lines = [l for l in text.splitlines() if "processing" in l.lower() or "seconds per minute" in l.lower()]
    assert lines and any("n/a" in l for l in lines)


def test_report_never_includes_image_paths_or_pixel_data(full, tmp_path):
    db, d, log = full
    text = render(db, d, cost_log=str(log))
    low = text.lower()
    assert ".jpg" not in low and ".png" not in low and ".jpeg" not in low
    assert "best_frame" not in low and "crop_path" not in low and "photo_path" not in low
    assert "base64" not in low and "data:image" not in low
    assert str(tmp_path).lower() not in low
    assert "storage" not in low
    assert not re.search(r"[a-z]:\\|[a-z]:/users", low)


def test_report_has_no_wall_clock_so_it_is_deterministic(full):
    db, d, log = full
    assert render(db, d, cost_log=str(log)) == render(db, d, cost_log=str(log))


def test_report_survives_cp1252(full):
    db, d, log = full
    render(db, d, cost_log=str(log)).encode("cp1252")


def test_render_markdown_does_not_touch_the_database(full):
    db, d, log = full
    before = db.fingerprint()
    render(db, d, cost_log=str(log))
    assert db.fingerprint() == before


# ------------------------------------------------------------------ the command line

def test_cli_prints_markdown_and_writes_no_file(full):
    db, d, log = full
    out = ROOT / "docs" / "eval-report.md"
    before = (out.exists(), out.stat().st_mtime_ns if out.exists() else None)
    cp = run_cli_v1(["--labels", d, "--db", db.path, "--cost-log", log])
    assert cp.returncode == 0, cp.stderr
    assert_clean(cp.stdout)
    assert "SAM only" in cp.stdout and "SAM + Muse Spark" in cp.stdout
    assert (out.exists(), out.stat().st_mtime_ns if out.exists() else None) == before, \
        "docs/eval-report.md is written only when --out is given"


def test_cli_output_equals_render_markdown(full):
    db, d, log = full
    cp = run_cli_v1(["--labels", d, "--db", db.path, "--cost-log", log])
    assert cp.returncode == 0
    assert cp.stdout.strip() == render(db, d, cost_log=str(log)).strip()


def test_cli_out_writes_the_file(full, tmp_path):
    db, d, log = full
    out = tmp_path / "report.md"
    cp = run_cli_v1(["--labels", d, "--db", db.path, "--cost-log", log, "--out", out])
    assert cp.returncode == 0, cp.stderr
    text = out.read_text(encoding="utf-8")
    assert text.strip() == render(db, d, cost_log=str(log)).strip()
    assert "Traceback" not in cp.stdout + cp.stderr


def test_cli_out_overwrites_an_existing_file(full, tmp_path):
    db, d, log = full
    out = tmp_path / "report.md"
    out.write_text("OLD CONTENT", encoding="utf-8")
    cp = run_cli_v1(["--labels", d, "--db", db.path, "--out", out])
    assert cp.returncode == 0
    assert "OLD CONTENT" not in out.read_text(encoding="utf-8")


def test_cli_out_in_a_new_folder_is_created_or_clearly_refused(full, tmp_path):
    db, d, log = full
    out = tmp_path / "newdir" / "report.md"
    cp = run_cli_v1(["--labels", d, "--db", db.path, "--out", out])
    if cp.returncode == 0:
        assert out.is_file()
    else:
        assert cp.stderr.strip() and "Traceback" not in cp.stderr


@pytest.mark.parametrize("kind", ["missing", "empty", "examples"])
def test_cli_without_usable_labels_works(tmp_path, kind):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.candidate("d1", "pothole", 62.0)
    d = tmp_path / "lab"
    if kind == "empty":
        d.mkdir()
    if kind == "examples":
        write_labels(d, "IMG_1", [seg("A", "00:00", "02:30", "EXAMPLE ROW: x"),
                                  haz("A", "01:12", "01:16", "pothole", "1", "H1", "EXAMPLE ROW: y")])
    cp = run_cli_v1(["--labels", d, "--db", db.path])
    assert cp.returncode == 0, cp.stderr
    assert "recall not measured" in cp.stdout
    assert "Traceback" not in cp.stdout + cp.stderr


def test_cli_empty_database(tmp_path):
    db = make_db(tmp_path)
    cp = run_cli_v1(["--labels", tmp_path / "nolabels", "--db", db.path])
    assert cp.returncode == 0, cp.stderr
    assert "recall not measured" in cp.stdout


def test_cli_bad_label_file_fails_clearly_and_writes_nothing(full, tmp_path):
    db, _, log = full
    bad = tmp_path / "badlabels"
    write_labels(bad, "IMG_9001", [seg("A", "00:00", "01:00"), haz("A", "00:10", "00:12", "sinkhole")])
    out = tmp_path / "out.md"
    cp = run_cli_v1(["--labels", bad, "--db", db.path, "--out", out])
    assert cp.returncode != 0
    assert "IMG_9001" in cp.stderr and re.search(r"(?i)(line\s*3\b|:3\b)", cp.stderr), cp.stderr
    assert "Traceback" not in cp.stderr
    assert not out.exists(), "no half report"
    assert cp.stdout.strip() == ""


def test_cli_missing_db_fails_and_creates_nothing(tmp_path):
    missing = tmp_path / "gone.sqlite"
    cp = run_cli_v1(["--labels", tmp_path / "labels", "--db", missing])
    assert cp.returncode != 0
    assert not missing.exists()
    assert "Traceback" not in cp.stderr


def test_cli_invalid_tolerance_is_a_usage_error(full):
    db, d, log = full
    assert run_cli_v1(["--labels", d, "--db", db.path, "--tolerance", "soon"]).returncode != 0


def test_cli_tolerance_changes_the_matching(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz("A", "01:00", "01:05", "pothole")])
    db.candidate("d1", "pothole", 67.5)
    narrow = run_cli_v1(["--labels", d, "--db", db.path, "--tolerance", "2"])
    wide = run_cli_v1(["--labels", d, "--db", db.path, "--tolerance", "3"])
    assert narrow.returncode == 0 and wide.returncode == 0
    assert narrow.stdout != wide.stdout
    assert run_cli_v1(["--labels", d, "--db", db.path]).stdout == narrow.stdout


def test_cli_cost_log_changes_the_cost_line(full):
    db, d, log = full
    without = run_cli_v1(["--labels", d, "--db", db.path])
    with_log = run_cli_v1(["--labels", d, "--db", db.path, "--cost-log", log])
    assert without.returncode == 0 and with_log.returncode == 0
    assert without.stdout != with_log.stdout
    assert re.search(r"0\.33", with_log.stdout)


def test_cli_uses_roadscout_db_when_db_is_not_given(full):
    db, d, log = full
    cp = run_cli_v1(["--labels", d], env_over={"ROADSCOUT_DB": str(db.path)})
    assert cp.returncode == 0, cp.stderr
    assert "SAM only" in cp.stdout


def test_cli_default_labels_dir_is_the_repos_labels(tmp_path):
    db = make_db(tmp_path)
    cp = run_cli_v1(["--db", db.path], cwd=ROOT)
    assert cp.returncode == 0, cp.stderr
    assert_clean(cp.stdout)


def test_cli_never_writes_the_database_or_needs_a_key(full):
    db, d, log = full
    before = db.fingerprint()
    for extra in ([], ["--cost-log", log], ["--tolerance", "5"]):
        cp = run_cli_v1(["--labels", d, "--db", db.path, *extra],
                        env_over={"META_API_KEY": "", "ROADSCOUT_PROVIDER": "live"})
        assert cp.returncode == 0, cp.stderr
        assert "META_API_KEY" not in cp.stdout + cp.stderr
    assert db.fingerprint() == before
    assert not list(db.path.parent.glob("*.sqlite-wal"))


def test_cli_help_lists_the_options():
    cp = run_cli_v1(["--help"])
    assert cp.returncode == 0
    for opt in ("--labels", "--db", "--cost-log", "--tolerance", "--out"):
        assert opt in cp.stdout


def test_cli_output_has_no_paths_or_pixels(full):
    db, d, log = full
    cp = run_cli_v1(["--labels", d, "--db", db.path, "--cost-log", log])
    low = cp.stdout.lower()
    assert ".jpg" not in low and "base64" not in low and "best_frame" not in low
