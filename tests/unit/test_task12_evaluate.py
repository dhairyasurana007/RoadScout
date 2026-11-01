"""Task 12: evaluate() - label-based recall/precision, false alarms, review metrics, drive filter, read-only."""
from __future__ import annotations

import dataclasses
import inspect
import sqlite3

import pytest

from rs12_helpers import CLASSES, evaluate, haz, make_db, seg, write_labels

A = ("A", "00:00", "10:00")          # segment A = 0..600 s
LAB = ("A", "01:00", "01:05")        # a labelled hazard at 60..65 s


@pytest.fixture()
def world(tmp_path):
    """db + labels dir with drive d1 = IMG_1.MOV and a pothole label at 60..65 s inside segment A (0..600 s)."""
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz(*LAB, "pothole", "2", "H1")])
    return db, d


def ev(db, d, **kw):
    return evaluate(db, d, **kw)


def zero_class(m):
    return (m.labeled, m.matched, m.accepted_in_segments, m.matched_accepted) == (0, 0, 0, 0) \
        and m.recall is None and m.precision is None


# ------------------------------------------------------------------ the report shape

def test_report_types_and_all_classes_present(world):
    from backend.evaluation import ClassMetrics, EvalReport, ReviewMetrics

    db, d = world
    r = ev(db, d)
    assert isinstance(r, EvalReport) and dataclasses.is_dataclass(r)
    assert set(r.per_class) == set(CLASSES) and set(r.review) == set(CLASSES)
    assert all(isinstance(m, ClassMetrics) for m in r.per_class.values())
    assert all(isinstance(m, ReviewMetrics) for m in r.review.values())
    for k in ("unreviewed", "false_alarms", "clean_seconds", "false_alarms_per_hour", "has_labels", "skipped_drives"):
        assert hasattr(r, k)


def test_default_tolerance_is_two_seconds():
    from backend.evaluation import evaluate as f

    assert inspect.signature(f).parameters["tolerance"].default == 2.0
    assert inspect.signature(f).parameters["drive_id"].default is None


# ------------------------------------------------------------------ the basics

def test_matched_candidate_gives_recall_and_precision_of_one(world):
    db, d = world
    db.candidate("d1", "pothole", 62.0)
    r = ev(db, d)
    m = r.per_class["pothole"]
    assert (m.labeled, m.matched, m.accepted_in_segments, m.matched_accepted) == (1, 1, 1, 1)
    assert m.recall == pytest.approx(1.0) and m.precision == pytest.approx(1.0)
    assert r.has_labels is True
    for c in CLASSES[1:]:
        assert zero_class(r.per_class[c])
    assert r.false_alarms == 0


def test_no_candidates_means_zero_recall_and_unmeasured_precision(world):
    db, d = world
    r = ev(db, d)
    m = r.per_class["pothole"]
    assert (m.labeled, m.matched) == (1, 0)
    assert m.recall == 0.0, "zero matched of one labelled is a measured 0, not n/a"
    assert m.accepted_in_segments == 0 and m.precision is None


def test_a_class_without_labels_has_no_recall_and_never_crashes(world):
    db, d = world
    db.candidate("d1", "debris in lane", 300.0)       # accepted, inside the segment, nothing labelled
    r = ev(db, d)
    m = r.per_class["debris in lane"]
    assert m.labeled == 0 and m.matched == 0 and m.recall is None
    assert m.accepted_in_segments == 1 and m.matched_accepted == 0 and m.precision == 0.0


def test_fractional_recall(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz("A", "01:00", "01:05", "pothole"), haz("A", "05:00", "05:05", "pothole")])
    db.candidate("d1", "pothole", 61.0)
    m = ev(db, d).per_class["pothole"]
    assert (m.labeled, m.matched) == (2, 1) and m.recall == pytest.approx(0.5)


def test_fractional_precision(world):
    db, d = world
    db.candidate("d1", "pothole", 62.0)
    db.candidate("d1", "pothole", 400.0)
    db.candidate("d1", "pothole", 500.0)
    m = ev(db, d).per_class["pothole"]
    assert (m.accepted_in_segments, m.matched_accepted) == (3, 1)
    assert m.precision == pytest.approx(1 / 3)
    assert m.recall == pytest.approx(1.0)


def test_each_class_is_scored_independently(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz("A", "01:00", "01:05", "pothole"),
                              haz("A", "02:00", "02:05", "damaged guardrail"),
                              haz("A", "03:00", "03:05", "faded lane markings")])
    db.candidate("d1", "pothole", 62.0)
    db.candidate("d1", "damaged guardrail", 122.0)
    r = ev(db, d)
    assert r.per_class["pothole"].recall == 1.0
    assert r.per_class["damaged guardrail"].recall == 1.0
    assert r.per_class["faded lane markings"].recall == 0.0
    assert r.per_class["faded lane markings"].precision is None
    assert zero_class(r.per_class["debris in lane"])


# ------------------------------------------------------------------ matching rules

@pytest.mark.parametrize("t,hit", [(58.0, True), (57.9, False), (67.0, True), (67.1, False), (62.0, True)])
def test_tolerance_boundary_default(world, t, hit):
    db, d = world
    db.candidate("d1", "pothole", t)
    m = ev(db, d).per_class["pothole"]
    assert m.matched == (1 if hit else 0)
    assert m.matched_accepted == (1 if hit else 0)
    assert m.accepted_in_segments == 1, "inside the segment either way"


@pytest.mark.parametrize("tol,t,hit", [(0.0, 60.0, True), (0.0, 65.0, True), (0.0, 65.1, False), (0.0, 59.9, False),
                                       (0.5, 65.4, True), (0.5, 65.6, False), (10.0, 74.9, True), (10.0, 75.1, False),
                                       (10.0, 50.1, True), (10.0, 49.9, False)])
def test_tolerance_argument(world, tol, t, hit):
    db, d = world
    db.candidate("d1", "pothole", t)
    assert ev(db, d, tolerance=tol).per_class["pothole"].matched == (1 if hit else 0)


def test_wrong_class_near_the_right_time_does_not_match(world):
    db, d = world
    db.candidate("d1", "debris in lane", 62.0)
    r = ev(db, d)
    assert r.per_class["pothole"].matched == 0 and r.per_class["pothole"].recall == 0.0
    deb = r.per_class["debris in lane"]
    assert (deb.accepted_in_segments, deb.matched_accepted) == (1, 0) and deb.precision == 0.0
    assert r.false_alarms == 0, "the segment is not clean, so this is not a clean-segment false alarm"


def test_duplicate_candidates_on_one_label_count_once_for_recall(world):
    db, d = world
    db.candidate("d1", "pothole", 61.0)
    db.candidate("d1", "pothole", 63.0)
    db.candidate("d1", "pothole", 66.0)
    m = ev(db, d).per_class["pothole"]
    assert (m.labeled, m.matched) == (1, 1) and m.recall == 1.0
    assert (m.accepted_in_segments, m.matched_accepted) == (3, 3) and m.precision == 1.0


def test_one_candidate_between_two_labels_matches_both_and_counts_once(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz("A", "01:00", "01:05", "pothole"), haz("A", "01:06", "01:10", "pothole")])
    db.candidate("d1", "pothole", 65.5)
    m = ev(db, d).per_class["pothole"]
    assert (m.labeled, m.matched) == (2, 2)
    assert (m.accepted_in_segments, m.matched_accepted) == (1, 1)


def test_candidate_matches_only_labels_of_its_own_drive(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.drive("d2", "IMG_2.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz(*LAB, "pothole")])
    write_labels(d, "IMG_2", [seg(*A)])
    db.candidate("d2", "pothole", 62.0)
    r = ev(db, d)
    m = r.per_class["pothole"]
    assert m.labeled == 1 and m.matched == 0, "a candidate of drive 2 cannot satisfy drive 1's label"
    assert m.accepted_in_segments == 1 and m.matched_accepted == 0
    assert r.false_alarms == 1, "drive 2's segment is clean"


def test_not_accepted_candidates_are_ignored(world):
    db, d = world
    db.candidate("d1", "pothole", 62.0, accepted=0)
    db.candidate("d1", "pothole", 63.0, verdict=False)
    r = ev(db, d)
    m = r.per_class["pothole"]
    assert m.matched == 0 and m.accepted_in_segments == 0 and m.precision is None
    assert r.false_alarms == 0


def test_candidates_outside_every_segment_are_not_counted(world):
    db, d = world
    db.candidate("d1", "pothole", 900.0)
    db.candidate("d1", "pothole", 601.0)
    r = ev(db, d)
    m = r.per_class["pothole"]
    assert m.accepted_in_segments == 0 and m.precision is None and r.false_alarms == 0


def test_accepted_in_segments_has_no_tolerance_at_the_segment_edge(world):
    db, d = world
    db.candidate("d1", "pothole", 600.0)      # exactly at the end: inside
    db.candidate("d1", "pothole", 600.5)      # just outside: not counted
    assert ev(db, d).per_class["pothole"].accepted_in_segments == 1


def test_candidate_at_the_start_edge_counts(world):
    db, d = world
    db.candidate("d1", "pothole", 0.0)
    assert ev(db, d).per_class["pothole"].accepted_in_segments == 1


# ------------------------------------------------------------------ clean segments

def test_false_alarm_in_a_clean_segment(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz(*LAB, "pothole"), seg("B", "12:00", "13:00")])   # B: 60 s, clean
    db.candidate("d1", "pothole", 62.0)
    db.candidate("d1", "debris in lane", 750.0)       # inside the clean segment
    r = ev(db, d)
    assert r.false_alarms == 1
    assert r.clean_seconds == pytest.approx(60.0)
    assert r.false_alarms_per_hour == pytest.approx(60.0)
    deb = r.per_class["debris in lane"]
    assert deb.accepted_in_segments == 1 and deb.matched_accepted == 0 and deb.precision == 0.0


def test_clean_segment_without_alarms_is_zero_per_hour(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg("B", "00:00", "30:00")])
    r = ev(db, d)
    assert r.clean_seconds == pytest.approx(1800.0)
    assert r.false_alarms == 0 and r.false_alarms_per_hour == 0.0


def test_no_clean_segments_means_no_rate(world):
    db, d = world
    r = ev(db, d)
    assert r.clean_seconds == 0.0 and r.false_alarms_per_hour is None


def test_clean_time_adds_up_over_segments_and_drives(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.drive("d2", "IMG_2.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg("B", "00:00", "30:00"), seg("C", "40:00", "50:00")])
    write_labels(d, "IMG_2", [seg("B", "00:00", "20:00")])
    db.candidate("d1", "pothole", 100.0)
    db.candidate("d1", "pothole", 2500.0)
    db.candidate("d2", "faded lane markings", 5.0)
    r = ev(db, d)
    assert r.clean_seconds == pytest.approx(1800 + 600 + 1200)
    assert r.false_alarms == 3
    assert r.false_alarms_per_hour == pytest.approx(3 / (3600 / 3600.0))


def test_several_alarms_in_one_clean_hour(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg("B", "00:00", "1:00:00")])
    for t in (10, 20, 30, 40):
        db.candidate("d1", "pothole", t)
    r = ev(db, d)
    assert r.false_alarms == 4 and r.false_alarms_per_hour == pytest.approx(4.0)


# ------------------------------------------------------------------ which drives and labels

def test_label_file_matches_by_video_filename_stem_case_insensitively(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_6513.MOV")
    db.drive("d2", "clip.Final.mp4")
    d = tmp_path / "labels"
    write_labels(d, "img_6513", [seg(*A), haz(*LAB, "pothole")])
    write_labels(d, "CLIP.FINAL", [seg(*A), haz(*LAB, "debris in lane")])
    db.candidate("d1", "pothole", 62.0)
    db.candidate("d2", "debris in lane", 62.0)
    r = ev(db, d)
    assert r.per_class["pothole"].matched == 1 and r.per_class["debris in lane"].matched == 1
    assert r.skipped_drives == []


def test_label_file_matches_by_drive_id(tmp_path):
    db = make_db(tmp_path)
    db.drive("0123abcd", None)                       # no video_filename
    d = tmp_path / "labels"
    write_labels(d, "0123abcd", [seg(*A), haz(*LAB, "pothole")])
    db.candidate("0123abcd", "pothole", 61.0)
    r = ev(db, d)
    assert r.per_class["pothole"].matched == 1 and r.skipped_drives == []


def test_stem_is_the_name_without_the_last_extension_only(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1.MOV", [seg(*A), haz(*LAB, "pothole")])     # stem "img_1.mov" must NOT match
    db.candidate("d1", "pothole", 62.0)
    r = ev(db, d)
    assert r.per_class["pothole"].matched == 0
    assert r.skipped_drives == ["d1"]


def test_drive_without_a_label_file_is_skipped_for_label_metrics(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.drive("d2", "IMG_2.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz(*LAB, "pothole")])
    db.candidate("d1", "pothole", 62.0)
    db.candidate("d2", "pothole", 62.0)
    db.candidate("d2", "debris in lane", 100.0)
    r = ev(db, d)
    assert r.skipped_drives == ["d2"]
    m = r.per_class["pothole"]
    assert m.accepted_in_segments == 1 and m.precision == 1.0
    assert r.per_class["debris in lane"].accepted_in_segments == 0
    assert r.false_alarms == 0


def test_empty_database_and_labels(tmp_path):
    db = make_db(tmp_path)
    r = ev(db, tmp_path / "no_labels_here")
    assert r.has_labels is False and r.false_alarms == 0 and r.unreviewed == 0
    assert all(zero_class(m) for m in r.per_class.values())
    assert all((m.approved, m.rejected, m.precision) == (0, 0, None) for m in r.review.values())
    assert r.false_alarms_per_hour is None and r.clean_seconds == 0.0


def test_drive_filter_restricts_labels_and_candidates(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.drive("d2", "IMG_2.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz(*LAB, "pothole")])
    write_labels(d, "IMG_2", [seg(*A), haz("A", "02:00", "02:05", "debris in lane"), seg("B", "20:00", "21:00")])
    db.candidate("d1", "pothole", 62.0)
    db.candidate("d2", "debris in lane", 122.0)
    db.candidate("d2", "pothole", 1230.0)
    r1 = ev(db, d, drive_id="d1")
    assert r1.per_class["pothole"].labeled == 1 and r1.per_class["pothole"].matched == 1
    assert r1.per_class["debris in lane"].labeled == 0
    assert r1.false_alarms == 0 and r1.clean_seconds == 0.0
    r2 = ev(db, d, drive_id="d2")
    assert r2.per_class["pothole"].labeled == 0
    assert r2.per_class["debris in lane"].matched == 1
    assert r2.false_alarms == 1 and r2.clean_seconds == pytest.approx(60.0)
    both = ev(db, d)
    assert both.per_class["pothole"].labeled == 1 and both.per_class["debris in lane"].labeled == 1


def test_drive_filter_by_a_drive_without_labels(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.drive("d2", "IMG_2.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz(*LAB, "pothole")])
    r = ev(db, d, drive_id="d2")
    assert r.skipped_drives == ["d2"]
    assert r.per_class["pothole"].labeled == 0


def test_unknown_drive_is_an_error_naming_the_id(world):
    db, d = world
    with pytest.raises(Exception) as ei:
        ev(db, d, drive_id="no-such-drive")
    assert "no-such-drive" in str(ei.value)


# ------------------------------------------------------------------ review-based metrics

def test_review_precision_per_class(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    for status, cls in [("approved", "pothole"), ("approved", "pothole"), ("rejected", "pothole"),
                        ("new", "pothole"), ("rejected", "debris in lane"), ("new", "debris in lane"),
                        ("new", "damaged guardrail")]:
        h = db.hazard(cls, status)
        db.candidate("d1", cls, 100.0, hazard_id=h)
    r = ev(db, tmp_path / "none")
    p = r.review["pothole"]
    assert (p.approved, p.rejected) == (2, 1) and p.precision == pytest.approx(2 / 3)
    deb = r.review["debris in lane"]
    assert (deb.approved, deb.rejected) == (0, 1) and deb.precision == 0.0
    g = r.review["damaged guardrail"]
    assert (g.approved, g.rejected) == (0, 0) and g.precision is None
    assert r.unreviewed == 3, "new hazards are excluded from the ratio and counted separately"
    assert r.has_labels is False


def test_review_metrics_work_without_a_label_file_for_the_drive(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.drive("d2", "IMG_2.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz(*LAB, "pothole")])
    h = db.hazard("pothole", "approved")
    db.candidate("d2", "pothole", 100.0, hazard_id=h)
    r = ev(db, d)
    assert "d2" in r.skipped_drives
    assert r.review["pothole"].approved == 1 and r.review["pothole"].precision == 1.0


def test_review_metrics_follow_the_drive_filter(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.drive("d2", "IMG_2.MOV")
    h1 = db.hazard("pothole", "approved")
    h2 = db.hazard("pothole", "rejected")
    db.candidate("d1", "pothole", 100.0, hazard_id=h1)
    db.candidate("d2", "pothole", 100.0, hazard_id=h2)
    r1 = ev(db, tmp_path / "none", drive_id="d1")
    assert (r1.review["pothole"].approved, r1.review["pothole"].rejected) == (1, 0)
    r2 = ev(db, tmp_path / "none", drive_id="d2")
    assert (r2.review["pothole"].approved, r2.review["pothole"].rejected) == (0, 1)
    allr = ev(db, tmp_path / "none")
    assert (allr.review["pothole"].approved, allr.review["pothole"].rejected) == (1, 1)


def test_a_hazard_with_two_candidates_is_counted_once(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    h = db.hazard("pothole", "approved")
    db.candidate("d1", "pothole", 100.0, hazard_id=h)
    db.candidate("d1", "pothole", 110.0, hazard_id=h)
    assert ev(db, tmp_path / "none").review["pothole"].approved == 1


def test_review_and_label_metrics_are_independent(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz(*LAB, "pothole")])
    h = db.hazard("pothole", "rejected")
    db.candidate("d1", "pothole", 62.0, hazard_id=h)       # matches the label, but the user rejected the hazard
    r = ev(db, d)
    assert r.per_class["pothole"].recall == 1.0 and r.per_class["pothole"].precision == 1.0
    assert r.review["pothole"].approved == 0 and r.review["pothole"].rejected == 1


# ------------------------------------------------------------------ read-only, tolerant of old databases

def test_evaluate_never_writes_the_database(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz(*LAB, "pothole")])
    h = db.hazard("pothole", "approved")
    db.candidate("d1", "pothole", 62.0, hazard_id=h)
    before = db.fingerprint()
    ev(db, d)
    ev(db, d, drive_id="d1", tolerance=5.0)
    assert db.fingerprint() == before
    assert not list(tmp_path.glob("*.sqlite-wal")) and not list(tmp_path.glob("*.sqlite-journal"))


def test_a_missing_database_is_an_error_and_is_not_created(tmp_path):
    from backend.evaluation import evaluate as f, load_labels

    missing = tmp_path / "nope.sqlite"
    with pytest.raises(Exception):
        f(str(missing), load_labels(tmp_path / "labels"))
    assert not missing.exists()


def test_an_old_database_without_the_video_filename_column_still_works(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", None)
    d = tmp_path / "labels"
    write_labels(d, "d1", [seg(*A), haz(*LAB, "pothole")])        # matched by drive id
    db.candidate("d1", "pothole", 62.0)
    con = sqlite3.connect(str(db.path))
    try:
        try:
            con.execute("ALTER TABLE drives DROP COLUMN video_filename")
        except sqlite3.OperationalError as e:
            pytest.skip("this SQLite cannot drop a column (%s)" % e)
        con.commit()
    finally:
        con.close()
    before = db.fingerprint()
    r = ev(db, d)
    assert r.per_class["pothole"].matched == 1
    assert db.fingerprint() == before, "read-only: the missing column must not be added"


def test_no_provider_and_no_key_are_needed(world, monkeypatch):
    db, d = world
    monkeypatch.setenv("ROADSCOUT_PROVIDER", "live")
    monkeypatch.delenv("META_API_KEY", raising=False)
    db.candidate("d1", "pothole", 62.0)
    assert ev(db, d).per_class["pothole"].recall == 1.0


def test_results_are_deterministic(world):
    db, d = world
    db.candidate("d1", "pothole", 62.0)
    db.candidate("d1", "debris in lane", 200.0)
    assert dataclasses.asdict(ev(db, d)) == dataclasses.asdict(ev(db, d))
