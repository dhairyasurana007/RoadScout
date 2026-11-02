"""Task 20: evaluate_v1 - the ablation of SAM alone against SAM + Muse Spark on the labeled segments."""
from __future__ import annotations

import dataclasses
import inspect

import pytest

from rs20_helpers import CLASSES, evaluate_v1, haz, make_db, seg, write_labels

A = ("A", "00:00", "10:00")          # labeled segment A = 0..600 s
LAB = ("A", "01:00", "01:05")        # labeled pothole at 60..65 s


def build(tmp_path, rows=None, cands=(), drive="d1", stem="IMG_1"):
    db = make_db(tmp_path)
    db.drive(drive, stem + ".MOV")
    d = tmp_path / "labels"
    write_labels(d, stem, rows if rows is not None else [seg(*A), haz(*LAB, "pothole", "2", "H1")])
    for cls, t, kw in cands:
        db.candidate(drive, cls, t, **kw)
    return db, d


def rows(r, cls):
    return r.ablation[cls]["sam_only"], r.ablation[cls]["sam_muse"]


# ------------------------------------------------------------------ shape

def test_report_shape_and_keys(tmp_path):
    from backend.eval_v1 import EvalV1Report

    db, d = build(tmp_path)
    r = evaluate_v1(db, d)
    assert isinstance(r, EvalV1Report) and dataclasses.is_dataclass(r)
    assert set(r.ablation) == set(CLASSES) | {"overall"}
    for key, per in r.ablation.items():
        assert set(per) == {"sam_only", "sam_muse"}, key
    for name in ("review", "unreviewed", "dedup", "cost", "timing", "has_labels", "skipped_drives", "clean_seconds"):
        assert hasattr(r, name), name
    row = r.ablation["pothole"]["sam_only"]
    for f in ("labeled", "matched", "recall", "detections", "matched_detections", "precision", "false_alarms",
              "false_alarms_per_hour"):
        assert hasattr(row, f), f
    assert hasattr(r.ablation["pothole"]["sam_muse"], "removed_by_verifier")


def test_signature_defaults():
    from backend.eval_v1 import evaluate_v1 as ev, render_markdown

    p = inspect.signature(ev).parameters
    assert p["tolerance"].default == 2.0 and p["cost_log"].default is None and p["drive_id"].default is None
    assert callable(render_markdown)


# ------------------------------------------------------------------ the core ablation

def mixed(tmp_path):
    return build(tmp_path, cands=[
        ("pothole", 62.0, {}),                                   # true, accepted
        ("pothole", 400.0, {"accepted": 0}),                     # false, verifier rejected it
        ("pothole", 450.0, {"verdict": False}),                  # false, never verified
        ("debris in lane", 300.0, {}),                           # false, accepted
    ])


def test_sam_only_counts_every_candidate_and_muse_only_accepted(tmp_path):
    db, d = mixed(tmp_path)
    r = evaluate_v1(db, d)
    sam, muse = rows(r, "pothole")
    assert (sam.labeled, sam.matched) == (1, 1) and sam.recall == pytest.approx(1.0)
    assert (muse.labeled, muse.matched) == (1, 1) and muse.recall == pytest.approx(1.0)
    assert (sam.detections, sam.matched_detections) == (3, 1)
    assert sam.precision == pytest.approx(1 / 3)
    assert (muse.detections, muse.matched_detections) == (1, 1)
    assert muse.precision == pytest.approx(1.0)
    assert muse.removed_by_verifier == 2


def test_other_class_and_overall_rows(tmp_path):
    db, d = mixed(tmp_path)
    r = evaluate_v1(db, d)
    sam, muse = rows(r, "debris in lane")
    assert (sam.labeled, sam.recall) == (0, None) and (muse.labeled, muse.recall) == (0, None)
    assert (sam.detections, muse.detections) == (1, 1)
    assert sam.precision == 0.0 and muse.precision == 0.0
    assert muse.removed_by_verifier == 0
    osam, omuse = rows(r, "overall")
    assert (osam.detections, osam.matched_detections) == (4, 1)
    assert (omuse.detections, omuse.matched_detections) == (2, 1)
    assert osam.precision == pytest.approx(1 / 4) and omuse.precision == pytest.approx(1 / 2)
    assert omuse.removed_by_verifier == 2
    assert (osam.labeled, osam.matched, osam.recall) == (1, 1, pytest.approx(1.0))


def test_classes_without_anything_have_zero_counts_and_no_rates(tmp_path):
    db, d = mixed(tmp_path)
    r = evaluate_v1(db, d)
    for cls in ("damaged guardrail", "fallen or damaged sign", "faded lane markings"):
        for row in rows(r, cls):
            assert (row.labeled, row.matched, row.detections, row.matched_detections) == (0, 0, 0, 0)
            assert row.recall is None and row.precision is None
        assert rows(r, cls)[1].removed_by_verifier == 0


def test_the_verifier_can_lose_a_true_hazard(tmp_path):
    db, d = build(tmp_path, cands=[("pothole", 62.0, {"accepted": 0})])
    sam, muse = rows(evaluate_v1(db, d), "pothole")
    assert sam.recall == pytest.approx(1.0)
    assert muse.recall == 0.0, "a measured zero, not n/a: one labeled hazard and none matched"
    assert muse.detections == 0 and muse.precision is None
    assert muse.removed_by_verifier == 1


def test_no_candidates_at_all(tmp_path):
    db, d = build(tmp_path)
    sam, muse = rows(evaluate_v1(db, d), "pothole")
    assert sam.recall == 0.0 and muse.recall == 0.0
    assert sam.precision is None and muse.precision is None
    assert muse.removed_by_verifier == 0


def test_candidates_without_a_verdict_row_are_sam_only(tmp_path):
    db, d = build(tmp_path, cands=[("pothole", 62.0, {"verdict": False})])
    sam, muse = rows(evaluate_v1(db, d), "pothole")
    assert sam.matched == 1 and muse.matched == 0
    assert sam.detections == 1 and muse.detections == 0


def test_candidates_outside_every_segment_are_ignored_in_both_columns(tmp_path):
    db, d = build(tmp_path, cands=[("pothole", 900.0, {}), ("pothole", 900.0, {"accepted": 0}),
                                   ("pothole", 601.0, {})])
    r = evaluate_v1(db, d)
    for row in rows(r, "pothole"):
        assert row.detections == 0 and row.precision is None
    assert rows(r, "pothole")[1].removed_by_verifier == 0
    assert rows(r, "overall")[0].false_alarms == 0


def test_wrong_class_near_the_right_time_is_not_a_match(tmp_path):
    db, d = build(tmp_path, cands=[("debris in lane", 62.0, {})])
    r = evaluate_v1(db, d)
    for row in rows(r, "pothole"):
        assert row.matched == 0 and row.recall == 0.0
    for row in rows(r, "debris in lane"):
        assert (row.detections, row.matched_detections, row.precision) == (1, 0, 0.0)
    assert rows(r, "overall")[0].false_alarms == 0, "the segment is not clean"


def test_duplicate_detections_on_one_label(tmp_path):
    db, d = build(tmp_path, cands=[("pothole", 61.0, {}), ("pothole", 63.0, {}), ("pothole", 64.0, {"accepted": 0})])
    sam, muse = rows(evaluate_v1(db, d), "pothole")
    assert (sam.labeled, sam.matched, sam.detections, sam.matched_detections) == (1, 1, 3, 3)
    assert (muse.matched, muse.detections, muse.matched_detections) == (1, 2, 2)
    assert muse.removed_by_verifier == 1


@pytest.mark.parametrize("t,hit", [(58.0, True), (57.9, False), (67.0, True), (67.1, False)])
def test_default_tolerance_boundary_for_both_columns(tmp_path, t, hit):
    db, d = build(tmp_path, cands=[("pothole", t, {})])
    for row in rows(evaluate_v1(db, d), "pothole"):
        assert row.matched == (1 if hit else 0)


def test_tolerance_argument(tmp_path):
    db, d = build(tmp_path, cands=[("pothole", 70.0, {})])
    assert rows(evaluate_v1(db, d), "pothole")[0].matched == 0
    assert rows(evaluate_v1(db, d, tolerance=5.0), "pothole")[0].matched == 1
    assert rows(evaluate_v1(db, d, tolerance=4.9), "pothole")[0].matched == 0


def test_overall_rates_come_from_sums_not_averages(tmp_path):
    rws = [seg(*A), haz("A", "01:00", "01:05", "pothole"), haz("A", "01:40", "01:41", "debris in lane"),
           haz("A", "03:20", "03:21", "debris in lane"), haz("A", "05:00", "05:01", "debris in lane")]
    db, d = build(tmp_path, rows=rws, cands=[("pothole", 62.0, {}), ("debris in lane", 450.0, {}),
                                              ("debris in lane", 460.0, {})])
    r = evaluate_v1(db, d)
    o = rows(r, "overall")[0]
    assert (o.labeled, o.matched) == (4, 1) and o.recall == pytest.approx(0.25)
    assert (o.detections, o.matched_detections) == (3, 1) and o.precision == pytest.approx(1 / 3)


def test_each_class_has_its_own_recall(tmp_path):
    rws = [seg(*A), haz("A", "01:00", "01:05", "pothole"), haz("A", "02:00", "02:05", "damaged guardrail"),
           haz("A", "03:00", "03:05", "damaged guardrail")]
    db, d = build(tmp_path, rows=rws, cands=[("pothole", 62.0, {}), ("damaged guardrail", 122.0, {})])
    r = evaluate_v1(db, d)
    assert rows(r, "pothole")[0].recall == 1.0
    assert rows(r, "damaged guardrail")[0].recall == pytest.approx(0.5)
    assert rows(r, "overall")[0].recall == pytest.approx(2 / 3)


# ------------------------------------------------------------------ false alarms per hour of clean time

def clean_world(tmp_path):
    rws = [seg(*A), haz(*LAB, "pothole"), seg("B", "12:00", "13:00")]        # B = 720..780 s, 60 s clean
    return build(tmp_path, rows=rws, cands=[
        ("pothole", 62.0, {}), ("pothole", 720.0, {}), ("debris in lane", 730.0, {"accepted": 0}),
        ("debris in lane", 740.0, {"verdict": False})])


def test_false_alarms_in_a_clean_segment_per_hour(tmp_path):
    db, d = clean_world(tmp_path)
    r = evaluate_v1(db, d)
    assert r.clean_seconds == pytest.approx(60.0)
    osam, omuse = rows(r, "overall")
    assert osam.false_alarms == 3 and osam.false_alarms_per_hour == pytest.approx(180.0)
    assert omuse.false_alarms == 1 and omuse.false_alarms_per_hour == pytest.approx(60.0)
    psam, pmuse = rows(r, "pothole")
    assert psam.false_alarms == 1 and pmuse.false_alarms == 1
    dsam, dmuse = rows(r, "debris in lane")
    assert dsam.false_alarms == 2 and dmuse.false_alarms == 0
    assert dmuse.false_alarms_per_hour == 0.0, "clean time exists, so zero is a measured zero"


def test_clean_alarms_also_count_as_unmatched_detections(tmp_path):
    db, d = clean_world(tmp_path)
    psam, pmuse = rows(evaluate_v1(db, d), "pothole")
    assert (psam.detections, psam.matched_detections) == (2, 1)
    assert pmuse.precision == pytest.approx(1 / 2)


def test_no_clean_segments_means_no_false_alarm_rate(tmp_path):
    db, d = mixed(tmp_path)
    r = evaluate_v1(db, d)
    assert r.clean_seconds == 0.0
    for row in rows(r, "overall"):
        assert row.false_alarms == 0 and row.false_alarms_per_hour is None


def test_clean_time_adds_up_over_drives(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.drive("d2", "IMG_2.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg("B", "00:00", "30:00")])
    write_labels(d, "IMG_2", [seg("B", "00:00", "30:00")])
    db.candidate("d1", "pothole", 10.0)
    db.candidate("d2", "pothole", 20.0, accepted=0)
    r = evaluate_v1(db, d)
    assert r.clean_seconds == pytest.approx(3600.0)
    osam, omuse = rows(r, "overall")
    assert osam.false_alarms == 2 and osam.false_alarms_per_hour == pytest.approx(2.0)
    assert omuse.false_alarms == 1 and omuse.false_alarms_per_hour == pytest.approx(1.0)


# ------------------------------------------------------------------ which drives and labels

def test_drive_without_a_label_file_is_skipped_for_the_ablation(tmp_path):
    db, d = build(tmp_path, cands=[("pothole", 62.0, {})])
    db.drive("d2", "IMG_2.MOV")
    db.candidate("d2", "pothole", 62.0)
    db.candidate("d2", "debris in lane", 100.0)
    r = evaluate_v1(db, d)
    assert r.skipped_drives == ["d2"]
    assert rows(r, "pothole")[0].detections == 1
    assert rows(r, "debris in lane")[0].detections == 0


def test_candidate_matches_only_its_own_drives_labels(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.drive("d2", "IMG_2.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz(*LAB, "pothole")])
    write_labels(d, "IMG_2", [seg(*A)])
    db.candidate("d2", "pothole", 62.0)
    sam, _ = rows(evaluate_v1(db, d), "pothole")
    assert sam.labeled == 1 and sam.matched == 0 and sam.detections == 1


def test_label_files_match_by_stem_case_insensitively_or_by_drive_id(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_6513.MOV")
    db.drive("0123abcd", None)
    d = tmp_path / "labels"
    write_labels(d, "img_6513", [seg(*A), haz(*LAB, "pothole")])
    write_labels(d, "0123abcd", [seg(*A), haz(*LAB, "debris in lane")])
    db.candidate("d1", "pothole", 62.0)
    db.candidate("0123abcd", "debris in lane", 62.0)
    r = evaluate_v1(db, d)
    assert r.skipped_drives == []
    assert rows(r, "pothole")[0].matched == 1 and rows(r, "debris in lane")[0].matched == 1


def test_drive_filter(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.drive("d2", "IMG_2.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz(*LAB, "pothole")])
    write_labels(d, "IMG_2", [seg(*A), haz("A", "02:00", "02:05", "debris in lane")])
    db.candidate("d1", "pothole", 62.0)
    db.candidate("d2", "debris in lane", 122.0)
    r1 = evaluate_v1(db, d, drive_id="d1")
    assert rows(r1, "pothole")[0].labeled == 1 and rows(r1, "debris in lane")[0].labeled == 0
    r2 = evaluate_v1(db, d, drive_id="d2")
    assert rows(r2, "pothole")[0].labeled == 0 and rows(r2, "debris in lane")[0].matched == 1


def test_unknown_drive_is_an_error_naming_the_id(tmp_path):
    db, d = build(tmp_path)
    with pytest.raises(Exception) as ei:
        evaluate_v1(db, d, drive_id="no-such-drive")
    assert "no-such-drive" in str(ei.value)


# ------------------------------------------------------------------ labels are optional

@pytest.mark.parametrize("kind", ["missing", "empty", "examples"])
def test_no_usable_labels(tmp_path, kind):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.candidate("d1", "pothole", 62.0)
    db.candidate("d1", "debris in lane", 100.0, accepted=0)
    d = tmp_path / "labels"
    if kind == "empty":
        d.mkdir()
    if kind == "examples":
        write_labels(d, "IMG_1", [seg("A", "00:00", "02:30", "EXAMPLE ROW: x"),
                                  haz("A", "01:12", "01:16", "pothole", "1", "H1", "EXAMPLE ROW: y")])
    r = evaluate_v1(db, d)
    assert r.has_labels is False
    assert r.clean_seconds == 0.0
    for cls, per in r.ablation.items():
        for row in per.values():
            assert row.labeled == 0 and row.matched == 0 and row.recall is None
            assert row.detections == 0 and row.precision is None
            assert row.false_alarms == 0 and row.false_alarms_per_hour is None


def test_empty_database(tmp_path):
    db = make_db(tmp_path)
    r = evaluate_v1(db, tmp_path / "nolabels")
    assert r.has_labels is False and r.unreviewed == 0 and r.skipped_drives == []
    assert r.dedup.merged_correctly == r.dedup.split_wrongly == r.dedup.merged_wrongly == 0
    assert r.cost.usd_per_minute is None and r.timing.seconds_per_minute is None


# ------------------------------------------------------------------ review-based precision

def test_review_precision_counts_reported_as_approved(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    for status, cls in [("approved", "pothole"), ("reported", "pothole"), ("rejected", "pothole"),
                        ("new", "pothole"), ("rejected", "debris in lane"), ("new", "debris in lane")]:
        h = db.hazard(cls, status)
        db.candidate("d1", cls, 100.0, hazard_id=h)
    r = evaluate_v1(db, tmp_path / "none")
    p = r.review["pothole"]
    assert (p.approved, p.rejected) == (2, 1) and p.precision == pytest.approx(2 / 3)
    deb = r.review["debris in lane"]
    assert (deb.approved, deb.rejected, deb.precision) == (0, 1, 0.0)
    g = r.review["damaged guardrail"]
    assert (g.approved, g.rejected, g.precision) == (0, 0, None)
    assert r.unreviewed == 2


def test_review_metrics_work_without_labels_and_for_unlabelled_drives(tmp_path):
    db, d = build(tmp_path)
    db.drive("d2", "IMG_2.MOV")
    h = db.hazard("pothole", "approved")
    db.candidate("d2", "pothole", 100.0, hazard_id=h)
    r = evaluate_v1(db, d)
    assert "d2" in r.skipped_drives
    assert r.review["pothole"].approved == 1


def test_review_follows_the_drive_filter(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.drive("d2", "IMG_2.MOV")
    db.candidate("d1", "pothole", 1.0, hazard_id=db.hazard("pothole", "approved"))
    db.candidate("d2", "pothole", 1.0, hazard_id=db.hazard("pothole", "rejected"))
    r = evaluate_v1(db, tmp_path / "none", drive_id="d2")
    assert (r.review["pothole"].approved, r.review["pothole"].rejected) == (0, 1)


# ------------------------------------------------------------------ read-only

def test_evaluate_v1_never_writes_the_database(tmp_path):
    db, d = mixed(tmp_path)
    db.candidate("d1", "pothole", 100.0, hazard_id=db.hazard("pothole", "approved"))
    before = db.fingerprint()
    evaluate_v1(db, d)
    evaluate_v1(db, d, drive_id="d1", tolerance=5.0)
    assert db.fingerprint() == before
    assert not list(tmp_path.glob("*.sqlite-wal")) and not list(tmp_path.glob("*.sqlite-journal"))


def test_missing_database_is_an_error_and_is_not_created(tmp_path):
    from backend.eval_v1 import evaluate_v1 as ev
    from backend.evaluation import load_labels

    missing = tmp_path / "nope.sqlite"
    with pytest.raises(Exception):
        ev(str(missing), load_labels(tmp_path / "labels"))
    assert not missing.exists()


def test_old_database_without_video_filename_still_works(tmp_path):
    import sqlite3

    db, d = build(tmp_path, stem="d1", cands=[("pothole", 62.0, {})])
    con = sqlite3.connect(str(db.path))
    try:
        try:
            con.execute("ALTER TABLE drives DROP COLUMN video_filename")
        except sqlite3.OperationalError as e:
            pytest.skip("cannot drop a column (%s)" % e)
        con.commit()
    finally:
        con.close()
    before = db.fingerprint()
    r = evaluate_v1(db, d)
    assert rows(r, "pothole")[0].matched == 1
    assert db.fingerprint() == before


def test_deterministic(tmp_path):
    db, d = mixed(tmp_path)
    assert dataclasses.asdict(evaluate_v1(db, d)) == dataclasses.asdict(evaluate_v1(db, d))
