"""Task 20: evaluate_v1 - dedup accuracy, cost per minute of video, processing time per minute of video."""
from __future__ import annotations

import dataclasses

import pytest

from rs20_helpers import evaluate_v1, haz, make_db, seg, write_labels, write_cost_log

A = ("A", "00:00", "10:00")


# ------------------------------------------------------------------ dedup accuracy

def two_drives(tmp_path, ids1, ids2=None, t1=("01:00", "01:05"), t2=("01:00", "01:05"), cls="pothole"):
    """Drive d1 (IMG_1) and d2 (IMG_2); each label has the given hazard_id. Returns (db, labels_dir)."""
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.drive("d2", "IMG_2.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A)] + [haz("A", t1[0], t1[1], cls, "2", hid) for hid in ids1])
    write_labels(d, "IMG_2", [seg(*A)] + [haz("A", t2[0], t2[1], cls, "2", hid) for hid in (ids2 or [])])
    return db, d


def counts(r):
    return (r.dedup.merged_correctly, r.dedup.split_wrongly, r.dedup.merged_wrongly)


def test_same_hazard_seen_in_two_drives_and_merged_into_one_row(tmp_path):
    db, d = two_drives(tmp_path, ["H1"], ["H1"])
    row = db.hazard("pothole", "new")
    db.candidate("d1", "pothole", 62.0, hazard_id=row)
    db.candidate("d2", "pothole", 62.0, hazard_id=row)
    assert counts(evaluate_v1(db, d)) == (1, 0, 0)


def test_same_hazard_split_into_two_rows(tmp_path):
    db, d = two_drives(tmp_path, ["H1"], ["H1"])
    db.candidate("d1", "pothole", 62.0, hazard_id=db.hazard("pothole"))
    db.candidate("d2", "pothole", 62.0, hazard_id=db.hazard("pothole"))
    assert counts(evaluate_v1(db, d)) == (0, 1, 0)


def test_different_hazards_merged_into_one_row(tmp_path):
    db, d = two_drives(tmp_path, ["H1"], ["H2"])
    row = db.hazard("pothole")
    db.candidate("d1", "pothole", 62.0, hazard_id=row)
    db.candidate("d2", "pothole", 62.0, hazard_id=row)
    assert counts(evaluate_v1(db, d)) == (0, 0, 1)


def test_different_hazards_in_separate_rows_is_neither_wrong_nor_counted(tmp_path):
    db, d = two_drives(tmp_path, ["H1"], ["H2"])
    db.candidate("d1", "pothole", 62.0, hazard_id=db.hazard("pothole"))
    db.candidate("d2", "pothole", 62.0, hazard_id=db.hazard("pothole"))
    assert counts(evaluate_v1(db, d)) == (0, 0, 0)


def test_wrong_merge_inside_one_drive(tmp_path):
    db, d = two_drives(tmp_path, ["H1", "H2"], [], t1=("01:00", "01:05"))
    # two labels at the same place and time would be ambiguous: put them 30 s apart
    d2 = tmp_path / "labels2"
    write_labels(d2, "IMG_1", [seg(*A), haz("A", "01:00", "01:05", "pothole", "2", "H1"),
                               haz("A", "02:00", "02:05", "pothole", "2", "H2")])
    row = db.hazard("pothole")
    db.candidate("d1", "pothole", 62.0, hazard_id=row)
    db.candidate("d1", "pothole", 122.0, hazard_id=row)
    assert counts(evaluate_v1(db, d2)) == (0, 0, 1)


def test_several_groups_are_counted_separately(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.drive("d2", "IMG_2.MOV")
    d = tmp_path / "labels"
    write_labels(d, "IMG_1", [seg(*A), haz("A", "01:00", "01:05", "pothole", "2", "H1"),
                              haz("A", "03:00", "03:05", "pothole", "2", "H3"),
                              haz("A", "05:00", "05:05", "debris in lane", "1", "H5")])
    write_labels(d, "IMG_2", [seg(*A), haz("A", "01:00", "01:05", "pothole", "2", "H1"),
                              haz("A", "03:00", "03:05", "pothole", "2", "H3"),
                              haz("A", "05:00", "05:05", "debris in lane", "1", "H6")])
    r1 = db.hazard("pothole")                       # H1 merged correctly
    db.candidate("d1", "pothole", 62.0, hazard_id=r1)
    db.candidate("d2", "pothole", 62.0, hazard_id=r1)
    db.candidate("d1", "pothole", 182.0, hazard_id=db.hazard("pothole"))      # H3 split wrongly
    db.candidate("d2", "pothole", 182.0, hazard_id=db.hazard("pothole"))
    r5 = db.hazard("debris in lane")                # H5 and H6 merged wrongly
    db.candidate("d1", "debris in lane", 302.0, hazard_id=r5)
    db.candidate("d2", "debris in lane", 302.0, hazard_id=r5)
    assert counts(evaluate_v1(db, d)) == (1, 1, 1)


def test_unmatched_labels_are_excluded(tmp_path):
    db, d = two_drives(tmp_path, ["H1"], ["H1"])
    db.candidate("d1", "pothole", 62.0, hazard_id=db.hazard("pothole"))      # d2 has no candidate at all
    assert counts(evaluate_v1(db, d)) == (0, 0, 0), "a group with fewer than two resolvable labels is not scored"


def test_only_accepted_candidates_resolve_a_label(tmp_path):
    db, d = two_drives(tmp_path, ["H1"], ["H1"])
    row = db.hazard("pothole")
    db.candidate("d1", "pothole", 62.0, hazard_id=row)
    db.candidate("d2", "pothole", 62.0, hazard_id=row, accepted=0)
    assert counts(evaluate_v1(db, d)) == (0, 0, 0)


def test_labels_without_a_hazard_id_are_not_scored(tmp_path):
    db, d = two_drives(tmp_path, [""], [""])
    row = db.hazard("pothole")
    db.candidate("d1", "pothole", 62.0, hazard_id=row)
    db.candidate("d2", "pothole", 62.0, hazard_id=row)
    assert counts(evaluate_v1(db, d)) == (0, 0, 0)


def test_a_hazard_id_seen_in_one_drive_only_counts_for_wrong_merges(tmp_path):
    db, d = two_drives(tmp_path, ["H9"], [])
    db.candidate("d1", "pothole", 62.0, hazard_id=db.hazard("pothole"))
    assert counts(evaluate_v1(db, d)) == (0, 0, 0)


def test_three_drives_one_hazard(tmp_path):
    db = make_db(tmp_path)
    d = tmp_path / "labels"
    row = db.hazard("pothole")
    for i in (1, 2, 3):
        db.drive(f"d{i}", f"IMG_{i}.MOV")
        write_labels(d, f"IMG_{i}", [seg(*A), haz("A", "01:00", "01:05", "pothole", "2", "H1")])
        db.candidate(f"d{i}", "pothole", 62.0, hazard_id=row)
    assert counts(evaluate_v1(db, d)) == (1, 0, 0)


def test_three_drives_partly_merged_counts_as_split(tmp_path):
    db = make_db(tmp_path)
    d = tmp_path / "labels"
    rows = [db.hazard("pothole"), db.hazard("pothole")]
    for i, row in zip((1, 2, 3), (rows[0], rows[0], rows[1])):
        db.drive(f"d{i}", f"IMG_{i}.MOV")
        write_labels(d, f"IMG_{i}", [seg(*A), haz("A", "01:00", "01:05", "pothole", "2", "H1")])
        db.candidate(f"d{i}", "pothole", 62.0, hazard_id=row)
    assert counts(evaluate_v1(db, d)) == (0, 1, 0)


def test_dedup_without_labels_is_all_zero(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV")
    db.candidate("d1", "pothole", 62.0, hazard_id=db.hazard("pothole"))
    r = evaluate_v1(db, tmp_path / "nolabels")
    assert counts(r) == (0, 0, 0)
    assert dataclasses.is_dataclass(r.dedup)


# ------------------------------------------------------------------ cost per minute of video

def world(tmp_path, durations=(60.0, 120.0), **kw):
    db = make_db(tmp_path)
    for i, dur in enumerate(durations, 1):
        db.drive(f"d{i}", f"IMG_{i}.MOV", duration=dur, **kw)
    return db


def test_cost_is_total_log_cost_over_total_video_minutes(tmp_path):
    db = world(tmp_path)
    log = write_cost_log(tmp_path / "cost.jsonl", [0.5, 0.25, 0.15])        # 0.90 USD over 3 minutes
    r = evaluate_v1(db, tmp_path / "labels", cost_log=str(log))
    assert r.cost.total_usd == pytest.approx(0.90)
    assert r.cost.minutes_of_video == pytest.approx(3.0)
    assert r.cost.usd_per_minute == pytest.approx(0.30)


def test_cost_log_accepts_a_path_object(tmp_path):
    db = world(tmp_path)
    log = write_cost_log(tmp_path / "cost.jsonl", [0.6])
    assert evaluate_v1(db, tmp_path / "labels", cost_log=log).cost.usd_per_minute == pytest.approx(0.2)


def test_no_cost_log_means_not_available(tmp_path):
    db = world(tmp_path)
    r = evaluate_v1(db, tmp_path / "labels")
    assert r.cost.usd_per_minute is None
    assert r.cost.total_usd in (None, 0.0)


def test_empty_cost_log_means_not_available(tmp_path):
    db = world(tmp_path)
    log = tmp_path / "cost.jsonl"
    log.write_text("", encoding="utf-8")
    assert evaluate_v1(db, tmp_path / "labels", cost_log=str(log)).cost.usd_per_minute is None
    log.write_text("\n\n", encoding="utf-8")
    assert evaluate_v1(db, tmp_path / "labels", cost_log=str(log)).cost.usd_per_minute is None


def test_missing_cost_log_file_is_not_available_or_a_clear_error(tmp_path):
    db = world(tmp_path)
    try:
        r = evaluate_v1(db, tmp_path / "labels", cost_log=str(tmp_path / "does_not_exist.jsonl"))
    except Exception as e:  # a clear error is acceptable
        assert str(e).strip()
    else:
        assert r.cost.usd_per_minute is None


def test_drive_filter_makes_cost_not_available(tmp_path):
    db = world(tmp_path)
    log = write_cost_log(tmp_path / "cost.jsonl", [0.9])
    r = evaluate_v1(db, tmp_path / "labels", cost_log=str(log), drive_id="d1")
    assert r.cost.usd_per_minute is None, "the log has no drive id"
    assert r.cost.minutes_of_video == pytest.approx(1.0)


def test_cost_without_any_video_minutes_is_not_available(tmp_path):
    db = make_db(tmp_path)
    log = write_cost_log(tmp_path / "cost.jsonl", [0.9])
    r = evaluate_v1(db, tmp_path / "labels", cost_log=str(log))
    assert r.cost.usd_per_minute is None


def test_drives_without_a_duration_do_not_count_as_minutes(tmp_path):
    db = world(tmp_path, durations=(60.0, 120.0))
    db.drive("d3", "IMG_3.MOV", duration=None)
    log = write_cost_log(tmp_path / "cost.jsonl", [0.9])
    r = evaluate_v1(db, tmp_path / "labels", cost_log=str(log))
    assert r.cost.minutes_of_video == pytest.approx(3.0)
    assert r.cost.usd_per_minute == pytest.approx(0.30)


def test_cost_numbers_are_exact_not_invented(tmp_path):
    db = world(tmp_path, durations=(600.0,))
    log = write_cost_log(tmp_path / "cost.jsonl", [1.0])
    assert evaluate_v1(db, tmp_path / "labels", cost_log=str(log)).cost.usd_per_minute == pytest.approx(0.1)


def test_cost_log_blank_lines_are_ignored(tmp_path):
    db = world(tmp_path)
    log = write_cost_log(tmp_path / "cost.jsonl", [0.3, 0.3, 0.3], blank_lines=True)
    assert evaluate_v1(db, tmp_path / "labels", cost_log=str(log)).cost.total_usd == pytest.approx(0.9)


# ------------------------------------------------------------------ processing time per minute of video

def test_timing_is_finished_minus_created_over_video_minutes(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV", duration=60.0, created="2026-10-20T12:00:00+00:00",
             finished="2026-10-20T12:02:00+00:00")                         # 120 s for 1 min
    db.drive("d2", "IMG_2.MOV", duration=120.0, created="2026-10-20T13:00:00+00:00",
             finished="2026-10-20T13:01:00+00:00")                         # 60 s for 2 min
    r = evaluate_v1(db, tmp_path / "labels")
    assert r.timing.seconds_per_minute == pytest.approx(180.0 / 3.0)


def test_timing_for_one_drive(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV", duration=300.0, created="2026-10-20T12:00:00+00:00",
             finished="2026-10-20T12:10:00+00:00")
    assert evaluate_v1(db, tmp_path / "labels").timing.seconds_per_minute == pytest.approx(120.0)


def test_timing_not_available_when_nothing_finished(tmp_path):
    db = world(tmp_path)
    r = evaluate_v1(db, tmp_path / "labels")
    assert r.timing.seconds_per_minute is None


def test_timing_skips_drives_that_never_finished(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV", duration=60.0, created="2026-10-20T12:00:00+00:00",
             finished="2026-10-20T12:02:00+00:00")
    db.drive("d2", "IMG_2.MOV", duration=600.0, created="2026-10-20T13:00:00+00:00")
    r = evaluate_v1(db, tmp_path / "labels")
    assert r.timing.seconds_per_minute is not None and r.timing.seconds_per_minute > 0


def test_timing_follows_the_drive_filter(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV", duration=60.0, created="2026-10-20T12:00:00+00:00",
             finished="2026-10-20T12:02:00+00:00")
    db.drive("d2", "IMG_2.MOV", duration=60.0, created="2026-10-20T13:00:00+00:00",
             finished="2026-10-20T13:04:00+00:00")
    assert evaluate_v1(db, tmp_path / "labels", drive_id="d1").timing.seconds_per_minute == pytest.approx(120.0)
    assert evaluate_v1(db, tmp_path / "labels", drive_id="d2").timing.seconds_per_minute == pytest.approx(240.0)
    assert evaluate_v1(db, tmp_path / "labels").timing.seconds_per_minute == pytest.approx(180.0)


def test_negative_or_missing_durations_do_not_crash(tmp_path):
    db = make_db(tmp_path)
    db.drive("d1", "IMG_1.MOV", duration=None, created="2026-10-20T12:00:00+00:00",
             finished="2026-10-20T12:02:00+00:00")
    r = evaluate_v1(db, tmp_path / "labels")
    assert r.timing.seconds_per_minute is None
