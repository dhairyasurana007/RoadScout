"""Task 14: field values of the hazard rows (sightings, confidence, severity, photo, times, position)."""
from datetime import timedelta

import pytest

from dd14_helpers import (T_A, T_B, T_C, VA, add_cand, add_drive, ang_diff, dedup, dedup_all, execute, haversine,
                          hazard_of, hazards, make_db, offset, parse_time)


@pytest.fixture
def db(tmp_path, monkeypatch):
    d = make_db(tmp_path, monkeypatch)
    add_drive(d, "A", T_A)
    add_drive(d, "B", T_B)
    add_drive(d, "C", T_C)
    return d


# ------------------------------------------------------------------ single candidate

def test_single_candidate_fields(db):
    frame = str(db.parent / "best.jpg")
    c = add_cand(db, "A", VA[0], VA[1], 123.0, cls="pothole", severity=3, conf=0.7, score=0.6, best_ts=12.0,
                 frame_path=frame)
    dedup(db, "A")
    (h,) = hazards(db)
    assert h["cls"] == "pothole"
    assert abs(h["lat"] - VA[0]) < 1e-6 and abs(h["lon"] - VA[1]) < 1e-6
    assert ang_diff(h["direction"], 123.0) < 1e-6
    assert h["severity"] == 3
    assert 0.7 - 1e-9 <= h["confidence"] <= 1.0
    assert h["confidence"] <= 0.7 + 1e-6, "a single sighting should not raise confidence"
    assert h["sightings"] == 1
    assert h["status"] == "new"
    assert h["photo_path"] == frame
    fs, ls = parse_time(h["first_seen"]), parse_time(h["last_seen"])
    assert fs <= ls
    assert T_A <= fs <= T_A + timedelta(hours=1)
    assert hazard_of(db, c) == h["id"]


def test_photo_falls_back_to_a_real_candidate_image(db):
    frame = str(db.parent / "f.jpg")
    crop = str(db.parent / "c.jpg")
    add_cand(db, "A", VA[0], VA[1], 0.0, frame_path=frame, crop_path=crop)
    dedup(db, "A")
    (h,) = hazards(db)
    assert h["photo_path"] in (frame, crop)


# ------------------------------------------------------------------ position, direction

def test_merged_position_lies_between_members(db):
    lat2, lon2 = offset(VA[0], VA[1], 10.0, 8.0)
    add_cand(db, "A", VA[0], VA[1], 0.0, score=0.5)
    add_cand(db, "B", lat2, lon2, 0.0, score=0.6)
    dedup_all(db, ["A", "B"])
    (h,) = hazards(db)
    assert min(VA[0], lat2) - 1e-9 <= h["lat"] <= max(VA[0], lat2) + 1e-9
    assert min(VA[1], lon2) - 1e-9 <= h["lon"] <= max(VA[1], lon2) + 1e-9
    assert haversine(h["lat"], h["lon"], VA[0], VA[1]) <= 13.0
    assert haversine(h["lat"], h["lon"], lat2, lon2) <= 13.0


def test_direction_of_merged_hazard_is_between_member_headings(db):
    add_cand(db, "A", VA[0], VA[1], 80.0)
    add_cand(db, "B", VA[0], VA[1], 100.0)
    dedup_all(db, ["A", "B"])
    (h,) = hazards(db)
    assert 0.0 <= h["direction"] < 360.0 or h["direction"] == 360.0
    assert ang_diff(h["direction"], 80.0) <= 20.5 and ang_diff(h["direction"], 100.0) <= 20.5


def test_direction_handles_wrap_around(db):
    add_cand(db, "A", VA[0], VA[1], 359.0)
    add_cand(db, "B", VA[0], VA[1], 1.0)
    dedup_all(db, ["A", "B"])
    (h,) = hazards(db)
    assert ang_diff(h["direction"], 0.0) <= 1.5, "direction %r is not near north" % h["direction"]


# ------------------------------------------------------------------ sightings

def test_cross_drive_sightings_count_two(db):
    add_cand(db, "A", VA[0], VA[1], 0.0)
    add_cand(db, "B", VA[0], VA[1], 0.0)
    dedup_all(db, ["A", "B"])
    (h,) = hazards(db)
    assert h["sightings"] == 2


def test_three_drives_sightings_count_three(db):
    for d in "ABC":
        add_cand(db, d, VA[0], VA[1], 0.0)
    dedup_all(db, ["A", "B", "C"])
    (h,) = hazards(db)
    assert h["sightings"] == 3


def test_sightings_semantics_are_consistent(db):
    """Sightings count either distinct drives or merged candidates, and the same rule everywhere."""
    lat, lon = offset(VA[0], VA[1], 6.0)
    add_cand(db, "A", VA[0], VA[1], 0.0, best_ts=10.0)
    add_cand(db, "A", lat, lon, 0.0, best_ts=30.0)
    add_cand(db, "B", VA[0], VA[1], 0.0)
    dedup_all(db, ["A", "B"])
    (h,) = hazards(db)
    assert h["sightings"] in (2, 3), "2 = distinct drives, 3 = candidates; got %r" % h["sightings"]
    mode = "drives" if h["sightings"] == 2 else "candidates"

    # same rule inside one drive (fresh hazard far away): one candidate -> 1; two -> 1 (drives) or 2
    la, lo = offset(VA[0], VA[1], 500.0)
    lb, lob = offset(la, lo, 5.0)
    add_cand(db, "C", la, lo, 0.0, best_ts=10.0)
    add_cand(db, "C", lb, lob, 0.0, best_ts=30.0)
    dedup(db, "C")
    far = [x for x in hazards(db) if x["id"] != h["id"]]
    assert len(far) == 1
    assert far[0]["sightings"] == (1 if mode == "drives" else 2)


def test_sightings_do_not_double_count_on_rerun(db):
    add_cand(db, "A", VA[0], VA[1], 0.0)
    add_cand(db, "B", VA[0], VA[1], 0.0)
    for _ in range(3):
        dedup_all(db, ["A", "B"])
    (h,) = hazards(db)
    assert h["sightings"] == 2


def test_separate_hazards_have_their_own_sightings(db):
    far = offset(VA[0], VA[1], 400.0)
    add_cand(db, "A", VA[0], VA[1], 0.0)
    add_cand(db, "B", VA[0], VA[1], 0.0)
    add_cand(db, "C", far[0], far[1], 0.0)
    dedup_all(db, ["A", "B", "C"])
    hs = sorted(hazards(db), key=lambda x: x["sightings"])
    assert [x["sightings"] for x in hs] == [1, 2]


# ------------------------------------------------------------------ confidence

def test_repeat_sightings_raise_confidence(db):
    add_cand(db, "A", VA[0], VA[1], 0.0, conf=0.6)
    dedup(db, "A")
    c1 = hazards(db)[0]["confidence"]
    add_cand(db, "B", VA[0], VA[1], 0.0, conf=0.6)
    dedup(db, "B")
    c2 = hazards(db)[0]["confidence"]
    add_cand(db, "C", VA[0], VA[1], 0.0, conf=0.6)
    dedup(db, "C")
    (h,) = hazards(db)
    c3 = h["confidence"]
    assert c2 > c1 + 1e-6
    assert c3 > c2 + 1e-9 or c3 >= c2
    assert c3 > c1 + 1e-6
    assert c3 <= 1.0 + 1e-12


def test_confidence_never_below_best_single_and_never_above_one(db):
    add_cand(db, "A", VA[0], VA[1], 0.0, conf=0.9)
    add_cand(db, "B", VA[0], VA[1], 0.0, conf=0.5)
    dedup_all(db, ["A", "B"])
    (h,) = hazards(db)
    assert 0.9 - 1e-9 <= h["confidence"] <= 1.0 + 1e-12


def test_confidence_capped_at_one_with_certain_candidates(db):
    for d in "ABC":
        add_cand(db, d, VA[0], VA[1], 0.0, conf=1.0)
    dedup_all(db, ["A", "B", "C"])
    (h,) = hazards(db)
    assert 1.0 - 1e-9 <= h["confidence"] <= 1.0 + 1e-12


def test_confidence_capped_with_many_drives(db):
    ids = []
    for i in range(12):
        did = "D%d" % i
        add_drive(db, did, T_A + timedelta(days=i))
        add_cand(db, did, VA[0], VA[1], 0.0, conf=0.95)
        ids.append(did)
    dedup_all(db, ids)
    (h,) = hazards(db)
    assert h["sightings"] == 12
    assert 0.95 <= h["confidence"] <= 1.0 + 1e-12


def test_merging_raises_confidence_above_unmerged_neighbour(db):
    far = offset(VA[0], VA[1], 500.0)
    add_cand(db, "A", VA[0], VA[1], 0.0, conf=0.7)
    add_cand(db, "B", VA[0], VA[1], 0.0, conf=0.7)
    add_cand(db, "C", far[0], far[1], 0.0, conf=0.7)
    dedup_all(db, ["A", "B", "C"])
    merged = max(hazards(db), key=lambda x: x["sightings"])
    single = min(hazards(db), key=lambda x: x["sightings"])
    assert merged["confidence"] > single["confidence"]


# ------------------------------------------------------------------ severity

def test_severity_uniform(db):
    add_cand(db, "A", VA[0], VA[1], 0.0, severity=2)
    add_cand(db, "B", VA[0], VA[1], 0.0, severity=2)
    dedup_all(db, ["A", "B"])
    assert hazards(db)[0]["severity"] == 2


def test_severity_is_max_or_mode_and_in_range(db):
    add_cand(db, "A", VA[0], VA[1], 0.0, severity=1)
    add_cand(db, "B", VA[0], VA[1], 0.0, severity=1)
    add_cand(db, "C", VA[0], VA[1], 0.0, severity=3)
    dedup_all(db, ["A", "B", "C"])
    (h,) = hazards(db)
    assert h["severity"] in (1, 3), "expected the max (3) or the mode (1), got %r" % h["severity"]


def test_severity_when_worst_is_also_most_common(db):
    add_cand(db, "A", VA[0], VA[1], 0.0, severity=3)
    add_cand(db, "B", VA[0], VA[1], 0.0, severity=3)
    add_cand(db, "C", VA[0], VA[1], 0.0, severity=1)
    dedup_all(db, ["A", "B", "C"])
    assert hazards(db)[0]["severity"] == 3


# ------------------------------------------------------------------ photo

def test_photo_is_best_scoring_candidates_frame(db):
    paths = {d: str(db.parent / ("frame_%s.jpg" % d)) for d in "ABC"}
    add_cand(db, "A", VA[0], VA[1], 0.0, score=0.3, conf=0.6, frame_path=paths["A"])
    add_cand(db, "B", VA[0], VA[1], 0.0, score=0.9, conf=0.9, frame_path=paths["B"])
    add_cand(db, "C", VA[0], VA[1], 0.0, score=0.5, conf=0.7, frame_path=paths["C"])
    dedup_all(db, ["A", "B", "C"])
    (h,) = hazards(db)
    assert h["photo_path"] == paths["B"]


def test_photo_updates_when_a_better_sighting_arrives(db):
    pa, pb = str(db.parent / "fa.jpg"), str(db.parent / "fb.jpg")
    add_cand(db, "A", VA[0], VA[1], 0.0, score=0.4, conf=0.6, frame_path=pa)
    dedup(db, "A")
    assert hazards(db)[0]["photo_path"] == pa
    add_cand(db, "B", VA[0], VA[1], 0.0, score=0.8, conf=0.8, frame_path=pb)
    dedup(db, "B")
    assert hazards(db)[0]["photo_path"] == pb


# ------------------------------------------------------------------ first / last seen

def test_first_and_last_seen_span_the_drives(db):
    add_cand(db, "B", VA[0], VA[1], 0.0, best_ts=20.0)
    add_cand(db, "A", VA[0], VA[1], 0.0, best_ts=10.0)
    add_cand(db, "C", VA[0], VA[1], 0.0, best_ts=30.0)
    dedup_all(db, ["B", "A", "C"])
    (h,) = hazards(db)
    fs, ls = parse_time(h["first_seen"]), parse_time(h["last_seen"])
    assert T_A <= fs <= T_A + timedelta(hours=1)
    assert T_C <= ls <= T_C + timedelta(hours=1)
    assert fs < ls


def test_first_seen_before_last_seen_within_one_drive(db):
    lat, lon = offset(VA[0], VA[1], 4.0)
    add_cand(db, "A", VA[0], VA[1], 0.0, best_ts=10.0)
    add_cand(db, "A", lat, lon, 0.0, best_ts=90.0)
    dedup(db, "A")
    (h,) = hazards(db)
    fs, ls = parse_time(h["first_seen"]), parse_time(h["last_seen"])
    assert fs <= ls
    assert T_A <= fs and ls <= T_A + timedelta(hours=1)


def test_last_seen_advances_when_a_later_drive_sees_it(db):
    add_cand(db, "A", VA[0], VA[1], 0.0)
    dedup(db, "A")
    before = hazards(db)[0]
    add_cand(db, "B", VA[0], VA[1], 0.0)
    dedup(db, "B")
    (after,) = hazards(db)
    assert parse_time(after["first_seen"]) == parse_time(before["first_seen"])
    assert parse_time(after["last_seen"]) > parse_time(before["last_seen"])
