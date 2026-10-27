"""Task 19: when a hazard becomes possibly_fixed, what is kept, and when it comes back."""
from __future__ import annotations

import random
from datetime import datetime, timezone

import pytest

from rs19_helpers import (BASE, DEFAULT_MISSES, HAZARD_AT, UNTOUCHED, day, known_classes, parse_time,  # noqa: F401
                          rs19_env)

HL, HO = HAZARD_AT
STARTS = ["new", "approved", "reported"]


def flagged(env, hid):
    return env.status(hid) == "possibly_fixed"


# ------------------------------------------------------------------ the threshold

@pytest.mark.parametrize("start", STARTS)
def test_default_is_three_consecutive_misses(rs19_env, start):
    env = rs19_env
    hid = env.make_hazard(status=start)
    env.miss_drives(hid, DEFAULT_MISSES - 1)
    env.check()
    assert env.status(hid) == start
    assert env.misses(hid) == DEFAULT_MISSES - 1
    assert env.flagged_at(hid) is None
    env.miss_drives(hid, 1, first=DEFAULT_MISSES)
    env.check()
    assert env.status(hid) == "possibly_fixed"
    assert env.misses(hid) == DEFAULT_MISSES
    assert env.flagged_at(hid) is not None


def test_no_later_drives_means_no_change(rs19_env):
    env = rs19_env
    hid = env.make_hazard(status="approved")
    env.check()
    assert env.status(hid) == "approved"
    assert env.misses(hid) == 0
    assert env.flagged_at(hid) is None


@pytest.mark.parametrize("n", [1, 2, 5])
def test_threshold_is_configurable(rs19_env, n):
    env = rs19_env
    hid = env.make_hazard()
    env.configure(misses=n)
    if n > 1:
        env.miss_drives(hid, n - 1)
        env.check()
        assert env.status(hid) == "new", "flagged one miss early with threshold %d" % n
        assert env.misses(hid) == n - 1
    env.miss_drives(hid, 1, first=n)
    env.check()
    assert env.status(hid) == "possibly_fixed"


def test_more_misses_after_flagging_keep_it_flagged(rs19_env):
    env = rs19_env
    hid = env.make_hazard(status="approved")
    env.miss_drives(hid, 3)
    env.check()
    at = env.flagged_at(hid)
    assert at
    env.miss_drives(hid, 2, first=4)
    env.check()
    assert env.status(hid) == "possibly_fixed"
    assert env.misses(hid) >= 3
    assert env.flagged_at(hid) == at, "the time of flagging moved on a later run"


def test_flag_time_is_a_utc_timestamp_from_now(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    env.miss_drives(hid, 3)
    before = datetime.now(timezone.utc)
    env.check()
    at = parse_time(env.flagged_at(hid))
    assert at.tzinfo is not None
    assert abs((at - before).total_seconds()) < 3600, "flag time %s is not about now (%s)" % (at, before)


# ------------------------------------------------------------------ what is remembered and kept

@pytest.mark.parametrize("start", STARTS)
def test_previous_status_is_remembered(rs19_env, start):
    env = rs19_env
    hid = env.make_hazard(status=start)
    env.miss_drives(hid, 3)
    env.check()
    assert env.status(hid) == "possibly_fixed"
    row = env.row(hid)
    cols = env.prev_status_cols()
    assert cols, "no hazards column remembers the previous status (columns: %s)" % sorted(row)
    assert any(row[c] == start for c in cols), {c: row[c] for c in cols}


def test_review_fields_survive_flagging(rs19_env):
    env = rs19_env
    cls = [c for c in known_classes() if c != "pothole"][0]
    hid = env.make_hazard(status="approved", severity=2)
    env.patch_ok(hid, hazard_class=cls, severity=3, agency="MDOT SHA")
    before = env.snapshot(hid)
    assert before["reviewed_at"] and before["class_edited"] == 1 and before["severity_edited"] == 1
    assert before["agency_override"] == "MDOT SHA"
    env.miss_drives(hid, 3)
    env.check()
    assert env.status(hid) == "possibly_fixed"
    assert env.snapshot(hid) == before


def test_reported_hazard_keeps_its_review_time(rs19_env):
    env = rs19_env
    hid = env.make_hazard(status="reported")
    before = env.snapshot(hid)
    assert before["reviewed_at"]
    env.miss_drives(hid, 3)
    env.check()
    assert env.status(hid) == "possibly_fixed"
    assert env.snapshot(hid) == before


def test_flagging_changes_only_status_and_bookkeeping(rs19_env):
    env = rs19_env
    hid = env.make_hazard(status="approved")
    before = env.snapshot(hid)
    item_before = env.item(hid)
    env.miss_drives(hid, 3)
    env.check()
    assert env.snapshot(hid) == before
    item = env.item(hid)
    for key in ("id", "hazard_class", "lat", "lon", "direction", "severity", "confidence", "sightings",
                "first_seen", "last_seen", "photo_url", "crop_url", "agency", "reviewed_at"):
        assert item[key] == item_before[key], key
    assert item["status"] == "possibly_fixed"


def test_rejected_hazards_are_never_changed(rs19_env):
    env = rs19_env
    hid = env.make_hazard(status="rejected")
    before = env.snapshot(hid)
    env.miss_drives(hid, 5)
    env.check()
    assert env.status(hid) == "rejected"
    assert env.flagged_at(hid) is None
    assert env.snapshot(hid) == before


def test_rejected_next_to_an_approved_hazard(rs19_env):
    env = rs19_env
    ok = env.make_hazard(status="approved")
    rej = env.make_hazard(*[HL + 0.01, HO], heading=0.0, status="rejected")
    env.miss_drives(ok, 3)
    h2 = env.row(rej)
    for k in range(1, 4):
        env.add_pass_drive("r%d" % k, day(k, hour=15), h2["lat"], h2["lon"], 0.0)
    env.check()
    assert env.status(ok) == "possibly_fixed"
    assert env.status(rej) == "rejected"


# ------------------------------------------------------------------ idempotent and deterministic

def test_running_the_check_again_changes_nothing(rs19_env):
    env = rs19_env
    flagged_h = env.make_hazard(status="approved")
    pending = env.make_hazard(*[HL + 0.02, HO], heading=0.0)
    env.miss_drives(flagged_h, 3)
    h2 = env.row(pending)
    for k in range(1, 3):
        env.add_pass_drive("p%d" % k, day(k, hour=16), h2["lat"], h2["lon"], 0.0)
    env.check()
    first = {r["id"]: {k: v for k, v in r.items() if k != "updated_at"} for r in env.rows()}
    items = env.hazards()
    for _ in range(3):
        env.check()
    second = {r["id"]: {k: v for k, v in r.items() if k != "updated_at"} for r in env.rows()}
    assert second == first
    assert env.hazards() == items
    assert env.misses(pending) == 2
    assert env.misses(flagged_h) == 3


def test_drives_are_counted_once_however_often_the_check_runs(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    env.miss_drives(hid, 2)
    for _ in range(5):
        env.check()
    assert env.misses(hid) == 2
    assert env.status(hid) == "new"


def test_result_does_not_depend_on_drive_ids_or_insertion_order(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    h = env.row(hid)
    # chronological: miss(k1), miss(k2); ids and insertion order are scrambled
    spec = [(2, "zz-a"), (1, "aa-z"), (3, "mm-m")]
    random.Random(7).shuffle(spec)
    for k, did in spec:
        env.add_pass_drive(did, day(k), h["lat"], h["lon"], 0.0)
    env.check()
    assert env.misses(hid) == 3
    assert env.status(hid) == "possibly_fixed"


def test_no_hazards_and_no_drives_is_fine(rs19_env):
    env = rs19_env
    env.check()
    env.add_drive("alone", day(1), gpx="missing")
    env.check()
    assert env.hazards() == []


# ------------------------------------------------------------------ a later sighting resets the count

def test_a_sighting_resets_the_miss_count(rs19_env):
    env = rs19_env
    hid = env.make_hazard(status="approved")
    env.miss_drives(hid, 2)                       # day 1, 2: misses
    env.check()
    assert env.misses(hid) == 2
    env.sight(hid, "seen3", day(3))               # day 3: seen again (deduplication moves last_seen)
    env.miss_drives(hid, 2, first=4)              # day 4, 5: misses
    env.check()
    assert env.status(hid) == "approved"
    assert env.misses(hid) == 2
    env.miss_drives(hid, 1, first=6)
    env.check()
    assert env.status(hid) == "possibly_fixed"
    assert env.misses(hid) == 3


def test_misses_and_sightings_in_scrambled_order(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    h = env.row(hid)
    env.add_pass_drive("m3", day(1), h["lat"], h["lon"], 0.0)
    env.add_pass_drive("m1", day(2), h["lat"], h["lon"], 0.0)
    env.sight(hid, "m5", day(3))
    env.add_pass_drive("m2", day(4), h["lat"], h["lon"], 0.0)
    env.add_pass_drive("m4", day(5), h["lat"], h["lon"], 0.0)
    env.check()
    assert env.status(hid) == "new"
    assert env.misses(hid) == 2


def test_a_drive_with_a_member_candidate_is_a_sighting_not_a_miss(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    h = env.row(hid)
    env.add_pass_drive("a1", day(1), h["lat"], h["lon"], 0.0)               # miss
    env.add_pass_drive("a2", day(2), h["lat"], h["lon"], 0.0)               # saw it (candidate, below)
    env.link_candidate(hid, "a2")
    env.add_pass_drive("a3", day(3), h["lat"], h["lon"], 0.0)               # miss
    env.add_pass_drive("a4", day(4), h["lat"], h["lon"], 0.0)               # miss
    env.check()
    assert env.status(hid) == "new", "a drive that has a member candidate must not count as a miss"
    assert env.misses(hid) == 2


def test_sighting_drive_alone_never_flags(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    h = env.row(hid)
    for k in range(1, 5):
        did = "seen%d" % k
        env.add_pass_drive(did, day(k), h["lat"], h["lon"], 0.0)
        env.link_candidate(hid, did)
    env.check()
    assert env.status(hid) == "new"
    assert env.misses(hid) == 0


# ------------------------------------------------------------------ coming back

@pytest.mark.parametrize("start", STARTS)
def test_seen_again_restores_the_previous_status(rs19_env, start):
    env = rs19_env
    hid = env.make_hazard(status=start)
    env.miss_drives(hid, 3)
    env.check()
    assert env.status(hid) == "possibly_fixed"
    row_before = env.snapshot(hid)
    env.sight(hid, "back", day(4))
    env.check()
    assert env.status(hid) == start
    assert env.misses(hid) == 0
    assert env.flagged_at(hid) is None
    assert env.row(hid)["sightings"] == 2
    snap = env.snapshot(hid)
    for key in ("reviewed_at", "class_edited", "severity_edited", "agency_override", "original_hazard_class"):
        assert snap[key] == row_before[key], key


def test_after_coming_back_it_takes_the_full_count_to_flag_again(rs19_env):
    env = rs19_env
    hid = env.make_hazard(status="approved")
    env.miss_drives(hid, 3)
    env.check()
    env.sight(hid, "back", day(4))
    env.check()
    assert env.status(hid) == "approved"
    env.miss_drives(hid, 2, first=5)
    env.check()
    assert env.status(hid) == "approved"
    assert env.misses(hid) == 2
    env.miss_drives(hid, 1, first=7)
    env.check()
    assert env.status(hid) == "possibly_fixed"
    again = env.row(hid)
    cols = env.prev_status_cols()
    assert any(again[c] == "approved" for c in cols)


def test_restoring_does_not_depend_on_running_the_check_twice(rs19_env):
    env = rs19_env
    hid = env.make_hazard(status="approved")
    env.miss_drives(hid, 3)
    env.check()
    env.sight(hid, "back", day(4))
    env.check()
    env.check()
    assert env.status(hid) == "approved"
    assert env.row(hid)["status"] != "possibly_fixed"
