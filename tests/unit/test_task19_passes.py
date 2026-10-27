"""Task 19: what counts as a later drive 'passing' a hazard without seeing it (one miss per drive)."""
from __future__ import annotations

from datetime import timedelta

import pytest

from rs19_helpers import (BASE, HAZARD_AT, M_PER_DEG, day, gpx_doc, gpx_pass, offset, pass_points,  # noqa: F401
                          rs19_env)

HL, HO = HAZARD_AT


def one_drive_misses(env, hid, **kw):
    env.miss_drives(hid, 1, **kw)
    env.check()
    return env.misses(hid)


# ------------------------------------------------------------------ the basic pass

def test_a_pass_without_a_sighting_is_one_miss(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    assert env.misses(hid) == 0
    assert one_drive_misses(env, hid) == 1
    assert env.status(hid) == "new"


def test_several_distinct_drives_count_one_each(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    env.miss_drives(hid, 2)
    env.check()
    assert env.misses(hid) == 2
    env.miss_drives(hid, 1, first=3)
    env.check()
    assert env.misses(hid) == 3


def test_a_drive_passing_twice_counts_once(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    # two same-direction passes in one recording (two segments; the gap between them is interpolated)
    seg1 = pass_points(HL, HO, 0.0, started=day(1), t_pass=10, t_from=-10, t_to=20)
    seg2 = pass_points(HL, HO, 0.0, started=day(1), t_pass=50, t_from=30, t_to=70)
    env.add_drive("loop1", day(1), gpx=gpx_doc([seg1, seg2]))
    env.check()
    assert env.misses(hid) == 1
    env.miss_drives(hid, 1, first=2)
    env.check()
    assert env.misses(hid) == 2


def test_drives_that_never_come_near_do_not_count(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    for k, lateral in enumerate((60.0, 150.0, 1000.0), start=1):
        env.add_pass_drive("far%d" % k, day(k), HL, HO, 0.0, lateral=lateral)
    env.check()
    assert env.misses(hid) == 0
    assert env.status(hid) == "new"


@pytest.mark.parametrize("lateral,counts", [(0.0, True), (10.0, True), (15.0, True), (40.0, False), (120.0, False)])
def test_default_radius_is_about_25_metres(rs19_env, lateral, counts):
    env = rs19_env
    hid = env.make_hazard()
    assert one_drive_misses(env, hid, lateral=lateral) == (1 if counts else 0)


def test_radius_is_configurable(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    env.configure(radius=80.0)
    assert one_drive_misses(env, hid, lateral=40.0) == 1


def test_a_small_radius_excludes_a_close_pass(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    env.configure(radius=5.0)
    assert one_drive_misses(env, hid, lateral=15.0) == 0


def test_check_works_with_a_hazard_far_from_the_equator_line(rs19_env):
    env = rs19_env
    hid = env.make_hazard(lat=39.2, lon=-76.6)
    assert one_drive_misses(env, hid, lateral=12.0) == 1


# ------------------------------------------------------------------ direction

@pytest.mark.parametrize("hazard_dir,drive_heading,counts", [
    (0.0, 0.0, True),
    (90.0, 120.0, True),
    (90.0, 60.0, True),
    (90.0, 150.0, False),       # 60 degrees apart
    (0.0, 180.0, False),        # the other side of a divided road
    (270.0, 90.0, False),
    (350.0, 20.0, True),        # wrap-around: 30 degrees apart
    (10.0, 340.0, True),
    (350.0, 80.0, False),       # 90 degrees apart
    (180.0, 0.0, False),
])
def test_heading_must_match_the_hazard_direction(rs19_env, hazard_dir, drive_heading, counts):
    env = rs19_env
    hid = env.make_hazard(heading=hazard_dir)
    got = one_drive_misses(env, hid, heading=drive_heading)
    assert got == (1 if counts else 0), "hazard %s, drive %s" % (hazard_dir, drive_heading)


def test_heading_tolerance_is_configurable(rs19_env):
    env = rs19_env
    hid = env.make_hazard(heading=90.0)
    env.configure(tol=90.0)
    assert one_drive_misses(env, hid, heading=150.0) == 1


def test_a_narrow_heading_tolerance_excludes_a_slightly_different_heading(rs19_env):
    env = rs19_env
    hid = env.make_hazard(heading=90.0)
    env.configure(tol=10.0)
    assert one_drive_misses(env, hid, heading=120.0) == 0


# ------------------------------------------------------------------ speed

def test_a_crawl_past_the_hazard_is_not_a_pass(rs19_env):
    env = rs19_env
    hid = env.make_hazard(heading=0.0)
    # 0.3 m/s the whole time, 5 m beside the hazard: below the 0.5 m/s stationary threshold
    assert one_drive_misses(env, hid, heading=0.0, speed=0.3, lateral=5.0) == 0


def test_a_stationary_threshold_from_config_is_respected(rs19_env):
    env = rs19_env
    hid = env.make_hazard(heading=0.0)
    env.mp.setenv("ROADSCOUT_STATIONARY_SPEED_MPS", "0.1")
    assert one_drive_misses(env, hid, heading=0.0, speed=0.3, lateral=5.0) == 1


def test_a_moving_pass_counts_even_when_the_car_was_stopped_earlier(rs19_env):
    env = rs19_env
    hid = env.make_hazard(heading=0.0)
    # stopped 300 m before the hazard for the first 20 s, then drives past it at 10 m/s
    pts = pass_points(HL, HO, 0.0, started=day(1), t_pass=40, t_from=20, t_to=70, speed=10.0)
    still = [(day(1) + timedelta(seconds=t), pts[0][1], pts[0][2]) for t in range(0, 20)]
    env.add_drive("stopgo", day(1), gpx=gpx_doc([still + pts]))
    env.check()
    assert env.misses(hid) == 1


# ------------------------------------------------------------------ time span and clock offset

def test_a_pass_after_the_recording_ended_is_not_a_pass(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    # the video lasts 20 s, the GPX goes on and passes the hazard at 30 s
    assert one_drive_misses(env, hid, duration=20.0, t_pass=30, t_to=80) == 0


def test_a_pass_before_the_recording_started_is_not_a_pass(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    assert one_drive_misses(env, hid, t_pass=-8, t_from=-30, t_to=60) == 0


def test_a_pass_inside_a_short_recording_counts(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    assert one_drive_misses(env, hid, duration=40.0, t_pass=30) == 1


def test_the_clock_offset_is_applied(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    # GPX time = started_at + offset + video time; the pass is at video second 30
    assert one_drive_misses(env, hid, offset_s=120.0) == 1


def test_a_negative_clock_offset_is_applied(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    assert one_drive_misses(env, hid, offset_s=-45.0) == 1


def test_the_offset_moves_a_pass_out_of_the_recording(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    # GPX built as if there were no offset, but the drive says the GPS clock is 120 s ahead:
    # the pass lands at video second -90
    g = gpx_pass(HL, HO, 0.0, started=day(1), offset_s=0.0)
    env.add_drive("skew", day(1), offset_s=120.0, gpx=g)
    env.check()
    assert env.misses(hid) == 0


# ------------------------------------------------------------------ which drives take part

@pytest.mark.parametrize("status", ["failed", "processing", "uploaded"])
def test_drives_that_are_not_done_never_count(rs19_env, status):
    env = rs19_env
    hid = env.make_hazard()
    assert one_drive_misses(env, hid, status=status) == 0


def test_only_done_drives_count_in_a_mix(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    env.add_pass_drive("ok1", day(1), HL, HO, 0.0)
    env.add_pass_drive("bad1", day(2), HL, HO, 0.0, status="failed")
    env.add_pass_drive("ok2", day(3), HL, HO, 0.0)
    env.add_pass_drive("bad2", day(4), HL, HO, 0.0, status="processing")
    env.add_pass_drive("bad3", day(5), HL, HO, 0.0, status="uploaded")
    env.check()
    assert env.misses(hid) == 2
    assert env.status(hid) == "new"


def test_drives_before_the_last_sighting_never_count(rs19_env):
    env = rs19_env
    hid = env.make_hazard()          # last_seen = BASE + 10 s
    for k, back in enumerate((1, 3, 10), start=1):
        env.add_pass_drive("old%d" % k, day(-back), HL, HO, 0.0)
    env.check()
    assert env.misses(hid) == 0
    assert env.status(hid) == "new"


def test_the_drive_that_made_the_sighting_is_not_a_later_drive(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    # the sighting drive itself, with a GPX that passes the hazard
    env.add_pass_drive("s00", BASE, HL, HO, 0.0)
    env.check()
    assert env.misses(hid) == 0


# ------------------------------------------------------------------ bad GPX files

def test_missing_gpx_is_skipped_not_a_miss(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    env.add_drive("nogpx", day(1), gpx="missing")
    env.check()
    assert env.misses(hid) == 0
    env.miss_drives(hid, 1, first=2)
    env.check()
    assert env.misses(hid) == 1


@pytest.mark.parametrize("payload", [b"this is not xml at all", b"", b"<gpx></gpx>", b"<gpx><trk><trkseg><trkpt lat=",
                                     b"\x00\xff\xfe binary \x01"])
def test_unparseable_gpx_is_skipped_without_crashing(rs19_env, payload):
    env = rs19_env
    hid = env.make_hazard()
    env.add_drive("bad", day(1), gpx=payload)
    env.check()
    assert env.misses(hid) == 0
    assert env.status(hid) == "new"


def test_empty_gpx_path_is_skipped(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    env.add_drive("nopath", day(1), gpx="")
    env.check()
    assert env.misses(hid) == 0


def test_bad_drives_do_not_break_the_good_ones(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    env.add_drive("bad1", day(1), gpx=b"garbage")
    env.add_drive("bad2", day(2), gpx="missing")
    env.miss_drives(hid, 3, first=3)
    env.check()
    assert env.misses(hid) == 3
    assert env.status(hid) == "possibly_fixed"


# ------------------------------------------------------------------ several hazards

def test_hazards_are_judged_independently(rs19_env):
    env = rs19_env
    a = env.make_hazard(lat=HL, lon=HO, heading=0.0)
    b = env.make_hazard(*offset(HL, HO, north_m=2000.0), heading=0.0)
    env.miss_drives(a, 3)
    env.check()
    assert env.status(a) == "possibly_fixed"
    assert env.status(b) == "new"
    assert env.misses(b) == 0


def test_one_drive_can_pass_two_hazards(rs19_env):
    env = rs19_env
    a = env.make_hazard(lat=HL, lon=HO, heading=0.0)
    lat2, lon2 = offset(HL, HO, north_m=300.0)
    b = env.make_hazard(lat2, lon2, heading=0.0)
    for k in range(1, 4):
        # north-bound at 10 m/s, passes A at video second 20 and B (300 m further) at second 50
        pts = pass_points(HL, HO, 0.0, started=day(k), t_pass=20, t_from=-10, t_to=70, speed=10.0)
        env.add_drive("two%d" % k, day(k), gpx=gpx_doc([pts]))
    env.check()
    assert env.status(a) == "possibly_fixed"
    assert env.status(b) == "possibly_fixed"
    assert env.misses(a) == env.misses(b) == 3
