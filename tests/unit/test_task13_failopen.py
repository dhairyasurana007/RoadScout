"""Task 13 hidden tests: no GPS, bad GPX, out-of-range timestamps and the clock offset.

Fail open: when speed cannot be determined for a frame, the frame is kept. The run still succeeds.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

import rs13_helpers as H
from rs13_helpers import N_CHUNKS, N_FRAMES, T0, chunk_index_of, is_done, run_scenario, ts_set


def off():
    return run_scenario("off", env={H.ENV_SWITCH: "0"})


def _assert_nothing_dropped(r):
    base = off()
    assert is_done(r), (r["status"], r["exception"], r["drive_json"].get("error"))
    assert len(r["calls"]) == N_CHUNKS
    assert len(r["detections"]) == N_FRAMES
    assert ts_set(r) == ts_set(base)
    assert {chunk_index_of(c) for c in r["calls"]} == set(range(N_CHUNKS))
    counts = H.recorded_counts(r)
    assert all(v == 0 for v in counts.values()), "something was recorded as skipped: %s" % counts


# ---- tamper helpers (run after upload, before processing)

def _delete_gpx(did, gpx_path, db):
    gpx_path.unlink()


def _garbage_gpx(did, gpx_path, db):
    gpx_path.write_bytes(b"this is not xml \x00\x01 at all <<<")


def _empty_gpx(did, gpx_path, db):
    gpx_path.write_bytes(b"")


def _no_points_gpx(did, gpx_path, db):
    gpx_path.write_bytes(b'<?xml version="1.0"?><gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">'
                         b"<trk><trkseg></trkseg></trk></gpx>")


def _null_gpx_path(did, gpx_path, db):
    import sqlite3

    con = sqlite3.connect(str(db))
    try:
        try:
            con.execute("UPDATE drives SET gpx_path = NULL WHERE id = ?", (did,))
            con.commit()
        except sqlite3.IntegrityError:
            con.execute("UPDATE drives SET gpx_path = '' WHERE id = ?", (did,))
            con.commit()
    finally:
        con.close()


def test_missing_gpx_file_keeps_every_frame():
    _assert_nothing_dropped(run_scenario("gpx_deleted", tamper=_delete_gpx))


def test_unparseable_gpx_keeps_every_frame():
    _assert_nothing_dropped(run_scenario("gpx_garbage", tamper=_garbage_gpx))


def test_empty_gpx_keeps_every_frame():
    _assert_nothing_dropped(run_scenario("gpx_empty", tamper=_empty_gpx))


def test_gpx_without_points_keeps_every_frame():
    _assert_nothing_dropped(run_scenario("gpx_nopoints", tamper=_no_points_gpx))


def test_drive_without_gpx_path_keeps_every_frame():
    _assert_nothing_dropped(run_scenario("gpx_null_path", tamper=_null_gpx_path))


def test_gpx_for_a_different_day_keeps_every_frame():
    # every frame maps outside the GPX range, although the GPX is a stop-and-go pattern
    r = run_scenario("gpx_next_day", gpx=H.gpx_xml(start=T0 + timedelta(days=1)))
    _assert_nothing_dropped(r)


def test_gpx_entirely_before_the_video_keeps_every_frame():
    r = run_scenario("gpx_before", gpx=H.gpx_xml(segments=[(10_000, 0.0)], seconds=30,
                                                  start=T0 - timedelta(seconds=600)))
    _assert_nothing_dropped(r)


def test_all_stationary_gpx_but_wrong_offset_is_out_of_range():
    # GPX is a permanent stop covering 60 s, a +120 s clock offset pushes every frame out of range
    r = run_scenario("gpx_stop_offset_far", gpx=H.gpx_xml(segments=[(10_000, 0.0)]), offset=120)
    _assert_nothing_dropped(r)


# ---- partial coverage

PARTIAL = [(12, H.MOVE), (10_000, H.STOP)]  # moving 0-12 s, then stopped; the GPX ends at 20 s


def test_gpx_ending_early_drops_covered_stop_but_keeps_uncovered_frames():
    r = run_scenario("gpx_ends_at_20s", gpx=H.gpx_xml(segments=PARTIAL, seconds=20))
    assert is_done(r), (r["status"], r["exception"])
    have = ts_set(r)
    for t in H.all_frame_times():
        if t > 21.0:
            assert t in have, "frame at %.2f s is past the end of the GPX and must be kept" % t
        elif H.definite_moving(t, PARTIAL):
            assert t in have, "moving frame at %.2f s lost" % t
        elif 13.0 <= t <= 19.0:
            assert t not in have, "covered stationary frame at %.2f s was not dropped" % t


def test_gpx_starting_late_keeps_uncovered_frames():
    # GPX starts 20 s into the video and is stopped for its whole 20 s; the stretch before it is kept
    r = run_scenario("gpx_starts_at_20s", gpx=H.gpx_xml(segments=[(10_000, 0.0)], seconds=20,
                                                         start=T0 + timedelta(seconds=20)))
    assert is_done(r), (r["status"], r["exception"])
    have = ts_set(r)
    for t in H.all_frame_times():
        if t < 19.0 or t > 41.0:
            assert t in have, "frame at %.2f s is outside the GPX and must be kept" % t
        elif 21.0 <= t <= 39.0:
            assert t not in have, "covered stationary frame at %.2f s was not dropped" % t


# ---- clock offset and clip start time

def test_clock_offset_is_applied_per_frame():
    # drive offset +6 s: GPX time = video time + 6, so the stop is video 6..30 s instead of 12..36 s
    r = run_scenario("offset_6", offset=6)
    assert is_done(r), (r["status"], r["exception"])
    have = ts_set(r)
    for t in H.all_frame_times():
        if H.definite_stationary(t, shift=6.0):
            assert t not in have, "frame %.2f s is stopped once the +6 s offset is applied" % t
        if H.definite_moving(t, shift=6.0):
            assert t in have, "frame %.2f s is moving once the +6 s offset is applied" % t


def test_negative_clock_offset_is_applied_per_frame():
    # offset -6: GPX time = video time - 6 -> the stop is video 18..42 s; GPX covers 0..60 so the first 6 s
    # of video (GPX time < 0) are out of range and must be kept
    r = run_scenario("offset_minus_6", offset=-6)
    assert is_done(r), (r["status"], r["exception"])
    have = ts_set(r)
    for t in H.all_frame_times():
        if t < 5.0:
            assert t in have
        if H.definite_stationary(t, shift=-6.0) and t >= 7.0:
            assert t not in have, t
        if H.definite_moving(t, shift=-6.0) and t >= 7.0:
            assert t in have, t


def test_offset_changes_which_frames_are_dropped():
    a, b = run_scenario("on"), run_scenario("offset_6")
    assert ts_set(a) != ts_set(b)


# ---- the whole drive is moving

def test_always_moving_drive_is_untouched():
    r = run_scenario("all_moving", gpx=H.gpx_xml(segments=[(10_000, 15.0)]))
    _assert_nothing_dropped(r)


def test_creeping_traffic_is_not_dropped_with_the_default_threshold():
    r = run_scenario("creep_3mps", gpx=H.gpx_xml(segments=[(10_000, 3.0)]))
    _assert_nothing_dropped(r)
