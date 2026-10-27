"""Task 8: geotagging the best frame through the GPX track (clock offset, video start, range handling)."""
from __future__ import annotations

from datetime import timedelta

import pytest

from rs8_helpers import (  # noqa: F401
    FPC, M_PER_DEG, T0, angdiff, make_spec, make_track, rs8_cache, rs8_env, ts_of, write_gpx_file,
)

SMALL = (60, 10, 80, 40)


def add_winner_track(e, chunk, winner, track_id=1, frames=range(5)):
    """A track whose unique best frame is ``winner`` (much bigger mask, everything else equal)."""
    for f in frames:
        e.add_det(chunk, f, track_id, (30, 10, 140, 40) if f == winner else SMALL)


def expected(track, ts, start, offset):
    return track.position_at(ts, video_start=start, clock_offset_seconds=offset)


def assert_located(c, pos):
    assert c["lat"] == pytest.approx(pos.latitude, abs=2e-6)
    assert c["lon"] == pytest.approx(pos.longitude, abs=2e-6)
    assert angdiff(float(c["heading"]), pos.heading_deg) < 1.0


def test_candidate_is_geotagged_at_the_best_frames_timestamp_from_drive_row_defaults(rs8_env):
    """No gpx/video_start/offset arguments: everything comes from the drives row and its GPX file."""
    e = rs8_env
    start, offset = T0 + timedelta(seconds=27), 0.5
    write_gpx_file(e.storage / "drive-a" / "track.gpx")
    e.add_drive("drive-a", started_at=start.isoformat(), offset=offset, gpx_key="drive-a/track.gpx")
    e.use_spec(make_spec(n_chunks=3))
    add_winner_track(e, 0, 2, track_id=1)   # best frame: chunk 0, frame 2   -> ts 0.667, GPX 28.17 s (north)
    add_winner_track(e, 2, 3, track_id=1)   # best frame: chunk 2, frame 3   -> ts 6.333, GPX 33.83 s (east)
    e.run(gpx=None)
    cs = {c["track_key"]: c for c in e.cands()}
    assert set(cs) == {"0:1", "2:1"}
    ref = make_track()
    p0 = expected(ref, ts_of(0, 2), start, offset)
    p2 = expected(ref, ts_of(2, 3), start, offset)
    assert_located(cs["0:1"], p0)
    assert_located(cs["2:1"], p2)
    assert angdiff(float(cs["0:1"]["heading"]), 0.0) < 1.0, "northbound before the turn"
    assert angdiff(float(cs["2:1"]["heading"]), 90.0) < 1.0, "eastbound after the turn"


def test_explicit_gpx_start_and_offset_override_the_drive_row(rs8_env):
    e = rs8_env
    e.add_drive("drive-a", started_at=T0.isoformat(), offset=0.0)  # deliberately different
    e.use_spec(make_spec(n_chunks=2))
    add_winner_track(e, 1, 4)
    track = make_track()
    start, offset = T0 + timedelta(seconds=10), -3.0
    e.run(gpx=track, video_start=start, offset=offset)
    (c,) = e.cands()
    assert_located(c, expected(track, ts_of(1, 4), start, offset))


def test_location_is_the_best_frame_not_the_first_or_last_frame(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    add_winner_track(e, 0, 2, frames=range(6))
    track = make_track()
    start = T0 + timedelta(seconds=5)
    e.run(gpx=track, video_start=start, offset=0.0)
    (c,) = e.cands()
    at_best = expected(track, ts_of(0, 2), start, 0.0)
    at_first = expected(track, ts_of(0, 0), start, 0.0)
    at_last = expected(track, ts_of(0, 5), start, 0.0)
    assert_located(c, at_best)
    assert abs(c["lat"] - at_first.latitude) * M_PER_DEG > 3
    assert abs(c["lat"] - at_last.latitude) * M_PER_DEG > 3


def test_clock_offset_shifts_the_location(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    add_winner_track(e, 0, 2)
    track = make_track()
    start = T0 + timedelta(seconds=10)
    e.run(gpx=track, video_start=start, offset=0.0)
    lat_a = e.cands()[0]["lat"]
    e.run(gpx=track, video_start=start, offset=5.0)
    lat_b = e.cands()[0]["lat"]
    # 5 s at 10 m/s northbound is about 50 m
    assert (lat_b - lat_a) * M_PER_DEG == pytest.approx(50.0, abs=2.0)


def test_video_start_shifts_the_location(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    add_winner_track(e, 0, 2)
    track = make_track()
    e.run(gpx=track, video_start=T0 + timedelta(seconds=4), offset=0.0)
    lat_a = e.cands()[0]["lat"]
    e.run(gpx=track, video_start=T0 + timedelta(seconds=10), offset=0.0)
    lat_b = e.cands()[0]["lat"]
    assert (lat_b - lat_a) * M_PER_DEG == pytest.approx(60.0, abs=2.0)


def test_gpx_track_parsed_from_a_gpx_file(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    add_winner_track(e, 0, 3)
    path = write_gpx_file(e.tmp / "t.gpx")
    start = T0 + timedelta(seconds=8)
    from backend.gpx_track import GpxTrack

    e.run(gpx=GpxTrack.from_file(path), video_start=start, offset=0.0)
    (c,) = e.cands()
    assert_located(c, expected(make_track(), ts_of(0, 3), start, 0.0))


def test_heading_flips_between_the_two_legs(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=3))
    add_winner_track(e, 0, 1)
    add_winner_track(e, 2, 1)
    track = make_track()
    e.run(gpx=track, video_start=T0 + timedelta(seconds=27), offset=0.0)
    cs = {c["track_key"]: c for c in e.cands()}
    assert angdiff(float(cs["0:1"]["heading"]), 0.0) < 1.0
    assert angdiff(float(cs["2:1"]["heading"]), 90.0) < 1.0


# ----------------------------------------------------------------------------- GPS range handling


def located_or_clamped(c, track):
    if c["lat"] is None and c["lon"] is None and c["heading"] is None:
        return "none"
    first = track.position_at(0.0, video_start=track.start_time, clock_offset_seconds=0.0)
    last = track.position_at(track.duration_seconds, video_start=track.start_time, clock_offset_seconds=0.0)
    for end in (first, last):
        if (abs(c["lat"] - end.latitude) < 2e-6 and abs(c["lon"] - end.longitude) < 2e-6):
            return "clamped"
    raise AssertionError("out-of-range candidate has location %r, %r: neither empty nor clamped to a track end"
                         % (c["lat"], c["lon"]))


@pytest.mark.parametrize("shift", [1000.0, -1000.0])
def test_best_frame_outside_the_gpx_range_does_not_crash(rs8_env, shift):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    add_winner_track(e, 0, 2)
    track = make_track()
    e.run(gpx=track, video_start=T0 + timedelta(seconds=shift), offset=0.0)
    cs = e.cands()
    assert len(cs) == 1, "the candidate must be kept even without a usable location"
    (c,) = cs
    assert located_or_clamped(c, track) in ("none", "clamped")
    assert e.resolve(c["best_path"]).is_file() and e.resolve(c["crop_path"]).is_file()
    assert c["frames"] == 5


def test_only_the_best_frame_matters_for_the_range(rs8_env):
    """The first frames fall before the GPX start, the best frame is inside the range: it is geotagged."""
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    add_winner_track(e, 0, 5, frames=range(6))
    track = make_track()
    start = T0 - timedelta(seconds=1.2)  # frame 0 -> GPX -1.2 s (outside), frame 5 -> 0.467 s (inside)
    e.run(gpx=track, video_start=start, offset=0.0)
    (c,) = e.cands()
    assert_located(c, expected(track, ts_of(0, 5), start, 0.0))


def test_out_of_range_for_one_candidate_does_not_affect_another(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=3))
    add_winner_track(e, 0, 2)
    add_winner_track(e, 2, 3)
    track = make_track(seconds=20, turn_at=100)  # GPX covers 20 s only
    start = T0 + timedelta(seconds=14)           # chunk 0 -> ~14.7 s (in), chunk 2 best -> 20.3 s (out)
    e.run(gpx=track, video_start=start, offset=0.0)
    cs = {c["track_key"]: c for c in e.cands()}
    assert set(cs) == {"0:1", "2:1"}
    assert_located(cs["0:1"], expected(track, ts_of(0, 2), start, 0.0))
    assert located_or_clamped(cs["2:1"], track) in ("none", "clamped")
