"""Task 13 hidden tests: the per-frame stationary flag function (pure logic, no ffmpeg, no pipeline)."""
from __future__ import annotations

from datetime import timedelta

import pytest

import rs13_helpers as H
from rs13_helpers import T0, call_flags, pattern_track, tag_track


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(H.ENV_THRESHOLD, raising=False)
    monkeypatch.delenv(H.ENV_SWITCH, raising=False)


def test_pattern_flags_follow_gps_speed():
    ts = [i / 3.0 for i in range(0, 60 * 3)]
    flags = call_flags(ts, pattern_track())
    assert len(flags) == len(ts)
    for t, f in zip(ts, flags):
        if H.definite_moving(t):
            assert f is False, "t=%.2f moving at 15 m/s but flagged stationary" % t
        if H.definite_stationary(t):
            assert f is True, "t=%.2f stationary but not flagged" % t
    assert any(flags) and not all(flags)


def test_result_matches_input_order_and_length():
    ts = [40.0, 20.0, 5.0, 30.0, 45.0, 14.0]  # unsorted on purpose
    flags = call_flags(ts, pattern_track())
    assert flags == [False, True, False, True, False, True]


def test_empty_timestamps():
    assert call_flags([], pattern_track()) == []


def test_single_timestamp():
    assert call_flags([20.0], pattern_track()) == [True]
    assert call_flags([3.0], pattern_track()) == [False]


def test_no_track_fails_open():
    ts = [0.0, 10.0, 20.0, 30.0]
    assert call_flags(ts, None) == [False] * 4


def test_timestamps_outside_track_fail_open():
    # track covers 0..30 s of GPX time and is stationary the whole time
    track = tag_track(0.0, seconds=30)
    ts = [-20.0, -1.0, 5.0, 15.0, 29.0, 31.0, 45.0, 600.0]
    flags = call_flags(ts, track)
    assert flags == [False, False, True, True, True, False, False, False]


def test_partially_covered_track_only_flags_covered_frames():
    # video starts 10 s into the track: video t maps to GPX t+10, so the track ends at video t=20
    track = tag_track(0.0, seconds=30)
    ts = [-12.0, -5.0, 0.0, 10.0, 19.0, 21.0, 40.0]
    flags = call_flags(ts, track, video_start=T0 + timedelta(seconds=10))
    assert flags == [False, True, True, True, True, False, False]


def test_clock_offset_shifts_the_lookup():
    # GPX time = video_start + offset + t; stationary stretch is GPX 12..36 s
    ts = [float(t) for t in range(0, 48)]
    flags = call_flags(ts, pattern_track(), offset=6.0)
    for t, f in zip(ts, flags):
        if H.definite_stationary(t, shift=6.0):
            assert f is True, t
        if H.definite_moving(t, shift=6.0):
            assert f is False, t
    # the same video times without the offset disagree around t = 8..11 (moving) vs (stationary) at 13..
    base = call_flags(ts, pattern_track(), offset=0.0)
    assert base != flags


def test_negative_clock_offset():
    ts = [float(t) for t in range(10, 58)]
    flags = call_flags(ts, pattern_track(seconds=80), offset=-6.0)
    for t, f in zip(ts, flags):
        if H.definite_stationary(t, shift=-6.0):
            assert f is True, t
        if H.definite_moving(t, shift=-6.0):
            assert f is False, t


def test_video_start_shifts_the_lookup():
    ts = [float(t) for t in range(0, 40)]
    flags = call_flags(ts, pattern_track(), video_start=T0 + timedelta(seconds=6), offset=0.0)
    for t, f in zip(ts, flags):
        if H.definite_stationary(t, shift=6.0):
            assert f is True, t
        if H.definite_moving(t, shift=6.0):
            assert f is False, t


@pytest.mark.parametrize("threshold", [0.3, 0.5, 1.0, 2.0])
def test_speed_exactly_at_threshold_is_moving(threshold):
    track = tag_track(threshold)
    assert call_flags([1.0, 10.0, 20.0], track, threshold=threshold) == [False, False, False]


@pytest.mark.parametrize("threshold", [0.3, 0.5, 1.0, 2.0])
def test_speed_just_below_threshold_is_stationary(threshold):
    track = tag_track(threshold - 0.01)
    assert call_flags([1.0, 10.0, 20.0], track, threshold=threshold) == [True, True, True]


@pytest.mark.parametrize("threshold", [0.3, 0.5, 1.0, 2.0])
def test_speed_just_above_threshold_is_moving(threshold):
    track = tag_track(threshold + 0.01)
    assert call_flags([1.0, 10.0, 20.0], track, threshold=threshold) == [False, False, False]


def test_default_threshold_is_small():
    # GPS jitter while parked is flagged; a crawl and normal driving are not
    assert call_flags([5.0, 15.0], tag_track(0.0)) == [True, True]
    assert call_flags([5.0, 15.0], tag_track(0.1)) == [True, True]
    assert call_flags([5.0, 15.0], tag_track(1.5)) == [False, False]
    assert call_flags([5.0, 15.0], tag_track(13.0)) == [False, False]


def test_default_threshold_between_03_and_1_mps():
    # pins the default into the agreed range: 0.25 m/s is stationary, 1.05 m/s is moving
    assert call_flags([5.0], tag_track(0.25)) == [True]
    assert call_flags([5.0], tag_track(1.05)) == [False]


def test_env_threshold_is_read_at_call_time(monkeypatch):
    track = tag_track(3.0)
    assert call_flags([5.0], track) == [False]
    monkeypatch.setenv(H.ENV_THRESHOLD, "5")
    assert call_flags([5.0], track) == [True]
    monkeypatch.setenv(H.ENV_THRESHOLD, "2")
    assert call_flags([5.0], track) == [False]
    monkeypatch.delenv(H.ENV_THRESHOLD)
    assert call_flags([5.0], track) == [False]


def test_explicit_threshold_beats_env(monkeypatch):
    track = tag_track(3.0)
    monkeypatch.setenv(H.ENV_THRESHOLD, "5")
    assert call_flags([5.0], track, threshold=1.0) == [False]


@pytest.mark.parametrize("bad", ["", "abc", "nan-ish", "  "])
def test_garbage_env_threshold_falls_back_to_default(monkeypatch, bad):
    monkeypatch.setenv(H.ENV_THRESHOLD, bad)
    assert call_flags([5.0], tag_track(0.0)) == [True]
    assert call_flags([5.0], tag_track(13.0)) == [False]


def test_speed_from_positions_when_no_tags():
    # standard pattern has no speed tags; speed comes from positions (15 m/s vs 0 m/s)
    flags = call_flags([2.5, 20.5, 40.5], pattern_track())
    assert flags == [False, True, False]


def test_slow_drift_counts_as_stationary():
    drift = [(12, 0.1), (10_000, 0.1)]
    assert call_flags([5.0, 20.0, 40.0], pattern_track(drift)) == [True, True, True]


def test_creeping_traffic_counts_as_moving():
    crawl = [(10_000, 3.0)]
    assert call_flags([5.0, 20.0, 40.0], pattern_track(crawl)) == [False, False, False]


def test_input_is_not_mutated():
    ts = [3.0, 20.0, 40.0]
    keep = list(ts)
    f = H.find_flag_function()
    import inspect

    sig = inspect.signature(f)
    kw = {H._pick(sig, H.START_PARAMS): T0}
    f(ts, pattern_track(), **kw)
    assert ts == keep


def test_returns_python_bools_or_equivalent():
    flags = call_flags([3.0, 20.0], pattern_track())
    assert all(isinstance(x, bool) for x in flags)
