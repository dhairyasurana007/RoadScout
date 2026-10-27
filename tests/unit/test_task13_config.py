"""Task 13 hidden tests: configuration switches, whole-drive stops and the recorded skip counts."""
from __future__ import annotations

import pytest

import rs13_helpers as H
from rs13_helpers import N_CHUNKS, N_FRAMES, chunk_index_of, chunk_set, is_done, run_scenario, ts_set

STOP_ALL = H.gpx_xml(segments=[(10_000, 0.0)])
DRIFT = H.gpx_xml(segments=[(10_000, 0.1)])
MOVE_ALL = H.gpx_xml(segments=[(10_000, 15.0)])
CREEP = H.gpx_xml(segments=[(10_000, 3.0)])


def off():
    return run_scenario("off", env={H.ENV_SWITCH: "0"})


def on():
    return run_scenario("on")


def all_stop():
    return run_scenario("all_stop", gpx=STOP_ALL)


# ---------------------------------------------------------------- whole-drive stationary video

def test_whole_drive_stationary_yields_zero_detections_and_drive_is_done():
    r = all_stop()
    assert is_done(r), (r["status"], r["exception"], r["drive_json"].get("error"))
    assert r["detections"] == []
    assert r["candidates"] == []
    assert r["verdicts"] == []


def test_whole_drive_stationary_never_calls_the_provider():
    r = all_stop()
    assert r["calls"] == []
    assert r["verify_calls"] == 0


def test_whole_drive_stationary_records_everything_skipped():
    r = all_stop()
    counts = H.recorded_counts(r)
    fr, ch = H.frame_counts(counts), H.chunk_counts(counts)
    assert fr or ch, "no stationary/skipped count recorded: %s" % counts
    for k, v in fr.items():
        assert v == N_FRAMES, (k, v)
    for k, v in ch.items():
        assert v == N_CHUNKS, (k, v)


def test_whole_drive_stationary_status_has_no_error():
    r = all_stop()
    assert not r["drive_json"].get("error")


def test_gps_drift_while_parked_counts_as_stationary():
    r = run_scenario("slow_drift", gpx=DRIFT)
    assert is_done(r)
    assert r["calls"] == [] and r["detections"] == []


# ---------------------------------------------------------------- the on/off switch

@pytest.mark.parametrize("value", ["0", "false", "off", "no", "FALSE", "Off"])
def test_switch_off_sends_everything(value):
    r = run_scenario("switch_" + value, env={H.ENV_SWITCH: value}, gpx=STOP_ALL)
    assert is_done(r), (r["status"], r["exception"])
    assert len(r["calls"]) == N_CHUNKS
    assert len(r["detections"]) == N_FRAMES
    assert {chunk_index_of(c) for c in r["calls"]} == set(range(N_CHUNKS))


@pytest.mark.parametrize("value", ["1", "true", "on"])
def test_switch_on_values_skip(value):
    r = run_scenario("switch_" + value, env={H.ENV_SWITCH: value}, gpx=STOP_ALL)
    assert is_done(r)
    assert r["calls"] == [] and r["detections"] == []


def test_switch_defaults_to_on():
    # env var unset (run_scenario clears it): the stop is skipped
    assert len(on()["calls"]) < N_CHUNKS


def test_switch_off_with_standard_pattern_equals_a_run_without_gps_knowledge():
    r = off()
    assert ts_set(r) == set(H.all_frame_times())
    counts = H.recorded_counts(r)
    assert all(v == 0 for v in counts.values()), counts


def test_switch_off_results_are_complete_and_done():
    r = off()
    assert is_done(r)
    assert len(r["candidates"]) >= 1


# ---------------------------------------------------------------- threshold

def test_threshold_env_above_driving_speed_drops_everything():
    r = run_scenario("thr_20", env={H.ENV_THRESHOLD: "20"}, gpx=MOVE_ALL)
    assert is_done(r), (r["status"], r["exception"])
    assert r["calls"] == [] and r["detections"] == []


def test_threshold_env_drops_creeping_traffic():
    r = run_scenario("creep_thr_5", env={H.ENV_THRESHOLD: "5"}, gpx=CREEP)
    assert is_done(r)
    assert r["calls"] == [] and r["detections"] == []


def test_threshold_env_lower_than_drift_keeps_everything():
    r = run_scenario("drift_thr_005", env={H.ENV_THRESHOLD: "0.05"}, gpx=DRIFT)
    assert is_done(r)
    assert len(r["calls"]) == N_CHUNKS and len(r["detections"]) == N_FRAMES


def test_threshold_env_does_not_change_a_clearly_moving_drive():
    r = run_scenario("thr_1", env={H.ENV_THRESHOLD: "1.0"}, gpx=MOVE_ALL)
    assert is_done(r)
    assert len(r["calls"]) == N_CHUNKS and len(r["detections"]) == N_FRAMES


def test_garbage_threshold_env_behaves_like_the_default():
    r = run_scenario("thr_garbage", env={H.ENV_THRESHOLD: "abc"})
    assert is_done(r), (r["status"], r["exception"])
    assert ts_set(r) == ts_set(on())


def test_threshold_applies_with_standard_pattern_too():
    # a threshold above 15 m/s turns the whole standard drive into a stop
    r = run_scenario("thr_20_std", env={H.ENV_THRESHOLD: "20"})
    assert is_done(r)
    assert r["detections"] == []


# ---------------------------------------------------------------- recorded counts

def _json_counts(r):
    return {k: v for k, v in H.recorded_counts(r).items() if k.startswith("json:")}


def test_counts_are_visible_in_the_drive_json():
    counts = _json_counts(on())
    assert H.frame_counts(counts) or H.chunk_counts(counts), \
        "GET /api/drives/{id} shows no integer field with 'stationary'/'skipped' and 'frame'/'chunk': %s" \
        % sorted(on()["drive_json"])


def test_frame_count_matches_the_frames_actually_dropped():
    base, r = off(), on()
    dropped = len(ts_set(base) - ts_set(r))
    assert dropped >= 48  # at least the frames of the four fully-stopped chunks
    fr = H.frame_counts(_json_counts(r))
    for k, v in fr.items():
        assert v == dropped, "%s=%s but %d frames were dropped" % (k, v, dropped)


def test_chunk_count_matches_the_chunks_never_sent():
    base, r = off(), on()
    never_sent = N_CHUNKS - len({chunk_index_of(c) for c in r["calls"]})
    assert never_sent >= 4
    ch = H.chunk_counts(_json_counts(r))
    for k, v in ch.items():
        assert v == never_sent, "%s=%s but %d chunks were never sent" % (k, v, never_sent)
        assert v == len(chunk_set(base) - chunk_set(r))


def test_at_least_one_count_is_recorded_and_positive():
    counts = _json_counts(on())
    assert any(v > 0 for v in counts.values()), counts


def test_counts_are_non_negative_integers():
    for k, v in H.recorded_counts(on()).items():
        assert isinstance(v, int) and v >= 0, (k, v)


def test_counts_are_zero_when_nothing_was_skipped():
    for r in (off(), run_scenario("all_moving", gpx=MOVE_ALL)):
        counts = H.recorded_counts(r)
        assert all(v == 0 for v in counts.values()), counts


def test_counts_match_the_row_in_the_database():
    r = on()
    row_counts = {k.split(":", 1)[1]: v for k, v in H.recorded_counts(r).items() if k.startswith("row:")}
    json_counts = {k.split(":", 1)[1]: v for k, v in H.recorded_counts(r).items() if k.startswith("json:")}
    for k in set(row_counts) & set(json_counts):
        assert row_counts[k] == json_counts[k]


def test_skipping_does_not_flag_the_drive_as_failed():
    for r in (on(), all_stop()):
        assert r["status"] in H.DONE
        assert not r["drive_json"].get("error")
