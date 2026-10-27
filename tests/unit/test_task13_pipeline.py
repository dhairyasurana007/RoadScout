"""Task 13 hidden tests: stationary frames are not sent to SAM 3.1 (fake provider), moving frames unaffected."""
from __future__ import annotations

import pytest

import rs13_helpers as H
from rs13_helpers import FPS, N_CHUNKS, N_FRAMES, chunk_index_of, is_done, run_scenario, ts_set


def chunk_times(c):
    return [round(c * H.CHUNK_SECONDS + k / FPS, 4) for k in range(H.FRAMES_PER_CHUNK)]


def off():
    return run_scenario("off", env={H.ENV_SWITCH: "0"})


def on():
    return run_scenario("on")


def det_key(d):
    return (round(float(d["frame_timestamp"]), 4), int(d["chunk"]), int(d["frame_index"]), d["hazard_class"],
            d["box"], d["score"])


# chunks lying entirely inside the definitely-stationary / definitely-moving stretches of the standard pattern
STOPPED_CHUNKS = [c for c in range(N_CHUNKS) if all(H.definite_stationary(t) for t in chunk_times(c))]
MOVING_CHUNKS = [c for c in range(N_CHUNKS) if all(H.definite_moving(t) for t in chunk_times(c))]


def test_scenario_assumptions_hold():
    assert STOPPED_CHUNKS == [4, 5, 6, 7]
    assert MOVING_CHUNKS == [0, 1, 10, 11]


def test_baseline_without_the_feature_sees_every_frame():
    r = off()
    assert is_done(r), (r["status"], r["exception"])
    assert len(r["calls"]) == N_CHUNKS
    assert len(r["detections"]) == N_FRAMES
    assert ts_set(r) == set(H.all_frame_times())


def test_default_run_succeeds():
    r = on()
    assert is_done(r), (r["status"], r["exception"], r["drive_json"].get("error"))


def test_provider_is_called_fewer_times():
    assert len(on()["calls"]) < len(off()["calls"])


def test_provider_never_called_for_fully_stationary_chunks():
    r = on()
    called = {chunk_index_of(c) for c in r["calls"]}
    assert None not in called, "provider was handed a file that is not a chunk_NNNN file: %s" % r["calls"]
    for c in STOPPED_CHUNKS:
        assert c not in called, "chunk %d lies fully inside the stop but was sent to the provider" % c
    assert len(r["calls"]) <= N_CHUNKS - len(STOPPED_CHUNKS)


def test_fully_moving_chunks_are_still_sent():
    called = {chunk_index_of(c) for c in on()["calls"]}
    for c in MOVING_CHUNKS:
        assert c in called, "moving chunk %d was not sent" % c


def test_no_stored_detection_on_a_stationary_frame():
    r = on()
    bad = [d["frame_timestamp"] for d in r["detections"] if H.definite_stationary(float(d["frame_timestamp"]))]
    assert bad == [], "detections stored on stationary frames: %s" % bad[:10]


def test_no_detections_in_fully_stationary_chunks():
    r = on()
    for c in STOPPED_CHUNKS:
        assert [d for d in r["detections"] if int(d["chunk"]) == c] == []


def test_detection_count_drops():
    assert len(on()["detections"]) < len(off()["detections"])


def test_every_definitely_moving_frame_keeps_its_detection():
    base, r = off(), on()
    have = ts_set(r)
    for t in H.all_frame_times():
        if H.definite_moving(t):
            assert t in have, "moving frame at %.3f s lost its detection" % t


def test_moving_frames_have_identical_detections():
    base, r = off(), on()
    want = sorted(det_key(d) for d in base["detections"] if H.definite_moving(float(d["frame_timestamp"])))
    got = sorted(det_key(d) for d in r["detections"] if H.definite_moving(float(d["frame_timestamp"])))
    assert got == want and len(want) > 0


def test_kept_timestamps_are_a_subset_of_baseline_global_timestamps():
    base, r = off(), on()
    assert ts_set(r) <= ts_set(base)


def test_frame_index_and_chunk_stay_global():
    r = on()
    for d in r["detections"]:
        t = round(float(d["frame_timestamp"]), 4)
        assert int(d["frame_index"]) == round(t * FPS)
        assert int(d["chunk"]) == int(t // H.CHUNK_SECONDS)


def test_partial_chunks_keep_only_or_at_least_their_moving_frames():
    r = on()
    have = ts_set(r)
    # chunk 2 (8-12 s) straddles the start of the stop; chunk 9 (36-40 s) straddles its end
    for t in chunk_times(2) + chunk_times(9):
        if H.definite_moving(t):
            assert t in have
        if H.definite_stationary(t):
            assert t not in have


def test_drive_still_ends_done_and_has_candidates_for_moving_chunks():
    r = on()
    assert is_done(r)
    assert len(r["candidates"]) >= 1
    firsts = [float(c["first_timestamp"]) for c in r["candidates"]]
    for c in MOVING_CHUNKS:
        assert any(c * H.CHUNK_SECONDS - 1e-6 <= f < (c + 1) * H.CHUNK_SECONDS for f in firsts), \
            "no candidate starts inside moving chunk %d" % c


def test_no_candidate_lies_in_the_stop():
    for c in on()["candidates"]:
        assert not H.definite_stationary(float(c["first_timestamp"]))
        assert not H.definite_stationary(float(c["last_timestamp"]))


def test_candidates_in_moving_chunks_match_baseline():
    base, r = off(), on()

    def by_first(rec):
        return {round(float(c["first_timestamp"]), 3): c for c in rec["candidates"]}

    b, o = by_first(base), by_first(r)
    for ch in MOVING_CHUNKS:
        key = round(ch * H.CHUNK_SECONDS, 3)
        assert key in b and key in o, (ch, sorted(b), sorted(o))
        assert int(o[key]["frames_seen"]) == int(b[key]["frames_seen"])
        assert float(o[key]["last_timestamp"]) == pytest.approx(float(b[key]["last_timestamp"]))
        assert float(o[key]["lat"]) == pytest.approx(float(b[key]["lat"]), abs=1e-7)
        assert float(o[key]["lon"]) == pytest.approx(float(b[key]["lon"]), abs=1e-7)


def test_geotags_still_match_the_gps_track_after_skipping():
    r = on()
    lats = H.lats_for(H.STD_SEGMENTS, 60)
    tol = 3.0 / H.M_PER_DEG
    assert r["candidates"]
    for c in r["candidates"]:
        assert c["lat"] is not None and c["lon"] is not None
        lo = H.lat_at(lats, float(c["first_timestamp"]))
        hi = H.lat_at(lats, float(c["last_timestamp"]))
        assert lo - tol <= float(c["lat"]) <= hi + tol, (c["first_timestamp"], c["last_timestamp"],
                                                              c["lat"], lo, hi)


def test_verification_only_for_surviving_candidates():
    base, r = off(), on()
    assert r["verify_calls"] <= base["verify_calls"]
    assert len(r["verdicts"]) <= len(base["verdicts"])
    ids = {c["id"] for c in r["candidates"]}
    assert {v["candidate_id"] for v in r["verdicts"]} <= ids


def test_original_video_untouched():
    r = on()
    assert r["video_sha_before"] == r["video_sha_after"]


def test_frame_timestamps_json_still_lists_every_frame():
    r = on()
    docs = [d for d in r["ft_docs"] if isinstance(d, dict) and isinstance(d.get("frames"), list)]
    if not docs:
        pytest.skip("frame_timestamps.json not found on disk after the run")
    doc = docs[0]
    assert len(doc["frames"]) == N_FRAMES
    assert [round(float(f["timestamp"]), 4) for f in doc["frames"]] == H.all_frame_times()


def test_http_process_route_also_skips():
    base = off()
    r = run_scenario("on_http", via_http=True, spy=False)
    assert r["final"]["status"].lower() in H.DONE, r["final"]
    assert 0 < len(r["detections"]) < len(base["detections"])
    assert not [d for d in r["detections"] if H.definite_stationary(float(d["frame_timestamp"]))]
    assert ts_set(r) >= {t for t in H.all_frame_times() if H.definite_moving(t)}


def test_rerun_does_not_accumulate():
    r = run_scenario("on_twice", twice=True)
    if r.get("second_ok") is not True:
        pytest.skip("pipeline refuses a second run of a finished drive: %s" % r.get("second_exception"))
    assert len(r["detections"]) == r["after_first"]["detections"]
    one = run_scenario("on")
    assert len(r["detections"]) == len(one["detections"])
    first_counts = H.recorded_counts({"drive_json": None, "row": r["after_first"]["drive"]})
    now_counts = H.recorded_counts(r)
    assert {k.split(":", 1)[1]: v for k, v in now_counts.items() if k.startswith("row")} == \
        {k.split(":", 1)[1]: v for k, v in first_counts.items() if k.startswith("row")}
