"""Task 10: the pipeline runner called directly (run_pipeline and aliases)."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from rs10_helpers import (CHUNKS, DETECTIONS, DONE, FAILED, INITIAL, ROOT, T0, err_of, gpx_bytes,  # noqa: F401
                          patch_everywhere, rs10_env, sha_file, spy, stage_pair, status_of)


def run_ok(e, drive_id, provider=None):
    res, exc = e.run(drive_id, provider)
    if exc is not None:
        raise exc
    return res


def test_runner_function_exists_and_takes_a_drive_id(rs10_env):
    from rs10_helpers import find_runner
    import inspect

    f = find_runner()
    assert len(inspect.signature(f).parameters) >= 1


def test_success_end_to_end(rs10_env):
    e = rs10_env
    did = e.upload()
    assert status_of(e.get(did)) in INITIAL | {"uploaded"}
    p = spy()
    run_ok(e, did, p)
    d = e.get(did)
    assert status_of(d) in DONE, d
    assert err_of(d) == ""
    c = e.counts(did)
    assert c["detections"] == DETECTIONS
    assert c["candidates"] == CHUNKS
    assert c["verdicts"] == CHUNKS
    assert [v["candidate_id"] for v in e.verdicts(did)] == [x["id"] for x in e.candidates(did)]
    assert all(v["is_hazard"] and v["accepted"] for v in e.verdicts(did))
    assert len(p.detect_calls) == CHUNKS
    assert len(p.verify_calls) == CHUNKS
    # a stored drive keeps its identity and its inputs
    row = e.row(did)
    assert row["video_path"] and row["gpx_path"]


def test_candidates_are_geotagged_from_the_drives_gpx(rs10_env):
    e = rs10_env
    did = e.upload()
    run_ok(e, did, spy())
    cands = e.candidates(did)
    assert len(cands) == CHUNKS
    for c in cands:
        assert c["geotag_status"] == "ok", c
        assert c["lat"] is not None and c["lon"] is not None
        assert 38.8 < c["lat"] < 39.0 and -77.4 < c["lon"] < -77.2


def test_stages_run_in_order_and_status_moves_through_stages(rs10_env):
    e = rs10_env
    did = e.upload()
    snaps = {}

    def snap(tag):
        d = e.get(did)
        return {"get": d, "row": e.row(did), "counts": e.counts(did)}

    def on_detect(p, n, video_path):
        if n == 0:
            s = snap("detect")
            s["chunk_exists"] = Path(str(video_path)).is_file()
            s["chunk_path"] = Path(str(video_path))
            snaps["detect"] = s

    def on_verify(p, n):
        if n == 0:
            snaps["verify"] = snap("verify")

    p = spy(on_detect=on_detect, on_verify=on_verify)
    run_ok(e, did, p)

    sd, sv = snaps["detect"], snaps["verify"]
    # preprocessing already happened: real chunk files exist when detection is called
    assert sd["chunk_exists"], "detect must receive an existing chunk produced by preprocessing"
    # while detecting: not the initial value, not terminal; visible through GET and in the DB
    for s in (sd, sv):
        st = status_of(s["get"])
        assert st not in INITIAL and st not in DONE and st not in FAILED, s["get"]
        assert status_of(s["row"]) == st
    # detection stage: nothing aggregated or verified yet
    assert sd["counts"]["candidates"] == 0 and sd["counts"]["verdicts"] == 0
    # verification stage: detections and candidates are in the DB, verdicts are not yet
    assert sv["counts"]["detections"] == DETECTIONS
    assert sv["counts"]["candidates"] == CHUNKS
    assert sv["counts"]["verdicts"] == 0
    # the visible stage differs between detecting and verifying (status or a stage/progress field)
    assert stage_pair(sd["get"]) != stage_pair(sv["get"]), (sd["get"], sv["get"])
    # all chunks detected before any verification
    assert p.events == ["detect"] * CHUNKS + ["verify"] * CHUNKS


def test_status_values_are_strings_and_final_status_is_stable(rs10_env):
    e = rs10_env
    did = e.upload()
    run_ok(e, did, spy())
    a = e.get(did)
    b = e.get(did)
    assert status_of(a) in DONE and status_of(a) == status_of(b)
    lst = e.client.get("/api/drives").json()
    assert [status_of(x) for x in lst if x["id"] == did][0] in DONE


def test_uses_the_drives_clock_offset(rs10_env):
    """GPX starts 100 s after the video; offset 100 aligns them. Offset 0 must not geotag."""
    e = rs10_env
    gpx = gpx_bytes(start=T0 + timedelta(seconds=100), seconds=60)
    good = e.upload(offset=100, gpx=gpx)
    run_ok(e, good, spy())
    assert status_of(e.get(good)) in DONE
    assert [c["geotag_status"] for c in e.candidates(good)] == ["ok"] * CHUNKS
    bad = e.upload(offset=0, gpx=gpx)
    run_ok(e, bad, spy())
    d = e.get(bad)
    assert status_of(d) in DONE, "an out-of-range GPX is not a pipeline failure"
    cands = e.candidates(bad)
    assert len(cands) == CHUNKS
    assert all(c["geotag_status"] != "ok" and c["lat"] is None for c in cands)
    assert len(e.verdicts(bad)) == CHUNKS


@pytest.mark.parametrize("scenario,cands,hazard", [
    ("default", CHUNKS, True),
    ("false_positive", CHUNKS, False),
    ("empty", 0, None),
    ("short_track", 0, None),
])
def test_default_provider_follows_env_scenario(rs10_env, monkeypatch, scenario, cands, hazard):
    e = rs10_env
    monkeypatch.setenv("ROADSCOUT_FAKE_SCENARIO", scenario)
    did = e.upload()
    run_ok(e, did)  # provider=None -> get_provider() -> fake; no network, no key (guarded in the fixture)
    d = e.get(did)
    assert status_of(d) in DONE, d
    assert err_of(d) == ""
    assert len(e.candidates(did)) == cands
    vs = e.verdicts(did)
    assert len(vs) == cands
    if hazard is not None:
        assert all(bool(v["is_hazard"]) is hazard for v in vs)


def test_no_candidates_means_verification_makes_no_calls(rs10_env):
    e = rs10_env
    did = e.upload()
    p = spy("empty")
    run_ok(e, did, p)
    assert status_of(e.get(did)) in DONE
    assert len(p.detect_calls) == CHUNKS
    assert p.verify_calls == []


def test_rerun_replaces_results(rs10_env):
    e = rs10_env
    did = e.upload()
    run_ok(e, did, spy())
    first = e.counts(did)
    first_ids = [c["id"] for c in e.candidates(did)]
    run_ok(e, did, spy())
    d = e.get(did)
    assert status_of(d) in DONE
    again = e.counts(did)
    assert again == first
    assert again["detections"] == DETECTIONS
    assert len({c["id"] for c in e.candidates(did)}) == CHUNKS
    assert e.orphan_verdicts() == []
    assert [v["candidate_id"] for v in e.verdicts(did)] == [c["id"] for c in e.candidates(did)]
    assert len(first_ids) == CHUNKS


def test_rerun_can_change_the_outcome(rs10_env):
    """Second run with a different scenario replaces the earlier results entirely."""
    e = rs10_env
    did = e.upload()
    run_ok(e, did, spy("default"))
    assert e.counts(did)["candidates"] == CHUNKS
    run_ok(e, did, spy("empty"))
    c = e.counts(did)
    assert c["detections"] == 0 and c["candidates"] == 0 and c["verdicts"] == 0
    assert status_of(e.get(did)) in DONE


def test_two_drives_do_not_mix(rs10_env):
    e = rs10_env
    a, b = e.upload(), e.upload()
    run_ok(e, a, spy("default"))
    run_ok(e, b, spy("false_positive"))
    assert all(v["is_hazard"] for v in e.verdicts(a))
    assert all(not v["is_hazard"] for v in e.verdicts(b))
    assert e.counts(a)["candidates"] == CHUNKS and e.counts(b)["candidates"] == CHUNKS
    assert {c["drive_id"] for c in e.candidates(a)} == {a}
    assert {c["drive_id"] for c in e.candidates(b)} == {b}
    assert status_of(e.get(a)) in DONE and status_of(e.get(b)) in DONE


def test_intermediate_files_live_under_the_data_dir_keyed_by_drive_id(rs10_env):
    e = rs10_env
    did = e.upload()
    original = e.video_file(did)
    gpx = e.gpx_file(did)
    h_video, h_gpx = sha_file(original), sha_file(gpx)
    p = spy()
    run_ok(e, did, p)
    # chunks that detection received
    assert len(p.detect_calls) == CHUNKS
    for path in p.detect_calls:
        pp = Path(path).resolve()
        assert pp.is_file()
        assert e.tmp.resolve() in pp.parents, "chunks must be written under the configured data dir, not elsewhere"
        assert did in pp.as_posix(), "chunk paths are keyed by the drive id"
        assert pp != original.resolve()
    # candidate images
    for c in e.candidates(did):
        for key in ("best_frame_path", "crop_path"):
            if c[key]:
                pp = Path(c[key]).resolve()
                assert pp.is_file()
                assert e.tmp.resolve() in pp.parents and did in pp.as_posix()
    # nothing leaked into the repository working tree
    if ROOT is not None:
        leaked = [x for x in (ROOT / "data").rglob(did + "*")] if (ROOT / "data").exists() else []
        assert leaked == []
    # the upload is untouched and never deleted
    assert original.is_file() and gpx.is_file()
    assert sha_file(original) == h_video and sha_file(gpx) == h_gpx
    run_ok(e, did, spy())
    assert original.is_file() and sha_file(original) == h_video and sha_file(gpx) == h_gpx


def test_video_is_kept_after_success(rs10_env):
    e = rs10_env
    did = e.upload()
    run_ok(e, did, spy())
    assert e.video_file(did).is_file()
    assert e.row(did)["video_path"]


def test_unknown_drive_does_not_create_anything(rs10_env):
    e = rs10_env
    p = spy()
    res, exc = e.run("no-such-drive", p)
    # either an exception or a quiet return is acceptable; but no model call and no rows
    assert p.detect_calls == [] and p.verify_calls == []
    assert e.row("no-such-drive") is None
    assert e.counts("no-such-drive") == {"detections": 0, "candidates": 0, "verdicts": 0, "all_verdicts": 0}


def test_runner_does_not_touch_other_drives(rs10_env):
    e = rs10_env
    a, b = e.upload(), e.upload()
    run_ok(e, a, spy())
    assert status_of(e.get(b)) in INITIAL | {"uploaded"}
    assert e.counts(b) == {"detections": 0, "candidates": 0, "verdicts": 0, "all_verdicts": 0}
