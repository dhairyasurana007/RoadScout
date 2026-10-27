"""Task 10: the HTTP side. POST /api/drives/{id}/process starts a background job; GET shows status."""
from __future__ import annotations

import pytest

from rs10_helpers import (CHUNKS, DETECTIONS, DONE, FAILED, INITIAL, err_of, rs10_env, spy, stage_pair,  # noqa: F401
                          status_of)


def test_process_route_exists(rs10_env):
    assert rs10_env.process_route().startswith("/api/drives/")


def test_process_unknown_drive_is_404(rs10_env):
    e = rs10_env
    r = e.post_process("nope")
    assert r.status_code == 404
    assert e.q("SELECT COUNT(*) n FROM drives")[0]["n"] == 0


def test_process_starts_job_and_reports_drive(rs10_env):
    e = rs10_env
    did = e.upload()
    r = e.process_via_http(did)
    body = r.json()
    assert isinstance(body, dict)
    got_id = body.get("id", body.get("drive_id"))
    if got_id is None and isinstance(body.get("drive"), dict):
        got_id = body["drive"].get("id")
    assert got_id == did, body
    assert "status" in body or "status" in (body.get("drive") or {}), body
    d = e.wait(did)
    assert status_of(d) in DONE, d
    assert err_of(d) == ""
    c = e.counts(did)
    assert c["detections"] == DETECTIONS and c["candidates"] == CHUNKS and c["verdicts"] == CHUNKS


def test_process_with_default_fake_provider_needs_no_key_or_network(rs10_env):
    e = rs10_env
    assert "META_API_KEY" not in __import__("os").environ
    did = e.upload()
    e.process_via_http(did)
    d = e.wait(did)
    assert status_of(d) in DONE
    assert all(v["model"] == "fake-muse-spark" for v in e.verdicts(did))


def test_process_no_body_required(rs10_env):
    e = rs10_env
    did = e.upload()
    r = e.post_process(did)  # no body, no content type
    assert r.status_code in (200, 201, 202)
    e.wait(did)


def test_rerun_finished_drive_replaces_results(rs10_env):
    e = rs10_env
    did = e.upload()
    e.process_via_http(did)
    assert status_of(e.wait(did)) in DONE
    first = e.counts(did)
    r = e.process_via_http(did)
    assert r.status_code in (200, 201, 202)
    d = e.wait(did)
    assert status_of(d) in DONE
    assert e.counts(did) == first
    assert e.orphan_verdicts() == []


def test_failure_via_http_is_recorded_and_server_survives(rs10_env):
    e = rs10_env
    did = e.upload()
    e.use_provider(spy(fail_detect_on=0))
    r = e.post_process(did)
    assert r.status_code in (200, 201, 202), "starting the job succeeds; the failure belongs to the job"
    d = e.wait(did)
    assert status_of(d) in FAILED, d
    msg = err_of(d)
    assert "boom-detect-xyz" in msg and "detect" in msg.lower()
    assert e.client.get("/health").status_code == 200
    # listed drives show the same failure
    lst = {x["id"]: x for x in e.client.get("/api/drives").json()}
    assert status_of(lst[did]) in FAILED and err_of(lst[did]) == msg
    assert e.counts(did) == {"detections": 0, "candidates": 0, "verdicts": 0, "all_verdicts": 0}


def test_retry_after_failure_via_http(rs10_env):
    e = rs10_env
    did = e.upload()
    e.use_provider(spy(fail_verify_on=0))
    e.process_via_http(did)
    assert status_of(e.wait(did)) in FAILED
    e.use_provider(spy())
    r = e.post_process(did)
    assert r.status_code in (200, 201, 202), "a failed drive can be re-run"
    d = e.wait(did)
    assert status_of(d) in DONE and err_of(d) == ""
    assert e.counts(did)["verdicts"] == CHUNKS


def test_corrupt_video_via_http(rs10_env):
    e = rs10_env
    did = e.upload()
    e.video_file(did).write_bytes(b"not a video at all" * 100)
    r = e.post_process(did)
    assert r.status_code in (200, 201, 202)
    d = e.wait(did)
    assert status_of(d) in FAILED
    assert "preprocess" in err_of(d).lower()
    assert e.client.get("/api/drives/%s" % did).status_code == 200
    assert e.video_file(did).is_file()


def test_a_failed_job_does_not_stop_the_next_drive(rs10_env):
    e = rs10_env
    bad, good = e.upload(), e.upload()
    e.video_file(bad).write_bytes(b"junk")
    e.process_via_http(bad)
    assert status_of(e.wait(bad)) in FAILED
    e.process_via_http(good)
    assert status_of(e.wait(good)) in DONE


def test_process_does_not_change_the_upload_endpoint_contract(rs10_env):
    e = rs10_env
    did = e.upload()
    d = e.get(did)
    for k in ("id", "started_at", "video_path", "gpx_path", "status", "clock_offset_seconds", "created_at"):
        assert k in d
    assert status_of(d) in INITIAL | {"uploaded"}
    assert not err_of(d)
