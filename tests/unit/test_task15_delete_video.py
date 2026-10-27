"""Task 15: the Delete video button's backend (removes the raw video and extracted frames, keeps the rest)."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from rs15_helpers import (DONE, FAILED, VIDEO_EXT, make_stub, rs15_done, rs15_env, sha, status_of)  # noqa: F401


def norm_path(p):
    return os.path.normcase(os.path.abspath(str(p)))


def video_deleted(d):
    if "video_deleted" in d:
        return bool(d["video_deleted"])
    return bool(d.get("video_deleted_at"))


def snapshot(env, did):
    return {
        "counts": env.q("SELECT (SELECT COUNT(*) FROM detections WHERE drive_id = ?) d, "
                        "(SELECT COUNT(*) FROM candidates WHERE drive_id = ?) c, "
                        "(SELECT COUNT(*) FROM verdicts WHERE drive_id = ?) v, "
                        "(SELECT COUNT(*) FROM hazards) h", (did, did, did))[0],
        "hazards": env.q("SELECT * FROM hazards ORDER BY id"),
        "list": env.hazards_list(did),
        "blurred": {norm_path(p): sha(p) for p in env.blurred_paths(did)},
        "photos": {c["id"]: env.photo(did, c["id"]).content for c in env.accepted(did)},
        "gpx": sha(env.gpx_file(did)),
    }


# ------------------------------------------------------------------ the happy path

def test_delete_route_exists(rs15_env):
    method, path = rs15_env.delete_route()
    assert method in ("delete", "post") and "{" in path


def test_drive_json_carries_video_deleted_false_before(rs15_done):
    env, did, stub = rs15_done
    d = env.get(did)
    assert "video_deleted" in d, "drive JSON must carry a video_deleted flag (the Delete video button uses it): %s" % sorted(d)
    assert d["video_deleted"] is False
    mine = [x for x in env.drive_list() if x["id"] == did][0]
    assert mine.get("video_deleted") is False


def test_delete_removes_video_chunks_and_unblurred_frames_but_keeps_the_rest(rs15_done):
    env, did, stub = rs15_done
    video = env.video_file(did)
    gpx = env.gpx_file(did)
    chunks = [Path(p) for p in stub.video_calls]
    unblurred = env.unblurred_paths(did)
    before = snapshot(env, did)
    assert video.is_file() and chunks and all(c.is_file() for c in chunks)
    assert unblurred and all(p.is_file() for p in unblurred)
    assert before["blurred"] and before["counts"]["c"] > 0 and before["counts"]["h"] > 0

    r = env.delete_video(did)
    assert r.status_code in (200, 202, 204), r.text

    # raw video, extracted frames/chunks and the unblurred candidate JPEGs are gone
    assert not video.exists(), "raw video still on disk"
    for c in chunks:
        assert not c.exists(), "work chunk still on disk: %s" % c
    for p in unblurred:
        assert not p.exists(), "unblurred candidate image still on disk: %s" % p
    blurred = set(before["blurred"])
    for f in env.drive_files(did):
        ext = f.suffix.lower()
        if ext in VIDEO_EXT:
            pytest.fail("a video file is still stored for the drive: %s" % f)
        if ext in (".jpg", ".jpeg", ".png"):
            assert norm_path(f) in blurred, "an unblurred image is still stored: %s" % f

    # GPX, DB records and blurred images stay exactly as they were
    assert gpx.is_file() and sha(gpx) == before["gpx"]
    after = snapshot(env, did)
    assert after["counts"] == before["counts"]
    assert after["hazards"] == before["hazards"]
    assert after["blurred"] == before["blurred"], "blurred images were touched"
    assert after["photos"] == before["photos"], "photos served after deletion differ"
    assert after["list"] == before["list"]


def test_drive_json_shows_the_video_is_deleted(rs15_done):
    env, did, stub = rs15_done
    r = env.delete_video(did)
    assert r.status_code in (200, 202, 204)
    d = env.get(did)
    assert video_deleted(d), d
    assert d.get("video_deleted") is True
    mine = [x for x in env.drive_list() if x["id"] == did][0]
    assert mine.get("video_deleted") is True
    assert status_of(d) in DONE, "deleting the video must not change the processing status"
    if r.status_code != 204 and "json" in r.headers.get("content-type", ""):
        body = r.json()
        if isinstance(body, dict) and ("video_deleted" in body or "id" in body):
            assert body.get("video_deleted", True) is True


def test_drive_record_survives_and_is_still_listed(rs15_done):
    env, did, stub = rs15_done
    env.delete_video(did)
    assert env.client.get("/api/drives/%s" % did).status_code == 200
    assert did in {x["id"] for x in env.drive_list()}
    assert env.client.get("/api/drives/%s/hazards" % did).status_code == 200


def test_other_drives_are_untouched(rs15_env):
    env = rs15_env
    sa, sb = make_stub(), make_stub()
    a = env.process_ok(sa)
    b = env.process_ok(sb)
    vb, gb = env.video_file(b), env.gpx_file(b)
    ub = env.unblurred_paths(b)
    bb = {norm_path(p): sha(p) for p in env.blurred_paths(b)}
    chunks_b = [Path(p) for p in sb.video_calls]
    assert env.delete_video(a).status_code in (200, 202, 204)
    assert vb.is_file() and gb.is_file()
    assert all(p.is_file() for p in ub + chunks_b)
    assert {norm_path(p): sha(p) for p in env.blurred_paths(b)} == bb
    assert env.get(b)["video_deleted"] is False


def test_delete_on_a_drive_that_was_never_processed(rs15_env):
    env = rs15_env
    did = env.upload()
    video = env.video_file(did)
    gpx = env.gpx_file(did)
    assert video.is_file()
    r = env.delete_video(did)
    assert r.status_code in (200, 202, 204), r.text
    assert not video.exists() and gpx.is_file()
    d = env.get(did)
    assert video_deleted(d) and status_of(d) == "uploaded"


def test_delete_on_a_failed_drive(rs15_env):
    env = rs15_env
    stub = make_stub(fail_video_on=0)
    did = env.process(stub)
    assert status_of(env.get(did)) in FAILED
    video = env.video_file(did)
    r = env.delete_video(did)
    assert r.status_code in (200, 202, 204), r.text
    assert not video.exists()
    d = env.get(did)
    assert status_of(d) in FAILED and video_deleted(d)


# ------------------------------------------------------------------ errors and idempotence

def test_unknown_drive_is_404(rs15_env):
    env = rs15_env
    for bad in ("no-such-drive", "00000000-0000-0000-0000-000000000000", "999999"):
        assert env.delete_video(bad).status_code == 404, bad


def test_traversal_in_drive_id_deletes_nothing(rs15_done):
    env, did, stub = rs15_done
    sentinel = env.storage / "sentinel-task15.txt"
    sentinel.write_bytes(b"keep me")
    video = env.video_file(did)
    method, path = env.delete_route()
    for bad in ("..", "%2e%2e", "..%2f..%2f", "%2e%2e%2f%2e%2e%2f", "..%5c", "%00"):
        r = getattr(env.client, method)(re.sub(r"\{[^}/]+\}", bad, path))
        assert r.status_code in (400, 404, 405, 422), (bad, r.status_code)
    assert sentinel.is_file() and video.is_file() and env.db.is_file()


def test_deleting_twice_is_idempotent_and_never_a_server_error(rs15_done):
    env, did, stub = rs15_done
    first = env.delete_video(did)
    assert first.status_code in (200, 202, 204)
    before = snapshot(env, did)
    second = env.delete_video(did)
    assert second.status_code in (200, 202, 204, 404, 409, 410), second.status_code
    assert second.status_code < 500
    after = snapshot(env, did)
    assert after["blurred"] == before["blurred"] and after["counts"] == before["counts"]
    assert env.get(did)["video_deleted"] is True


def test_delete_when_the_files_are_already_missing_is_not_a_server_error(rs15_done):
    env, did, stub = rs15_done
    env.video_file(did).unlink()
    for p in env.unblurred_paths(did):
        p.unlink()
    r = env.delete_video(did)
    assert r.status_code < 500, r.text
    assert r.status_code in (200, 202, 204, 404, 409, 410)
    assert all(p.is_file() for p in env.blurred_paths(did))


def test_get_on_the_delete_route_does_not_delete(rs15_done):
    env, did, stub = rs15_done
    method, path = env.delete_route()
    video = env.video_file(did)
    url = re.sub(r"\{[^}/]+\}", did, path)
    for other in ("put", "patch"):
        r = getattr(env.client, other)(url)
        assert r.status_code in (404, 405)
    r = env.client.get(url)
    assert r.status_code in (200, 404, 405)
    if r.status_code == 200:
        assert r.content != b"" and video.is_file()
    assert video.is_file()
    assert env.get(did)["video_deleted"] is False


# ------------------------------------------------------------------ while processing

def test_processing_drive_cannot_have_its_video_deleted(rs15_done):
    env, did, stub = rs15_done
    video = env.video_file(did)
    env.x("UPDATE drives SET status = 'processing' WHERE id = ?", (did,))
    r = env.delete_video(did)
    assert r.status_code == 409, r.text
    assert video.is_file()
    assert all(p.is_file() for p in env.unblurred_paths(did))
    assert env.get(did)["video_deleted"] is False


def test_delete_during_a_live_run_is_refused_and_the_run_finishes(rs15_env):
    env = rs15_env
    codes = []

    def mid_run(stub, n, path):
        if n == 0:
            r = env.delete_video(env._did)
            codes.append(r.status_code)
            codes.append(env.video_file(env._did).is_file())

    stub = make_stub(on_video=mid_run)
    did = env.upload()
    env._did = did
    env.process(stub, did=did)
    assert codes[0] == 409, codes
    assert codes[1] is True, "video was deleted while the pipeline was running"
    d = env.get(did)
    assert status_of(d) in DONE and d["video_deleted"] is False

    stub2 = make_stub(on_image=lambda s, n, p: codes.append(("blur", env.delete_video(did).status_code))
                      if n == 0 else None)
    env.process(stub2, did=did)
    assert ("blur", 409) in codes, codes
    assert status_of(env.get(did)) in DONE
    assert env.video_file(did).is_file()


# ------------------------------------------------------------------ after deletion

def test_process_endpoint_refuses_after_deletion(rs15_done):
    env, did, stub = rs15_done
    before = snapshot(env, did)
    assert env.delete_video(did).status_code in (200, 202, 204)
    probe = make_stub()
    env.use_provider(probe)
    r = env.post_process(did)
    assert r.status_code in (409, 410), "re-processing a drive whose video is gone must be refused, got %s" % r.status_code
    assert probe.video_calls == [] and probe.image_calls == [], "the provider was called without a video"
    d = env.get(did)
    assert status_of(d) in DONE and d["video_deleted"] is True
    after = snapshot(env, did)
    assert after["counts"] == before["counts"] and after["blurred"] == before["blurred"]
    assert after["photos"] == before["photos"]


def test_process_endpoint_refuses_after_deletion_on_unprocessed_drive(rs15_env):
    env = rs15_env
    did = env.upload()
    assert env.delete_video(did).status_code in (200, 202, 204)
    probe = make_stub()
    env.use_provider(probe)
    r = env.post_process(did)
    assert r.status_code in (409, 410), r.status_code
    assert probe.video_calls == []
    assert status_of(env.get(did)) == "uploaded"


def test_hazards_and_blurred_photos_still_work_after_deletion(rs15_done):
    env, did, stub = rs15_done
    items_before = env.hazards_list(did)
    assert items_before
    env.delete_video(did)
    items = env.hazards_list(did)
    assert items == items_before
    for it in items:
        r = env.client.get(it["photo_url"])
        assert r.status_code == 200 and r.headers["content-type"].split(";")[0] == "image/jpeg"
        assert r.content != b""
        if it.get("crop_url"):
            assert env.client.get(it["crop_url"]).status_code == 200


def test_unblurred_images_are_not_served_after_deletion_either(rs15_done):
    env, did, stub = rs15_done
    c = env.accepted(did)[0]
    orig_bytes = env.path_of(c["best_frame_path"]).read_bytes()
    env.delete_video(did)
    r = env.photo(did, c["id"])
    assert r.status_code == 200 and r.content != orig_bytes
    env.x("UPDATE candidates SET blurred_frame_path = NULL WHERE id = ?", (c["id"],))
    assert env.photo(did, c["id"]).status_code == 404


def test_delete_after_rerun_still_keeps_the_new_blurred_images(rs15_env):
    env = rs15_env
    did = env.process_ok(make_stub())
    env.process(make_stub(), did=did)
    blurred = {norm_path(p): sha(p) for p in env.blurred_paths(did)}
    assert blurred
    assert env.delete_video(did).status_code in (200, 202, 204)
    assert {norm_path(p): sha(p) for p in env.blurred_paths(did)} == blurred
