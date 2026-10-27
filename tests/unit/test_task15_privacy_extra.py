"""Task 15 review round 2: extra privacy and dedup checks against the delivered design."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rs15_helpers import (OK_STATUS, as_list, make_stub, rs15_done, rs15_env, sha, status_of)  # noqa: F401
from rs15_helpers import FAILED


def norm_path(p):
    return os.path.normcase(os.path.abspath(str(p)))


def _all_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _all_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_strings(v)


def _no_fs_paths(env, text, what):
    tmp = str(env.tmp)
    assert tmp not in text and tmp.replace("\\", "/") not in text and tmp.replace("\\", "\\\\") not in text, \
        "%s exposes a filesystem path under the data dir: %s" % (what, text[:300])
    low = text.lower()
    assert "c:\\\\" not in low and "c:/" not in low and "/users/" not in low and "\\\\users\\\\" not in low, \
        "%s exposes an absolute path: %s" % (what, text[:300])


# ------------------------------------------------------------------ no paths in any JSON

def test_drive_and_hazard_json_have_no_filesystem_paths(rs15_done):
    env, did, stub = rs15_done
    for url in ("/api/drives/%s" % did, "/api/drives", "/api/drives/%s/hazards" % did):
        r = env.client.get(url)
        assert r.status_code == 200
        _no_fs_paths(env, r.text, url)
        body = r.json()
        for s in _all_strings(body):
            assert not s.lower().endswith((".jpg", ".jpeg", ".png")) or s.startswith("/api/"), (url, s)
    # stored keys such as video_path must at least be relative
    d = env.get(did)
    for k in ("video_path", "gpx_path"):
        if d.get(k):
            assert not Path(str(d[k])).is_absolute(), "%s is an absolute path: %s" % (k, d[k])


def test_blur_failure_error_has_no_filesystem_path(rs15_env):
    env = rs15_env
    did = env.process(make_stub(fail_image_on=0))
    d = env.get(did)
    assert status_of(d) in FAILED
    _no_fs_paths(env, json.dumps(d), "failed drive JSON")
    assert str(d["error"]).startswith("blurring:")


def test_delete_failure_detail_does_not_leak_a_path(rs15_done, monkeypatch):
    """A file that cannot be removed must not put an absolute path into the HTTP answer."""
    env, did, stub = rs15_done
    real = Path.unlink

    def boom(self, *a, **k):
        if self.name.endswith(".jpg") and "candidates" in self.parts and "_blur" not in self.name:
            raise PermissionError(13, "Access is denied", str(self))
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", boom)
    r = env.delete_video(did)
    monkeypatch.setattr(Path, "unlink", real)
    if r.status_code == 200:
        pytest.skip("deletion did not go through Path.unlink")
    assert r.status_code == 500
    _no_fs_paths(env, r.text, "500 answer of Delete video")


def test_delete_retry_after_a_failed_removal_finishes_the_job(rs15_done, monkeypatch):
    env, did, stub = rs15_done
    real = Path.unlink

    def boom(self, *a, **k):
        if self.name.endswith(".jpg") and "candidates" in self.parts and "_blur" not in self.name:
            raise PermissionError(13, "Access is denied", str(self))
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", boom)
    first = env.delete_video(did)
    monkeypatch.setattr(Path, "unlink", real)
    if first.status_code == 200:
        pytest.skip("deletion did not go through Path.unlink")
    blurred = {norm_path(p): sha(p) for p in env.blurred_paths(did)}
    second = env.delete_video(did)
    assert second.status_code == 200
    assert all(not p.exists() for p in env.unblurred_paths(did))
    assert {norm_path(p): sha(p) for p in env.blurred_paths(did)} == blurred


# ------------------------------------------------------------------ photo route is fail closed

def test_blurred_path_equal_to_original_is_never_served(rs15_done):
    env, did, stub = rs15_done
    c = env.accepted(did)[0]
    env.x("UPDATE candidates SET blurred_frame_path = best_frame_path WHERE id = ?", (c["id"],))
    r = env.photo(did, c["id"])
    assert r.status_code == 404
    assert r.content != env.path_of(c["best_frame_path"]).read_bytes()


def test_null_blur_status_with_a_blurred_path_is_not_served(rs15_done):
    env, did, stub = rs15_done
    c = env.accepted(did)[0]
    assert env.photo(did, c["id"]).status_code == 200
    env.x("UPDATE candidates SET blur_status = NULL WHERE id = ?", (c["id"],))
    assert env.photo(did, c["id"]).status_code == 404
    assert env.photo(did, c["id"], crop=True).status_code == 404
    env.x("UPDATE candidates SET blur_status = 'pending' WHERE id = ?", (c["id"],))
    assert env.photo(did, c["id"]).status_code == 404


def test_nothing_unblurred_is_served_by_any_documented_route(rs15_done):
    env, did, stub = rs15_done
    from backend.main import app

    originals = {}
    for c in env.cands(did):
        for k in ("best_frame_path", "crop_path"):
            if c.get(k):
                originals[norm_path(env.path_of(c[k]))] = env.path_of(c[k]).read_bytes()
    blobs = set(originals.values())
    ids = {"drive_id": did, "id": did, "candidate_id": str(env.accepted(did)[0]["id"]),
           "cid": str(env.accepted(did)[0]["id"])}
    import re
    for path, ops in app.openapi().get("paths", {}).items():
        if "get" not in ops:
            continue
        url = re.sub(r"\{([^}/]+)\}", lambda m: ids.get(m.group(1), "1"), path)
        for suffix in ("", "?variant=crop", "?variant=best"):
            r = env.client.get(url + suffix)
            assert r.status_code < 500, (url, r.status_code)
            assert r.content not in blobs, "route %s serves an unblurred candidate image" % url


# ------------------------------------------------------------------ dedup rules from the design

def test_dedup_skips_failed_blur_and_never_falls_back(tmp_path, monkeypatch):
    from dd14_helpers import VA, add_cand, dedup, execute, hazard_of, hazards, make_db

    db = make_db(tmp_path, monkeypatch)
    frame, crop = str(tmp_path / "f.jpg"), str(tmp_path / "c.jpg")
    good = add_cand(db, "A", VA[0], VA[1], 0.0, frame_path=frame, crop_path=crop)
    bad = add_cand(db, "B", VA[0] + 1e-5, VA[1], 0.0, frame_path=str(tmp_path / "bad.jpg"))
    nul = add_cand(db, "C", VA[0] + 5e-4, VA[1], 0.0, frame_path=str(tmp_path / "nul.jpg"))
    execute(db, "UPDATE candidates SET blur_status = 'failed' WHERE id = ?", (bad,))
    execute(db, "UPDATE candidates SET blur_status = NULL, blurred_frame_path = NULL, blurred_crop_path = NULL "
                "WHERE id = ?", (nul,))
    for d in ("A", "B", "C"):
        dedup(db, d)
    hs = hazards(db)
    assert hazard_of(db, bad) is None, "a failed-blur candidate must not become a hazard"
    assert hazard_of(db, good) is not None
    for h in hs:
        assert h["photo_path"] != str(tmp_path / "nul.jpg"), "fell back to the unblurred path"
        assert h["photo_path"] != str(tmp_path / "bad.jpg")
    nh = hazard_of(db, nul)
    if nh is not None:
        row = [h for h in hs if h["id"] == nh][0]
        assert row["photo_path"] in (None, ""), "no blurred image, so no photo"
