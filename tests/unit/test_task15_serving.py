"""Task 15: the review UI / results API serve only blurred images and never expose an unblurred path."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from rs15_helpers import (OK_STATUS, as_list, cid_of, make_stub, rs15_done, rs15_env, sha)  # noqa: F401


RAW_PATH_KEYS = {"best_frame_path", "crop_path", "blurred_frame_path", "blurred_crop_path", "photo_path"}


def norm_path(p):
    return os.path.normcase(os.path.abspath(str(p)))


def test_photo_bytes_are_the_blurred_file_never_the_original(rs15_done):
    env, did, stub = rs15_done
    acc = env.accepted(did)
    assert acc
    for c in acc:
        r = env.photo(did, c["id"])
        assert r.status_code == 200, r.text[:200]
        assert r.headers["content-type"].split(";")[0].strip() == "image/jpeg"
        blurred = env.path_of(c["blurred_frame_path"])
        original = env.path_of(c["best_frame_path"])
        assert r.content == blurred.read_bytes()
        assert r.content != original.read_bytes(), "the unblurred original was served"


def test_crop_bytes_are_the_blurred_crop_never_the_original(rs15_done):
    env, did, stub = rs15_done
    seen = 0
    for c in env.accepted(did):
        if not c.get("crop_path"):
            continue
        r = env.photo(did, c["id"], crop=True)
        assert r.status_code == 200, r.text[:200]
        assert r.content == env.path_of(c["blurred_crop_path"]).read_bytes()
        assert r.content != env.path_of(c["crop_path"]).read_bytes()
        seen += 1
    if not seen:
        pytest.skip("no crops saved")


def test_list_urls_serve_the_blurred_bytes(rs15_done):
    env, did, stub = rs15_done
    items = env.hazards_list(did)
    assert items
    rows = {str(c["id"]): c for c in env.cands(did)}
    for it in items:
        c = rows[str(cid_of(it))]
        url = it.get("photo_url")
        assert url and url.startswith("/api/"), it
        r = env.client.get(url)
        assert r.status_code == 200
        assert r.content == env.path_of(c["blurred_frame_path"]).read_bytes()
        if it.get("crop_url"):
            r = env.client.get(it["crop_url"])
            assert r.status_code == 200
            assert r.content == env.path_of(c["blurred_crop_path"]).read_bytes()


@pytest.mark.parametrize("variant", ["original", "raw", "unblurred", "full", "source", "best"])
def test_no_variant_unlocks_the_unblurred_image(rs15_done, variant):
    env, did, stub = rs15_done
    c = env.accepted(did)[0]
    orig = env.path_of(c["best_frame_path"]).read_bytes()
    r = env.client.get("/api/drives/%s/hazards/%s/photo?variant=%s" % (did, c["id"], variant))
    assert r.status_code in (200, 400, 404, 422)
    assert r.content != orig


def test_candidate_without_blurred_image_is_404_even_though_original_exists(rs15_done):
    env, did, stub = rs15_done
    c = env.accepted(did)[0]
    orig = env.path_of(c["best_frame_path"])
    assert orig.is_file()
    assert env.photo(did, c["id"]).status_code == 200
    env.x("UPDATE candidates SET blurred_frame_path = NULL WHERE id = ?", (c["id"],))
    r = env.photo(did, c["id"])
    assert r.status_code == 404, "fell back to the unblurred file (%s)" % r.status_code
    assert r.content != orig.read_bytes()
    assert orig.is_file()


def test_crop_without_blurred_crop_is_404(rs15_done):
    env, did, stub = rs15_done
    cs = [c for c in env.accepted(did) if c.get("crop_path")]
    if not cs:
        pytest.skip("no crops saved")
    c = cs[0]
    assert env.photo(did, c["id"], crop=True).status_code == 200
    env.x("UPDATE candidates SET blurred_crop_path = NULL WHERE id = ?", (c["id"],))
    r = env.photo(did, c["id"], crop=True)
    assert r.status_code == 404
    assert r.content != env.path_of(c["crop_path"]).read_bytes()
    # the best frame is unaffected
    assert env.photo(did, c["id"]).status_code == 200


def test_blur_status_not_ok_is_not_served(rs15_done):
    env, did, stub = rs15_done
    c = env.accepted(did)[0]
    env.x("UPDATE candidates SET blur_status = 'failed', blurred_frame_path = NULL, blurred_crop_path = NULL "
          "WHERE id = ?", (c["id"],))
    assert env.photo(did, c["id"]).status_code == 404
    assert env.photo(did, c["id"], crop=True).status_code == 404


def test_blurred_file_missing_on_disk_is_404_not_500_and_not_the_original(rs15_done):
    env, did, stub = rs15_done
    c = env.accepted(did)[0]
    env.path_of(c["blurred_frame_path"]).unlink()
    r = env.photo(did, c["id"])
    assert r.status_code == 404
    assert r.content != env.path_of(c["best_frame_path"]).read_bytes()


def test_list_never_offers_a_broken_or_unblurred_link(rs15_done):
    env, did, stub = rs15_done
    victim = env.accepted(did)[0]
    env.x("UPDATE candidates SET blurred_frame_path = NULL, blurred_crop_path = NULL, blur_status = 'failed' "
          "WHERE id = ?", (victim["id"],))
    r = env.client.get("/api/drives/%s/hazards" % did)
    assert r.status_code == 200
    for it in as_list(r.json()):
        for key in ("photo_url", "crop_url"):
            u = it.get(key)
            if u:
                resp = env.client.get(u)
                if str(cid_of(it)) == str(victim["id"]):
                    assert resp.status_code == 404
                else:
                    assert resp.status_code == 200


def test_hazards_list_json_exposes_no_unblurred_path(rs15_done):
    env, did, stub = rs15_done
    r = env.client.get("/api/drives/%s/hazards" % did)
    assert r.status_code == 200
    body = r.text
    for c in env.cands(did):
        for k in ("best_frame_path", "crop_path"):
            if c.get(k):
                p = env.path_of(c[k])
                assert p.name not in body, "unblurred file name leaked: %s" % p.name
                assert str(p) not in body and str(p).replace("\\", "/") not in body
    assert str(env.tmp) not in body and str(env.tmp).replace("\\", "/") not in body
    for it in as_list(r.json()):
        for k, v in it.items():
            if isinstance(v, str) and not v.startswith("/api/"):
                assert not v.lower().endswith((".jpg", ".jpeg", ".png")), (k, v)
            assert k.lower() not in RAW_PATH_KEYS, "raw path field %r exposed" % k


def test_drive_json_exposes_no_unblurred_path(rs15_done):
    env, did, stub = rs15_done
    texts = [env.client.get("/api/drives/%s" % did).text, env.client.get("/api/drives").text]
    for body in texts:
        for c in env.cands(did):
            for k in ("best_frame_path", "crop_path"):
                if c.get(k):
                    assert env.path_of(c[k]).name not in body


def test_no_static_mount_exposes_the_storage_folder(rs15_done):
    env, did, stub = rs15_done
    from backend.main import app

    storage = env.storage.resolve()
    for route in app.routes:
        sub = getattr(route, "app", None)
        directory = getattr(sub, "directory", None)
        dirs = list(getattr(sub, "all_directories", []) or ([directory] if directory else []))
        for d in dirs:
            p = Path(str(d)).resolve()
            assert not (p == storage or storage in p.parents or p in storage.parents),                 "route %s statically serves %s (contains unblurred images)" % (getattr(route, "path", route), p)


@pytest.mark.parametrize("root", ["storage", "data", "files", "static", "media", "images", "uploads"])
def test_unblurred_candidate_file_not_reachable_through_guessed_static_urls(rs15_done, root):
    env, did, stub = rs15_done
    c = env.accepted(did)[0]
    orig = env.path_of(c["best_frame_path"])
    rel = orig.relative_to(env.storage).as_posix()
    orig_bytes = orig.read_bytes()
    for url in ("/%s/%s" % (root, rel), "/%s/%s/candidates/%s" % (root, did, orig.name),
                "/api/%s/%s" % (root, rel)):
        r = env.client.get(url)
        assert r.content != orig_bytes, "unblurred image reachable at %s" % url
