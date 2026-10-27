"""Task 11 backend: serving the best-frame and crop photos of verified hazards.

Updated for task 15: the photo route serves only the BLURRED copies (blurred_frame_path / blurred_crop_path).
"""
from __future__ import annotations

import re

import pytest

from rs11_helpers import (cid_of, is_jpeg, rs11_done, rs11_env)  # noqa: F401


def _replace_segment(url, old, new):
    """Replace the last path segment (or query value) equal to `old` with `new`."""
    old = str(old)
    pat = re.compile(r"(?<=[/=])%s(?=$|[/?&.])" % re.escape(old))
    ms = list(pat.finditer(url))
    assert ms, "candidate id %r not found as a path segment/query value in %r" % (old, url)
    m = ms[-1]
    return url[:m.start()] + str(new) + url[m.end():]


def _first(env, did):
    items = env.hazards(did)
    assert items
    return items[0]


def test_photo_served_as_jpeg_matching_best_frame(rs11_done):
    env, did = rs11_done
    rows = {str(r["id"]): r for r in env.all_candidates(did)}
    items = env.hazards(did)
    assert items
    for it in items:
        url = env.photo_url(it)
        r = env.client.get(url)
        assert r.status_code == 200, (url, r.status_code, r.text[:200])
        assert r.headers["content-type"].split(";")[0].strip() == "image/jpeg"
        assert is_jpeg(r.content), "not a valid JPEG"
        f = env.resolve_file(rows[str(cid_of(it))]["blurred_frame_path"])
        assert f is not None, "blurred_frame_path %r not found on disk" % rows[str(cid_of(it))]["blurred_frame_path"]
        assert r.content == f.read_bytes()


def test_photo_url_is_relative_api_url(rs11_done):
    env, did = rs11_done
    for it in env.hazards(did):
        url = env.photo_url(it)
        assert url.startswith("/"), url
        assert url.startswith("/api/"), "photo URL should live under /api/ so the Vite proxy serves it: %s" % url


def test_distinct_candidates_have_distinct_photo_urls(rs11_done):
    env, did = rs11_done
    urls = [env.photo_url(i) for i in env.hazards(did)]
    assert len(set(urls)) == len(urls)


def test_crop_served_if_offered(rs11_done):
    env, did = rs11_done
    rows = {str(r["id"]): r for r in env.all_candidates(did)}
    seen = 0
    for it in env.hazards(did):
        url = env.photo_url(it, crop=True)
        if not url:
            continue
        seen += 1
        r = env.client.get(url)
        assert r.status_code == 200, (url, r.status_code)
        assert r.headers["content-type"].split(";")[0].strip() == "image/jpeg"
        assert is_jpeg(r.content)
        f = env.resolve_file(rows[str(cid_of(it))]["blurred_crop_path"])
        assert f is not None
        assert r.content == f.read_bytes()
    if not seen:
        pytest.skip("no crop URL offered (optional)")


def test_unknown_candidate_is_404(rs11_done):
    env, did = rs11_done
    it = _first(env, did)
    url = env.photo_url(it)
    for bad in ("999999", "no-such-candidate", "0", "-1"):
        r = env.client.get(_replace_segment(url, cid_of(it), bad))
        assert r.status_code in (404, 422), (bad, r.status_code)
        assert r.status_code != 200


def test_unknown_drive_photo_is_404(rs11_done):
    env, did = rs11_done
    it = _first(env, did)
    url = env.photo_url(it)
    r = env.client.get(_replace_segment(url, did, "no-such-drive"))
    assert r.status_code == 404


def test_candidate_of_other_drive_is_404(rs11_env):
    env = rs11_env
    a = env.upload_and_process()
    b = env.upload_and_process()
    item_a = _first(env, a)
    url_a = env.photo_url(item_a)
    assert env.client.get(url_a).status_code == 200
    cross = _replace_segment(url_a, a, b)  # drive B + candidate of drive A
    r = env.client.get(cross)
    assert r.status_code == 404, "candidate of drive A served under drive B (%s)" % cross


def test_crop_of_other_drive_is_404(rs11_env):
    env = rs11_env
    a = env.upload_and_process()
    b = env.upload_and_process()
    item_a = _first(env, a)
    url = env.photo_url(item_a, crop=True)
    if not url:
        pytest.skip("no crop URL offered (optional)")
    assert env.client.get(_replace_segment(url, a, b)).status_code == 404


def test_rejected_candidate_photo_is_not_served(rs11_done):
    """A photo must not be reachable for a candidate that is not a verified hazard (unblurred frames stay minimal)."""
    env, did = rs11_done
    it = _first(env, did)
    url = env.photo_url(it)
    assert env.client.get(url).status_code == 200
    env.x("UPDATE verdicts SET accepted = 0 WHERE candidate_id = ?", (cid_of(it),))
    r = env.client.get(url)
    # Strict reading of "list of verified hazards with photos": rejected candidates are not exposed.
    assert r.status_code == 404


TRAVERSAL_IDS = [
    "..", "../..", "..%2f..%2f", "%2e%2e%2f%2e%2e%2f", "..%5c..%5c", "%2e%2e",
    "....//", "%252e%252e%252f", "..%2fdb%2frs11.sqlite", "/etc/passwd", "C:%5cWindows%5cwin.ini",
    "%00", "1%00.jpg", "1/../../..",
]


@pytest.mark.parametrize("bad", TRAVERSAL_IDS)
def test_path_traversal_in_candidate_id_impossible(rs11_done, bad):
    env, did = rs11_done
    it = _first(env, did)
    url = env.photo_url(it)
    secret = env.storage / "secret-task11.txt"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_bytes(b"TOP-SECRET-TASK11")
    r = env.client.get(_replace_segment(url, cid_of(it), bad))
    assert r.status_code in (400, 404, 405, 422), (bad, r.status_code)
    assert b"TOP-SECRET-TASK11" not in r.content
    assert not r.content.startswith(b"SQLite format 3")
    assert not r.content.startswith(b"\xff\xd8")


@pytest.mark.parametrize("bad", ["..", "..%2f", "%2e%2e%2f", "..%5c", "%2e%2e%2f%2e%2e%2fdb"])
def test_path_traversal_in_drive_id_impossible(rs11_done, bad):
    env, did = rs11_done
    it = _first(env, did)
    url = env.photo_url(it)
    r = env.client.get(_replace_segment(url, did, bad))
    assert r.status_code in (400, 404, 405, 422), (bad, r.status_code)
    assert not r.content.startswith(b"SQLite format 3")
    assert not r.content.startswith(b"\xff\xd8")


def test_only_candidates_folder_is_served(rs11_done):
    """Even if a DB row points elsewhere (tampered / bad data), files outside the candidates folder are not served."""
    env, did = rs11_done
    it = _first(env, did)
    cid = cid_of(it)
    url = env.photo_url(it)

    secret = env.storage / "outside-candidates-task11.jpg"
    secret.parent.mkdir(parents=True, exist_ok=True)
    # a valid JPEG so only the location can be the reason for refusing it
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (250, 0, 0)).save(buf, "JPEG")
    secret.write_bytes(buf.getvalue())

    targets = [str(secret), str(env.resolve_file(env.all_candidates(did)[0]["best_frame_path"]).parent.parent
                              / "outside-candidates-task11.jpg"),
               "../outside-candidates-task11.jpg", "../../outside-candidates-task11.jpg",
               str(env.db)]
    original = [r for r in env.all_candidates(did) if str(r["id"]) == str(cid)][0]["blurred_frame_path"]
    for t in targets:
        env.x("UPDATE candidates SET blurred_frame_path = ? WHERE id = ?", (t, cid))
        r = env.client.get(url)
        assert r.status_code != 200 or r.content != buf.getvalue(), "served a file outside the candidates folder: %s" % t
        assert not r.content.startswith(b"SQLite format 3")
        assert r.status_code < 500 or r.status_code == 500 and False, "server error for tampered path %s" % t
        assert r.status_code in (400, 403, 404), (t, r.status_code)
    env.x("UPDATE candidates SET blurred_frame_path = ? WHERE id = ?", (original, cid))
    assert env.client.get(url).status_code == 200


def test_missing_file_on_disk_is_404_not_500(rs11_done):
    env, did = rs11_done
    it = _first(env, did)
    row = [r for r in env.all_candidates(did) if str(r["id"]) == str(cid_of(it))][0]
    f = env.resolve_file(row["blurred_frame_path"])
    f.unlink()
    r = env.client.get(env.photo_url(it))
    assert r.status_code == 404


def test_photo_is_get_only(rs11_done):
    env, did = rs11_done
    url = env.photo_url(_first(env, did))
    for m in ("post", "put", "delete", "patch"):
        r = getattr(env.client, m)(url)
        assert r.status_code in (404, 405), (m, r.status_code)
    assert env.client.get(url).status_code == 200
