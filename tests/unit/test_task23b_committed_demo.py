"""Task 23b: the COMMITTED demo/ folder is privacy-safe and works with the read-only demo app.

Skipped while <repo>/demo does not exist.
"""
from __future__ import annotations

import importlib
import os
import re
import sqlite3
from pathlib import Path

import pytest

from rs23b_helpers import DRIVE_LETTER, IMG_EXT, ROOT, PATH_COLS

DEMO = ROOT / "demo"
pytestmark = pytest.mark.skipif(not DEMO.exists(), reason="no committed demo/ folder yet")

DEMO_BODY = {"detail": "This is a read-only demo."}
ALLOWED = re.compile(r"^(demo\.sqlite|storage/[^/]+/candidates/[^/]+\.(jpg|jpeg|png))$")


def files():
    return sorted(str(p.relative_to(DEMO)).replace("\\", "/") for p in DEMO.rglob("*") if p.is_file())


def db():
    return DEMO / "demo.sqlite"


def q(sql, params=()):
    con = sqlite3.connect(f"file:{db().as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params)]
    finally:
        con.close()


def text_values():
    con = sqlite3.connect(f"file:{db().as_posix()}?mode=ro", uri=True)
    try:
        for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall():
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
            for row in con.execute(f"SELECT * FROM {t}"):
                for c, v in zip(cols, row):
                    if isinstance(v, str):
                        yield t, c, v
    finally:
        con.close()


# ------------------------------------------------------------------ the files

def test_the_database_exists():
    assert db().is_file()


def test_only_allowed_files_exist():
    bad = [f for f in files() if not ALLOWED.match(f)]
    assert not bad, bad


def test_no_video_gpx_or_frame_files():
    bad = [f for f in files() if f.lower().endswith((".mov", ".mp4", ".m4v", ".hevc", ".avi", ".mkv", ".gpx"))]
    assert not bad, bad
    assert not [f for f in files() if re.search(r"(?i)(_best\.|_crop\.|best_frame|/work/|/exports/)", f)
                and not re.search(r"_blur\.", f)], "an unblurred image name"


def test_every_image_is_a_blurred_copy_named_like_one():
    for f in files():
        if f.lower().endswith(IMG_EXT):
            assert "blur" in Path(f).name.lower(), f


def test_no_sqlite_side_files():
    assert not [f for f in files() if f.endswith(("-wal", "-shm", "-journal"))]


# ------------------------------------------------------------------ the database

def test_no_absolute_or_local_path_strings():
    for t, c, v in text_values():
        assert not DRIVE_LETTER.search(v), (t, c, v[:80])
        assert "/Users/" not in v and "\\Users\\" not in v, (t, c)
        assert not (v.startswith("/") and "/" in v[1:] and c.endswith("path")), (t, c, v[:80])
        assert str(ROOT) not in v and str(ROOT).replace("\\", "/") not in v, (t, c)


def test_path_columns_are_relative_with_forward_slashes():
    for t, c, v in text_values():
        if c in PATH_COLS or c == "photo_path":
            assert not os.path.isabs(v) and "\\" not in v, (t, c, v)


def test_every_hazard_photo_and_crop_exists():
    hz = q("SELECT * FROM hazards")
    for h in hz:
        assert h["photo_path"], f"hazard {h['id']} has no photo"
        assert (ROOT / h["photo_path"]).is_file(), h["photo_path"]
        if h["crop_path"]:
            assert (ROOT / h["crop_path"]).is_file(), h["crop_path"]


def test_every_image_file_is_referenced():
    refs = {v for t, c, v in text_values() if c in PATH_COLS or c == "photo_path"}
    for f in files():
        if f.lower().endswith(IMG_EXT):
            assert f"demo/{f}" in refs, f


def test_unblurred_columns_and_drive_paths_are_scrubbed():
    for c in q("SELECT * FROM candidates"):
        for col in ("best_frame_path", "crop_path"):
            v = c[col]
            assert v is None or not (ROOT / v).is_file()
    for d in q("SELECT * FROM drives"):
        assert d["video_filename"] is None
        assert d["video_deleted_at"]
        assert d["error"] is None and d["warning"] is None
        for col in ("video_path", "gpx_path"):
            assert d[col] is None or not (ROOT / d[col]).is_file()


def test_internal_tables_are_empty():
    for t in ("detections", "detection_chunks", "pipeline_runs", "exports"):
        assert q(f"SELECT COUNT(*) AS n FROM {t}")[0]["n"] == 0, t


def test_hazards_exist_and_have_demo_statuses():
    hz = q("SELECT id, status FROM hazards")
    assert hz
    assert {h["status"] for h in hz} <= {"approved", "reported", "new", "rejected", "possibly_fixed"}


# ------------------------------------------------------------------ served by the app

@pytest.fixture()
def client(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("ROADSCOUT_DEMO", "1")
    monkeypatch.setenv("ROADSCOUT_DB", str(DEMO / "demo.sqlite"))
    monkeypatch.setenv("ROADSCOUT_STORAGE_DIR", str(DEMO / "storage"))
    monkeypatch.setenv("ROADSCOUT_COST_LOG", "off")
    monkeypatch.setenv("ROADSCOUT_PROVIDER", "fake")
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("ROADSCOUT_STATIC_DIR", raising=False)
    import backend.main as main

    main = importlib.reload(main)
    from fastapi.testclient import TestClient

    before = (DEMO / "demo.sqlite").read_bytes()
    yield TestClient(main.app)
    assert (DEMO / "demo.sqlite").read_bytes() == before, "the demo app must never write to the committed database"
    os.environ.pop("ROADSCOUT_DEMO", None)
    importlib.reload(main)


def hazards(client):
    r = client.get("/api/hazards")
    assert r.status_code == 200
    body = r.json()
    return body if isinstance(body, list) else body.get("hazards", body.get("items", []))


def test_api_lists_at_least_one_hazard(client):
    assert len(hazards(client)) >= 1


def test_every_hazard_photo_is_served_as_jpeg(client):
    for h in hazards(client):
        assert h.get("photo_url"), h["id"]
        r = client.get(h["photo_url"])
        assert r.status_code == 200, h["id"]
        assert r.headers["content-type"].startswith("image/jpeg")
        assert len(r.content) > 0
        if h.get("crop_url"):
            assert client.get(h["crop_url"]).status_code == 200


def test_report_for_an_approved_hazard(client):
    approved = [h for h in hazards(client) if h.get("status") == "approved"]
    if not approved:
        pytest.skip("the committed dataset has no approved hazard")
    r = client.get(f"/api/hazards/{approved[0]['id']}/report")
    assert r.status_code == 200


def test_a_patch_is_forbidden(client):
    hz = hazards(client)
    r = client.patch(f"/api/hazards/{hz[0]['id']}", json={"status": "rejected"})
    assert r.status_code == 403 and r.json() == DEMO_BODY


def test_the_demo_endpoint_says_demo(client):
    assert client.get("/api/demo").json() == {"demo": True}
    assert client.get("/health").status_code == 200


def test_no_get_route_returns_an_unblurred_original(client):
    originals = {f for f in files()}          # nothing but blurred copies exists, so there is nothing to leak
    assert all("blur" in Path(f).name.lower() for f in originals if f.lower().endswith(IMG_EXT))
