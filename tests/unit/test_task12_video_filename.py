"""Task 12: drives.video_filename - the column, its migration, and the upload route storing it."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

import rs12_helpers as H  # noqa: F401  (sets up sys.path for the repo)

_UNIT = Path(r"C:\Users\dhair\RoadScout\tests\unit")
if _UNIT.is_dir() and str(_UNIT) not in sys.path:
    sys.path.append(str(_UNIT))


def cols(path):
    con = sqlite3.connect(str(path))
    try:
        return {r[1]: r for r in con.execute("PRAGMA table_info(drives)")}
    finally:
        con.close()


def test_column_exists_nullable_text(tmp_path):
    from backend.db import init_db

    p = tmp_path / "a.sqlite"
    init_db(p)
    c = cols(p)
    assert "video_filename" in c
    _cid, _name, ctype, notnull, _default, _pk = c["video_filename"]
    assert ctype.upper() == "TEXT" and notnull == 0


def test_migration_adds_the_column_to_an_old_database_and_keeps_rows(tmp_path):
    db = H.make_db(tmp_path)
    db.drive("old-1", None)
    con = sqlite3.connect(str(db.path))
    try:
        try:
            con.execute("ALTER TABLE drives DROP COLUMN video_filename")
        except sqlite3.OperationalError as e:
            pytest.skip("this SQLite cannot drop a column (%s)" % e)
        con.commit()
    finally:
        con.close()
    assert "video_filename" not in cols(db.path)
    from backend.db import init_db

    init_db(db.path)
    init_db(db.path)
    assert "video_filename" in cols(db.path)
    con = sqlite3.connect(str(db.path))
    try:
        row = con.execute("SELECT id, status, video_filename FROM drives WHERE id = 'old-1'").fetchone()
    finally:
        con.close()
    assert row == ("old-1", "done", None)


def test_init_db_twice_keeps_one_column(tmp_path):
    from backend.db import init_db

    p = tmp_path / "b.sqlite"
    init_db(p)
    init_db(p)
    assert [c for c in cols(p) if c == "video_filename"] == ["video_filename"]


@pytest.fixture()
def upload_env(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    import shutil

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not on PATH")
    from rs10_helpers import clip_bytes, gpx_bytes

    monkeypatch.setenv("ROADSCOUT_DB", str(tmp_path / "up.sqlite"))
    monkeypatch.setenv("ROADSCOUT_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("ROADSCOUT_PROVIDER", "fake")
    monkeypatch.delenv("META_API_KEY", raising=False)
    from fastapi.testclient import TestClient

    from backend.db import init_db
    from backend.main import app

    init_db()
    with TestClient(app) as client:
        yield client, tmp_path / "up.sqlite", clip_bytes(), gpx_bytes()


def upload(client, clip, gpx, filename):
    files = {"video": (filename, clip, "video/mp4"), "gpx": ("track.gpx", gpx, "application/gpx+xml")}
    r = client.post("/api/drives", files=files)
    assert r.status_code == 201, r.text
    return r.json()


def stored(db, drive_id):
    con = sqlite3.connect(str(db))
    try:
        return con.execute("SELECT video_filename FROM drives WHERE id = ?", (drive_id,)).fetchone()[0]
    finally:
        con.close()


def test_upload_stores_the_original_file_name(upload_env):
    client, db, clip, gpx = upload_env
    d = upload(client, clip, gpx, "IMG_6513.MOV")
    assert stored(db, d["id"]) == "IMG_6513.MOV"


@pytest.mark.parametrize("sent", ["some/dir/IMG_6513.MOV", "../../etc/IMG_6513.MOV", "C:\\Users\\me\\Movies\\IMG_6513.MOV"])
def test_upload_stores_the_basename_only(upload_env, sent):
    client, db, clip, gpx = upload_env
    d = upload(client, clip, gpx, sent)
    assert stored(db, d["id"]) == "IMG_6513.MOV"


def test_upload_keeps_dots_and_spaces_in_the_name(upload_env):
    client, db, clip, gpx = upload_env
    d = upload(client, clip, gpx, "my drive.final.v2.mp4")
    assert stored(db, d["id"]) == "my drive.final.v2.mp4"


def test_the_file_name_is_not_exposed_in_the_drive_json(upload_env):
    client, db, clip, gpx = upload_env
    d = upload(client, clip, gpx, "IMG_6513.MOV")
    assert "video_filename" not in d
    assert "video_filename" not in client.get("/api/drives/%s" % d["id"]).json()
    body = client.get("/api/drives").json()
    items = body if isinstance(body, list) else body.get("drives", body.get("items", []))
    assert all("video_filename" not in x for x in items)


def test_a_label_file_named_after_the_upload_is_matched_end_to_end(upload_env, tmp_path):
    """Upload IMG_6513.MOV, seed a candidate, and the label file IMG_6513.csv is applied to that drive."""
    from backend.evaluation import evaluate, load_labels
    from rs12_helpers import haz, seg, write_labels

    client, db, clip, gpx = upload_env
    d = upload(client, clip, gpx, "IMG_6513.MOV")
    con = sqlite3.connect(str(db))
    try:
        with con:
            con.execute(
                "INSERT INTO candidates (drive_id, hazard_class, track_key, track_id, chunk, first_timestamp, "
                "last_timestamp, frames_seen, max_mask_area, best_frame_index, best_timestamp, best_box, "
                "best_frame_path, lat, lon, heading, score, geotag_status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (d["id"], "pothole", "0:1", 1, 0, 60.0, 64.0, 5, 400.0, 180, 62.0, "[1,2,3,4]", "x.jpg", 38.8, -77.3,
                 0.0, 0.7, "ok"))
            cid = con.execute("SELECT id FROM candidates").fetchone()[0]
            con.execute(
                "INSERT INTO verdicts (candidate_id, drive_id, is_hazard, hazard_class, severity, lane, confidence, "
                "reason, report_description, model, prompt_version, accepted, created_at) VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (cid, d["id"], 1, "pothole", 2, "center", 0.9, "r", "d", "m", "v1", 1, "2026-10-20T12:00:00+00:00"))
    finally:
        con.close()
    labels = tmp_path / "labels"
    write_labels(labels, "img_6513", [seg("A", "00:00", "05:00"), haz("A", "01:00", "01:05", "pothole")])
    r = evaluate(str(db), load_labels(labels))
    assert r.skipped_drives == []
    m = r.per_class["pothole"]
    assert (m.labeled, m.matched) == (1, 1)
