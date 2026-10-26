"""The drives table in SQLite and the rows the upload endpoint writes."""
import sqlite3

import pytest

from rs4_fixtures import *  # noqa: F401,F403
from rs4_helpers import (
    DEFAULT_CREATION_TIME,
    col,
    db_rows,
    drive_id,
    make_gpx,
    ok_status,
    parse_ts,
    resolve_stored,
    secs,
    sha,
    table_columns,
    unwrap_drive,
    upload,
)

META_TIME = parse_ts(DEFAULT_CREATION_TIME)


def _init(path):
    from backend.db import init_db

    init_db(path)


def test_init_db_creates_drives_table_with_key_fields(rs4_env, rs4_app):
    _init(rs4_env.db)
    cols = [c["name"] for c in table_columns(rs4_env.db)]
    assert cols, "no drives table"
    assert "id" in cols
    assert col(cols, r"^started_at$|^start"), cols
    assert col(cols, r"offset"), cols
    assert col(cols, r"video.*(path|key|uri|file)|(path|key|uri|file).*video"), cols
    assert col(cols, r"gpx.*(path|key|uri|file)|(path|key|uri|file).*gpx"), cols
    assert "status" in cols, cols


def test_id_is_the_primary_key(rs4_env, rs4_app):
    _init(rs4_env.db)
    info = {c["name"]: c for c in table_columns(rs4_env.db)}
    assert info["id"]["pk"] >= 1


def test_status_is_not_nullable_or_has_default(rs4_env, rs4_app):
    _init(rs4_env.db)
    info = {c["name"]: c for c in table_columns(rs4_env.db)}
    st = info["status"]
    assert st["notnull"] == 1 or st["dflt_value"] is not None


def test_init_db_is_idempotent_and_keeps_data(rs4_env, rs4_app):
    _init(rs4_env.db)
    _init(rs4_env.db)
    cols = [c["name"] for c in table_columns(rs4_env.db)]
    assert "status" in cols


def test_init_db_keeps_the_task_1_meta_table(rs4_env, rs4_app):
    _init(rs4_env.db)
    con = sqlite3.connect(rs4_env.db)
    try:
        row = con.execute("SELECT value FROM app_meta WHERE key='schema_version'").fetchone()
    finally:
        con.close()
    assert row is not None


def test_init_db_upgrades_an_existing_task_1_database(rs4_env, rs4_app):
    """A database created before this task (only app_meta) gains the drives table."""
    rs4_env.db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(rs4_env.db)
    con.execute("CREATE TABLE app_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    con.execute("INSERT INTO app_meta VALUES ('schema_version', '1')")
    con.commit()
    con.close()
    _init(rs4_env.db)
    assert [c["name"] for c in table_columns(rs4_env.db)]
    con = sqlite3.connect(rs4_env.db)
    try:
        assert con.execute("SELECT count(*) FROM app_meta").fetchone()[0] >= 1
    finally:
        con.close()


def test_app_startup_creates_the_drives_table(rs4_client, rs4_env):
    """With the lifespan run (client fixture) the table exists in the configured DB file."""
    assert rs4_env.db.exists(), "app startup did not create the SQLite file at ROADSCOUT_DB"
    assert [c["name"] for c in table_columns(rs4_env.db)]


def test_upload_writes_one_row_matching_the_response(rs4_client, rs4_api, rs4_clip, rs4_env):
    video = rs4_clip("mp4-h264")
    gpx = make_gpx()
    r = upload(rs4_client, rs4_api, ("clip.mp4", video, "video/mp4"),
               ("t.gpx", gpx, "application/gpx+xml"))
    assert ok_status(r.status_code), r.text
    api_drive = unwrap_drive(r.json())
    rows = db_rows(rs4_env)
    assert len(rows) == 1
    row = rows[0]
    assert str(row["id"]) == str(drive_id(api_drive))
    assert row["status"] == api_drive["status"]

    cols = list(row)
    started = col(cols, r"^started_at$|^start")
    ts = parse_ts(row[started])
    assert ts is not None and abs(secs(ts, META_TIME)) <= 1.5

    offset_col = col(cols, r"offset")
    assert row[offset_col] in (None, 0, 0.0)

    vcol = col(cols, r"video.*(path|key|uri|file)|(path|key|uri|file).*video")
    gcol = col(cols, r"gpx.*(path|key|uri|file)|(path|key|uri|file).*gpx")
    assert row[vcol] and row[gcol] and row[vcol] != row[gcol]
    vfile = resolve_stored(rs4_env, row[vcol])
    gfile = resolve_stored(rs4_env, row[gcol])
    assert vfile is not None, f"video path {row[vcol]!r} does not point at a stored file"
    assert gfile is not None, f"gpx path {row[gcol]!r} does not point at a stored file"
    assert sha(vfile.read_bytes()) == sha(video)
    assert sha(gfile.read_bytes()) == sha(gpx)


def test_offset_is_written_to_the_row(rs4_client, rs4_api, rs4_clip, rs4_env):
    r = upload(rs4_client, rs4_api, ("clip.mp4", rs4_clip("mp4-h264"), "video/mp4"),
               ("t.gpx", make_gpx(), "application/gpx+xml"), offset=-12.75)
    assert ok_status(r.status_code), r.text
    (row,) = db_rows(rs4_env)
    assert float(row[col(list(row), r"offset")]) == pytest.approx(-12.75)


def test_each_drive_gets_its_own_files_and_row(rs4_client, rs4_api, rs4_clip, rs4_env):
    vids = [rs4_clip("mp4-h264", c) for c in ("red", "green")]
    for v in vids:
        assert ok_status(upload(rs4_client, rs4_api, ("clip.mp4", v, "video/mp4"),
                                ("t.gpx", make_gpx(), "application/gpx+xml")).status_code)
    rows = db_rows(rs4_env)
    assert len(rows) == 2
    assert len({str(r["id"]) for r in rows}) == 2
    cols = list(rows[0])
    vcol = col(cols, r"video.*(path|key|uri|file)|(path|key|uri|file).*video")
    paths = {r[vcol] for r in rows}
    assert len(paths) == 2, "two drives share one stored video path"
    stored = {sha(resolve_stored(rs4_env, p).read_bytes()) for p in paths}
    assert stored == {sha(v) for v in vids}
