"""Task 8: the candidates table."""
from __future__ import annotations

import sqlite3

import pytest

from rs8_helpers import make_spec, rs8_cache, rs8_env  # noqa: F401

REQUIRED_ANY_OF = {
    "id": ["id"],
    "drive_id": ["drive_id"],
    "hazard_id": ["hazard_id"],
    "class": ["hazard_class", "class", "class_name", "label"],
    "track": ["track_key", "track_id"],
    "first timestamp": ["first_timestamp", "first_ts", "first_seen", "first_time", "start_timestamp"],
    "last timestamp": ["last_timestamp", "last_ts", "last_seen", "last_time", "end_timestamp"],
    "frames seen": ["frames_seen", "frame_count", "n_frames", "num_frames"],
    "best-frame path": ["best_frame_path", "best_frame", "best_frame_image", "best_image_path"],
    "crop path": ["crop_path", "crop", "best_crop_path"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng"],
    "heading": ["heading", "heading_deg"],
}


def info(db):
    conn = sqlite3.connect(str(db))
    try:
        return {r[1]: r for r in conn.execute("PRAGMA table_info(candidates)")}
    finally:
        conn.close()


def test_init_db_creates_candidates_table_with_plan_columns(rs8_env):
    cols = info(rs8_env.db)
    assert cols, "init_db() must create the candidates table"
    for label, names in REQUIRED_ANY_OF.items():
        assert any(n in cols for n in names), "candidates has no %s column (one of %s); columns: %s" % (
            label, names, list(cols))


def test_hazard_id_is_nullable_and_empty_until_dedup(rs8_env):
    e = rs8_env
    cols = info(e.db)
    assert cols["hazard_id"][3] == 0, "hazard_id must allow NULL"
    e.use_spec(make_spec(n_chunks=1))
    e.add_track(0, range(5), 1, (60, 10, 100, 40))
    e.run()
    assert [r["hazard_id"] for r in e.raw()] == [None]


def test_init_db_is_repeatable_and_keeps_candidates(rs8_env):
    from backend.db import init_db

    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    e.add_track(0, range(5), 1, (60, 10, 100, 40))
    e.run()
    init_db()
    init_db()
    assert len(e.raw()) == 1


def test_candidate_ids_are_unique(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=2))
    e.add_track(0, range(5), 1, (60, 10, 100, 40))
    e.add_track(0, range(5), 2, (10, 10, 30, 40))
    e.add_track(1, range(5), 1, (60, 10, 100, 40))
    e.run()
    ids = [r["id"] for r in e.raw()]
    assert len(ids) == 3 and len(set(ids)) == 3 and None not in ids
