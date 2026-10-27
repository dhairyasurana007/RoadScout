"""Task 14: the hazards table."""
import pytest

from dd14_helpers import (VA, add_cand, add_drive, class_col, dedup, execute, hazards, make_db, offset, rows)


@pytest.fixture
def db(tmp_path, monkeypatch):
    return make_db(tmp_path, monkeypatch)


REQUIRED = {"id", "lat", "lon", "direction", "severity", "confidence", "sightings", "first_seen", "last_seen",
            "photo_path", "status"}


def test_hazards_table_exists_with_columns(db):
    cols = {r["name"] for r in rows(db, "PRAGMA table_info(hazards)")}
    assert cols, "hazards table does not exist after init_db"
    assert REQUIRED <= cols, "missing columns: %s" % sorted(REQUIRED - cols)
    assert "hazard_class" in cols or "class" in cols


def test_id_is_primary_key(db):
    pk = [r["name"] for r in rows(db, "PRAGMA table_info(hazards)") if r["pk"]]
    assert pk == ["id"]


def test_status_default_is_new(db):
    info = {r["name"]: r for r in rows(db, "PRAGMA table_info(hazards)")}
    dflt = info["status"]["dflt_value"]
    assert dflt is not None and "new" in str(dflt).lower(), "status column should DEFAULT 'new'"


def test_candidates_hazard_id_column_kept(db):
    cols = {r["name"] for r in rows(db, "PRAGMA table_info(candidates)")}
    assert "hazard_id" in cols


def test_init_db_is_idempotent_and_keeps_hazards(db):
    from backend.db import init_db

    add_drive(db, "A")
    add_cand(db, "A", VA[0], VA[1], 0.0)
    dedup(db, "A")
    before = hazards(db)
    assert len(before) == 1
    init_db(db)
    init_db(db)
    assert hazards(db) == before


def test_old_database_without_hazards_is_migrated(db):
    from backend.db import init_db

    execute(db, "DROP TABLE hazards")
    assert not rows(db, "SELECT name FROM sqlite_master WHERE type='table' AND name='hazards'")
    init_db(db)
    cols = {r["name"] for r in rows(db, "PRAGMA table_info(hazards)")}
    assert REQUIRED <= cols


def test_dedup_creates_table_even_on_fresh_unmigrated_db(tmp_path, monkeypatch):
    """deduplicate_drive on a database that only has the older tables must not crash."""
    import sqlite3

    from backend.db import init_db

    db = tmp_path / "old.sqlite"
    init_db(db)
    con = sqlite3.connect(str(db))
    con.execute("DROP TABLE hazards")
    con.commit()
    con.close()
    monkeypatch.setenv("ROADSCOUT_DB", str(db))
    dedup(db, "does-not-exist")
    assert class_col(db)
    assert hazards(db) == []


def test_unknown_or_empty_drive_is_a_noop(db):
    dedup(db, "no-such-drive")
    assert hazards(db) == []


def test_new_hazard_status_is_new_and_fields_not_null(db):
    add_drive(db, "A")
    add_cand(db, "A", VA[0], VA[1], 10.0)
    dedup(db, "A")
    (h,) = hazards(db)
    assert h["status"] == "new"
    for k in ("id", "lat", "lon", "direction", "severity", "confidence", "sightings", "first_seen", "last_seen",
              "photo_path", "cls"):
        assert h[k] is not None, k
