"""SQLite helpers. Later tasks add their tables in init_db()."""
import sqlite3
from pathlib import Path

from backend.config import db_path


# One row per uploaded drive. video_path and gpx_path are storage keys (see backend/storage.py).
# clock_offset_seconds is added to video time to get GPX time (applied in task 6).
DRIVES_SCHEMA = """
CREATE TABLE IF NOT EXISTS drives (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    started_at_source TEXT NOT NULL,
    clock_offset_seconds REAL NOT NULL DEFAULT 0,
    video_path TEXT NOT NULL,
    gpx_path TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_seconds REAL,
    created_at TEXT NOT NULL
)
"""


# Raw SAM 3.1 output, one row per detection (task 7, written by backend/detection.py).
# frame_index is the global frame index and frame_timestamp the global time in seconds from the
# source video start (both from frame_timestamps.json); frame_in_chunk is the position inside the
# chunk file. track_id is the id the provider gave inside that chunk and track_key is
# "<chunk>:<track_id>", unique across the drive because tracks never span chunks here.
# box is JSON [x1, y1, x2, y2] in chunk pixels; mask is the RLE dict as JSON text, or NULL.
# No foreign key to drives: the table is a raw log keyed by drive id.
DETECTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drive_id TEXT NOT NULL,
    chunk INTEGER NOT NULL,
    frame_index INTEGER NOT NULL,
    frame_in_chunk INTEGER NOT NULL,
    frame_timestamp REAL NOT NULL,
    hazard_class TEXT NOT NULL,
    class_prompt TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    track_key TEXT NOT NULL,
    box TEXT NOT NULL,
    mask TEXT,
    score REAL NOT NULL
)
"""
DETECTIONS_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_detections_drive ON detections (drive_id, chunk, frame_index)"
)


def get_connection(path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection (rows behave like dicts, foreign keys on)."""
    conn = sqlite3.connect(str(path if path is not None else db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path: Path | str | None = None) -> Path:
    """Create the database file, its parent folder and base tables. Safe to call repeatedly."""
    target = Path(path if path is not None else db_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(target)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS app_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO app_meta (key, value) VALUES ('schema_version', '1')"
        )
        conn.execute(DRIVES_SCHEMA)
        conn.execute(DETECTIONS_SCHEMA)
        conn.execute(DETECTIONS_INDEX)
        conn.commit()
    finally:
        conn.close()
    return target
