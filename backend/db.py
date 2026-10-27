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
    created_at TEXT NOT NULL,
    stage TEXT,
    error TEXT,
    failed_stage TEXT,
    updated_at TEXT,
    finished_at TEXT
)
"""

# Columns added to drives by the pipeline runner (task 10). init_db adds any that an older
# database lacks (ALTER TABLE ... ADD COLUMN), so existing databases keep working.
#   stage         current pipeline stage: queued, preprocessing, detecting, aggregating,
#                 verifying, done or failed; NULL until a run is requested
#   error         "<stage>: <message>" of the last failed run; NULL on success and while running
#   failed_stage  the stage that raised, set only when stage is 'failed'
#   updated_at    UTC ISO time of the last status or stage change
#   finished_at   UTC ISO time the last run ended (done or failed); NULL while running
DRIVES_PIPELINE_COLUMNS = {
    "stage": "TEXT",
    "error": "TEXT",
    "failed_stage": "TEXT",
    "updated_at": "TEXT",
    "finished_at": "TEXT",
}

# Columns added to drives by the stationary-frame filter (task 13). Integers default to 0 (also for
# drives created before the migration) and are reset to 0 when a run starts, then written when the
# detection stage finishes. Same idempotent migration as above.
#   frames_total                sampled frames in all chunks of the run
#   stationary_frames_skipped   frames whose GPS speed was below the threshold (0 when the filter is
#                               off, the GPS is unusable or nothing was stationary)
#   stationary_chunks_skipped   chunks never sent to the provider because every frame was stationary
#   chunks_sent                 chunks sent to the provider (skipped + sent = all chunks)
#   stationary_note             NULL, or one line why the filter was off or the GPS not (fully) usable
DRIVES_STATIONARY_COLUMNS = {
    "frames_total": "INTEGER DEFAULT 0",
    "stationary_frames_skipped": "INTEGER DEFAULT 0",
    "stationary_chunks_skipped": "INTEGER DEFAULT 0",
    "chunks_sent": "INTEGER DEFAULT 0",
    "stationary_note": "TEXT",
}


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


# One row per tracked object worth looking at (task 8, written by backend/candidates.py).
# Tracks are grouped per (hazard_class, track_key), so a physical object that the provider loses
# at a chunk boundary becomes two candidates (stitching is left to deduplication, task 14).
# hazard_id stays NULL until deduplication (task 14) links the candidate to a hazard (no foreign
# key yet: the hazards table does not exist). Timestamps are seconds from the source video start;
# best_frame_index is the global frame index. best_frame_path and crop_path are absolute JPEG
# paths inside the git-ignored data dir. lat/lon/heading are NULL unless geotag_status is 'ok';
# 'out_of_range' means the best frame's time falls outside the GPX track, 'no_gps' that the drive
# has no usable GPX. geotag_note says why. max_mask_area is in pixels (box area when no mask).
CANDIDATES_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drive_id TEXT NOT NULL,
    hazard_id INTEGER,
    hazard_class TEXT NOT NULL,
    track_key TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    chunk INTEGER NOT NULL,
    first_timestamp REAL NOT NULL,
    last_timestamp REAL NOT NULL,
    frames_seen INTEGER NOT NULL,
    max_mask_area REAL NOT NULL,
    best_frame_index INTEGER NOT NULL,
    best_timestamp REAL NOT NULL,
    best_box TEXT NOT NULL,
    best_frame_path TEXT NOT NULL,
    crop_path TEXT,
    lat REAL,
    lon REAL,
    heading REAL,
    score REAL NOT NULL,
    geotag_status TEXT NOT NULL,
    geotag_note TEXT
)
"""
CANDIDATES_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_candidates_drive ON candidates (drive_id, first_timestamp)"
)


# One Muse Spark verdict per candidate (task 9, written by backend/verification.py). The row is
# deleted with its candidate (re-running aggregation drops the drive's old verdicts). hazard_class
# is the class the model gave (a known class when is_hazard is 1; anything for a non-hazard);
# severity is 1 to 3 for a hazard (0 allowed only when not a hazard). model and prompt_version
# say which model and which verification prompt produced the verdict. accepted is 1 when
# is_hazard and confidence >= the threshold used; rejected verdicts are kept for the eval ablation.
VERDICTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS verdicts (
    candidate_id INTEGER PRIMARY KEY REFERENCES candidates(id) ON DELETE CASCADE,
    drive_id TEXT NOT NULL,
    is_hazard INTEGER NOT NULL,
    hazard_class TEXT NOT NULL,
    severity INTEGER NOT NULL,
    lane TEXT,
    confidence REAL NOT NULL,
    reason TEXT NOT NULL,
    report_description TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    accepted INTEGER NOT NULL,
    created_at TEXT NOT NULL
)
"""
VERDICTS_INDEX = "CREATE INDEX IF NOT EXISTS idx_verdicts_drive ON verdicts (drive_id)"


def get_connection(path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection (rows behave like dicts, foreign keys on)."""
    conn = sqlite3.connect(str(path if path is not None else db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _add_missing_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """Idempotent migration: add each column the table does not have yet."""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, decl in columns.items():
        if name in existing:
            continue
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        except sqlite3.OperationalError as exc:
            # Another connection added it between our check and the ALTER.
            if "duplicate column" not in str(exc).lower():
                raise


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
        _add_missing_columns(conn, "drives", DRIVES_PIPELINE_COLUMNS)
        _add_missing_columns(conn, "drives", DRIVES_STATIONARY_COLUMNS)
        conn.execute(DETECTIONS_SCHEMA)
        conn.execute(DETECTIONS_INDEX)
        conn.execute(CANDIDATES_SCHEMA)
        conn.execute(CANDIDATES_INDEX)
        conn.execute(VERDICTS_SCHEMA)
        conn.execute(VERDICTS_INDEX)
        conn.commit()
    finally:
        conn.close()
    return target
