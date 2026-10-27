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
#                 verifying, deduplicating, done or failed; NULL until a run is requested
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


# Column added to drives by the Delete video button (task 15), same idempotent migration.
#   video_deleted_at  UTC ISO time the raw video, work folder and unblurred frames were deleted on
#                     request; NULL while they exist. video_path keeps the stored key either way.
DRIVES_VIDEO_COLUMNS = {
    "video_deleted_at": "TEXT",
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
# key: the link is recomputed by deduplication and may briefly point at a deleted hazard, which is
# then treated as NULL). Timestamps are seconds from the source video start;
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

# Columns added to candidates by the privacy blur (task 15, written by backend/privacy.py), same
# idempotent migration. Only the blurred images are ever served or used for hazards:
#   blurred_frame_path  JPEG of the best frame with faces and license plates blurred (NULL unless ok)
#   blurred_crop_path   the same for the crop (NULL unless ok, or when the candidate has no crop)
#   blur_status         NULL (not blurred yet / not a verified hazard), 'ok' or 'failed' (fail closed:
#                       no blurred image exists and nothing is served for the candidate)
#   blur_note           NULL, or one line why the blur failed
CANDIDATES_BLUR_COLUMNS = {
    "blurred_frame_path": "TEXT",
    "blurred_crop_path": "TEXT",
    "blur_status": "TEXT",
    "blur_note": "TEXT",
}
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


# One row per real-world hazard (task 14, written by backend/dedup.py; status is also set by the
# review UI). Candidates link to it through candidates.hazard_id (no foreign key: the link is
# recomputed by deduplication and must never block deleting candidates). All of the columns below
# except id, status and created_at are derived from the member candidates and rewritten by every
# deduplication run; id and status survive. The rules are in backend/dedup.py.
#   direction     degrees, 0 = north, clockwise, [0, 360): circular mean of the members' headings
#   severity      1 to 3, the largest severity among the members
#   confidence    0 to 1, noisy-OR of the best confidence per drive
#   sightings     number of distinct drives that saw it; candidate_count is the number of members
#   first_seen    UTC ISO time of the earliest member (drive start + best-frame time)
#   last_seen     UTC ISO time of the latest member
#   photo_path    BLURRED best-frame JPEG of the best-scoring member (crop_path: its blurred crop, may be
#                 NULL); candidates whose blur failed do not take part (task 15)
#   status        new, approved, rejected, reported or possibly_fixed (enforced by a CHECK)
HAZARD_STATUSES = ("new", "approved", "rejected", "reported", "possibly_fixed")
HAZARDS_SCHEMA = """
CREATE TABLE IF NOT EXISTS hazards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hazard_class TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    direction REAL,
    severity INTEGER NOT NULL,
    confidence REAL NOT NULL,
    sightings INTEGER NOT NULL,
    candidate_count INTEGER NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    photo_path TEXT,
    crop_path TEXT,
    status TEXT NOT NULL DEFAULT 'new'
        CHECK (status IN ('new', 'approved', 'rejected', 'reported', 'possibly_fixed')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""
# Columns added to hazards by the review actions (task 17, written by backend/dedup.py review_hazard),
# same idempotent migration (existing rows get the defaults):
#   reviewed_at            UTC ISO time status was last set to approved or rejected; NULL while the status
#                          is anything else (set back to new clears it)
#   class_edited           1 while the user's class differs from the automatic one, else 0
#   severity_edited        1 while the user's severity differs from the automatic one, else 0
#   original_hazard_class  the automatic class while class_edited = 1, else NULL (for the eval)
#   original_severity      the automatic severity while severity_edited = 1, else NULL
# Deduplication never overwrites an edited class or severity (it keeps the original_* values current).
HAZARDS_REVIEW_COLUMNS = {
    "reviewed_at": "TEXT",
    "class_edited": "INTEGER NOT NULL DEFAULT 0",
    "severity_edited": "INTEGER NOT NULL DEFAULT 0",
    "original_hazard_class": "TEXT",
    "original_severity": "INTEGER",
}
HAZARDS_INDEX = "CREATE INDEX IF NOT EXISTS idx_hazards_class ON hazards (hazard_class, lat, lon)"
CANDIDATES_HAZARD_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_candidates_hazard ON candidates (hazard_id)"
)


def get_connection(path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection (rows behave like dicts, foreign keys on)."""
    # A generous busy timeout: concurrent writers (review clicks, a pipeline run) wait for the lock.
    conn = sqlite3.connect(str(path if path is not None else db_path()), timeout=30)
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
        _add_missing_columns(conn, "drives", DRIVES_VIDEO_COLUMNS)
        conn.execute(DETECTIONS_SCHEMA)
        conn.execute(DETECTIONS_INDEX)
        conn.execute(CANDIDATES_SCHEMA)
        _add_missing_columns(conn, "candidates", CANDIDATES_BLUR_COLUMNS)
        conn.execute(CANDIDATES_INDEX)
        conn.execute(VERDICTS_SCHEMA)
        conn.execute(VERDICTS_INDEX)
        conn.execute(HAZARDS_SCHEMA)
        _add_missing_columns(conn, "hazards", HAZARDS_REVIEW_COLUMNS)
        conn.execute(HAZARDS_INDEX)
        conn.execute(CANDIDATES_HAZARD_INDEX)
        conn.commit()
    finally:
        conn.close()
    return target
