"""Checkpoints of the pipeline (task 26c): what lets an interrupted drive resume where it stopped.

Tables (``backend/db.py``): ``pipeline_runs`` holds one row per drive, the latest run: a fingerprint
of the settings that decide whether earlier work can be reused, ``complete`` (1 once the run ended
``done``) and ``stages``, a JSON map of the stages that completed with a small info dict each.
``detection_chunks`` marks chunks whose detections are committed (see ``backend/detection.py``).

Resume rule (``plan_resume``): a run resumes only if the drive has a ``pipeline_runs`` row that is NOT
complete (the earlier run failed or was interrupted), the fingerprint still matches and there is
something to reuse (a completed stage or a committed chunk). Otherwise the run starts clean. The
fingerprint covers the provider name, the hash of the hazard prompts file, the preprocessing fps,
chunk seconds and max height, the stationary filter switch and speed threshold, the drive's clock
offset and the identity (size and modification time, never the path) of the raw video.

Helpers only touch the database; they never call a provider and never see a key or an image.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .db import get_connection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prompts_hash() -> str:
    """Hash of the hazard prompts file content ("unreadable" when it cannot be read)."""
    try:
        return hashlib.sha256(config.hazard_prompts_path().read_bytes()).hexdigest()
    except OSError:
        return "unreadable"


def video_identity(path: str | Path | None) -> list[int] | None:
    """[size, modification time in ns] of the raw video, or None when it is missing."""
    try:
        st = Path(path).stat() if path is not None else None
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns] if st is not None else None


def compute_fingerprint(
    *,
    provider_name: str,
    fps: float | None,
    chunk_seconds: float | None,
    max_height: int | None,
    clock_offset_seconds: float,
    video: str | Path | None,
    skip_stationary: bool | None,
    stationary_speed_mps: float | None,
) -> str:
    """Hash of everything that decides whether earlier work can be reused (see the module docstring)."""
    parts: dict[str, Any] = {
        "provider": str(provider_name),
        "prompts": prompts_hash(),
        "fps": float(config.preprocess_fps() if fps is None else fps),
        "chunk_seconds": float(config.chunk_seconds() if chunk_seconds is None else chunk_seconds),
        "max_height": int(config.preprocess_max_height() if max_height is None else max_height),
        "clock_offset": float(clock_offset_seconds or 0.0),
        "video": video_identity(video),
        "skip_stationary": bool(config.skip_stationary_enabled() if skip_stationary is None else skip_stationary),
        "stationary_speed": float(
            config.stationary_speed_mps() if stationary_speed_mps is None else stationary_speed_mps
        ),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()


def detections_signature(drive_id: str, db_path: Path) -> str:
    """Hash of the drive's stored detections (chunk, frame, track, class, box, score; not the masks)."""
    h = hashlib.sha256()
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT chunk, frame_index, track_key, hazard_class, box, score FROM detections "
            "WHERE drive_id = ? ORDER BY chunk, frame_index, id", (drive_id,))
        for row in cur:
            h.update(repr(tuple(row)).encode("utf-8"))
    finally:
        conn.close()
    return h.hexdigest()


def load_run(db_path: Path, drive_id: str) -> dict[str, Any] | None:
    """The drive's pipeline_runs row as ``{"fingerprint", "complete", "stages"}``, or None."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT fingerprint, complete, stages FROM pipeline_runs WHERE drive_id = ?", (drive_id,)
        ).fetchone()
        chunks = conn.execute(
            "SELECT COUNT(*) FROM detection_chunks WHERE drive_id = ?", (drive_id,)).fetchone()[0]
    finally:
        conn.close()
    if row is None:
        return None
    try:
        stages = json.loads(row["stages"] or "{}")
    except ValueError:
        stages = {}
    return {"fingerprint": row["fingerprint"], "complete": bool(row["complete"]),
            "stages": stages if isinstance(stages, dict) else {}, "chunks_done": int(chunks)}


def plan_resume(db_path: Path, drive_id: str, fingerprint: str) -> dict[str, Any] | None:
    """The earlier run's record when this run may resume it, else None (start clean)."""
    run = load_run(db_path, drive_id)
    if run is None or run["complete"] or run["fingerprint"] != fingerprint:
        return None
    if not run["stages"] and not run["chunks_done"]:
        return None  # nothing was finished, so there is nothing to reuse
    return run


def start_run(db_path: Path, drive_id: str, fingerprint: str) -> None:
    """Begin a clean run: replace the drive's run record (no stages complete)."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO pipeline_runs (drive_id, fingerprint, complete, stages, updated_at) "
                "VALUES (?, ?, 0, '{}', ?)", (drive_id, fingerprint, _now()))
    finally:
        conn.close()


def mark_stage(db_path: Path, drive_id: str, stage: str, info: dict[str, Any] | None = None) -> None:
    """Record that ``stage`` completed (with an info dict)."""
    conn = get_connection(db_path)
    try:
        with conn:
            row = conn.execute("SELECT stages FROM pipeline_runs WHERE drive_id = ?", (drive_id,)).fetchone()
            if row is None:
                return
            try:
                stages = json.loads(row["stages"] or "{}")
            except ValueError:
                stages = {}
            stages[stage] = info or {}
            conn.execute("UPDATE pipeline_runs SET stages = ?, updated_at = ? WHERE drive_id = ?",
                         (json.dumps(stages), _now(), drive_id))
    finally:
        conn.close()


def mark_complete(db_path: Path, drive_id: str) -> None:
    """The run ended ``done``: the next run of this drive starts clean."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("UPDATE pipeline_runs SET complete = 1, updated_at = ? WHERE drive_id = ?",
                         (_now(), drive_id))
    finally:
        conn.close()


def delete_checkpoints(conn, drive_id: str) -> None:
    """Remove the drive's checkpoints (call inside the caller's transaction)."""
    conn.execute("DELETE FROM pipeline_runs WHERE drive_id = ?", (drive_id,))
    conn.execute("DELETE FROM detection_chunks WHERE drive_id = ?", (drive_id,))
