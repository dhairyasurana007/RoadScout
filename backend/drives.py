"""Drive upload and lookup endpoints (task 4)."""
import math
import re
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.db import get_connection, init_db
from backend.media import InvalidMedia, ProbeUnavailable, parse_gpx, probe_video
from backend.storage import Storage, get_storage

router = APIRouter(prefix="/api/drives", tags=["drives"])

STATUS_UPLOADED = "uploaded"
MAX_CLOCK_OFFSET_SECONDS = 24 * 3600

SOURCE_VIDEO_METADATA = "video_metadata"
SOURCE_GPX_START = "gpx_start"
SOURCE_UPLOAD_TIME = "upload_time"


def _parse_offset(raw: str | None) -> float:
    if raw is None or not raw.strip():
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        raise HTTPException(422, "clock_offset_seconds must be a number") from None
    if not math.isfinite(value) or abs(value) > MAX_CLOCK_OFFSET_SECONDS:
        raise HTTPException(422, "clock_offset_seconds must be within +/- 86400 seconds")
    return value


def _extension(filename: str | None) -> str:
    match = re.search(r"\.([A-Za-z0-9]{1,8})$", filename or "")
    return f".{match.group(1).lower()}" if match else ".bin"


def _row_to_drive(row) -> dict:
    drive = dict(row)
    started = datetime.fromisoformat(drive["started_at"])
    drive["effective_started_at"] = (
        started + timedelta(seconds=drive["clock_offset_seconds"])
    ).isoformat()
    drive["started_at_fell_back"] = drive["started_at_source"] != SOURCE_VIDEO_METADATA
    return drive


@router.post("", status_code=201)
def create_drive(
    video: UploadFile = File(...),
    gpx: UploadFile = File(...),
    clock_offset_seconds: str | None = Form(None),
    storage: Storage = Depends(get_storage),
) -> dict:
    """Upload a video and a GPX file. Files are streamed to storage in chunks.

    clock_offset_seconds: seconds to add to video time to get GPX time (default 0).
    Returns the drive record. Nothing is left behind if validation fails.
    """
    init_db()
    offset = _parse_offset(clock_offset_seconds)
    drive_id = uuid.uuid4().hex
    try:
        video_key = storage.save(drive_id, "video" + _extension(video.filename), video.file)
        gpx_key = storage.save(drive_id, "track.gpx", gpx.file)
        try:
            video_info = probe_video(storage.local_path(video_key))
            gpx_info = parse_gpx(storage.local_path(gpx_key))
        except InvalidMedia as exc:
            raise HTTPException(400, str(exc)) from exc
        except ProbeUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc

        now = datetime.now(timezone.utc)
        if video_info.created_at:
            started_at, source = video_info.created_at, SOURCE_VIDEO_METADATA
        elif gpx_info.first_time:
            started_at, source = gpx_info.first_time, SOURCE_GPX_START
        else:
            started_at, source = now, SOURCE_UPLOAD_TIME

        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO drives (id, started_at, started_at_source, clock_offset_seconds,"
                " video_path, gpx_path, status, duration_seconds, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    drive_id, started_at.isoformat(), source, offset, video_key, gpx_key,
                    STATUS_UPLOADED, video_info.duration_seconds, now.isoformat(),
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM drives WHERE id = ?", (drive_id,)).fetchone()
        finally:
            conn.close()
    except BaseException:
        storage.delete_drive(drive_id)
        raise
    return _row_to_drive(row)


@router.get("")
def list_drives() -> list[dict]:
    init_db()
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM drives ORDER BY created_at DESC, id").fetchall()
    finally:
        conn.close()
    return [_row_to_drive(r) for r in rows]


@router.get("/{drive_id}")
def get_drive(drive_id: str) -> dict:
    init_db()
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM drives WHERE id = ?", (drive_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(404, "Drive not found")
    return _row_to_drive(row)
