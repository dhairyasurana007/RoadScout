"""Read endpoints for the results page (task 11).

    GET /api/drives/{id}/hazards                          the drive's accepted (verified) hazards
    GET /api/drives/{id}/hazards/{cid}/photo              best-frame JPEG (add ?variant=crop for the crop)

The photos are NOT blurred until task 15, so these routes are for the local machine only: nothing
here makes them public, and the hosted demo (task 23) must not mount this router.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Any, Literal

from backend.db import get_connection, init_db
from backend.storage import Storage, get_storage
from backend.verification import load_accepted_candidates

router = APIRouter(prefix="/api/drives", tags=["results"])

CANDIDATES_DIR = "candidates"  # <storage>/<drive id>/candidates/, where task 8 writes the JPEGs


def _require_drive(drive_id: str) -> None:
    init_db()
    conn = get_connection()
    try:
        row = conn.execute("SELECT 1 FROM drives WHERE id = ?", (drive_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(404, "Drive not found")


def _photo_url(drive_id: str, candidate_id: int, variant: str | None = None) -> str:
    url = f"/api/drives/{drive_id}/hazards/{candidate_id}/photo"
    return url + f"?variant={variant}" if variant else url


def _hazard_json(drive_id: str, cand: dict[str, Any]) -> dict[str, Any]:
    verdict = cand["verdict"]
    return {
        "id": cand["id"],
        "candidate_id": cand["id"],
        "drive_id": drive_id,
        "hazard_class": verdict["hazard_class"],
        "detector_class": cand["hazard_class"],
        "severity": verdict["severity"],
        "confidence": verdict["confidence"],
        "lane": verdict["lane"],
        "reason": verdict["reason"],
        "report_description": verdict["report_description"],
        "lat": cand["lat"],
        "lon": cand["lon"],
        "heading": cand["heading"],
        "geotag_status": cand["geotag_status"],
        "best_timestamp": cand["best_timestamp"],
        "first_timestamp": cand["first_timestamp"],
        "last_timestamp": cand["last_timestamp"],
        "frames_seen": cand["frames_seen"],
        "photo_url": _photo_url(drive_id, cand["id"]),
        "crop_url": _photo_url(drive_id, cand["id"], "crop") if cand.get("crop_path") else None,
    }


@router.get("/{drive_id}/hazards")
def list_hazards(drive_id: str) -> list[dict[str, Any]]:
    """The drive's accepted hazards (candidates whose Muse Spark verdict was accepted), in
    drive-time order. An empty list when there are none. 404 for an unknown drive."""
    _require_drive(drive_id)
    return [
        _hazard_json(drive_id, c) for c in load_accepted_candidates(drive_id) if c["verdict"]["is_hazard"]
    ]


@router.get("/{drive_id}/hazards/{candidate_id}/photo")
def get_candidate_photo(
    drive_id: str,
    candidate_id: int,
    variant: Literal["best", "crop"] = Query("best"),
    storage: Storage = Depends(get_storage),
) -> FileResponse:
    """The best-frame JPEG of an accepted hazard (``variant=crop`` for the crop). 404 for an unknown
    drive, an unknown candidate, a candidate of another drive, a candidate whose verdict is rejected
    or missing, a candidate without that image, or a file that is missing or lies outside the
    drive's candidates folder. A non-integer id is a 422."""
    _require_drive(drive_id)
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT c.best_frame_path, c.crop_path FROM candidates c JOIN verdicts v ON v.candidate_id = c.id"
            " WHERE c.id = ? AND c.drive_id = ? AND v.accepted = 1 AND v.is_hazard = 1",
            (candidate_id, drive_id),
        ).fetchone()
    finally:
        conn.close()
    stored = None if row is None else row["crop_path" if variant == "crop" else "best_frame_path"]
    if not stored:
        raise HTTPException(404, "Photo not found")
    try:
        folder = storage.local_path(f"{drive_id}/{CANDIDATES_DIR}").resolve()
        path = Path(stored).resolve()
    except (ValueError, OSError):
        raise HTTPException(404, "Photo not found") from None
    if not path.is_relative_to(folder) or not path.is_file():
        raise HTTPException(404, "Photo not found")
    # no-store: the file is replaced when a drive is processed again, and it is unblurred.
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
