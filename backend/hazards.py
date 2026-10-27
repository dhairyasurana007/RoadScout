"""Read endpoints for the hazard map (task 16): the deduplicated hazards across all drives.

    GET /api/hazards                       the hazards table as JSON (``?status=``, ``?drive_id=`` filters)
    GET /api/hazards/{id}/photo            the hazard's BLURRED photo (``?variant=crop`` for the crop)

List: a bare JSON list ordered by ``id`` ascending, ``[]`` when empty, every status included unless
``?status=`` (one of ``new, approved, rejected, reported, possibly_fixed``; anything else is a 422) asks
for one. ``?drive_id=`` keeps the hazards that have a member candidate (``candidates.hazard_id``) in that
drive; an unknown drive gives ``[]``. A hazard without a finite ``lat`` and ``lon`` is never listed (a pin
needs a position; it is never returned as 0,0). ``photo_url`` / ``crop_url`` are ``null`` when the
hazard has no such image or its file is not servable, so the list never points at a 404 and never fails
because a file is missing. No filesystem path is returned. GET only; nothing is written.

Photo route, FAIL CLOSED. It serves ``hazards.photo_path`` / ``crop_path`` (the BLURRED paths task 14
stores) only when ALL hold, else 404: the hazard exists, the column is not NULL, the path (after resolving
symlinks and ``..``) is a file directly in ``<storage>/<any drive id>/candidates/``, and it is NOT the
resolved ``best_frame_path`` or ``crop_path`` of ANY candidate of ANY drive (the unblurred originals).
Judging by the candidates table instead of a file-name pattern means no spelling of an original's path
gets through. The route does not look at the hazard's drive. The file is read on every request,
``Cache-Control: no-store``, ``image/jpeg``. A non-integer id or an unknown ``variant`` is a 422.
"""
import math
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path as PathParam, Query
from fastapi.responses import FileResponse

from backend.db import HAZARD_STATUSES, get_connection, init_db
from backend.dedup import load_hazards
from backend.storage import Storage, get_storage

router = APIRouter(prefix="/api/hazards", tags=["hazards"])

CANDIDATES_DIR = "candidates"  # <storage>/<drive id>/candidates/
MAX_ID = 2**63 - 1  # SQLite integers are 64-bit; larger ids are a 422, not a server error

Status = Literal["new", "approved", "rejected", "reported", "possibly_fixed"]
assert Status.__args__ == HAZARD_STATUSES  # keep the filter in step with the table's CHECK constraint


def _photo_url(hazard_id: int, variant: str | None = None) -> str:
    url = f"/api/hazards/{hazard_id}/photo"
    return url + f"?variant={variant}" if variant else url


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _resolve(text: Any) -> Path | None:
    try:
        return Path(text).resolve() if text else None
    except (ValueError, OSError, TypeError):
        return None


def _original_paths() -> set[Path]:
    """The resolved unblurred best-frame and crop paths of every candidate of every drive."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT best_frame_path, crop_path FROM candidates").fetchall()
    finally:
        conn.close()
    found = set()
    for row in rows:
        for column in ("best_frame_path", "crop_path"):
            path = _resolve(row[column])
            if path is not None:
                found.add(path)
    return found


def _servable(stored: Any, root: Path, originals: set[Path]) -> Path | None:
    """The file to serve for a stored path, or None (see the module docstring)."""
    path = _resolve(stored)
    if path is None:
        return None
    try:
        relative = path.relative_to(root)
        if len(relative.parts) != 3 or relative.parts[1] != CANDIDATES_DIR:
            return None
        if path in originals or not path.is_file():
            return None
    except (ValueError, OSError):
        return None
    return path


def _storage_root(storage: Storage) -> Path:
    return storage.local_path("").resolve()


def _hazard_json(row: dict[str, Any], root: Path, originals: set[Path]) -> dict[str, Any]:
    has_photo = _servable(row["photo_path"], root, originals) is not None
    has_crop = _servable(row["crop_path"], root, originals) is not None
    return {
        "id": row["id"],
        "hazard_class": row["hazard_class"],
        "lat": row["lat"],
        "lon": row["lon"],
        "direction": row["direction"],
        "severity": row["severity"],
        "confidence": row["confidence"],
        "sightings": row["sightings"],
        "candidate_count": row["candidate_count"],
        "first_seen": row["first_seen"],
        "last_seen": row["last_seen"],
        "status": row["status"],
        "photo_url": _photo_url(row["id"]) if has_photo else None,
        "crop_url": _photo_url(row["id"], "crop") if has_crop else None,
    }


@router.get("")
def list_hazards(
    status: Status | None = Query(None),
    drive_id: str | None = Query(None),
    storage: Storage = Depends(get_storage),
) -> list[dict[str, Any]]:
    """The deduplicated hazards that have a position, ordered by id."""
    rows = [h for h in load_hazards(status=status) if _finite(h["lat"]) and _finite(h["lon"])]
    if drive_id is not None:
        conn = get_connection()
        try:
            wanted = {
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT hazard_id FROM candidates WHERE drive_id = ? AND hazard_id IS NOT NULL",
                    (drive_id,),
                )
            }
        finally:
            conn.close()
        rows = [h for h in rows if h["id"] in wanted]
    root = _storage_root(storage)
    originals = _original_paths()
    return [_hazard_json(h, root, originals) for h in rows]


@router.get("/{hazard_id}/photo")
def get_hazard_photo(
    hazard_id: int = PathParam(..., le=MAX_ID),
    variant: Literal["best", "crop"] = Query("best"),
    storage: Storage = Depends(get_storage),
) -> FileResponse:
    """The BLURRED photo of a hazard, or 404 (see the module docstring for every 404 case)."""
    init_db()
    conn = get_connection()
    try:
        row = conn.execute("SELECT photo_path, crop_path FROM hazards WHERE id = ?", (hazard_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(404, "Photo not found")
    path = _servable(row["crop_path" if variant == "crop" else "photo_path"], _storage_root(storage), _original_paths())
    if path is None:
        raise HTTPException(404, "Photo not found")
    # no-store: the file is replaced when a drive is processed again.
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
