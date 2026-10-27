"""Read endpoints for the hazard map (task 16): the deduplicated hazards across all drives.

    GET /api/hazards                       the hazards table as JSON (``?status=``, ``?drive_id=`` filters)
    GET /api/hazards/{id}/photo            the hazard's BLURRED photo (``?variant=crop`` for the crop)
    PATCH /api/hazards/{id}                review: set status / class / severity (task 17, see below)

List: a bare JSON list ordered by ``id`` ascending, ``[]`` when empty, every status included unless
``?status=`` (one of ``new, approved, rejected, reported, possibly_fixed``; anything else is a 422) asks
for one. ``?drive_id=`` keeps the hazards that have a member candidate (``candidates.hazard_id``) in that
drive; an unknown drive gives ``[]``. A hazard without a finite ``lat`` and ``lon`` is never listed (a pin
needs a position; it is never returned as 0,0). ``photo_url`` / ``crop_url`` are ``null`` when the
hazard has no such image or its file is not servable, so the list never points at a 404 and never fails
because a file is missing. No filesystem path is returned. The list and photo routes only read.
Besides the keys of task 16 every item has ``reviewed_at`` (UTC ISO text or null), ``class_edited`` and
``severity_edited`` (booleans) and ``original_hazard_class`` / ``original_severity`` (the automatic value
while the field is edited, else null).

PATCH /api/hazards/{id}, the review endpoint. JSON object body, every field optional but at least one is
required: ``status`` ("new", "approved" or "rejected"; "reported" and "possibly_fixed" are set by later
tasks and are a 422 here), ``hazard_class`` (one of the classes of config/hazard_prompts.json, exact
spelling) and ``severity`` (an integer 1 to 3; true, 2.0 and "2" are refused). Unknown fields, null values,
an empty object, an empty or non-object body: 422 (``detail`` is a message or FastAPI's list of errors).
Unknown hazard: 404 ``{"detail": "Hazard not found"}``. Validation happens before the lookup. Success: 200
with the updated hazard in the shape of a list item. The change is one transaction; ``updated_at`` moves,
``reviewed_at`` is set for approved / rejected and cleared for new, and an edited class or severity sets
``class_edited`` / ``severity_edited`` (back to the automatic value clears it). Re-running deduplication
keeps edited values. Any current status can be reviewed. Nothing is ever submitted anywhere.

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
from pydantic import BaseModel, ConfigDict, StrictInt

from backend.db import HAZARD_STATUSES, get_connection, init_db
from backend.dedup import load_hazards, review_hazard
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
        "reviewed_at": row.get("reviewed_at"),
        "class_edited": bool(row.get("class_edited")),
        "severity_edited": bool(row.get("severity_edited")),
        "original_hazard_class": row.get("original_hazard_class"),
        "original_severity": row.get("original_severity"),
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


class ReviewBody(BaseModel):
    """Body of PATCH /api/hazards/{id}. Only the types are checked here; the values are checked by
    ``review_hazard``. Fields that are absent are left alone (``model_fields_set``)."""

    model_config = ConfigDict(extra="forbid")

    status: str = None  # type: ignore[assignment]
    hazard_class: str = None  # type: ignore[assignment]
    severity: StrictInt = None  # type: ignore[assignment]


@router.patch("/{hazard_id}")
def review(
    body: ReviewBody,
    hazard_id: int = PathParam(..., le=MAX_ID),
    storage: Storage = Depends(get_storage),
) -> dict[str, Any]:
    """Approve, reject or reset a hazard and/or edit its class or severity. See the module docstring."""
    changes = {name: getattr(body, name) for name in body.model_fields_set}
    try:
        row = review_hazard(hazard_id, **changes)
    except LookupError:
        raise HTTPException(404, "Hazard not found") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    return _hazard_json(row, _storage_root(storage), _original_paths())
