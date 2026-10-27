"""Read endpoints for the hazard map (task 16): the deduplicated hazards across all drives.

    GET /api/hazards                       the hazards table as JSON (``?status=``, ``?drive_id=`` filters)
    GET /api/hazards/{id}/photo            the hazard's BLURRED photo (``?variant=crop`` for the crop)
    PATCH /api/hazards/{id}                review: set status / class / severity / agency (tasks 17 and 18, see below)
    GET /api/hazards/{id}/report           the prefilled agency report of an approved hazard (task 18, see below)

List: a bare JSON list ordered by ``id`` ascending, ``[]`` when empty, every status included unless
``?status=`` (one of ``new, approved, rejected, reported, possibly_fixed``; anything else is a 422) asks
for one. ``?drive_id=`` keeps the hazards that have a member candidate (``candidates.hazard_id``) in that
drive; an unknown drive gives ``[]``. A hazard without a finite ``lat`` and ``lon`` is never listed (a pin
needs a position; it is never returned as 0,0). ``photo_url`` / ``crop_url`` are ``null`` when the
hazard has no such image or its file is not servable, so the list never points at a 404 and never fails
because a file is missing. No filesystem path is returned. The list and photo routes only read.
Besides the keys of task 16 every item has ``agency`` (``"VDOT"``, ``"MDOT SHA"`` or ``null`` = no agency, the one
in use), ``agency_source`` (``"auto"`` or ``"user"``) and ``automatic_agency`` (task 18), ``reviewed_at`` (UTC ISO text or null), ``class_edited`` and
``severity_edited`` (booleans) and ``original_hazard_class`` / ``original_severity`` (the automatic value
while the field is edited, else null). Task 19 adds ``misses`` (integer: the consecutive later drives that passed
the hazard without seeing it), ``possibly_fixed_at`` (UTC ISO text, set while the status is possibly_fixed) and
``previous_status`` (the status to come back to when it is seen again; null unless flagged). ``possibly_fixed`` is
set only by the system (``backend/possibly_fixed.py``); a PATCH that changes the status of a possibly fixed hazard
dismisses the flag.

PATCH /api/hazards/{id}, the review endpoint. JSON object body, every field optional but at least one is
required: ``status`` ("new", "approved", "rejected" or "reported"; "reported" means the user says they submitted
the report themselves, is only allowed from approved or reported hazards (else 409) and leaves ``reviewed_at``
alone; "possibly_fixed" is set by a later task and is a 422 here), ``agency`` ("VDOT", "MDOT SHA", "none" = no
agency, or null / "auto" = the automatic agency; anything else is a 422; stored in ``hazards.agency_override``),
``hazard_class`` (one of the classes of config/hazard_prompts.json, exact
spelling) and ``severity`` (an integer 1 to 3; true, 2.0 and "2" are refused). Unknown fields, null values,
an empty object (only ``agency`` may be null), an empty or non-object body: 422 (``detail`` is a message or FastAPI's list of errors).
Unknown hazard: 404 ``{"detail": "Hazard not found"}``. Validation happens before the lookup. Success: 200
with the updated hazard in the shape of a list item. The change is one transaction; ``updated_at`` moves,
``reviewed_at`` is set for approved / rejected and cleared for new, and an edited class or severity sets
``class_edited`` / ``severity_edited`` (back to the automatic value clears it). Re-running deduplication
keeps edited values. Any current status can be reviewed. Nothing is ever submitted anywhere.

Report route (task 18). ``GET /api/hazards/{id}/report`` returns the prefilled report of an APPROVED or REPORTED
hazard as JSON (see ``backend/report.py`` for the keys); a new, rejected or possibly fixed hazard is a 409 with a
message, an unknown one a 404. ``?format=txt`` (or ``/report.txt``) returns the same report as ``text/plain`` with
``Content-Disposition: attachment; filename=hazard-<id>-report.txt``; an unknown ``format`` is a 422. Nothing is ever
sent to an agency: no HTTP client is used anywhere in this feature.

Photo route, FAIL CLOSED. It serves ``hazards.photo_path`` / ``crop_path`` (the BLURRED paths task 14
stores) only when ALL hold, else 404: the hazard exists, the column is not NULL, the path (after resolving
symlinks and ``..``) is a file directly in ``<storage>/<any drive id>/candidates/``, and it is NOT the
resolved ``best_frame_path`` or ``crop_path`` of ANY candidate of ANY drive (the unblurred originals).
Judging by the candidates table instead of a file-name pattern means no spelling of an original's path
gets through. The route does not look at the hazard's drive. The file is read on every request,
``Cache-Control: no-store``, ``image/jpeg``. A non-integer id or an unknown ``variant`` is a 422. ``?download=1`` adds
``Content-Disposition: attachment; filename=hazard-<id>-photo.jpg`` (same bytes, same rules).
"""
import math
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path as PathParam, Query
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, StrictInt

from backend.db import HAZARD_STATUSES, get_connection, init_db
from backend.dedup import ReviewConflict, get_hazard, load_hazards, review_hazard
from backend.report import ReportNotAvailable, build_report, effective_agency, load_report_verdict
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
    agency, automatic, source = effective_agency(row)
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
        "misses": int(row.get("fix_misses") or 0),
        "possibly_fixed_at": row.get("possibly_fixed_at"),
        "previous_status": row.get("previous_status"),
        "class_edited": bool(row.get("class_edited")),
        "severity_edited": bool(row.get("severity_edited")),
        "original_hazard_class": row.get("original_hazard_class"),
        "original_severity": row.get("original_severity"),
        "agency": agency,
        "agency_source": source,
        "automatic_agency": automatic,
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
    download: bool = Query(False),
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
    headers = {"Cache-Control": "no-store"}
    if download:
        suffix = "-crop" if variant == "crop" else ""
        headers["Content-Disposition"] = f"attachment; filename=hazard-{hazard_id}-photo{suffix}.jpg"
    return FileResponse(path, media_type="image/jpeg", headers=headers)


def _report_for(hazard_id: int, storage: Storage) -> dict[str, Any]:
    row = get_hazard(hazard_id)
    if row is None:
        raise HTTPException(404, "Hazard not found")
    has_photo = _servable(row["photo_path"], _storage_root(storage), _original_paths()) is not None
    try:
        return build_report(row, load_report_verdict(hazard_id), has_photo=has_photo)
    except ReportNotAvailable as exc:
        raise HTTPException(409, str(exc)) from None


def _report_text(report: dict[str, Any]) -> PlainTextResponse:
    return PlainTextResponse(
        report["text"],
        headers={
            "Content-Disposition": f"attachment; filename=hazard-{report['hazard_id']}-report.txt",
            "Cache-Control": "no-store",
        },
    )


@router.get("/{hazard_id}/report", response_model=None)
def get_hazard_report(
    hazard_id: int = PathParam(..., le=MAX_ID),
    format: Literal["json", "txt"] = Query("json"),
    storage: Storage = Depends(get_storage),
) -> Any:
    """The prefilled agency report of an approved or reported hazard (JSON, or ``?format=txt`` as a download)."""
    report = _report_for(hazard_id, storage)
    return _report_text(report) if format == "txt" else report


@router.get("/{hazard_id}/report.txt")
def get_hazard_report_txt(
    hazard_id: int = PathParam(..., le=MAX_ID),
    storage: Storage = Depends(get_storage),
) -> PlainTextResponse:
    """The same report as a plain-text download."""
    return _report_text(_report_for(hazard_id, storage))


class ReviewBody(BaseModel):
    """Body of PATCH /api/hazards/{id}. Only the types are checked here; the values are checked by
    ``review_hazard``. Fields that are absent are left alone (``model_fields_set``)."""

    model_config = ConfigDict(extra="forbid")

    status: str = None  # type: ignore[assignment]
    hazard_class: str = None  # type: ignore[assignment]
    severity: StrictInt = None  # type: ignore[assignment]
    agency: str | None = None  # "VDOT", "MDOT SHA", "none", "auto" or null (= automatic); checked by review_hazard


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
    except ReviewConflict as exc:
        raise HTTPException(409, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    return _hazard_json(row, _storage_root(storage), _original_paths())
