"""Routes for the annotated video exports (task 22). The export itself is in ``backend/export.py``.

    POST /api/hazards/{id}/export          make a blurred side-by-side clip of the hazard (201 + the export JSON)
    GET  /api/hazards/{id}/exports         the hazard's export records, newest first
    GET  /api/exports                      all export records, newest first (``?hazard_id=``, ``?drive_id=``)
    GET  /api/exports/{export_id}          one export record
    GET  /api/exports/{export_id}/video    the mp4 (``video/mp4``, HTTP Range requests, ``Cache-Control: no-store``)

POST body (optional JSON object; the body may be absent or ``{}``): ``pre_seconds`` and ``post_seconds`` (aliases ``before_seconds`` / ``after_seconds``), numbers
from 0 to 3600 (JSON numbers only: strings, booleans and NaN are refused), not both 0; other fields are refused.
Defaults and the cap on the clip length come from the environment (see ``backend/export.py``). The handler is a
plain ``def``, so FastAPI runs it in its threadpool: the request is SYNCHRONOUS and answers when the clip is done
(a few seconds for the default 8 s window). Answers: 201 with the export JSON; 404 unknown hazard; 409 with a clear
``detail`` when the export is refused (hazard without a member candidate, drive still processing, raw video deleted or
missing, window outside the video); 422 for an invalid body; 502 when the provider failed and 500 for any other
failure of the export itself, both as ``{"detail": "<short message>", "export": <the failed record>}`` (fail
closed: no file exists, no filesystem path or key is in the message). An export with ``status`` ``failed`` is still
listed by the GET routes.

Export JSON keys: ``id, drive_id, hazard_id, start_seconds, end_seconds, duration_seconds, status, error,
created_at, video_url`` (``/api/exports/<id>/video`` for an ok export, else null). No filesystem path appears.

Video route: 404 for an unknown id, a failed export, a missing file or a stored key that does not point at a file
directly inside ``<storage>/<drive id>/exports/``; a non-integer id is a 422. A ``Range: bytes=a-b`` (also ``a-`` and
``-n``) request gets 206 with ``Content-Range: bytes a-b/size``; an unsatisfiable range gets 416 with
``Content-Range: bytes */size``; a malformed or multi-range header is ignored (200, whole file). The file is read
in 64 KiB blocks, never loaded whole.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Annotated, Any, Iterator

from fastapi import APIRouter, Depends, HTTPException, Header, Path as PathParam, Query
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.export import (
    ExportError, ExportFailedError, ExportNotFoundError, ExportRefusedError, export_file_path, export_hazard_clip,
    get_export, list_exports,
)
from backend.db import get_connection, init_db
from backend.storage import Storage, get_storage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["exports"])

MAX_ID = 2**63 - 1
BLOCK_SIZE = 64 * 1024

Seconds = Annotated[float, Field(strict=True, ge=0, le=3600, allow_inf_nan=False)]


class ExportBody(BaseModel):
    """Body of POST /api/hazards/{id}/export."""

    model_config = ConfigDict(extra="forbid")

    pre_seconds: Seconds | None = None
    post_seconds: Seconds | None = None
    before_seconds: Seconds | None = None  # alias of pre_seconds
    after_seconds: Seconds | None = None   # alias of post_seconds

    @model_validator(mode="after")
    def _check(self) -> "ExportBody":
        if self.pre_seconds is not None and self.before_seconds is not None:
            raise ValueError("give pre_seconds or before_seconds, not both")
        if self.post_seconds is not None and self.after_seconds is not None:
            raise ValueError("give post_seconds or after_seconds, not both")
        pre = self.pre_seconds if self.pre_seconds is not None else self.before_seconds
        post = self.post_seconds if self.post_seconds is not None else self.after_seconds
        if pre == 0 and post == 0:
            raise ValueError("the seconds before and after must not both be 0")
        return self


@router.post("/hazards/{hazard_id}/export", status_code=201, response_model=None)
def export_hazard(
    body: ExportBody | None = None,
    hazard_id: int = PathParam(..., le=MAX_ID),
    storage: Storage = Depends(get_storage),
) -> Any:
    """Make a blurred side-by-side raw / annotated clip of a hazard (synchronous; see the module docstring)."""
    try:
        record = export_hazard_clip(
            hazard_id, storage=storage,
            pre_seconds=body.pre_seconds if body else None, post_seconds=body.post_seconds if body else None,
            before_seconds=body.before_seconds if body else None, after_seconds=body.after_seconds if body else None,
        )
    except ExportNotFoundError as exc:
        raise HTTPException(404, str(exc)) from None
    except ExportRefusedError as exc:
        raise HTTPException(409, str(exc)) from None
    except ExportFailedError as exc:
        return JSONResponse(
            status_code=502 if exc.stage == "provider" else 500,
            content={"detail": str(exc), "export": exc.record.to_dict() if exc.record else None},
            headers={"Cache-Control": "no-store"},
        )
    except ExportError as exc:
        raise HTTPException(500, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    except Exception:  # a bare exception must never escape the route (details are in the server log)
        logger.exception("export of hazard %s failed unexpectedly", hazard_id)
        raise HTTPException(500, "the export failed unexpectedly") from None
    return record.to_dict()


@router.get("/hazards/{hazard_id}/exports")
def hazard_exports(hazard_id: int = PathParam(..., le=MAX_ID)) -> list[dict[str, Any]]:
    """The export records of a hazard, newest first (404 when the hazard is unknown and has none)."""
    records = list_exports(hazard_id=hazard_id)
    if not records:
        conn = get_connection(init_db())
        try:
            known = conn.execute("SELECT 1 FROM hazards WHERE id = ?", (hazard_id,)).fetchone() is not None
        finally:
            conn.close()
        if not known:
            raise HTTPException(404, "Hazard not found")
    return [r.to_dict() for r in records]


@router.get("/exports")
def exports_list(
    hazard_id: int | None = Query(None, le=MAX_ID),
    drive_id: str | None = Query(None),
) -> list[dict[str, Any]]:
    """Every export record, newest first."""
    return [r.to_dict() for r in list_exports(hazard_id=hazard_id, drive_id=drive_id)]


@router.get("/exports/{export_id}")
def export_detail(export_id: int = PathParam(..., le=MAX_ID)) -> dict[str, Any]:
    record = get_export(export_id)
    if record is None:
        raise HTTPException(404, "Export not found")
    return record.to_dict()


_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def parse_range(header: str | None, size: int) -> tuple[int, int] | str | None:
    """``(first, last)`` for a satisfiable single range, ``"unsatisfiable"``, or None when the header is absent,
    malformed or asks for several ranges (then the whole file is served)."""
    if not header:
        return None
    m = _RANGE_RE.fullmatch(header.strip())
    if m is None:
        return None
    first, last = m.group(1), m.group(2)
    if not first and not last:
        return None
    if not first:  # suffix: the last n bytes
        n = int(last)
        if n == 0 or size == 0:
            return "unsatisfiable"
        return max(0, size - n), size - 1
    start = int(first)
    end = int(last) if last else size - 1
    if last and end < start:
        return None  # an invalid range is ignored
    if start >= size:
        return "unsatisfiable"
    return start, min(end, size - 1)


def _iter_file(path: Path, start: int, length: int) -> Iterator[bytes]:
    with open(path, "rb") as f:
        f.seek(start)
        left = length
        while left > 0:
            block = f.read(min(BLOCK_SIZE, left))
            if not block:
                break
            left -= len(block)
            yield block


@router.get("/exports/{export_id}/video", response_model=None)
def export_video(
    export_id: int = PathParam(..., le=MAX_ID),
    range_header: str | None = Header(None, alias="Range"),
    storage: Storage = Depends(get_storage),
) -> Response:
    """The mp4 of an ok export, with Range support (see the module docstring)."""
    record = get_export(export_id)
    path = export_file_path(record, storage) if record is not None else None
    if path is None:
        raise HTTPException(404, "Export video not found")
    try:
        size = path.stat().st_size
    except OSError:
        raise HTTPException(404, "Export video not found") from None
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-store"}
    parsed = parse_range(range_header, size)
    if parsed == "unsatisfiable":
        return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{size}"})
    if parsed is None:
        headers["Content-Length"] = str(size)
        return StreamingResponse(_iter_file(path, 0, size), status_code=200, media_type="video/mp4", headers=headers)
    first, last = parsed
    headers["Content-Length"] = str(last - first + 1)
    headers["Content-Range"] = f"bytes {first}-{last}/{size}"
    return StreamingResponse(_iter_file(path, first, last - first + 1), status_code=206, media_type="video/mp4",
                             headers=headers)
