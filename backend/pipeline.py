"""Pipeline runner (task 10): preprocess, detect, aggregate and verify one drive as a single job.

    from backend.pipeline import run_pipeline
    result = run_pipeline(drive_id)           # synchronous; the API runs it as a background task

``run_pipeline(drive_id, *, provider=None, db_path=None, storage=None, work_dir=None, fps=None,
chunk_seconds=None, height=None, min_frames=None, min_confidence=None, claimed=False)``

Stages, in order (each one is the function of an earlier task):

    preprocessing  preprocess_video   (task 5)  stored video -> <work_dir>/chunk_*.mp4 + frame_timestamps.json
    detecting      run_detection      (task 7)  chunks -> detections table (provider.detect per chunk)
    aggregating    aggregate_drive    (task 8)  detections -> candidates, geotagged with the drive's GPX,
                                                started_at and clock_offset_seconds
    verifying      verify_drive       (task 9)  candidates -> verdicts (chunks= passed, so the frames
                                                next to the best frame are attached)

Drive status and stage (columns of the ``drives`` table, also keys of the drive JSON):

    status  uploaded -> processing -> done | failed        (a finished or failed drive can run again)
    stage   NULL (never run), queued, preprocessing, detecting, aggregating, verifying, done, failed
    error   NULL unless status is failed; then "<stage>: <message>" (see below)
    failed_stage  the stage that raised (preprocessing, ...), only when status is failed
    updated_at    UTC ISO time of the last status or stage change
    finished_at   UTC ISO time the last run ended; NULL while running

Stationary stretches (task 13): the detecting stage drops frames whose GPS speed is below
ROADSCOUT_STATIONARY_SPEED_MPS (default 0.5 m/s; ``stationary_speed_mps=``), unless switched off with
ROADSCOUT_SKIP_STATIONARY=0 / ``skip_stationary=False`` (see ``backend/detection.py``; it fails open
when the GPX is missing or unusable). The counts are on the drive row (integers, 0 until detection
finishes and reset to 0 at the start of a run, so a re-run never accumulates): ``frames_total``, ``stationary_frames_skipped``, ``stationary_chunks_skipped``,
``chunks_sent`` and ``stationary_note``. The full summary is ``PipelineResult.detection``.

The status and stage are written and committed BEFORE each stage starts, so ``GET /api/drives/{id}``
shows progress while the job runs. Claiming a drive (status processing, stage queued) is one atomic
UPDATE, so two concurrent requests cannot both start it (the loser gets ``DriveBusyError`` / HTTP 409).

Provider: ``provider`` is any ``ModelProvider`` (default ``get_provider()``, the fake provider unless
ROADSCOUT_PROVIDER says otherwise). It is created when detection starts, so a configuration problem
(for example live without a key) is recorded as a ``detecting`` failure. No network call is made
here; no key is read or logged by this module.

Files, all under ``<storage>/<drive id>/`` (the same folder the upload lives in, so
``storage.delete_drive`` removes everything together):

    video.<ext>, track.gpx      the upload; NEVER modified or deleted by the pipeline
    work/                       chunk_0000.mp4 ..., frame_timestamps.json (kept after a run for reruns)
    candidates/                 best-frame and crop JPEGs written by aggregate_drive

Failure behaviour:

- ``run_pipeline`` does NOT raise for a stage failure: it records it and returns a ``PipelineResult``
  with ``status == "failed"``. The drive row gets status failed, stage failed, ``failed_stage`` and
  ``error = "<stage>: <message>"`` (whitespace collapsed to one line, cut to 500 characters, no
  stack trace, the Meta key and bearer tokens replaced by ``***``). The full traceback goes to the
  server log only, never to the API.
- It raises only for caller mistakes: ``DriveNotFoundError`` (unknown drive id) and
  ``DriveBusyError`` (the drive is already processing).
- What is kept after a failure: every run starts by clearing the drive's old derived results
  (detections, candidates with their verdicts and images, and the default work folder). So after a
  failure at stage S, the results of the stages BEFORE S from this run are kept and are consistent
  with each other, and nothing exists for S or later stages (no stale rows from an older run).
  Examples: verifying fails -> detections and candidates stay, no verdicts; detecting fails -> only
  the preprocessed chunks stay. Each stage is itself all-or-nothing (see its module).
- Re-running a done or failed drive clears and replaces its results. The upload is never touched.
- A server restart while a drive is processing leaves it stuck at ``processing``; ``recover_interrupted``
  (called when the app starts) marks such drives failed with "interrupted: ...". Run one server
  process only.

``claim_drive`` and ``recover_interrupted`` are the building blocks the API route uses.
"""
from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .candidates import AggregationSummary, aggregate_drive
from .db import get_connection, init_db
from .detection import DetectionSummary, run_detection
from .preprocess import PreprocessResult, preprocess_video
from .providers import ModelProvider, get_provider
from .verification import VerificationSummary, verify_drive

logger = logging.getLogger(__name__)

STATUS_UPLOADED = "uploaded"
STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

STAGE_QUEUED = "queued"
STAGE_PREPROCESSING = "preprocessing"
STAGE_DETECTING = "detecting"
STAGE_AGGREGATING = "aggregating"
STAGE_VERIFYING = "verifying"
STAGE_DONE = "done"
STAGE_FAILED = "failed"

STAGES = (STAGE_PREPROCESSING, STAGE_DETECTING, STAGE_AGGREGATING, STAGE_VERIFYING)

MAX_ERROR_CHARS = 500
WORK_DIR_NAME = "work"
CANDIDATES_DIR_NAME = "candidates"


class DriveNotFoundError(LookupError):
    """No drive with that id."""


class DriveBusyError(RuntimeError):
    """The drive is already processing."""


@dataclass
class PipelineResult:
    """What a ``run_pipeline`` call did. A summary is None when its stage did not complete."""

    drive_id: str
    status: str  # "done" or "failed"
    stage: str  # "done" or "failed"
    error: str | None = None
    failed_stage: str | None = None
    preprocess: PreprocessResult | None = None
    detection: DetectionSummary | None = None
    aggregation: AggregationSummary | None = None
    verification: VerificationSummary | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_DONE


# ----- helpers -----------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SECRET_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|token|authorization|secret)\b(\s*[=:]\s*)\S+"),
)


def clean_error_message(exc_or_text: BaseException | str, stage: str | None = None) -> str:
    """One-line, secret-free, length-capped error text: ``"<stage>: <message>"``.

    Whitespace (including newlines) is collapsed; the value of ``META_API_KEY`` and bearer tokens
    or ``api_key=...`` style values are replaced by ``***``; the result is at most
    ``MAX_ERROR_CHARS`` characters (ending in ``...`` when cut). An exception with no message
    shows its class name.
    """
    if isinstance(exc_or_text, BaseException):
        text = str(exc_or_text).strip() or type(exc_or_text).__name__
    else:
        text = str(exc_or_text)
    key = config.meta_api_key()
    if key and len(key) >= 4:
        text = text.replace(key, "***")
    text = _SECRET_PATTERNS[0].sub("Bearer ***", text)
    text = _SECRET_PATTERNS[1].sub(lambda m: f"{m.group(1)}{m.group(2)}***", text)
    text = " ".join(text.split())
    message = f"{stage}: {text}" if stage else text
    if len(message) > MAX_ERROR_CHARS:
        message = message[: MAX_ERROR_CHARS - 3].rstrip() + "..."
    return message


def _update(db: Path, drive_id: str, **fields: Any) -> None:
    """Write columns of one drive row and commit (a fresh short connection each time)."""
    fields["updated_at"] = _now()
    assignments = ", ".join(f"{name} = ?" for name in fields)
    conn = get_connection(db)
    try:
        with conn:
            conn.execute(f"UPDATE drives SET {assignments} WHERE id = ?", (*fields.values(), drive_id))
    finally:
        conn.close()


def _get_row(db: Path, drive_id: str):
    conn = get_connection(db)
    try:
        return conn.execute("SELECT * FROM drives WHERE id = ?", (drive_id,)).fetchone()
    finally:
        conn.close()


def claim_drive(drive_id: str, *, db_path: str | Path | None = None) -> dict[str, Any]:
    """Atomically start a run: status processing, stage queued, error cleared. Returns the drive row.

    One UPDATE decides the winner, so concurrent callers cannot both claim a drive.
    Raises ``DriveNotFoundError`` for an unknown id and ``DriveBusyError`` if already processing.
    """
    db = init_db(db_path)
    now = _now()
    conn = get_connection(db)
    try:
        with conn:
            cur = conn.execute(
                "UPDATE drives SET status = ?, stage = ?, error = NULL, failed_stage = NULL,"
                " finished_at = NULL, updated_at = ? WHERE id = ? AND status != ?",
                (STATUS_PROCESSING, STAGE_QUEUED, now, drive_id, STATUS_PROCESSING),
            )
            claimed = cur.rowcount == 1
        row = conn.execute("SELECT * FROM drives WHERE id = ?", (drive_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise DriveNotFoundError(f"drive not found: {drive_id}")
    if not claimed:
        raise DriveBusyError(f"drive {drive_id} is already processing")
    return dict(row)


def recover_interrupted(*, db_path: str | Path | None = None) -> int:
    """Mark drives left in status processing (the server stopped mid-run) as failed.
    Returns how many were fixed. Call once at start-up, with a single server process."""
    db = init_db(db_path)
    now = _now()
    conn = get_connection(db)
    try:
        with conn:
            cur = conn.execute(
                "UPDATE drives SET status = ?, failed_stage = CASE WHEN stage IN (?, ?, ?, ?) THEN stage END,"
                " stage = ?, error = ?, finished_at = ?, updated_at = ? WHERE status = ?",
                (STATUS_FAILED, *STAGES, STAGE_FAILED,
                 "interrupted: the server stopped while this drive was processing; run it again",
                 now, now, STATUS_PROCESSING),
            )
            return cur.rowcount
    finally:
        conn.close()


def clear_results(drive_id: str, *, db_path: str | Path | None = None, work_dir: Path | None = None) -> None:
    """Delete a drive's derived results: detections, candidates (their verdicts go with them),
    the candidate images and, when given, the work folder. Never touches the upload."""
    db = init_db(db_path)
    conn = get_connection(db)
    try:
        old = conn.execute(
            "SELECT best_frame_path, crop_path FROM candidates WHERE drive_id = ?", (drive_id,)
        ).fetchall()
        with conn:
            conn.execute("DELETE FROM verdicts WHERE drive_id = ?", (drive_id,))
            conn.execute("DELETE FROM candidates WHERE drive_id = ?", (drive_id,))
            conn.execute("DELETE FROM detections WHERE drive_id = ?", (drive_id,))
    finally:
        conn.close()
    for row in old:
        for path in row:
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    logger.warning("could not remove %s", path)
    if work_dir is not None:
        shutil.rmtree(work_dir, ignore_errors=True)


# ----- main --------------------------------------------------------------------------------


def run_pipeline(
    drive_id: str,
    *,
    provider: ModelProvider | None = None,
    db_path: str | Path | None = None,
    storage: Any = None,
    work_dir: str | Path | None = None,
    fps: float | None = None,
    chunk_seconds: float | None = None,
    height: int | None = None,
    min_frames: int | None = None,
    min_confidence: float | None = None,
    claimed: bool = False,
    skip_stationary: bool | None = None,
    stationary_speed_mps: float | None = None,
) -> PipelineResult:
    """Run the four stages for one drive and record progress, errors and the outcome in the
    ``drives`` row. See the module docstring.

    ``provider``, ``db_path`` and ``storage`` default to the configured ones. ``work_dir`` defaults to
    ``<storage>/<drive id>/work`` (only that default folder is wiped at the start of a run).
    ``fps``, ``chunk_seconds``, ``height``, ``min_frames`` and ``min_confidence`` are passed to the
    stages (None means their config defaults). ``claimed=True`` means the caller already did
    ``claim_drive`` (the API route does); otherwise this function claims the drive itself.
    ``skip_stationary`` / ``stationary_speed_mps`` switch and tune the stationary-frame filter of the
    detecting stage (task 13; None means ROADSCOUT_SKIP_STATIONARY / ROADSCOUT_STATIONARY_SPEED_MPS).
    """
    if drive_id is None or not str(drive_id).strip():
        raise ValueError("drive_id must not be blank")
    drive_id = str(drive_id).strip()
    db = init_db(db_path)
    if not claimed:
        claim_drive(drive_id, db_path=db)
    row = _get_row(db, drive_id)
    if row is None:
        raise DriveNotFoundError(f"drive not found: {drive_id}")
    drive = dict(row)

    stage = STAGE_PREPROCESSING
    result = PipelineResult(drive_id=drive_id, status=STATUS_PROCESSING, stage=STAGE_QUEUED)
    try:
        if storage is None:
            from .storage import get_storage

            storage = get_storage()
        default_work = work_dir is None
        work = Path(work_dir) if work_dir is not None else storage.local_path(f"{drive_id}/{WORK_DIR_NAME}")
        candidates_dir = storage.local_path(f"{drive_id}/{CANDIDATES_DIR_NAME}")

        # Start from a clean slate so a failure never leaves stale rows from an older run.
        clear_results(drive_id, db_path=db, work_dir=work if default_work else None)
        _update(db, drive_id, frames_total=0, stationary_frames_skipped=0, stationary_chunks_skipped=0,
                chunks_sent=0, stationary_note=None)

        stage = STAGE_PREPROCESSING
        _update(db, drive_id, status=STATUS_PROCESSING, stage=stage)
        video = storage.local_path(drive["video_path"])
        if not Path(video).is_file():
            raise FileNotFoundError("the stored video file is missing")
        result.preprocess = preprocess_video(
            video, work, fps=fps, chunk_seconds=chunk_seconds, height=height
        )

        stage = STAGE_DETECTING
        _update(db, drive_id, stage=stage)
        if provider is None:
            provider = get_provider()
        result.detection = run_detection(
            drive_id, result.preprocess, provider=provider, db_path=db,
            skip_stationary=skip_stationary, stationary_speed_mps=stationary_speed_mps,
            gpx_track=drive,  # the drive record: GPX path, started_at and clock offset (fails open)
            storage=storage,
        )
        _record_stationary_counts(db, drive_id, result.detection)

        stage = STAGE_AGGREGATING
        _update(db, drive_id, stage=stage)
        result.aggregation = aggregate_drive(
            drive_id,
            result.preprocess,
            gpx_track=drive,  # the drive record: GPX path, started_at and clock offset
            video_start=datetime.fromisoformat(drive["started_at"]),
            clock_offset_seconds=float(drive["clock_offset_seconds"] or 0.0),
            min_frames=min_frames,
            db_path=db,
            output_dir=candidates_dir,
            storage=storage,
        )

        stage = STAGE_VERIFYING
        _update(db, drive_id, stage=stage)
        result.verification = verify_drive(
            drive_id,
            chunks=result.preprocess,
            provider=provider,
            min_confidence=min_confidence,
            db_path=db,
        )
    except Exception as exc:
        message = clean_error_message(exc, stage)
        logger.warning("pipeline failed for drive %s at %s: %s", drive_id, stage, message,
                       exc_info=logger.isEnabledFor(logging.DEBUG))
        _record_failure(db, drive_id, stage, message)
        result.status, result.stage = STATUS_FAILED, STAGE_FAILED
        result.error, result.failed_stage = message, stage
        return result
    except BaseException:
        # KeyboardInterrupt, SystemExit: do not leave the drive stuck in processing, then propagate.
        _record_failure(db, drive_id, stage, f"{stage}: interrupted")
        raise

    now = _now()
    _update(db, drive_id, status=STATUS_DONE, stage=STAGE_DONE, error=None, failed_stage=None,
            finished_at=now)
    result.status, result.stage = STATUS_DONE, STAGE_DONE
    logger.info("pipeline done for drive %s: %d candidates, %d accepted", drive_id,
                result.verification.candidate_count, result.verification.accepted_count)
    return result


def _record_stationary_counts(db: Path, drive_id: str, detection: DetectionSummary) -> None:
    """Write the stationary-filter counts of this run onto the drive row (shown by GET /api/drives/{id})."""
    _update(db, drive_id, frames_total=detection.frames_total,
            stationary_frames_skipped=detection.stationary_frames_skipped, stationary_chunks_skipped=detection.stationary_chunks_skipped,
            chunks_sent=detection.chunks_sent, stationary_note=detection.stationary_note)


def _record_failure(db: Path, drive_id: str, stage: str, message: str) -> None:
    try:
        _update(db, drive_id, status=STATUS_FAILED, stage=STAGE_FAILED, failed_stage=stage,
                error=message, finished_at=_now())
    except Exception:  # never let bookkeeping crash the worker; recover_interrupted fixes it on restart
        logger.exception("could not record the failure of drive %s", drive_id)


def run_pipeline_in_background(drive_id: str, **kwargs: Any) -> None:
    """``run_pipeline(drive_id, claimed=True, **kwargs)`` for a background task: never raises."""
    try:
        run_pipeline(drive_id, claimed=True, **kwargs)
    except Exception:
        logger.exception("background pipeline for drive %s crashed", drive_id)
