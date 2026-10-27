"""Delete video (task 15): remove a drive's raw video and unblurred frames on request.

    from backend.video_deletion import delete_video
    drive = delete_video(drive_id)            # the updated drive row as a dict

Behind ``DELETE /api/drives/{id}/video`` (the Delete video button). Raw video is never deleted any
other way: the pipeline, re-runs and ``clear_results`` leave the upload alone.

``delete_video(drive_id, *, storage=None, db_path=None) -> dict``

Removes, from ``<storage>/<drive id>/``:

- the raw upload (the file behind ``drives.video_path``, and any other ``video.*`` file in the folder),
- ``work/`` (the preprocessed chunks and ``frame_timestamps.json``, i.e. the extracted frames),
- every UNBLURRED candidate image (the best-frame and crop JPEGs in ``candidates/``, and any other file
  there that is not a recorded blurred image or named ``*_blur.jpg``).

Keeps: the GPX (``track.gpx``), every database row (drive, detections, candidates, verdicts, hazards) and
the BLURRED images in ``candidates/`` (the recorded ``blurred_*_path`` files), so the results page, its photos and the
hazards keep working. ``candidates.best_frame_path`` / ``crop_path`` keep their (now dangling) values;
nothing reads them any more.

Effects on the ``drives`` row: ``video_deleted_at`` (UTC ISO time, new nullable column added by the
idempotent migration in ``init_db``) is set; ``video_path`` keeps the stored key (it is not nulled). The
drive JSON adds ``video_deleted`` (bool) and ``video_deleted_at``.

Rules:

- Unknown drive: ``DriveNotFoundError``. A drive that is processing: ``DriveBusyError`` (nothing is
  changed). The check and the mark are ONE atomic UPDATE (``status != 'processing'``), and
  ``claim_drive`` refuses a drive with ``video_deleted_at`` set, so a delete and a process request cannot
  both win.
- The mark is written first, then the files are removed. Idempotent: calling it again on a drive whose
  video is already deleted succeeds, keeps the original ``video_deleted_at`` and sweeps any leftovers
  (so it is also the retry after a failed removal). A file that cannot be removed raises ``OSError``
  after the mark is written (as ``DeleteIncomplete``, whose message names no path; the details are logged); a second call tries again.
- Works on a drive in any state except processing (uploaded, done, failed). Deleting the video of a
  drive that was never processed means it can never be processed.
- Only paths under the drive's own folder are touched.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .db import get_connection, init_db
from .pipeline import DriveBusyError, DriveNotFoundError, STATUS_PROCESSING

logger = logging.getLogger(__name__)

WORK_DIR_NAME = "work"
CANDIDATES_DIR_NAME = "candidates"


class DeleteIncomplete(OSError):
    """Some files could not be removed. The message is generic on purpose (no paths); see the log."""


def _inside(path: Path, folder: Path) -> bool:
    try:
        return path.resolve().is_relative_to(folder.resolve())
    except (OSError, ValueError):
        return False


def _unlink(path: Path, failures: list[str]) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        failures.append(f"{path.name}: {e}")


def _remove_files(drive_id: str, video_key: str | None, rows: list[Any], storage: Any) -> None:
    """Remove the raw video, the work folder and the unblurred candidate images. Raises OSError if any
    file is left behind."""
    failures: list[str] = []
    try:
        folder = Path(storage.local_path(drive_id))
    except ValueError:  # an id that is not a valid storage folder: nothing on disk to remove
        return

    if video_key:
        try:
            storage.delete(video_key)
        except (OSError, ValueError) as e:
            failures.append(f"{video_key}: {e}")
    if folder.is_dir():
        for extra in folder.glob("video.*"):  # any other raw upload left in the drive's folder
            if extra.is_file():
                _unlink(extra, failures)

    work = folder / WORK_DIR_NAME
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
        if work.exists():
            failures.append(f"{WORK_DIR_NAME}/: could not remove the folder")

    keep: set[Path] = set()
    for r in rows:  # every recorded blurred image, whatever its status
        keep.update(Path(p).resolve() for p in (r["blurred_frame_path"], r["blurred_crop_path"]) if p)
    candidates = folder / CANDIDATES_DIR_NAME
    unblurred = {Path(p) for r in rows for p in (r["best_frame_path"], r["crop_path"]) if p}
    for path in unblurred:
        if path.resolve() not in keep and _inside(path, folder):
            _unlink(path, failures)
    if candidates.is_dir():
        for path in candidates.iterdir():  # leftovers, partial files, files of older runs
            if path.is_file() and path.resolve() not in keep and not path.name.endswith("_blur.jpg"):
                _unlink(path, failures)

    if failures:
        # The details (names, errno text) go to the server log only; the error that reaches the API
        # carries no path or file name.
        logger.warning("could not delete everything for drive %s: %s", drive_id, "; ".join(failures[:10]))
        raise DeleteIncomplete("some files could not be removed")


def delete_video(drive_id: str, *, storage: Any = None, db_path: str | Path | None = None) -> dict[str, Any]:
    """Delete the raw video, work folder and unblurred frames of a drive. See the module docstring."""
    if drive_id is None or not str(drive_id).strip():
        raise ValueError("drive_id must not be blank")
    drive_id = str(drive_id).strip()
    if storage is None:
        from .storage import get_storage

        storage = get_storage()
    db = init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()

    conn = get_connection(db)
    try:
        with conn:
            conn.execute(
                "UPDATE drives SET video_deleted_at = ?, updated_at = ? "
                "WHERE id = ? AND video_deleted_at IS NULL AND status != ?",
                (now, now, drive_id, STATUS_PROCESSING),
            )
        drive = conn.execute("SELECT * FROM drives WHERE id = ?", (drive_id,)).fetchone()
        if drive is None:
            raise DriveNotFoundError(f"drive not found: {drive_id}")
        if not drive["video_deleted_at"]:  # the update did not apply: the drive is processing
            raise DriveBusyError(f"drive {drive_id} is processing; wait for it to finish")
        rows = conn.execute(
            "SELECT best_frame_path, crop_path, blurred_frame_path, blurred_crop_path, blur_status "
            "FROM candidates WHERE drive_id = ?", (drive_id,)
        ).fetchall()
    finally:
        conn.close()

    _remove_files(drive_id, drive["video_path"], rows, storage)
    logger.info("deleted the raw video and unblurred frames of drive %s", drive_id)
    return dict(drive)
