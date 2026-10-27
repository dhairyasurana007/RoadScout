"""Annotated video export (task 22): a side-by-side raw and annotated clip of a hazard, for the demo.

    from backend.export import export_hazard_clip, export_segment
    record = export_hazard_clip(hazard_id)                    # ExportRecord: a dict (status 'ok')
    record = export_segment(drive_id, 12.0, 20.0, track_key="3:1")

    python -m backend.export --hazard 7 [--pre 4] [--post 4] [--max 20]     # prints the export JSON

Only a SHORT SEGMENT of the raw video is ever cut and sent to the model (never the whole drive), so the
face and license plate blur costs one ``provider.detect`` call per clip. No unblurred face or plate appears
in the output: both halves of every frame are blurred.

``export_hazard_clip(hazard_id, *, provider=None, db_path=None, storage=None, pre_seconds=None, post_seconds=None,
max_seconds=None, before_seconds=None, after_seconds=None) -> ExportRecord``

Takes the hazard's best member candidate (highest ``score``, ties: verified and not blur-failed first, then highest
verdict confidence, then drive id / track key / time / id), its drive and ``best_timestamp`` ``t``, and exports the
window ``[t - pre, t + post]``. Defaults: ``pre`` 4 s (ROADSCOUT_EXPORT_BEFORE_SECONDS) and ``post`` 4 s
(ROADSCOUT_EXPORT_AFTER_SECONDS); ``before_seconds`` / ``after_seconds`` are aliases of the parameters.
The window is clipped to the video's bounds ``[0, duration]`` and THEN its total length is capped at ``max_seconds``
(default 20 s, ROADSCOUT_EXPORT_MAX_SECONDS): a longer window is shrunk around ``t`` keeping the ratio of the
parts before and after it. The hazard's own track (``candidates.track_key``) is highlighted.

``export_segment(drive_id, start_seconds, end_seconds, *, hazard_id=None, track_key=None, provider=None,
db_path=None, storage=None, max_seconds=None, center_seconds=None) -> ExportRecord`` is the lower level call. The
window is in source video seconds; it is clipped to ``[0, duration]`` and, when still longer than the cap, shrunk
around ``center_seconds`` (when given and inside the window) or cut off at ``start + max_seconds`` (a clamp, not
an error). ``track_key`` (``"<chunk>:<track id>"``) is the track to highlight.
Bad arguments raise ``ValueError`` before anything happens.

Pipeline of one export (everything unblurred lives in ONE private temp directory, ``tempfile.mkdtemp``, removed
in a ``finally`` whatever happens):

1. ffmpeg (argv list, no shell, streamed) cuts only the window from the raw video into ``<tmp>/segment.mp4``: no
   audio, rotation applied, at most 720 p high (never upscaled, even dimensions), constant frame rate (the
   source rate, at most 60 fps), H.264 yuv420p. This segment is the only unblurred artefact.
2. ONE ``provider.detect(<segment path>, ["face", "license plate"])`` call (prompts from
   ``config/privacy_prompts.json``, see ``backend/privacy.py``). The provider is ``get_provider()`` by default, so
   retries, timeouts and the spend cap apply. Frame ``timestamp`` values are relative to the segment.
3. The segment is decoded frame by frame through a pipe (never loaded whole into memory). Each frame is blurred
   with ``backend.privacy.blur_frame`` (the same blur as ``blur_image``, grown by ``ROADSCOUT_BLUR_DILATE_PX``)
   using the masks of the provider records that bracket the frame in time: the record at or before it and the
   record after it, when each is within 1.5 sampling intervals (at least 2 frame periods); an exact timestamp
   match uses only that record. A frame without a record in reach is not blurred (the provider reported
   nothing there). A detection without a mask is blurred by its box (segment pixels).
4. Composition: the output frame is ``[ RAW | ANNOTATED ]`` side by side, width = 2 x the single view, same
   frame rate. LEFT is the blurred frame with a small caption "RAW" (top left). RIGHT is the same blurred frame
   plus the annotation of the rows of the ``detections`` table of that drive whose sample time is the nearest
   one to the frame (within 0.75 s; a frame farther than that from every sample has NO overlay): a translucent
   tint of the mask, a box outline, the label ``<class> #<track id>`` and the caption "ANNOTATED". Colours are
   per class (a fixed palette indexed by the class name); the hazard's own track (``track_key``) is drawn with a
   thicker white-backed outline, a stronger tint and a filled label. Box and mask coordinates are scaled from
   the detection's mask size (chunk pixels) to the view; a detection without a mask is drawn at its own pixels.
5. ffmpeg encodes the piped frames as H.264 yuv420p (libx264, ``+faststart``), no audio, into the temp dir, and
   the finished file is copied into ``<storage>/<drive id>/exports/<name>.mp4`` (through a ``.part`` file that is
   renamed). Only blurred pixels ever reach the exports folder.

FAIL CLOSED. A provider error, a garbage result (not a detection result, a bad timestamp, an invalid mask or
box), an ffmpeg failure, a decode or encode problem or a failed copy leaves NO file in the exports folder (the
partial ``.part`` file is removed too), removes the temp dir, writes an ``exports`` row with ``status = 'failed'``
and a clear short ``error`` (no filesystem paths, no key) and raises ``ExportFailedError`` (``.record`` is that
row, ``.stage`` is one of ``cut, provider, blur, encode, save``).

Refusals (nothing is called, nothing is written, no record): ``ExportNotFoundError`` (a ``LookupError``) for an
unknown hazard or drive, ``ExportRefusedError`` (a ``RuntimeError``) for a hazard without a member candidate,
a drive that is still ``processing``, a raw video that was deleted (``drives.video_deleted_at``) or is missing or
unreadable, a candidate without a usable time, or a window outside the video. The provider is not even created
in these cases. All errors derive from ``ExportError``.

Registry: the ``exports`` table (``backend/db.py``): ``id, drive_id, hazard_id, start_seconds, end_seconds, key,
status, error, created_at``; ``key`` is INTERNAL (never in the JSON), storage relative (``<drive id>/exports/<name>.mp4``, NULL when failed);
``start_seconds`` / ``end_seconds`` are the window that was really cut (clipped, in source video seconds).
Re-exporting a hazard adds a NEW record and a new file; the earlier records and files are KEPT (nothing is
overwritten, so a clip someone is watching stays valid). Exports are blurred demo artefacts: deleting a drive's
video (``backend/video_deletion.py``) does NOT delete its exports.

``ExportRecord`` is a ``dict`` (also the API JSON, ``.to_dict()`` gives a plain copy; attribute access works too, and
``record.key`` is the internal storage key): ``id, drive_id, hazard_id, start_seconds, end_seconds, duration_seconds,
status, error, created_at, video_url`` (``/api/exports/<id>/video`` when ok, else null). It never contains a
filesystem path, an ``.mp4`` name or a file key. Routes are in ``backend/exports.py``. No network is used except the provider call;
nothing here reads or logs the API key.
"""
from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import config
from .db import get_connection, init_db
from .media import InvalidMedia, probe_video
from .privacy import (
    InvalidMask, PrivacyError, _box_array, _fit_mask, _note, _rle_to_array, blur_frame, load_privacy_prompts,
)
from .providers import ProviderError, get_provider, validate_rle

logger = logging.getLogger(__name__)

EXPORTS_DIR_NAME = "exports"
SEGMENT_NAME = "segment.mp4"
OUTPUT_NAME = "export.mp4"
MAX_HEIGHT = 720
MAX_FPS = 60
CAPTION_RAW = "RAW"
CAPTION_ANNOTATED = "ANNOTATED"
CUT_TIMEOUT_S = 600
ENCODE_TIMEOUT_S = 1800
MAX_ERROR_CHARS = 200
ANNOTATION_REACH_S = 0.75  # an overlay is shown on frames up to this far (seconds) from a sampled detection time
STAGES = ("cut", "provider", "blur", "encode", "save")

# BGR colours for the classes (picked by a checksum of the class name, so a class keeps its colour).
PALETTE = (
    (0, 0, 255), (0, 165, 255), (0, 220, 255), (0, 200, 0),
    (255, 200, 0), (255, 80, 0), (255, 0, 200), (160, 0, 255),
)
_NAME_RE = re.compile(r"[A-Za-z0-9._-]+\.mp4")
_DRIVE_RE = re.compile(r"[A-Za-z0-9_-]+")


# ----- errors and records --------------------------------------------------------------------


class ExportError(Exception):
    """Base class of every export error."""


class ExportNotFoundError(ExportError, LookupError):
    """The hazard or drive does not exist."""


class ExportRefusedError(ExportError, RuntimeError):
    """The export cannot be made now (no member candidate, drive processing, raw video deleted or missing)."""


class ExportFailedError(ExportError, RuntimeError):
    """The export was tried and failed (fail closed). ``record`` is the failed ``exports`` row (or None when even
    that could not be written); ``stage`` is one of ``cut, provider, blur, encode, save``."""

    def __init__(self, message: str, *, record: "ExportRecord | None" = None, stage: str = "encode"):
        super().__init__(message)
        self.record = record
        self.stage = stage


class _StageError(Exception):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


class ExportRecord(dict):
    """One export record: a plain ``dict`` (so ``record["status"]`` and ``json.dumps(record)`` work) that also allows
    attribute access (``record.status``). Keys: ``id, drive_id, hazard_id, start_seconds, end_seconds,
    duration_seconds, status, error, created_at, video_url``. The storage-relative file key of the ``exports``
    table is INTERNAL: it is the attribute ``record.key``, never a dict key, so the JSON has no file name."""

    def __init__(self, *, id: int, drive_id: str, hazard_id: int | None, start_seconds: float, end_seconds: float,
                 key: str | None, status: str, error: str | None, created_at: str):
        super().__init__(
            id=id, drive_id=drive_id, hazard_id=hazard_id, start_seconds=start_seconds, end_seconds=end_seconds,
            duration_seconds=round(end_seconds - start_seconds, 6), status=status, error=error, created_at=created_at,
            video_url=f"/api/exports/{id}/video" if status == "ok" and key else None,
        )
        self.key = key

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def to_dict(self) -> dict[str, Any]:
        """The export JSON: a plain dict copy (no filesystem path, no file key)."""
        return dict(self)


def _record_of(row: Any) -> ExportRecord:
    return ExportRecord(
        id=row["id"], drive_id=row["drive_id"], hazard_id=row["hazard_id"],
        start_seconds=row["start_seconds"], end_seconds=row["end_seconds"], key=row["key"],
        status=row["status"], error=row["error"], created_at=row["created_at"],
    )


def get_export(export_id: int, *, db_path: str | Path | None = None) -> ExportRecord | None:
    """One export record, or None."""
    conn = get_connection(init_db(db_path))
    try:
        row = conn.execute("SELECT * FROM exports WHERE id = ?", (export_id,)).fetchone()
    finally:
        conn.close()
    return _record_of(row) if row is not None else None


def list_exports(
    *, hazard_id: int | None = None, drive_id: str | None = None, db_path: str | Path | None = None
) -> list[ExportRecord]:
    """Export records, newest first (``created_at`` then ``id``, both descending); optional filters."""
    where, args = [], []
    if hazard_id is not None:
        where.append("hazard_id = ?")
        args.append(hazard_id)
    if drive_id is not None:
        where.append("drive_id = ?")
        args.append(drive_id)
    sql = "SELECT * FROM exports" + (" WHERE " + " AND ".join(where) if where else "")
    conn = get_connection(init_db(db_path))
    try:
        rows = conn.execute(sql + " ORDER BY created_at DESC, id DESC", args).fetchall()
    finally:
        conn.close()
    return [_record_of(r) for r in rows]


def export_file_path(record: ExportRecord, storage: Any = None) -> Path | None:
    """The mp4 file of an ok record, or None when the record failed, the key is not
    ``<drive id>/exports/<name>.mp4`` for the record's own drive, the path leaves the drive's exports folder
    or the file is missing."""
    if record.status != "ok" or not record.key:
        return None
    parts = record.key.split("/")
    if len(parts) != 3 or parts[0] != record.drive_id or parts[1] != EXPORTS_DIR_NAME \
            or not _NAME_RE.fullmatch(parts[2]) or not _DRIVE_RE.fullmatch(parts[0]):
        return None
    if storage is None:
        from .storage import get_storage

        storage = get_storage()
    try:
        folder = Path(storage.local_path(f"{record.drive_id}/{EXPORTS_DIR_NAME}")).resolve()
        path = Path(storage.local_path(record.key)).resolve()
        if not path.is_relative_to(folder) or path.parent != folder or not path.is_file():
            return None
    except (ValueError, OSError):
        return None
    return path


# ----- small helpers -------------------------------------------------------------------------

_WIN_PATH = re.compile(r"(?:(?<![A-Za-z0-9])[A-Za-z]:[\\/]|\\\\)[^\s'\"<>|]*")
_MP4_NAME = re.compile(r"\S*\.mp4\S*", re.IGNORECASE)
_POSIX_PATH = re.compile(r"(?<![\w:/.])/(?:[\w.~+-]+/)+[\w.~+-]*")


def _short_error(exc_or_text: Any, *scrub: Any) -> str:
    """One short line about a failure: no key, no bearer token, no filesystem path."""
    text = _note(exc_or_text) if isinstance(exc_or_text, BaseException) else " ".join(str(exc_or_text).split())
    for s in scrub:
        s = str(s)
        if len(s) >= 3:
            text = text.replace(s, "<path>")
    text = _WIN_PATH.sub("<path>", text)
    text = _POSIX_PATH.sub("<path>", text)
    text = _MP4_NAME.sub("<file>", text).replace("\\", "/")
    return text if len(text) <= MAX_ERROR_CHARS else text[: MAX_ERROR_CHARS - 3].rstrip() + "..."


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rmtree(path: Path) -> None:
    for attempt in range(8):  # Windows may hold a file a moment after a process ended
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.15 * (attempt + 1))
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        logger.warning("could not remove the export temp folder %s", path.name)


def _unlink(path: Path | None) -> None:
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("could not remove %s", path.name)


# ----- ffmpeg --------------------------------------------------------------------------------


def _probe_fps(path: Path, stream_index: int | None) -> Fraction:
    """The frame rate for the segment: the source's (average, else nominal) rate, at most MAX_FPS (60); 30 if unknown."""
    argv = ["ffprobe", "-v", "error", "-select_streams", str(stream_index if stream_index is not None else "v:0"),
            "-show_entries", "stream=avg_frame_rate,r_frame_rate", "-of", "json", str(path)]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=120).stdout
        stream = json.loads(out or "{}")["streams"][0]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError):
        return Fraction(30)
    for name in ("avg_frame_rate", "r_frame_rate"):
        try:
            fps = Fraction(str(stream.get(name, "0/0")))
        except (ValueError, ZeroDivisionError):
            continue
        if 0 < fps <= 1000:
            return min(fps, Fraction(MAX_FPS)).limit_denominator(1001)
    return Fraction(30)


def _probe_size(path: Path) -> tuple[int, int]:
    argv = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
            "-of", "json", str(path)]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=120).stdout
        stream = json.loads(out or "{}")["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError):
        raise _StageError("cut", "the cut segment could not be read back") from None
    if width < 2 or height < 2:
        raise _StageError("cut", "the cut segment has no picture")
    return width, height


def _cut_segment(src: Path, stream_index: int, start: float, duration: float, fps: Fraction, dst: Path) -> None:
    vf = (f"fps={fps.numerator}/{fps.denominator},"
          f"scale=-2:'trunc(min({MAX_HEIGHT},ih)/2)*2',setsar=1,format=yuv420p")
    argv = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src),
        "-map", f"0:{stream_index}", "-an", "-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1",
        "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-movflags", "+faststart", str(dst),
    ]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=CUT_TIMEOUT_S)
    except FileNotFoundError:
        raise _StageError("cut", "ffmpeg is not installed or not on PATH") from None
    except subprocess.TimeoutExpired:
        raise _StageError("cut", "cutting the segment timed out") from None
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        logger.warning("ffmpeg cut failed (exit %s): %s", proc.returncode, (proc.stderr or "").strip()[-300:])
        raise _StageError("cut", f"ffmpeg could not cut the segment (exit {proc.returncode})")


# ----- the provider result -> blur plan ------------------------------------------------------


class _BlurPlan:
    """The privacy masks of the provider's per-frame records, looked up by time in the segment."""

    def __init__(self, result: Any, fps: Fraction, width: int, height: int):
        self.width, self.height = width, height
        frames = getattr(result, "frames", None)
        if frames is None or isinstance(frames, (str, bytes, dict)):
            raise _StageError("blur", f"the provider returned {type(result).__name__}, not a detection result")
        records: list[tuple[float, list[tuple[str, Any]]]] = []
        try:
            for frame in frames:
                ts = getattr(frame, "timestamp", None)
                dets = getattr(frame, "detections", None)
                if not _is_number(ts) or ts < 0 or dets is None or isinstance(dets, (str, bytes, dict)):
                    raise ValueError("a frame record has a bad timestamp or detections")
                specs: list[tuple[str, Any]] = []
                for det in dets:
                    mask = getattr(det, "mask", None)
                    if mask is not None:
                        specs.append(("mask", validate_rle(dict(mask))))
                        continue
                    x1, y1, x2, y2 = (float(v) for v in det.box)
                    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                        raise ValueError("a detection has an invalid box")
                    specs.append(("box", (x1, y1, x2, y2)))
                records.append((float(ts), specs))
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise _StageError("blur", f"the provider returned an unusable result ({type(exc).__name__}: {exc})") from None
        records.sort(key=lambda r: r[0])
        self.times = [r[0] for r in records]
        self.specs = [r[1] for r in records]
        gaps = [b - a for a, b in zip(self.times, self.times[1:]) if b - a > 1e-6]
        interval = statistics.median(gaps) if gaps else 1.0 / 3.0
        self.limit = max(1.5 * interval, 2.0 / float(fps))
        self._cache_key: tuple[int, ...] | None = None
        self._cache_val: np.ndarray | None = None
        self.mask_count = sum(len(s) for s in self.specs)

    def _record_masks(self, index: int) -> list[np.ndarray]:
        out = []
        for kind, value in self.specs[index]:
            try:
                out.append(_rle_to_array(value) if kind == "mask" else _box_array(value, self.width, self.height))
            except InvalidMask as exc:
                raise _StageError("blur", f"the provider returned an unusable mask ({exc})") from None
        return out

    def union_at(self, t: float) -> np.ndarray | None:
        """The union mask (a (height, width) bool array) for the frame at ``t`` seconds, or None."""
        if not self.times:
            return None
        i = bisect.bisect_right(self.times, t + 1e-6)
        if i > 0 and abs(self.times[i - 1] - t) <= 1e-6:
            idx: tuple[int, ...] = (i - 1,)
        else:
            idx = tuple(j for j in (i - 1, i)
                        if 0 <= j < len(self.times) and abs(self.times[j] - t) <= self.limit)
        if idx != self._cache_key:
            union = None
            for j in idx:
                for arr in self._record_masks(j):
                    fitted = _fit_mask(arr, self.width, self.height)
                    union = fitted.copy() if union is None else union | fitted
            self._cache_key, self._cache_val = idx, union
        return self._cache_val


# ----- annotation drawing --------------------------------------------------------------------


@dataclass
class Overlay:
    """One detection to draw, in view pixels. ``mask`` is a (height, width) bool array or None."""

    hazard_class: str
    track_id: int
    track_key: str
    box: tuple[float, float, float, float]
    mask: np.ndarray | None = None
    own: bool = False


def class_color(hazard_class: str) -> tuple[int, int, int]:
    """The BGR colour of a class (stable: a checksum of the name picks one of ``PALETTE``)."""
    return PALETTE[zlib.crc32(str(hazard_class).encode("utf-8")) % len(PALETTE)]


def _scale_for(height: int) -> tuple[float, int]:
    scale = max(0.4, height / 720.0 * 0.7)
    return scale, max(1, int(round(scale * 2)))


def draw_caption(frame: np.ndarray, text: str) -> None:
    """Burn a small caption (white text on a dark box) into the top left corner, in place."""
    scale, thick = _scale_for(frame.shape[0])
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = max(3, th // 3)
    cv2.rectangle(frame, (0, 0), (tw + 2 * pad, th + base + 2 * pad), (0, 0, 0), -1)
    cv2.putText(frame, text, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_overlays(frame: np.ndarray, overlays: list[Overlay]) -> None:
    """Draw the detections on ``frame`` in place: mask tint, box outline, ``<class> #<track id>`` label. The
    hazard's own track (``own``) is drawn last, with a stronger tint, a thicker white-backed outline and a
    filled label."""
    height, width = frame.shape[:2]
    scale, thick = _scale_for(height)
    for ov in sorted(overlays, key=lambda o: o.own):
        color = class_color(ov.hazard_class)
        if ov.mask is not None and ov.mask.any():
            alpha = 0.5 if ov.own else 0.3
            region = frame[ov.mask]
            frame[ov.mask] = (region * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
        x1, y1, x2, y2 = (int(round(v)) for v in ov.box)
        x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
        y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
        line = max(1, thick) + (2 if ov.own else 0)
        if ov.own:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), line + 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line)
        label = f"{str(ov.hazard_class).replace('_', ' ')} #{ov.track_id}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ly = y1 - 2 if y1 - th - base - 4 >= 0 else min(height - 1, y1 + th + base + 4)
        lx = max(0, min(x1, width - tw - 2))
        if ov.own:
            cv2.rectangle(frame, (lx, ly - th - base), (lx + tw + 2, ly + base), color, -1)
            cv2.putText(frame, label, (lx + 1, ly), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
        else:
            cv2.putText(frame, label, (lx + 1, ly), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(frame, label, (lx + 1, ly), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def compose_side_by_side(raw: np.ndarray, annotated: np.ndarray) -> np.ndarray:
    """``[ raw | annotated ]``: a new array twice as wide as one view."""
    return np.ascontiguousarray(np.hstack((raw, annotated)))


class _Annotations:
    """The ``detections`` rows around the segment, looked up by source video time."""

    def __init__(self, rows: list[Any], track_key: str | None, width: int, height: int):
        self.track_key, self.width, self.height = track_key, width, height
        by_ts: dict[float, list[Any]] = {}
        for r in rows:
            by_ts.setdefault(float(r["frame_timestamp"]), []).append(r)
        self.times = sorted(by_ts)
        self.rows = [by_ts[t] for t in self.times]
        gaps = [b - a for a, b in zip(self.times, self.times[1:]) if b - a > 1e-6]
        self.interval = min(gaps) if gaps else 1.0 / config.preprocess_fps()
        self.reach = ANNOTATION_REACH_S
        self._cache_idx: int | None = None
        self._cache_val: list[Overlay] = []

    def at(self, source_t: float) -> list[Overlay]:
        if not self.times:
            return []
        i = bisect.bisect_left(self.times, source_t)
        near = [j for j in (i - 1, i) if 0 <= j < len(self.times)]
        j = min(near, key=lambda k: abs(self.times[k] - source_t))
        if abs(self.times[j] - source_t) > self.reach + 1e-6:
            return []
        if j != self._cache_idx:
            self._cache_idx, self._cache_val = j, self._build(self.rows[j])
        return self._cache_val

    def _build(self, rows: list[Any]) -> list[Overlay]:
        out: list[Overlay] = []
        for r in rows:
            try:
                box = [float(v) for v in json.loads(r["box"])]
                if len(box) != 4 or not all(math.isfinite(v) for v in box):
                    raise ValueError("bad box")
                mask = None
                sx = sy = 1.0
                if r["mask"]:
                    rle = validate_rle(json.loads(r["mask"]))
                    mh, mw = rle["size"]
                    sx, sy = self.width / mw, self.height / mh
                    mask = _fit_mask(_rle_to_array(rle), self.width, self.height)
                out.append(Overlay(
                    hazard_class=str(r["hazard_class"]), track_id=int(r["track_id"]), track_key=str(r["track_key"]),
                    box=(box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy), mask=mask,
                    own=self.track_key is not None and str(r["track_key"]) == self.track_key,
                ))
            except (ValueError, TypeError, KeyError, InvalidMask):
                logger.warning("skipped a detection row that cannot be drawn (id %s)", r["id"])
        return out


# ----- rendering -----------------------------------------------------------------------------


def _read_exact(stream: Any, n: int) -> bytes:
    return stream.read(n)


def _render(segment: Path, out_path: Path, width: int, height: int, fps: Fraction, plan: _BlurPlan,
            notes: _Annotations, start: float, dilate: int, tmp: Path) -> int:
    """Decode the segment, blur, compose and encode into ``out_path``. Returns the number of frames."""
    rate = f"{fps.numerator}/{fps.denominator}"
    decode = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(segment),
              "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    encode = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
              "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{2 * width}x{height}", "-framerate", rate, "-i", "-",
              "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", rate,
              "-movflags", "+faststart", str(out_path)]
    frame_bytes = width * height * 3
    dec_err = open(tmp / "decode.log", "wb")
    enc_err = open(tmp / "encode.log", "wb")
    dec = enc = None
    try:
        try:
            dec = subprocess.Popen(decode, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=dec_err)
            enc = subprocess.Popen(encode, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=enc_err)
        except FileNotFoundError:
            raise _StageError("encode", "ffmpeg is not installed or not on PATH") from None
        count = 0
        deadline = time.monotonic() + ENCODE_TIMEOUT_S
        while True:
            buf = _read_exact(dec.stdout, frame_bytes)
            if not buf:
                break
            if len(buf) != frame_bytes:
                raise _StageError("encode", "the cut segment ended in a truncated frame")
            if time.monotonic() > deadline:
                raise _StageError("encode", "rendering timed out")
            t = count / float(fps)
            raw = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
            union = plan.union_at(t)
            try:
                blurred = blur_frame(raw, [union], dilate_px=dilate) if union is not None else raw.copy()
            except PrivacyError as exc:
                raise _StageError("blur", f"the blur failed ({exc})") from None
            right = blurred.copy()
            draw_overlays(right, notes.at(start + t))
            draw_caption(blurred, CAPTION_RAW)
            draw_caption(right, CAPTION_ANNOTATED)
            try:
                enc.stdin.write(compose_side_by_side(blurred, right).tobytes())
            except (BrokenPipeError, OSError):
                raise _StageError("encode", "ffmpeg stopped while encoding") from None
            count += 1
        enc.stdin.close()
        if dec.wait(timeout=60) != 0:
            raise _StageError("encode", "ffmpeg could not decode the cut segment")
        if enc.wait(timeout=300) != 0:
            logger.warning("ffmpeg encode failed: %s", (tmp / "encode.log").read_bytes()[-300:].decode("utf-8", "replace"))
            raise _StageError("encode", "ffmpeg could not encode the export")
        if count == 0:
            raise _StageError("cut", "the segment has no frames")
        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise _StageError("encode", "ffmpeg wrote no output")
        return count
    except subprocess.TimeoutExpired:
        raise _StageError("encode", "ffmpeg timed out") from None
    finally:
        for proc in (enc, dec):
            if proc is not None:
                for pipe in (proc.stdin, proc.stdout):
                    try:
                        if pipe:
                            pipe.close()
                    except OSError:
                        pass
                if proc.poll() is None:
                    proc.kill()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        dec_err.close()
        enc_err.close()


# ----- the exports table ---------------------------------------------------------------------


def _insert_record(db: Path, drive_id: str, hazard_id: int | None, start: float, end: float, key: str | None,
                   status: str, error: str | None) -> ExportRecord:
    now = _utc_now()
    conn = get_connection(db)
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO exports (drive_id, hazard_id, start_seconds, end_seconds, key, status, error, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (drive_id, hazard_id, round(start, 6), round(end, 6), key, status, error, now),
            )
            row = conn.execute("SELECT * FROM exports WHERE id = ?", (cur.lastrowid,)).fetchone()
    finally:
        conn.close()
    return _record_of(row)


# ----- the export ----------------------------------------------------------------------------


def _check_window_args(start_seconds: Any, end_seconds: Any, max_seconds: Any) -> float:
    if not _is_number(start_seconds) or not _is_number(end_seconds):
        raise ValueError("start_seconds and end_seconds must be finite numbers")
    if not end_seconds > start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")
    if max_seconds is None:
        return config.export_max_seconds()
    if not _is_number(max_seconds) or not 0 < max_seconds <= 3600:
        raise ValueError(f"max_seconds must be a number from 0 to 3600, got {max_seconds!r}")
    return float(max_seconds)


def _prepare_drive(db: Path, storage: Any, drive_id: str) -> tuple[Path, Any]:
    """Refusal checks for the drive; returns (raw video path, ffprobe info)."""
    if not _DRIVE_RE.fullmatch(drive_id):
        raise ExportNotFoundError(f"drive not found: {drive_id}")
    conn = get_connection(db)
    try:
        drive = conn.execute("SELECT status, video_path, video_deleted_at FROM drives WHERE id = ?",
                             (drive_id,)).fetchone()
    finally:
        conn.close()
    if drive is None:
        raise ExportNotFoundError(f"drive not found: {drive_id}")
    if drive["status"] == "processing":
        raise ExportRefusedError(f"drive {drive_id} is processing; wait for it to finish")
    if drive["video_deleted_at"]:
        raise ExportRefusedError(f"the raw video of drive {drive_id} was deleted, so it cannot be exported")
    try:
        path = Path(storage.local_path(drive["video_path"])) if drive["video_path"] else None
        found = path is not None and path.is_file()
    except (ValueError, OSError):
        found = False
    if not found:
        raise ExportRefusedError(f"the raw video of drive {drive_id} is missing")
    try:
        info = probe_video(path)
    except (InvalidMedia, RuntimeError, OSError):
        raise ExportRefusedError(f"the raw video of drive {drive_id} cannot be read") from None
    if info.stream_index is None:
        raise ExportRefusedError(f"the raw video of drive {drive_id} has no video stream")
    return path, info


def export_segment(
    drive_id: str,
    start_seconds: float,
    end_seconds: float,
    *,
    hazard_id: int | None = None,
    track_key: str | None = None,
    provider: Any = None,
    db_path: str | Path | None = None,
    storage: Any = None,
    max_seconds: float | None = None,
    center_seconds: float | None = None,
) -> ExportRecord:
    """Export the window ``[start_seconds, end_seconds]`` of a drive's raw video as a blurred side-by-side clip.
    See the module docstring."""
    if drive_id is None or not str(drive_id).strip():
        raise ValueError("drive_id must not be blank")
    drive_id = str(drive_id).strip()
    cap = _check_window_args(start_seconds, end_seconds, max_seconds)
    if hazard_id is not None and (isinstance(hazard_id, bool) or not isinstance(hazard_id, int)):
        raise ValueError("hazard_id must be an integer")
    if track_key is not None and not isinstance(track_key, str):
        raise ValueError("track_key must be a string")
    if storage is None:
        from .storage import get_storage

        storage = get_storage()
    db = init_db(db_path)
    prompts = list(load_privacy_prompts())
    dilate = config.blur_dilate_px()

    src, info = _prepare_drive(db, storage, drive_id)
    start = max(0.0, float(start_seconds))
    end = float(end_seconds)
    if info.duration_seconds is not None and info.duration_seconds > 0:
        end = min(end, float(info.duration_seconds))
    if end - start > cap:  # cap AFTER clipping: around the centre when given, else keep the start
        if center_seconds is not None and _is_number(center_seconds) and start <= center_seconds <= end:
            start += (center_seconds - start) * (1.0 - cap / (end - start))
            start = max(0.0, start)
        end = start + cap
    if not end > start + 1e-3:
        raise ExportRefusedError("the requested window is outside the video")

    tmp = Path(tempfile.mkdtemp(prefix="roadscout-export-"))
    part: Path | None = None
    final: Path | None = None
    stage = "cut"
    try:
        try:
            stage = "cut"
            fps = _probe_fps(src, info.stream_index)
            segment = tmp / SEGMENT_NAME
            _cut_segment(src, info.stream_index, start, end - start, fps, segment)
            width, height = _probe_size(segment)

            stage = "provider"
            if provider is None:
                provider = get_provider()
            result = provider.detect(str(segment), list(prompts))  # the ONE model call, on the segment only

            stage = "blur"
            plan = _BlurPlan(result, fps, width, height)
            conn = get_connection(db)
            try:
                rows = conn.execute(
                    "SELECT id, frame_timestamp, hazard_class, track_id, track_key, box, mask FROM detections "
                    "WHERE drive_id = ? AND frame_timestamp >= ? AND frame_timestamp <= ? "
                    "ORDER BY frame_timestamp, id", (drive_id, start - 1.0, end + 1.0)).fetchall()
            finally:
                conn.close()
            notes = _Annotations(rows, track_key, width, height)

            stage = "encode"
            out_path = tmp / OUTPUT_NAME
            frames = _render(segment, out_path, width, height, fps, plan, notes, start, dilate, tmp)

            stage = "save"
            name = f"hazard-{hazard_id}-{uuid.uuid4().hex[:8]}.mp4" if hazard_id is not None \
                else f"segment-{uuid.uuid4().hex[:8]}.mp4"
            folder = Path(storage.local_path(f"{drive_id}/{EXPORTS_DIR_NAME}"))
            folder.mkdir(parents=True, exist_ok=True)
            final = folder / name
            part = folder / (name + ".part")
            shutil.copyfile(out_path, part)
            os.replace(part, final)
            part = None
            key = f"{drive_id}/{EXPORTS_DIR_NAME}/{name}"
            record = _insert_record(db, drive_id, hazard_id, start, start + frames / float(fps), key, "ok", None)
            final = None  # registered: keep it
            logger.info("exported %d frames of drive %s as export %s", frames, drive_id, record.id)
            return record
        except BaseException as exc:
            _unlink(part)
            _unlink(final)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, _StageError):
                stage, message = exc.stage, str(exc)
            elif isinstance(exc, (ProviderError, TimeoutError)) or stage == "provider":
                stage, message = "provider", f"the provider failed: {_short_error(exc, tmp, src)}"
            else:
                message = f"{stage} failed: {_short_error(exc, tmp, src)}"
            message = _short_error(message, tmp, src)
            logger.warning("export of drive %s failed at %s: %s", drive_id, stage, message)
            try:
                record = _insert_record(db, drive_id, hazard_id, start, end, None, "failed", message)
            except Exception:  # the registry itself is broken: still fail closed
                logger.exception("could not record the failed export")
                record = None
            raise ExportFailedError(message, record=record, stage=stage) from None
    finally:
        _rmtree(tmp)


def _pick_member(db: Path, hazard_id: int) -> Any:
    """The member candidate that supplies the hazard photo (see the module docstring), or None."""
    conn = get_connection(db)
    try:
        return conn.execute(
            "SELECT c.id, c.drive_id, c.track_key, c.best_timestamp, c.score, "
            "COALESCE(v.confidence, 0) AS confidence, "
            "(COALESCE(v.accepted, 0) = 1 AND COALESCE(v.is_hazard, 0) = 1 "
            " AND (c.blur_status IS NULL OR c.blur_status != 'failed')) AS eligible "
            "FROM candidates c LEFT JOIN verdicts v ON v.candidate_id = c.id "
            "WHERE c.hazard_id = ? "
            "ORDER BY c.score DESC, eligible DESC, confidence DESC, c.drive_id, c.track_key, "
            "c.best_timestamp, c.id LIMIT 1", (hazard_id,)).fetchone()
    finally:
        conn.close()


def export_hazard_clip(
    hazard_id: int,
    *,
    provider: Any = None,
    db_path: str | Path | None = None,
    storage: Any = None,
    pre_seconds: float | None = None,
    post_seconds: float | None = None,
    max_seconds: float | None = None,
    before_seconds: float | None = None,
    after_seconds: float | None = None,
) -> ExportRecord:
    """Export the short clip around a hazard's best frame. See the module docstring. ``before_seconds`` /
    ``after_seconds`` are aliases of ``pre_seconds`` / ``post_seconds`` (give one name or the other, not both)."""
    if before_seconds is not None:
        if pre_seconds is not None:
            raise ValueError("give pre_seconds or before_seconds, not both")
        pre_seconds = before_seconds
    if after_seconds is not None:
        if post_seconds is not None:
            raise ValueError("give post_seconds or after_seconds, not both")
        post_seconds = after_seconds
    if isinstance(hazard_id, bool) or not isinstance(hazard_id, int):
        raise ValueError("hazard_id must be an integer")
    pre = config.export_pre_seconds() if pre_seconds is None else pre_seconds
    post = config.export_post_seconds() if post_seconds is None else post_seconds
    for name, value in (("pre_seconds", pre), ("post_seconds", post)):
        if not _is_number(value) or not 0 <= value <= 3600:
            raise ValueError(f"{name} must be a number from 0 to 3600, got {value!r}")
    if not pre + post > 0:
        raise ValueError("pre_seconds and post_seconds must not both be 0")
    cap = _check_window_args(0.0, 1.0, max_seconds)
    db = init_db(db_path)

    conn = get_connection(db)
    try:
        hazard = conn.execute("SELECT id FROM hazards WHERE id = ?", (hazard_id,)).fetchone()
    finally:
        conn.close()
    if hazard is None:
        raise ExportNotFoundError(f"hazard not found: {hazard_id}")
    member = _pick_member(db, hazard_id)
    if member is None:
        raise ExportRefusedError(f"hazard {hazard_id} has no member candidate to export")
    t = member["best_timestamp"]
    if not _is_number(t) or t < 0:
        raise ExportRefusedError(f"the best frame of hazard {hazard_id} has no usable time")
    t = float(t)
    return export_segment(
        member["drive_id"], t - pre, t + post, hazard_id=hazard_id, track_key=member["track_key"],
        provider=provider, db_path=db, storage=storage, max_seconds=cap, center_seconds=t,
    )


# ----- command line --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m backend.export",
                                     description="Export a blurred side-by-side raw/annotated clip of a hazard.")
    parser.add_argument("--hazard", type=int, required=True, help="hazard id")
    parser.add_argument("--pre", "--before", type=float, default=None, help="seconds before the best frame")
    parser.add_argument("--post", "--after", type=float, default=None, help="seconds after the best frame")
    parser.add_argument("--max", type=float, default=None, dest="max_seconds", help="hard cap on the clip length")
    args = parser.parse_args(argv)
    try:
        record = export_hazard_clip(args.hazard, pre_seconds=args.pre, post_seconds=args.post,
                                    max_seconds=args.max_seconds)
    except ExportFailedError as exc:
        print(json.dumps({"error": str(exc), "stage": exc.stage,
                          "export": exc.record.to_dict() if exc.record else None}), file=sys.stderr)
        return 2
    except (ExportError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(record.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
