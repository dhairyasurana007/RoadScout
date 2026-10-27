"""Skip stationary stretches (task 13): find the sampled frames recorded while the car was not moving.

    from backend.stationary import flag_stationary_frames, find_stationary_frames, frame_speeds, resolve_track

``flag_stationary_frames(timestamps, gpx_track, *, video_start=None, clock_offset_seconds=None,
threshold_mps=None) -> list[bool]``

- ``timestamps``: global video seconds, any order (duplicates fine). The result has one bool per
  timestamp in the same order; True means stationary (drop the frame). GPX time = video_start +
  clock_offset_seconds + t.
- Speed ``< threshold`` is stationary; ``>= threshold`` (including exactly equal) is moving.
- ``gpx_track=None`` or a timestamp outside the track (``GpxRangeError``) gives False for that
  frame; in-range frames are still flagged.
- ``threshold_mps``: default ``config.stationary_speed_mps()``, read at call time from
  ROADSCOUT_STATIONARY_SPEED_MPS (default 0.5 m/s; a garbage env value falls back to 0.5). An explicit
  parameter beats the env var.

``find_stationary_frames(frame_timestamps, track, *, threshold_mps=None, video_start=None,
clock_offset_seconds=None) -> StationaryAnalysis``

- ``frame_timestamps``: ``{global frame index: seconds from the source video start}``, the
  ``frames`` list of ``frame_timestamps.json`` (task 5). Indices and times stay global; nothing is
  renumbered.
- ``track``: a ``GpxTrack`` (task 6) or None. ``video_start`` / ``clock_offset_seconds`` override
  the track's own defaults (a track built with ``GpxTrack.from_drive`` already has them).
- ``threshold_mps``: default ``config.stationary_speed_mps()`` (env ROADSCOUT_STATIONARY_SPEED_MPS,
  default 0.5 m/s). A frame is stationary when its GPS speed is strictly BELOW the threshold, so a
  speed exactly at the threshold counts as moving. Must be a finite number of 0 or more
  (``ValueError`` otherwise; 0 disables the filter in practice because no speed is below 0).
- The result has ``frames`` (the set of stationary global frame indices), ``speeds`` (global frame
  index -> m/s, or None when unknown) and ``note`` (None, or a one-line reason when the GPS could
  not be used).

Speed of a frame is ``GpxTrack.position_at(t).speed_mps`` at the frame's own global timestamp, with
the GPX time = drive ``started_at`` + ``clock_offset_seconds`` + t (the task 6 mapping).
``GpxTrack.positions_at`` is used first for the whole drive; if some frames are outside the GPX the
lookup falls back to one ``position_at`` call per frame.

Fail open (never drop a frame because we do not know): no track, a GPX that cannot be read, a track
without a start time, or a frame whose time is outside the GPX (``GpxRangeError``) all give "moving"
for the affected frames (speed None, not in ``frames``), never an exception.

``resolve_track(gpx_track, drive_id, db_path, storage) -> (GpxTrack | None, note | None)`` turns a
``GpxTrack``, a drive record (dict / sqlite row with ``gpx_path``, ``started_at``,
``clock_offset_seconds``) or None (look the drive up in the ``drives`` table) into a track. Any
problem (no drive row, missing or unparseable GPX) returns ``(None, reason)``; it does not raise.

How detection uses this is documented in ``backend/detection.py`` (``run_detection(...,
skip_stationary=...)``): a chunk whose frames are all stationary is not sent to the provider; a
partly stationary chunk is sent once and the detections on its stationary frames are dropped before
they are stored. Chunk files are never re-encoded.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from . import config
from .db import get_connection
from .gpx_track import GpxRangeError, GpxTrack

logger = logging.getLogger(__name__)

NOTE_MAX_CHARS = 300


@dataclass
class StationaryAnalysis:
    """Result of ``find_stationary_frames``."""

    frames: set[int] = field(default_factory=set)  # global indices of stationary frames
    speeds: dict[int, float | None] = field(default_factory=dict)  # global index -> m/s or None
    note: str | None = None  # one line when the GPS could not be (fully) used, else None


def validate_threshold(value: Any) -> float:
    """A finite number of 0 or more (not a bool), as a float. Raises ValueError / TypeError."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"stationary speed threshold must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"stationary speed threshold must be a finite number of 0 or more, got {value!r}")
    return value


def one_line(text: str, limit: int = NOTE_MAX_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def resolve_track(gpx_track: Any, drive_id: str, db_path: str | Path | None,
                  storage: Any) -> tuple[GpxTrack | None, str | None]:
    """Return ``(track, None)`` or ``(None, reason)``. Never raises (fail open)."""
    try:
        if isinstance(gpx_track, GpxTrack):
            return gpx_track, None
        record = gpx_track
        if record is None:
            conn = get_connection(db_path)
            try:
                record = conn.execute("SELECT * FROM drives WHERE id = ?", (drive_id,)).fetchone()
            finally:
                conn.close()
            if record is None:
                return None, f"drive {drive_id} has no record in the drives table, so there is no GPX"
        return GpxTrack.from_drive(record, storage), None
    except Exception as e:  # unreadable, missing or unparseable GPX: treat every frame as moving
        detail = type(e).__name__ if isinstance(e, OSError) else f"{type(e).__name__}: {e}"  # no file paths
        return None, one_line(f"GPX unusable for drive {drive_id}: {detail}")


def frame_speeds(frame_timestamps: Mapping[int, float], track: GpxTrack | None, *,
                 video_start=None, clock_offset_seconds: float | None = None
                 ) -> tuple[dict[int, float | None], str | None]:
    """Speed in m/s for every frame (None where unknown) and a note. Never raises."""
    items = sorted((int(i), float(t)) for i, t in frame_timestamps.items())
    unknown: dict[int, float | None] = {i: None for i, _ in items}
    if track is None:
        return unknown, "no GPX track, so no frame was treated as stationary"
    kwargs = {"video_start": video_start, "clock_offset_seconds": clock_offset_seconds}
    try:
        positions = track.positions_at([t for _, t in items], **kwargs)
        return {i: _finite(p.speed_mps) for (i, _), p in zip(items, positions)}, None
    except GpxRangeError:
        pass  # some frames are outside the GPX: look each one up on its own
    except Exception as e:
        return unknown, one_line(f"GPX speed lookup failed: {type(e).__name__}: {e}")

    speeds: dict[int, float | None] = {}
    outside = 0
    try:
        for i, t in items:
            try:
                speeds[i] = _finite(track.position_at(t, **kwargs).speed_mps)
            except GpxRangeError:
                speeds[i] = None
                outside += 1
    except Exception as e:
        return unknown, one_line(f"GPX speed lookup failed: {type(e).__name__}: {e}")
    return speeds, f"{outside} frame(s) fall outside the GPX track and were treated as moving"


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def find_stationary_frames(frame_timestamps: Mapping[int, float], track: GpxTrack | None, *,
                           threshold_mps: float | None = None, video_start=None,
                           clock_offset_seconds: float | None = None) -> StationaryAnalysis:
    """Global indices of the frames whose GPS speed is below the threshold. See the module docstring."""
    threshold = validate_threshold(config.stationary_speed_mps() if threshold_mps is None else threshold_mps)
    speeds, note = frame_speeds(frame_timestamps, track, video_start=video_start,
                                clock_offset_seconds=clock_offset_seconds)
    frames = {i for i, s in speeds.items() if s is not None and s < threshold}
    return StationaryAnalysis(frames=frames, speeds=speeds, note=note)


def flag_stationary_frames(timestamps, gpx_track: GpxTrack | None, *, video_start=None,
                           clock_offset_seconds: float | None = None,
                           threshold_mps: float | None = None) -> list[bool]:
    """One bool per timestamp (same order): True when the GPS speed there is below the threshold.
    Unknown (no track, outside the GPX, unreadable) is False, so nothing is dropped by mistake."""
    times = [float(t) for t in timestamps]
    analysis = find_stationary_frames(dict(enumerate(times)), gpx_track, threshold_mps=threshold_mps,
                                      video_start=video_start, clock_offset_seconds=clock_offset_seconds)
    return [i in analysis.frames for i in range(len(times))]
