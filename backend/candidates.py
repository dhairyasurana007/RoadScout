"""Track aggregation, best-frame selection and geotagging (task 8).

    from backend.candidates import aggregate_drive, load_candidates
    summary = aggregate_drive(drive_id, preprocess_result, gpx_track=track)
    rows = load_candidates(drive_id)

``aggregate_drive(drive_id, chunks=None, *, gpx_track=None, video_start=None,
clock_offset_seconds=None, min_frames=None, db_path=None, output_dir=None, storage=None)``

Turns the drive's stored detections (task 7) into one *candidate* per tracked object.

- ``chunks``: how to find the chunk videos. A ``PreprocessResult``, a path to
  ``frame_timestamps.json`` (or the folder holding it), or a sequence of ``Chunk`` objects, exactly
  as ``run_detection`` accepts them. Only needed (and only validated) when the drive has detections.
- ``gpx_track``: a ``GpxTrack`` (from ``GpxTrack.from_drive`` it already knows the drive's
  ``started_at`` and clock offset), or a drive record (dict / sqlite row with ``gpx_path``,
  ``started_at``, ``clock_offset_seconds``), or None to look the drive up in the ``drives`` table of
  ``db_path`` and load its GPX through ``storage`` (default ``get_storage()``). A plain track needs
  ``video_start``. ``video_start`` / ``clock_offset_seconds`` override the track's own values.
- ``min_frames``: a track seen in fewer frames is dropped; exactly ``min_frames`` is kept. Default
  ``config.min_track_frames()`` (env ``ROADSCOUT_MIN_TRACK_FRAMES``, 3). Whole number >= 1.
- ``db_path``: SQLite file (default ``config.db_path()``); ``init_db`` is run first.
- ``output_dir``: where the JPEGs go. Default ``<storage dir>/<drive id>/candidates/`` (inside the
  git-ignored data dir; ``storage.delete_drive`` removes it together with the drive).

Grouping: one candidate per (``hazard_class``, ``track_key``). ``track_key`` is
``"<chunk>:<track_id>"`` (task 7), so tracks are NOT stitched across chunks: an object that the
provider tracks through a chunk boundary becomes one candidate in each chunk, and deduplication
(task 14, same class, within about 25 m) merges them later. Because of that, a track can never
span more than one chunk, and ``frames_seen`` counts distinct frames inside that chunk.

Per candidate:

- ``first_timestamp`` / ``last_timestamp``: smallest and largest ``frame_timestamp`` (global
  seconds from the video start); ``frames_seen``: number of distinct frames the object appears in
  (two detections of one track in one frame count once; the larger one is used).
- ``max_mask_area``: the largest mask area in pixels over the track, counted from the RLE mask. A
  detection with no mask (or an empty mask) uses its box area instead.
- Best frame: the detection with the highest score (below); ties go to the earlier frame. The
  full frame is decoded from the chunk video and saved as a JPEG (``best_frame_path``), plus a JPEG
  of the box region (``crop_path``; NULL only if the box lies entirely outside the frame).
  File names are ``c<chunk>_t<track_id>_<class>_best.jpg`` and ``..._crop.jpg``.

Best-frame score, all parts in 0..1 (weights are the constants below, they sum to 1)::

    score = W_SIZE * size + W_LOW * low + W_SHARP * sharp
    size  = mask area of this detection / largest mask area in the track
    low   = box bottom edge y2 / frame height (1 = at the bottom of the frame = closest to the car)
    sharp = v / (v + SHARPNESS_HALF_POINT), v = variance of the Laplacian of the grayscale box
            region of the actual decoded frame (0 for a flat or empty region)

The box and mask are in the provider's frame pixels; if the decoded frame has a different size the
box is scaled to it (the reference size is the mask's ``size`` when there is one). The candidate's
``score`` is the best frame's score. Everything is deterministic: same inputs, same output.

Geotag: ``GpxTrack.position_at(best_timestamp, video_start=..., clock_offset_seconds=...)`` gives
``lat``, ``lon`` and ``heading`` (degrees, 0 = north, clockwise) and ``geotag_status = 'ok'``.
If the best frame's time is outside the GPX track (nothing is clamped), the candidate is kept with
NULL lat/lon/heading, ``geotag_status = 'out_of_range'`` and the reason in ``geotag_note``. If the
drive has no usable GPX (drive record missing, file missing or unparseable) the status is
``'no_gps'`` with a note. A track given without any start time raises ``ValueError`` up front.

Table ``candidates`` (columns in ``backend/db.py``): ``id``, ``drive_id``, ``hazard_id`` (NULL until
deduplication), ``hazard_class``, ``track_key``, ``track_id``, ``chunk``, ``first_timestamp``,
``last_timestamp``, ``frames_seen``, ``max_mask_area``, ``best_frame_index`` (global),
``best_timestamp``, ``best_box`` (JSON ``[x1, y1, x2, y2]`` in decoded-frame pixels),
``best_frame_path``, ``crop_path``, ``lat``, ``lon``, ``heading``, ``score``, ``geotag_status``,
``geotag_note``. Rows are ordered by ``first_timestamp``, then chunk, then track id, then class.

Rules:

- Idempotent per drive: the drive's old candidates are replaced in ONE transaction (other drives
  are untouched); images of the old candidates that the new run does not reuse are deleted.
- All-or-nothing: if a chunk file is missing or a frame cannot be decoded, ``CandidateError`` is
  raised before the database is touched and the drive's earlier candidates stay as they were.
- No detections (or every track too short) means zero candidates and no error; old candidates of
  the drive are cleared.
- Bad arguments (blank ``drive_id``, bad ``min_frames``) raise ``ValueError`` before any work.
- No network calls, no keys.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from . import config
from .db import get_connection, init_db
from .dedup import refresh_hazards_quietly
from .detection import _normalise_chunks, load_detections
from .gpx_track import GpxError, GpxRangeError, GpxTrack
from .providers.types import rle_area

logger = logging.getLogger(__name__)

# Best-frame score weights (sum to 1). Bigger, lower in the frame and sharper is better.
W_SIZE = 0.4
W_LOW = 0.3
W_SHARP = 0.3
# Laplacian variance at which the sharpness part reaches 0.5 (sharp = v / (v + this)).
SHARPNESS_HALF_POINT = 100.0
JPEG_QUALITY = 90

GEOTAG_OK = "ok"
GEOTAG_OUT_OF_RANGE = "out_of_range"
GEOTAG_NO_GPS = "no_gps"

_INSERT = (
    "INSERT INTO candidates (drive_id, hazard_id, hazard_class, track_key, track_id, chunk, "
    "first_timestamp, last_timestamp, frames_seen, max_mask_area, best_frame_index, best_timestamp, "
    "best_box, best_frame_path, crop_path, lat, lon, heading, score, geotag_status, geotag_note) "
    "VALUES (:drive_id, NULL, :hazard_class, :track_key, :track_id, :chunk, :first_timestamp, "
    ":last_timestamp, :frames_seen, :max_mask_area, :best_frame_index, :best_timestamp, :best_box, "
    ":best_frame_path, :crop_path, :lat, :lon, :heading, :score, :geotag_status, :geotag_note)"
)


class CandidateError(RuntimeError):
    """Aggregation failed (missing chunk video, undecodable frame); nothing was stored."""


@dataclass
class AggregationSummary:
    """What an ``aggregate_drive`` call stored."""

    drive_id: str
    min_frames: int
    detection_count: int  # detection rows read
    track_count: int  # distinct (class, track_key) groups
    dropped_track_count: int  # groups seen in fewer than min_frames frames
    candidate_count: int
    geotagged_count: int  # candidates with geotag_status 'ok'
    candidates: list[dict[str, Any]] = field(default_factory=list)  # as load_candidates returns


# ----- scoring -----------------------------------------------------------------------------


def sharpness_from_variance(variance: float) -> float:
    """Map a Laplacian variance (0 up) to 0..1."""
    v = max(0.0, float(variance))
    return v / (v + SHARPNESS_HALF_POINT)


def frame_score(size: float, low: float, sharp: float) -> float:
    """The documented best-frame score; every input is clamped to 0..1 first."""
    def clamp(x: float) -> float:
        return min(1.0, max(0.0, float(x)))

    return W_SIZE * clamp(size) + W_LOW * clamp(low) + W_SHARP * clamp(sharp)


def _box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _mask_or_box_area(det: dict[str, Any]) -> float:
    mask = det["mask"]
    if mask is not None:
        area = rle_area(mask)
        if area > 0:
            return float(area)
    return _box_area(det["box"])


def _reference_size(det: dict[str, Any], frame_w: int, frame_h: int) -> tuple[float, float]:
    """(width, height) of the frame the box is expressed in: the mask's size, else the real frame."""
    mask = det["mask"]
    if mask is not None:
        h, w = mask["size"]
        return float(w), float(h)
    return float(frame_w), float(frame_h)


def _crop_bounds(det: dict[str, Any], frame_w: int, frame_h: int) -> tuple[int, int, int, int, list[float]]:
    """Box scaled to the decoded frame, as (x1, y1, x2, y2) pixel bounds (may be empty) and floats."""
    ref_w, ref_h = _reference_size(det, frame_w, frame_h)
    sx, sy = frame_w / ref_w, frame_h / ref_h
    x1, y1, x2, y2 = det["box"]
    scaled = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
    ix1 = min(max(int(np.floor(scaled[0])), 0), frame_w)
    iy1 = min(max(int(np.floor(scaled[1])), 0), frame_h)
    ix2 = min(max(int(np.ceil(scaled[2])), 0), frame_w)
    iy2 = min(max(int(np.ceil(scaled[3])), 0), frame_h)
    return ix1, iy1, ix2, iy2, scaled


def _laplacian_variance(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ----- frames ------------------------------------------------------------------------------


def _iter_frames(path: Path, wanted: set[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Decode the chunk sequentially and yield (frame_in_chunk, BGR frame) for the wanted indexes."""
    if not path.is_file():
        raise CandidateError(f"chunk video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise CandidateError(f"cannot open chunk video: {path}")
        last = max(wanted)
        for k in range(last + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                missing = sorted(w for w in wanted if w >= k)
                raise CandidateError(
                    f"chunk video {path.name} ended at frame {k}; cannot read frame(s) {missing}"
                )
            if k in wanted:
                yield k, frame
    finally:
        cap.release()


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise CandidateError(f"could not encode {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(buf.tobytes())
    tmp.replace(path)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower() or "x"


def _dir_name(drive_id: str) -> str:
    """The drive id as a safe folder name (letters, digits, - and _ kept; anything else becomes _)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", drive_id)


# ----- geotag ------------------------------------------------------------------------------


def _resolve_track(gpx_track: Any, drive_id: str, db_path: Path, storage: Any) -> tuple[GpxTrack | None, str | None]:
    """Return (track, None) or (None, reason) when the drive has no usable GPX."""
    if isinstance(gpx_track, GpxTrack):
        return gpx_track, None
    record: Any = None
    if gpx_track is not None:
        if not isinstance(gpx_track, Mapping) and not hasattr(gpx_track, "keys"):
            raise TypeError("gpx_track must be a GpxTrack, a drive record, or None")
        record = gpx_track
    else:
        conn = get_connection(db_path)
        try:
            record = conn.execute("SELECT * FROM drives WHERE id = ?", (drive_id,)).fetchone()
        finally:
            conn.close()
        if record is None:
            return None, f"drive {drive_id} has no record in the drives table, so there is no GPX"
    try:
        return GpxTrack.from_drive(record, storage), None
    except (GpxError, OSError, KeyError, ValueError) as e:
        return None, f"GPX unusable for drive {drive_id}: {type(e).__name__}: {e}"


# ----- main --------------------------------------------------------------------------------


def aggregate_drive(
    drive_id: str,
    chunks: Any = None,
    *,
    gpx_track: Any = None,
    video_start: Any = None,
    clock_offset_seconds: float | None = None,
    min_frames: int | None = None,
    db_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    storage: Any = None,
) -> AggregationSummary:
    """Aggregate a drive's detections into geotagged candidates. See the module docstring."""
    if drive_id is None or not str(drive_id).strip():
        raise ValueError("drive_id must not be blank")
    drive_id = str(drive_id).strip()
    if min_frames is None:
        min_frames = config.min_track_frames()
    if isinstance(min_frames, bool) or not isinstance(min_frames, int) or min_frames < 1:
        raise ValueError(f"min_frames must be a whole number of 1 or more, got {min_frames!r}")

    target = init_db(db_path)
    detections = load_detections(drive_id, db_path=target)

    # Group per (class, track_key); within a group keep one detection per frame (the largest).
    groups: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for det in detections:
        det["area"] = _mask_or_box_area(det)
        frames = groups.setdefault((det["hazard_class"], det["track_key"]), {})
        current = frames.get(det["frame_index"])
        if current is None or (det["area"], det["score"]) > (current["area"], current["score"]):
            frames[det["frame_index"]] = det
    kept = {key: frames for key, frames in groups.items() if len(frames) >= min_frames}

    rows: list[dict[str, Any]] = []
    new_files: set[Path] = set()
    if kept:
        rows, new_files = _build_rows(drive_id, kept, chunks, gpx_track, video_start,
                                      clock_offset_seconds, target, output_dir, storage)

    _store(drive_id, rows, target, new_files)
    candidates = load_candidates(drive_id, db_path=target)
    summary = AggregationSummary(
        drive_id=drive_id,
        min_frames=min_frames,
        detection_count=len(detections),
        track_count=len(groups),
        dropped_track_count=len(groups) - len(kept),
        candidate_count=len(candidates),
        geotagged_count=sum(1 for c in candidates if c["geotag_status"] == GEOTAG_OK),
        candidates=candidates,
    )
    logger.info("aggregation for drive %s: %d detections, %d tracks, %d dropped, %d candidates",
                drive_id, summary.detection_count, summary.track_count,
                summary.dropped_track_count, summary.candidate_count)
    return summary


def _build_rows(drive_id, kept, chunks, gpx_track, video_start, clock_offset_seconds, target,
                output_dir, storage) -> tuple[list[dict[str, Any]], set[Path]]:
    infos, _ = _normalise_chunks(chunks)
    paths = {c.index: Path(c.path) for c in infos}
    for (_, track_key), frames in kept.items():
        chunk = next(iter(frames.values()))["chunk"]
        if chunk not in paths:
            raise CandidateError(f"track {track_key} refers to chunk {chunk}, which is not among the given chunks")

    if storage is None and gpx_track is None:
        from .storage import get_storage

        storage = get_storage()
    track, gps_problem = _resolve_track(gpx_track, drive_id, target, storage)
    start = effective_offset = None
    if track is not None:
        start = video_start if video_start is not None else track.video_start
        if start is None:
            raise ValueError("video_start is required: the GPX track has no drive start time")
        effective_offset = clock_offset_seconds if clock_offset_seconds is not None else (
            track.clock_offset_seconds if track.clock_offset_seconds is not None else 0.0)

    out_dir = Path(output_dir) if output_dir is not None else (
        config.storage_dir() / _dir_name(drive_id) / "candidates")
    out_dir = out_dir.resolve()

    # Pass 1: sharpness of every kept detection, one decode per chunk.
    by_chunk: dict[int, dict[int, list[tuple[tuple[str, str], dict[str, Any]]]]] = {}
    for key, frames in kept.items():
        for det in frames.values():
            by_chunk.setdefault(det["chunk"], {}).setdefault(det["frame_in_chunk"], []).append((key, det))
    sizes: dict[int, tuple[int, int]] = {}  # chunk -> (frame_w, frame_h)
    for chunk, per_frame in by_chunk.items():
        for k, frame in _iter_frames(paths[chunk], set(per_frame)):
            h, w = frame.shape[:2]
            sizes[chunk] = (w, h)
            for _, det in per_frame[k]:
                x1, y1, x2, y2, _scaled = _crop_bounds(det, w, h)
                det["sharp"] = sharpness_from_variance(_laplacian_variance(frame[y1:y2, x1:x2]))

    # Choose the best detection of every track.
    chosen: dict[tuple[str, str], dict[str, Any]] = {}
    for key, frames in kept.items():
        max_area = max(d["area"] for d in frames.values())
        best = None
        for det in frames.values():
            _, ref_h = _reference_size(det, *sizes[det["chunk"]])
            size = det["area"] / max_area if max_area > 0 else 0.0
            low = det["box"][3] / ref_h
            det["score_total"] = frame_score(size, low, det["sharp"])
            if best is None or (det["score_total"], -det["frame_index"]) > (best["score_total"], -best["frame_index"]):
                best = det
        chosen[key] = best

    # Pass 2: decode just the chosen frames and write the images.
    wanted: dict[int, dict[int, list[tuple[str, str]]]] = {}
    for key, det in chosen.items():
        wanted.setdefault(det["chunk"], {}).setdefault(det["frame_in_chunk"], []).append(key)
    images: dict[tuple[str, str], tuple[Path, np.ndarray, Path | None, np.ndarray | None, list[float]]] = {}
    for chunk, per_frame in wanted.items():
        for k, frame in _iter_frames(paths[chunk], set(per_frame)):
            h, w = frame.shape[:2]
            for key in per_frame[k]:
                det = chosen[key]
                x1, y1, x2, y2, scaled = _crop_bounds(det, w, h)
                stem = f"c{det['chunk']:04d}_t{det['track_id']}_{_slug(det['hazard_class'])}"
                crop = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
                images[key] = (out_dir / f"{stem}_best.jpg", frame.copy(),
                               out_dir / f"{stem}_crop.jpg" if crop is not None else None,
                               crop.copy() if crop is not None else None, scaled)

    new_files: set[Path] = set()
    rows: list[dict[str, Any]] = []
    try:
        for key, frames in kept.items():
            det = chosen[key]
            best_path, frame_img, crop_path, crop_img, scaled = images[key]
            _write_jpeg(best_path, frame_img)
            new_files.add(best_path)
            if crop_path is not None:
                _write_jpeg(crop_path, crop_img)
                new_files.add(crop_path)
            lat = lon = heading = None
            status, note = GEOTAG_NO_GPS, gps_problem
            if track is not None:
                try:
                    pos = track.position_at(det["frame_timestamp"], video_start=start,
                                            clock_offset_seconds=effective_offset)
                    lat, lon, heading = pos.latitude, pos.longitude, pos.heading_deg
                    status, note = GEOTAG_OK, None
                except GpxRangeError as e:
                    status, note = GEOTAG_OUT_OF_RANGE, str(e)
                    logger.warning("candidate %s of drive %s is outside the GPX range", key[1], drive_id)
            stamps = [d["frame_timestamp"] for d in frames.values()]
            rows.append({
                "drive_id": drive_id,
                "hazard_class": key[0],
                "track_key": key[1],
                "track_id": det["track_id"],
                "chunk": det["chunk"],
                "first_timestamp": min(stamps),
                "last_timestamp": max(stamps),
                "frames_seen": len(frames),
                "max_mask_area": max(d["area"] for d in frames.values()),
                "best_frame_index": det["frame_index"],
                "best_timestamp": det["frame_timestamp"],
                "best_box": json.dumps([round(v, 2) for v in scaled]),
                "best_frame_path": str(best_path),
                "crop_path": str(crop_path) if crop_path is not None else None,
                "lat": lat, "lon": lon, "heading": heading,
                "score": det["score_total"],
                "geotag_status": status,
                "geotag_note": note,
            })
    except BaseException:
        _remove_files(new_files)
        raise
    rows.sort(key=lambda r: (r["first_timestamp"], r["chunk"], r["track_id"], r["hazard_class"]))
    return rows, new_files


def _remove_files(paths) -> None:
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            logger.warning("could not remove %s", p)


def _store(drive_id: str, rows: list[dict[str, Any]], target: Path, new_files: set[Path]) -> None:
    """Replace the drive's candidates in one transaction; drop images only the old rows used."""
    conn = get_connection(target)
    try:
        old = conn.execute(
            "SELECT best_frame_path, crop_path FROM candidates WHERE drive_id = ?", (drive_id,)
        ).fetchall()
        try:
            with conn:
                conn.execute("DELETE FROM candidates WHERE drive_id = ?", (drive_id,))
                conn.executemany(_INSERT, rows)
        except BaseException:
            _remove_files(new_files - {Path(p) for r in old for p in r if p})
            raise
    finally:
        conn.close()
    stale = [p for r in old for p in r if p]
    keep = {str(p) for p in new_files}
    _remove_files(Path(p) for p in stale if p not in keep)
    if old:
        refresh_hazards_quietly(target)


def load_candidates(drive_id: str, *, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """The stored candidates of a drive as dicts (column names as keys; ``best_box`` a list of four
    floats), ordered by first_timestamp, chunk, track id, class."""
    conn = get_connection(init_db(db_path))
    try:
        cur = conn.execute(
            "SELECT * FROM candidates WHERE drive_id = ? "
            "ORDER BY first_timestamp, chunk, track_id, hazard_class, id",
            (str(drive_id),),
        )
        out = []
        for row in cur.fetchall():
            d = dict(row)
            d["best_box"] = json.loads(d["best_box"])
            out.append(d)
        return out
    finally:
        conn.close()
