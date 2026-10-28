"""SAM 3.1 detection per chunk (task 7): send every chunk to the provider, store the raw output.

    from backend.detection import run_detection, load_detections
    summary = run_detection(drive_id, preprocess_result)       # fake provider unless configured
    rows = load_detections(drive_id)

``run_detection(drive_id, chunks, *, provider=None, db_path=None, prompts=None)``

- ``chunks``: a ``PreprocessResult`` from ``preprocess_video``; or a path to its
  ``frame_timestamps.json`` (or the folder holding it); or a sequence of ``Chunk`` objects
  (index, path, start_frame, frame_count, start_time). Chunks are processed in index order.
- ``provider``: any ``ModelProvider``. Default ``get_provider()``, so the fake provider unless
  ROADSCOUT_PROVIDER says otherwise. This module makes no network calls itself and logs no keys.
- ``prompts``: a ``HazardPrompts`` (see ``backend/hazard_prompts.py``). Default: load the config
  file (``config/hazard_prompts.json``, override with ROADSCOUT_HAZARD_PROMPTS). The first
  phrasing of every class is sent, in one ``provider.detect(chunk_path, prompts)`` call per chunk.
- ``db_path``: SQLite file (default ``config.db_path()``); ``init_db`` is run first.

Mapping to the ``detections`` table (one row per ``Detection``; columns in ``backend/db.py``):

- ``chunk`` is the chunk index. ``frame_in_chunk`` is the frame index the provider reported,
  which counts from 0 at the start of the chunk file.
- ``frame_index`` is the global frame index: chunk ``start_frame`` + ``frame_in_chunk``.
- ``frame_timestamp`` is the global time in seconds from the source video start, looked up in
  ``frame_timestamps.json`` by global frame index. If the provider reports a frame past the
  chunk's last frame, or no timestamps record is available (a plain ``Chunk`` list), the time is
  chunk ``start_time`` + the provider's chunk-relative timestamp instead.
- ``class_prompt`` is the prompt text the provider says matched; ``hazard_class`` is the class
  that phrasing belongs to in the prompts config (the prompt text itself if the provider returns
  a phrasing the config does not know).
- ``track_id`` is the provider's id (unique only inside its chunk); ``track_key`` is
  ``"<chunk>:<track_id>"``, unique across the drive. Tracks are not stitched across chunks here.
- ``box`` is JSON ``[x1, y1, x2, y2]`` in chunk pixels; ``mask`` is the RLE dict as JSON text
  (NULL if the provider gave none); ``score`` is the provider's score.

Skipping stationary stretches (task 13, ``backend/stationary.py``). Extra optional parameters:
``skip_stationary`` (default ``config.skip_stationary_enabled()``, env ROADSCOUT_SKIP_STATIONARY,
on), ``stationary_speed_mps`` (default env ROADSCOUT_STATIONARY_SPEED_MPS, 0.5), ``gpx_track`` (a
``GpxTrack``, a drive record, or None to read the drive's row from the ``drives`` table of
``db_path``), ``video_start``, ``clock_offset_seconds`` (override the track's own) and ``storage``
(to find the GPX file; default ``get_storage()``). The speed of each sampled frame is the GPX speed at
its global timestamp (GPX time = started_at + clock_offset_seconds + timestamp). A frame is
stationary when its speed is below the threshold (exactly at the threshold counts as moving).

- A chunk whose frames are ALL stationary is not sent to the provider at all (no ``detect`` call, no
  cost). It writes no rows, appears in ``detections_per_chunk`` with 0 and in ``skipped_chunks``.
- A chunk with SOME stationary frames is still sent once, unchanged (chunks are never re-encoded),
  and any detection the provider reports on a stationary frame is dropped before storing
  (``stationary_detections_dropped`` counts them). Moving frames keep their global ``frame_index``
  and ``frame_timestamp``, so aggregation, geotagging and verification work as before.
- Fail open: no GPX, an unreadable/unparseable GPX, no frame timestamps, or a frame outside the GPX
  range means "moving": those frames are never dropped and the run still succeeds
  (``stationary_note`` says why). Disabled (``skip_stationary=False``) behaves exactly as before.
- The summary carries ``frames_total``, ``stationary_frames_skipped``, ``chunks_total``, ``chunks_sent``,
  ``stationary_chunks_skipped``. ``chunk_count`` is the number of chunks sent to the provider.

Resuming (task 26c). ``resume=True`` (the pipeline always passes it) changes only how results are stored:
each chunk's rows are committed in their own transaction together with a ``detection_chunks`` record
(chunk done, provider frame records, detections dropped on stationary frames). A chunk already
recorded as done is NOT sent to the provider again; rows of chunks not recorded as done are removed
first. The summary is computed from the stored rows, so it is identical to an uninterrupted run's
(stationary skips are recomputed from the GPX each time). If the provider fails at chunk k,
``DetectionError`` is raised and chunks before k stay stored and done. Without ``resume`` nothing below
changes (all-or-nothing, one transaction, the drive's rows and chunk records replaced).

Rules (without ``resume``):

- An empty result for a chunk is success and adds no rows. No chunks at all is also success.
- Idempotent per drive: a run first collects every chunk's detections in memory, then in ONE
  transaction deletes that drive's old rows and inserts the new ones. Re-running replaces the
  drive's detections; other drives are untouched.
- All-or-nothing: if the provider raises for any chunk (after its own retries), or returns data
  that cannot be mapped, nothing is written, the drive's previous detections stay as they were,
  and ``DetectionError`` is raised naming the chunk (``chunk_index``, ``chunk_path``; the
  original error is ``__cause__``). Chunks processed before the failure are not kept.
- Bad arguments (blank ``drive_id``, unusable ``chunks``) raise ``ValueError`` / ``TypeError``;
  an invalid prompts file raises ``PromptConfigError``. All of these happen before any provider call.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from . import config
from .db import get_connection, init_db
from .hazard_prompts import HazardPrompts, load_hazard_prompts
from .providers import ModelProvider, ProviderError, get_provider
from .stationary import find_stationary_frames, resolve_track, validate_threshold

logger = logging.getLogger(__name__)

_INSERT = (
    "INSERT INTO detections (drive_id, chunk, frame_index, frame_in_chunk, frame_timestamp, "
    "hazard_class, class_prompt, track_id, track_key, box, mask, score) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


class DetectionError(ProviderError):
    """Detection failed for one chunk; nothing was stored. ``__cause__`` is the original error."""

    def __init__(self, message: str, *, chunk_index: int | None = None, chunk_path: str | None = None):
        super().__init__(message)
        self.chunk_index = chunk_index
        self.chunk_path = chunk_path


@dataclass
class DetectionSummary:
    """What a ``run_detection`` call stored."""

    drive_id: str
    chunk_count: int  # chunks sent to the provider (same as chunks_sent)
    frame_count: int  # frame records the provider returned
    detection_count: int  # rows written
    track_count: int  # distinct track_key values
    prompts: list[str] = field(default_factory=list)  # the prompt texts sent to every chunk
    detections_per_chunk: dict[int, int] = field(default_factory=dict)  # chunk index -> rows (all chunks)
    # Stationary-stretch counts (task 13). With the filter off or no usable GPX: stationary_frames_skipped
    # and stationary_chunks_skipped are 0 and chunks_sent == chunks_total.
    frames_total: int = 0  # sampled frames in all chunks
    stationary_frames_skipped: int = 0  # of those, frames below the speed threshold
    chunks_total: int = 0  # all chunks of the drive
    chunks_sent: int = 0  # chunks sent to the provider
    stationary_chunks_skipped: int = 0  # chunks not sent because every frame was stationary
    skipped_chunks: list[int] = field(default_factory=list)  # their chunk indexes
    stationary_detections_dropped: int = 0  # detections dropped on stationary frames of sent chunks
    stationary_note: str | None = None  # why the filter was off or the GPS not (fully) usable


@dataclass
class _ChunkInfo:
    index: int
    path: str
    start_frame: int
    frame_count: int
    start_time: float
    end_time: float | None = None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _info_from_chunk(c: Any) -> _ChunkInfo:
    try:
        return _ChunkInfo(int(c.index), str(c.path), int(c.start_frame), int(c.frame_count),
                          float(c.start_time), _optional_float(getattr(c, "end_time", None)))
    except (AttributeError, TypeError, ValueError):
        raise TypeError(
            "chunks must be a PreprocessResult, a frame_timestamps.json path, or a sequence of "
            f"Chunk objects (index, path, start_frame, frame_count, start_time); got {c!r}"
        ) from None


def _read_record(path: Path) -> tuple[list[_ChunkInfo], dict[int, float]]:
    if path.is_dir():
        path = path / "frame_timestamps.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        base = path.parent
        infos = [
            _ChunkInfo(int(c["index"]), str(base / c["file"]), int(c["start_frame"]),
                       int(c["frame_count"]), float(c["start_time"]), _optional_float(c.get("end_time")))
            for c in doc["chunks"]
        ]
        stamps = {int(f["index"]): float(f["timestamp"]) for f in doc["frames"]}
    except FileNotFoundError:
        raise ValueError(f"frame timestamps file not found: {path}") from None
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError(f"cannot read frame timestamps file {path}: {e}") from None
    return infos, stamps


def _normalise_chunks(chunks: Any) -> tuple[list[_ChunkInfo], dict[int, float]]:
    """Return the chunks in index order and the global frame index -> timestamp map."""
    if hasattr(chunks, "chunks") and hasattr(chunks, "frame_timestamps_path"):  # PreprocessResult
        _, stamps = _read_record(Path(chunks.frame_timestamps_path))
        infos = [_info_from_chunk(c) for c in chunks.chunks]
    elif isinstance(chunks, (str, Path)):
        infos, stamps = _read_record(Path(chunks))
    elif isinstance(chunks, Sequence) and not isinstance(chunks, (bytes, bytearray)):
        infos, stamps = [_info_from_chunk(c) for c in chunks], {}
    else:
        raise TypeError(
            "chunks must be a PreprocessResult, a frame_timestamps.json path, or a sequence of "
            f"Chunk objects; got {type(chunks).__name__}"
        )
    infos.sort(key=lambda c: c.index)
    if len({c.index for c in infos}) != len(infos):
        raise ValueError("chunk indexes must be unique")
    return infos, stamps


def _map_chunk(drive_id: str, info: _ChunkInfo, result: Any, stamps: dict[int, float],
               prompts: HazardPrompts) -> tuple[list[tuple], int]:
    """Turn one chunk's DetectionResult into table rows. Returns (rows, frame records seen)."""
    rows: list[tuple] = []
    for frame in result.frames:
        k = frame.frame_index
        global_index = info.start_frame + k
        stamp = stamps.get(global_index) if k < info.frame_count else None
        if stamp is None:
            stamp = info.start_time + float(frame.timestamp)
        for det in frame.detections:
            hazard_class = prompts.class_for_prompt(det.prompt) or det.prompt
            rows.append((
                drive_id, info.index, global_index, k, round(float(stamp), 6),
                hazard_class, det.prompt, int(det.track_id), f"{info.index}:{det.track_id}",
                json.dumps(list(det.box)),
                json.dumps(det.mask, separators=(",", ":")) if det.mask is not None else None,
                float(det.score),
            ))
    return rows, len(result.frames)


def run_detection(
    drive_id: str,
    chunks: Any,
    *,
    provider: ModelProvider | None = None,
    db_path: str | Path | None = None,
    prompts: HazardPrompts | None = None,
    skip_stationary: bool | None = None,
    stationary_speed_mps: float | None = None,
    gpx_track: Any = None,
    video_start: Any = None,
    clock_offset_seconds: float | None = None,
    storage: Any = None,
    resume: bool = False,
) -> DetectionSummary:
    """Detect hazards in every chunk and store the raw output. See the module docstring.

    ``resume=True`` commits each chunk as soon as it is done and skips chunks already recorded as done
    (see "Resuming" in the module docstring); the default keeps the all-or-nothing behaviour.

    Stationary frames (GPS speed below ``stationary_speed_mps``) are skipped: a chunk with only
    stationary frames is not sent to the provider; a chunk with some is sent once, unchanged, and the
    detections on its stationary frames are dropped before storing. Anything unknown about the GPS
    counts as moving.
    """
    if drive_id is None or not str(drive_id).strip():
        raise ValueError("drive_id must not be blank")
    drive_id = str(drive_id).strip()
    if skip_stationary is not None and not isinstance(skip_stationary, bool):
        raise TypeError(f"skip_stationary must be True, False or None, got {skip_stationary!r}")
    skip_enabled = config.skip_stationary_enabled() if skip_stationary is None else skip_stationary
    if stationary_speed_mps is not None:
        threshold = validate_threshold(stationary_speed_mps)  # bad values fail even when the filter is off
    else:
        threshold = validate_threshold(config.stationary_speed_mps()) if skip_enabled else 0.0
    infos, stamps = _normalise_chunks(chunks)
    prompts = prompts if prompts is not None else load_hazard_prompts()
    sent = prompts.default_prompts()
    provider = provider if provider is not None else get_provider()

    stationary: set[int] = set()
    note: str | None = None
    if not skip_enabled:
        note = "stationary filter is off"
    else:
        frame_times = _frame_times(infos, stamps)
        if not frame_times:
            note = "no frame timestamps, so no frame was treated as stationary"
        else:
            track, note = resolve_track(gpx_track, drive_id, init_db(db_path), storage)
            if track is not None:
                analysis = find_stationary_frames(
                    frame_times, track, threshold_mps=threshold,
                    video_start=video_start, clock_offset_seconds=clock_offset_seconds,
                )
                stationary, note = analysis.frames, analysis.note
        if note:
            logger.info("stationary filter for drive %s: %s", drive_id, note)

    target = init_db(db_path)
    done: set[int] = set()
    if resume:
        done = _forget_unfinished(drive_id, target, {i.index for i in infos})

    all_rows: list[tuple] = []
    per_chunk: dict[int, int] = {}
    skipped: list[int] = []
    dropped = 0
    frame_total = 0
    for info in infos:
        if _all_stationary(info, stationary):
            per_chunk[info.index] = 0
            skipped.append(info.index)
            continue
        if info.index in done:  # committed by an earlier run: no provider call
            continue
        try:
            result = provider.detect(info.path, sent)
            rows, n_frames = _map_chunk(drive_id, info, result, stamps, prompts)
            chunk_dropped = 0
            if stationary:
                kept = [r for r in rows if not (r[3] < info.frame_count and r[2] in stationary)]
                chunk_dropped = len(rows) - len(kept)
                rows = kept
        except Exception as e:
            kept_note = "chunks before it were kept" if resume else "nothing was stored"
            raise DetectionError(
                f"detection failed for chunk {info.index} ({Path(info.path).name}) of drive "
                f"{drive_id}; {kept_note}: {type(e).__name__}: {e}",
                chunk_index=info.index, chunk_path=info.path,
            ) from e
        if resume:
            _commit_chunk(drive_id, info.index, rows, n_frames, chunk_dropped, target)
        all_rows.extend(rows)
        per_chunk[info.index] = len(rows)
        frame_total += n_frames
        dropped += chunk_dropped

    if resume:  # the stored rows are the truth: this run's numbers must equal an uninterrupted run's
        per_chunk, detection_count, track_count, frame_total, dropped = _stored_totals(
            drive_id, target, [i.index for i in infos])
        for index in skipped:
            per_chunk[index] = 0
    else:
        conn = get_connection(target)
        try:
            with conn:  # one transaction: commit on success, roll back on any error
                conn.execute("DELETE FROM detections WHERE drive_id = ?", (drive_id,))
                conn.execute("DELETE FROM detection_chunks WHERE drive_id = ?", (drive_id,))
                conn.executemany(_INSERT, all_rows)
        finally:
            conn.close()
        detection_count, track_count = len(all_rows), len({r[8] for r in all_rows})

    summary = DetectionSummary(
        drive_id=drive_id,
        chunk_count=len(infos) - len(skipped),
        frame_count=frame_total,
        detection_count=detection_count,
        track_count=track_count,
        prompts=list(sent),
        detections_per_chunk=per_chunk,
        frames_total=sum(i.frame_count for i in infos),
        stationary_frames_skipped=len(stationary),
        chunks_total=len(infos),
        chunks_sent=len(infos) - len(skipped),
        stationary_chunks_skipped=len(skipped),
        skipped_chunks=skipped,
        stationary_detections_dropped=dropped,
        stationary_note=note,
    )
    logger.info("detection for drive %s: %d/%d chunks sent, %d/%d frames stationary, %d detections, %d tracks",
                drive_id, summary.chunks_sent, summary.chunks_total, summary.stationary_frames_skipped,
                summary.frames_total, summary.detection_count, summary.track_count)
    return summary


def _forget_unfinished(drive_id: str, target: Path, chunk_indexes: set[int]) -> set[int]:
    """Resume: the chunks recorded as done. Detection rows of any other chunk (and records of chunks that
    no longer exist) are removed, so every stored row belongs to a chunk that finished."""
    conn = get_connection(target)
    try:
        with conn:
            done = {r["chunk"] for r in conn.execute(
                "SELECT chunk FROM detection_chunks WHERE drive_id = ?", (drive_id,))} & chunk_indexes
            marks = ",".join("?" for _ in done)
            keep = f" AND chunk NOT IN ({marks})" if done else ""
            conn.execute("DELETE FROM detections WHERE drive_id = ?" + keep, (drive_id, *sorted(done)))
            conn.execute("DELETE FROM detection_chunks WHERE drive_id = ?" + keep, (drive_id, *sorted(done)))
        return done
    finally:
        conn.close()


def _commit_chunk(drive_id: str, chunk: int, rows: list[tuple], frames: int, dropped: int, target: Path) -> None:
    """One transaction: replace this chunk's rows and record the chunk as done."""
    conn = get_connection(target)
    try:
        with conn:
            conn.execute("DELETE FROM detections WHERE drive_id = ? AND chunk = ?", (drive_id, chunk))
            conn.executemany(_INSERT, rows)
            conn.execute(
                "INSERT OR REPLACE INTO detection_chunks (drive_id, chunk, frames, dropped) VALUES (?, ?, ?, ?)",
                (drive_id, chunk, frames, dropped))
    finally:
        conn.close()


def _stored_totals(drive_id: str, target: Path, chunk_indexes: list[int]) -> tuple[dict[int, int], int, int, int, int]:
    """(rows per chunk, rows, distinct tracks, provider frame records, dropped detections) from the database."""
    conn = get_connection(target)
    try:
        per = {r["chunk"]: r["n"] for r in conn.execute(
            "SELECT chunk, COUNT(*) AS n FROM detections WHERE drive_id = ? GROUP BY chunk", (drive_id,))}
        tracks = conn.execute(
            "SELECT COUNT(DISTINCT track_key) FROM detections WHERE drive_id = ?", (drive_id,)).fetchone()[0]
        frames, dropped = conn.execute(
            "SELECT COALESCE(SUM(frames), 0), COALESCE(SUM(dropped), 0) FROM detection_chunks WHERE drive_id = ?",
            (drive_id,)).fetchone()
    finally:
        conn.close()
    per_chunk = {i: per.get(i, 0) for i in chunk_indexes}
    return per_chunk, sum(per_chunk.values()), int(tracks), int(frames), int(dropped)


def _frame_times(infos: list[_ChunkInfo], stamps: dict[int, float]) -> dict[int, float]:
    """Global frame index -> seconds from the video start, for every frame we can time.

    From ``frame_timestamps.json`` when present; otherwise spread evenly from the chunk's
    ``start_time`` to its ``end_time`` when the chunk objects carry one; else the frame is left out
    (and so is never treated as stationary).
    """
    out: dict[int, float] = {}
    for info in infos:
        step = None
        if info.end_time is not None and info.frame_count > 0 and info.end_time > info.start_time:
            step = (info.end_time - info.start_time) / info.frame_count
        for k in range(info.frame_count):
            g = info.start_frame + k
            if g in stamps:
                out[g] = stamps[g]
            elif step is not None:
                out[g] = info.start_time + k * step
    return out


def _all_stationary(info: _ChunkInfo, stationary: set[int]) -> bool:
    """True when the chunk has frames and every one of them is stationary."""
    if not stationary or info.frame_count <= 0:
        return False
    return all(info.start_frame + k in stationary for k in range(info.frame_count))


def load_detections(drive_id: str, *, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """The stored detections of a drive as dicts, ordered by chunk, frame, then row id.

    Keys are the column names; ``box`` is a list of four floats and ``mask`` a decoded RLE dict
    (or None).
    """
    conn = get_connection(init_db(db_path))
    try:
        cur = conn.execute(
            "SELECT * FROM detections WHERE drive_id = ? ORDER BY chunk, frame_index, id",
            (str(drive_id),),
        )
        out = []
        for row in cur.fetchall():
            d = dict(row)
            d["box"] = json.loads(d["box"])
            d["mask"] = json.loads(d["mask"]) if d["mask"] is not None else None
            out.append(d)
        return out
    finally:
        conn.close()
