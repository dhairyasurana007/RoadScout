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

Rules:

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

from .db import get_connection, init_db
from .hazard_prompts import HazardPrompts, load_hazard_prompts
from .providers import ModelProvider, ProviderError, get_provider

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
    chunk_count: int  # chunks sent to the provider
    frame_count: int  # frame records the provider returned
    detection_count: int  # rows written
    track_count: int  # distinct track_key values
    prompts: list[str] = field(default_factory=list)  # the prompt texts sent to every chunk
    detections_per_chunk: dict[int, int] = field(default_factory=dict)  # chunk index -> rows


@dataclass
class _ChunkInfo:
    index: int
    path: str
    start_frame: int
    frame_count: int
    start_time: float


def _info_from_chunk(c: Any) -> _ChunkInfo:
    try:
        return _ChunkInfo(int(c.index), str(c.path), int(c.start_frame), int(c.frame_count), float(c.start_time))
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
                       int(c["frame_count"]), float(c["start_time"]))
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
) -> DetectionSummary:
    """Detect hazards in every chunk and store the raw output. See the module docstring."""
    if drive_id is None or not str(drive_id).strip():
        raise ValueError("drive_id must not be blank")
    drive_id = str(drive_id).strip()
    infos, stamps = _normalise_chunks(chunks)
    prompts = prompts if prompts is not None else load_hazard_prompts()
    sent = prompts.default_prompts()
    provider = provider if provider is not None else get_provider()

    all_rows: list[tuple] = []
    per_chunk: dict[int, int] = {}
    frame_total = 0
    for info in infos:
        try:
            result = provider.detect(info.path, sent)
            rows, n_frames = _map_chunk(drive_id, info, result, stamps, prompts)
        except Exception as e:
            raise DetectionError(
                f"detection failed for chunk {info.index} ({Path(info.path).name}) of drive "
                f"{drive_id}; nothing was stored: {type(e).__name__}: {e}",
                chunk_index=info.index, chunk_path=info.path,
            ) from e
        all_rows.extend(rows)
        per_chunk[info.index] = len(rows)
        frame_total += n_frames

    target = init_db(db_path)
    conn = get_connection(target)
    try:
        with conn:  # one transaction: commit on success, roll back on any error
            conn.execute("DELETE FROM detections WHERE drive_id = ?", (drive_id,))
            conn.executemany(_INSERT, all_rows)
    finally:
        conn.close()

    summary = DetectionSummary(
        drive_id=drive_id,
        chunk_count=len(infos),
        frame_count=frame_total,
        detection_count=len(all_rows),
        track_count=len({r[8] for r in all_rows}),
        prompts=list(sent),
        detections_per_chunk=per_chunk,
    )
    logger.info("detection for drive %s: %d chunks, %d detections, %d tracks",
                drive_id, summary.chunk_count, summary.detection_count, summary.track_count)
    return summary


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
