"""Possibly-fixed detection (task 19): flag a hazard that later drives pass without seeing it.

    from backend.possibly_fixed import check_possibly_fixed
    summary = check_possibly_fixed()      # every hazard, every drive; also runs as pipeline stage 'checking_fixes'

``check_possibly_fixed(*, db_path=None, storage=None, radius_m=None, heading_tolerance_deg=None,
min_misses=None, include_drive_ids=None)`` returns a ``FixCheckSummary``. It is global like deduplication: it looks at
ALL hazards and ALL drives, recomputes everything from the stored data and writes only what changed, so running it
twice changes nothing and never double counts. No network calls, no keys, no model.

Parameters (an explicit value beats the environment variable; an invalid one raises ``ValueError`` before any work):

- ``radius_m``: env ``ROADSCOUT_FIXED_RADIUS_M``, default 25 (metres, >= 0, inclusive);
- ``heading_tolerance_deg``: env ``ROADSCOUT_FIXED_HEADING_TOLERANCE_DEG``, default 45 (0 to 180, inclusive,
  wrap-around aware: 350 and 10 differ by 20);
- ``min_misses``: env ``ROADSCOUT_FIXED_MISSES``, default 3 (a whole number >= 1);
- ``include_drive_ids``: drives that count as finished even though their status is not ``done`` yet (the pipeline
  passes the drive it is processing, because that drive only becomes ``done`` after this stage).
- The moving threshold is ``config.stationary_speed_mps()`` (env ROADSCOUT_STATIONARY_SPEED_MPS, default 0.5 m/s).

What counts. A drive *passes* a hazard when its GPX track (parsed once per run; a drive whose GPX is missing or
unparseable is skipped) goes within ``radius_m`` of the hazard's position (closest approach of each track segment,
so a sparse track still counts), at that point is moving at or above the stationary threshold (a speed exactly at the
threshold is moving), has a heading within ``heading_tolerance_deg`` of the hazard's ``direction`` (a hazard whose
direction is unknown does not restrict the heading), and the time of that point lies inside the drive's recorded span:
``started_at + clock_offset_seconds`` to that plus ``duration_seconds`` (the track's own span when the duration is
unknown). Only drives with status ``done`` (and ``include_drive_ids``) are looked at, and a drive is *later* than a
hazard when its ``started_at`` is after the hazard's ``last_seen`` and after ``fix_check_dismissed_at``.

A drive is a *sighting* when it has a member candidate of the hazard (``candidates.hazard_id``; it resets the count
even if ``last_seen`` was not updated and whether or not the track shows a pass), and a *miss* when it only passes. Walking the later drives in time order (ties by drive id), a miss adds one to the count and a sighting resets
it to 0; a drive is one pass however often it goes by, so re-processing a drive never counts it twice. A hazard that
reaches ``min_misses`` consecutive misses becomes ``possibly_fixed``.

Hazard columns (nullable, see ``backend/db.py``): ``fix_misses`` (the current count, also shown as ``misses``),
``previous_status``, ``possibly_fixed_at`` and ``fix_check_dismissed_at``.

- ``rejected`` hazards are never touched. ``new``, ``approved`` and ``reported`` hazards may be flagged: their status
  becomes ``possibly_fixed``, ``previous_status`` remembers the old one and ``possibly_fixed_at`` is set; ``reviewed_at``,
  edits and ``agency_override`` stay as they are.
- A flagged hazard that is seen again (its count falls below ``min_misses``, for example because a new sighting moved
  ``last_seen`` forward) goes back to ``previous_status`` (``new`` if that is unknown), and ``previous_status`` and
  ``possibly_fixed_at`` are cleared. It is never left ``possibly_fixed`` once seen. A hazard whose status was set to
  ``possibly_fixed`` by hand (``possibly_fixed_at`` NULL) is left alone, except that its count is kept up to date.
- Dismissal: when the user changes the status of a ``possibly_fixed`` hazard (PATCH, ``review_hazard``),
  ``fix_check_dismissed_at`` is set to that moment. Drives that started before it are ignored from then on, so the hazard is not
  flagged again at once; it needs ``min_misses`` further misses.

Everything is written in ONE transaction. Deduplication never writes these columns, so they stay on a hazard that
deduplication adopts after its drive was re-processed.
"""
from __future__ import annotations

import logging
import math
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from . import config
from .db import get_connection, init_db
from .dedup import EARTH_RADIUS_M, heading_diff_deg
from .gpx_track import GpxTrack

logger = logging.getLogger(__name__)

_EPS = 1e-9
_TIME_EPS_S = 1e-3
_CELL_DEG = 0.002  # grid cell of the per-drive segment index (about 220 m of latitude)
_MAX_CELLS_PER_SEGMENT = 64  # a longer segment (a GPS gap) is checked against every hazard instead
_M_PER_DEG_LAT_MIN = 110_000.0
_RESTORE_STATUSES = ("new", "approved", "rejected", "reported")


class FixCheckError(RuntimeError):
    """The check failed; the transaction was rolled back."""


@dataclass
class FixCheckSummary:
    radius_m: float
    heading_tolerance_deg: float
    min_misses: int
    hazard_count: int = 0  # hazards that were checked (not rejected, with a position and a usable last_seen)
    drive_count: int = 0  # finished drives whose GPX was usable
    skipped_drive_count: int = 0  # finished drives left out (GPX missing or unparseable, bad start time)
    flagged_count: int = 0  # hazards that became possibly_fixed in this run
    restored_count: int = 0  # hazards that went back to their previous status in this run
    updated_count: int = 0  # hazard rows written
    possibly_fixed_total: int = 0  # hazards with status possibly_fixed after the run
    flagged_ids: list[int] = field(default_factory=list)
    restored_ids: list[int] = field(default_factory=list)


# ----- arguments ---------------------------------------------------------------------------


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _number_in(value: Any, name: str, low: float, high: float) -> float:
    if not _finite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a number from {low:g} to {high:g}, got {value!r}")
    return float(value)


def _resolve_params(radius_m: Any, heading_tolerance_deg: Any, min_misses: Any) -> tuple[float, float, int]:
    radius = config.fix_radius_m() if radius_m is None else _number_in(radius_m, "radius_m", 0.0, 1e7)
    tol = (config.fix_heading_tolerance_deg() if heading_tolerance_deg is None
           else _number_in(heading_tolerance_deg, "heading_tolerance_deg", 0.0, 180.0))
    if min_misses is None:
        need = config.fix_min_misses()
    else:
        if not _finite(min_misses) or min_misses < 1 or min_misses != int(min_misses):
            raise ValueError(f"min_misses must be a whole number of 1 or more, got {min_misses!r}")
        need = int(min_misses)
    return radius, tol, need


def _parse_time(text: Any) -> datetime | None:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        parsed = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


# ----- one drive's track, indexed ----------------------------------------------------------


class _DriveTrack:
    """A finished drive: its start, its recorded span and its GPS track with a grid index of the track segments, so
    that a hazard only looks at the few segments near it. The GPX is parsed once, here."""

    def __init__(self, drive_id: str, started: datetime, span: tuple[datetime, datetime], track: GpxTrack):
        self.id = drive_id
        self.started = started
        self.span = span
        self.track = track
        samples = track.samples()
        self.rel = [s[0] for s in samples]
        self.lat = [s[1] for s in samples]
        self.lon = [s[2] for s in samples]
        self.segment_count = max(len(samples) - 1, 0)
        self.cells: dict[tuple[int, int], list[int]] = {}
        self.long_segments: list[int] = []
        for i in range(self.segment_count):
            la, lb = sorted((self.lat[i], self.lat[i + 1]))
            oa, ob = sorted((self.lon[i], self.lon[i + 1]))
            i0, i1 = math.floor(la / _CELL_DEG), math.floor(lb / _CELL_DEG)
            j0, j1 = math.floor(oa / _CELL_DEG), math.floor(ob / _CELL_DEG)
            if (i1 - i0 + 1) * (j1 - j0 + 1) > _MAX_CELLS_PER_SEGMENT:
                self.long_segments.append(i)
                continue
            for ci in range(i0, i1 + 1):
                for cj in range(j0, j1 + 1):
                    self.cells.setdefault((ci, cj), []).append(i)
        self.min_lat, self.max_lat = (min(self.lat), max(self.lat)) if self.lat else (0.0, 0.0)
        self.min_lon, self.max_lon = (min(self.lon), max(self.lon)) if self.lon else (0.0, 0.0)

    def _nearby_segments(self, lat0: float, lon0: float, radius_m: float) -> list[int]:
        pad = radius_m + 1.0
        dlat = pad / _M_PER_DEG_LAT_MIN
        dlon = pad / (_M_PER_DEG_LAT_MIN * max(math.cos(math.radians(lat0)), 0.01))
        if (lat0 + dlat < self.min_lat or lat0 - dlat > self.max_lat
                or lon0 + dlon < self.min_lon or lon0 - dlon > self.max_lon):
            return []
        i0, i1 = math.floor((lat0 - dlat) / _CELL_DEG), math.floor((lat0 + dlat) / _CELL_DEG)
        j0, j1 = math.floor((lon0 - dlon) / _CELL_DEG), math.floor((lon0 + dlon) / _CELL_DEG)
        if (i1 - i0 + 1) * (j1 - j0 + 1) > max(self.segment_count, 1):
            return list(range(self.segment_count))  # a huge radius: cheaper to look at every segment
        found = set(self.long_segments)
        for ci in range(i0, i1 + 1):
            for cj in range(j0, j1 + 1):
                found.update(self.cells.get((ci, cj), ()))
        return sorted(found)

    def latest_pass(self, lat0: float, lon0: float, direction: float | None, radius_m: float,
                    tolerance_deg: float, moving_mps: float) -> datetime | None:
        """The time of the last place the drive passed the hazard (see the module docstring), or None."""
        cos0 = math.cos(math.radians(lat0))
        best: datetime | None = None
        for i in self._nearby_segments(lat0, lon0, radius_m):
            ax = math.radians(self.lon[i] - lon0) * cos0 * EARTH_RADIUS_M
            ay = math.radians(self.lat[i] - lat0) * EARTH_RADIUS_M
            bx = math.radians(self.lon[i + 1] - lon0) * cos0 * EARTH_RADIUS_M
            by = math.radians(self.lat[i + 1] - lat0) * EARTH_RADIUS_M
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            u = 0.0 if length2 <= 0 else min(1.0, max(0.0, -(ax * dx + ay * dy) / length2))
            if math.hypot(ax + u * dx, ay + u * dy) > radius_m + _EPS:
                continue
            # The heading is that of this segment, so look slightly inside it (not at its end points).
            at = self.rel[i] + min(max(u, 1e-6), 1.0 - 1e-6) * (self.rel[i + 1] - self.rel[i])
            when = self.track.start_time + timedelta(seconds=at)
            if not (self.span[0] - timedelta(seconds=_TIME_EPS_S) <= when <= self.span[1] + timedelta(seconds=_TIME_EPS_S)):
                continue
            pos = self.track.position_at(at, video_start=self.track.start_time, clock_offset_seconds=0.0, clamp=True)
            if not _finite(pos.speed_mps) or pos.speed_mps < moving_mps:
                continue
            if direction is not None and heading_diff_deg(float(pos.heading_deg), direction) > tolerance_deg + _EPS:
                continue
            if best is None or when > best:
                best = when
        return best


def _load_drives(db: Path, storage: Any, include: set[str]) -> tuple[list[_DriveTrack], int]:
    conn = get_connection(db)
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM drives ORDER BY id")]
    finally:
        conn.close()
    rows = [r for r in rows if r["status"] == "done" or r["id"] in include]
    if rows and storage is None:
        from .storage import get_storage

        storage = get_storage()
    drives: list[_DriveTrack] = []
    skipped = 0
    for row in rows:
        try:
            started = _parse_time(row["started_at"])
            if started is None:
                raise ValueError("bad started_at")
            track = GpxTrack.from_drive(row, storage)
            offset = row.get("clock_offset_seconds")
            offset = float(offset) if _finite(offset) else 0.0
            duration = row.get("duration_seconds")
            if _finite(duration) and duration >= 0:
                begin = started + timedelta(seconds=offset)
                span = (begin, begin + timedelta(seconds=float(duration)))
            else:  # unknown duration: the track's own span
                span = (track.start_time, track.end_time)
            drives.append(_DriveTrack(row["id"], started, span, track))
        except Exception as exc:  # missing or unparseable GPX: this drive cannot say anything about a hazard
            skipped += 1
            logger.info("possibly-fixed check skips drive %s: %s", row.get("id"), type(exc).__name__)
    drives.sort(key=lambda d: (d.started, d.id))
    return drives, skipped


# ----- main --------------------------------------------------------------------------------


def check_possibly_fixed(
    *,
    db_path: str | Path | None = None,
    storage: Any = None,
    radius_m: float | None = None,
    heading_tolerance_deg: float | None = None,
    min_misses: int | None = None,
    include_drive_ids: Iterable[str] | None = None,
) -> FixCheckSummary:
    """Count, for every hazard, the later drives that passed it without a sighting, and flag or restore hazards
    accordingly. See the module docstring."""
    radius, tol, need = _resolve_params(radius_m, heading_tolerance_deg, min_misses)
    include = {str(d) for d in (include_drive_ids or ())}
    moving = config.stationary_speed_mps()
    target = init_db(db_path)
    summary = FixCheckSummary(radius_m=radius, heading_tolerance_deg=tol, min_misses=need)

    drives, summary.skipped_drive_count = _load_drives(target, storage, include)  # parsed outside the transaction
    summary.drive_count = len(drives)
    now_text = datetime.now(timezone.utc).isoformat()

    conn = get_connection(target)
    conn.isolation_level = None  # explicit transaction below: read, decide and write atomically
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            members: dict[int, set[str]] = {}
            for r in conn.execute("SELECT DISTINCT hazard_id, drive_id FROM candidates WHERE hazard_id IS NOT NULL"):
                members.setdefault(r["hazard_id"], set()).add(r["drive_id"])
            starts = [d.started for d in drives]
            for h in [dict(r) for r in conn.execute("SELECT * FROM hazards ORDER BY id")]:
                if h["status"] == "rejected" or not (_finite(h["lat"]) and _finite(h["lon"])):
                    continue
                last_seen = _parse_time(h["last_seen"])
                if last_seen is None:
                    continue
                summary.hazard_count += 1
                misses = _count_misses(h, drives, starts, last_seen, members.get(h["id"], set()),
                                       radius, tol, moving)
                changes = _decide(h, misses, need, now_text)
                if changes is None:
                    continue
                new_status = changes.get("status")
                if new_status == "possibly_fixed" and h["status"] != "possibly_fixed":
                    summary.flagged_count += 1
                    summary.flagged_ids.append(h["id"])
                elif h["status"] == "possibly_fixed" and new_status is not None:
                    summary.restored_count += 1
                    summary.restored_ids.append(h["id"])
                conn.execute(
                    "UPDATE hazards SET " + ", ".join(f"{k} = ?" for k in changes) + " WHERE id = ?",
                    (*changes.values(), h["id"]),
                )
                summary.updated_count += 1
            summary.possibly_fixed_total = conn.execute(
                "SELECT COUNT(*) FROM hazards WHERE status = 'possibly_fixed'").fetchone()[0]
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    except (ValueError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise FixCheckError(f"possibly-fixed check failed; nothing was changed: {type(exc).__name__}: {exc}") from exc
    finally:
        conn.close()
    logger.info("possibly-fixed check: %d hazards, %d drives, %d flagged, %d restored, %d possibly fixed in total",
                summary.hazard_count, summary.drive_count, summary.flagged_count, summary.restored_count,
                summary.possibly_fixed_total)
    return summary


def _count_misses(h: dict[str, Any], drives: list[_DriveTrack], starts: list[datetime], last_seen: datetime,
                  seen_by: set[str], radius: float, tol: float, moving: float) -> int:
    """Consecutive later drives that passed the hazard without a sighting (a sighting resets the count)."""
    dismissed = _parse_time(h.get("fix_check_dismissed_at"))
    direction = float(h["direction"]) % 360.0 if _finite(h["direction"]) else None
    after = max(last_seen, dismissed) if dismissed is not None else last_seen
    misses = 0
    for drive in drives[bisect_left(starts, after):]:
        if drive.started <= after:  # bisect_left keeps a drive that started exactly at `after`: not later
            continue
        if drive.id in seen_by:  # a member candidate is a sighting, whether or not the track shows a pass
            misses = 0
            continue
        if drive.latest_pass(float(h["lat"]), float(h["lon"]), direction, radius, tol, moving) is not None:
            misses += 1
    return misses


def _decide(h: dict[str, Any], misses: int, need: int, now_text: str) -> dict[str, Any] | None:
    """The column changes for one hazard, or None when nothing changes."""
    status = h["status"]
    stored = int(h.get("fix_misses") or 0)
    changes: dict[str, Any] = {}
    if status != "possibly_fixed":
        if misses >= need:
            changes.update(status="possibly_fixed", previous_status=status, possibly_fixed_at=now_text,
                           fix_misses=misses, updated_at=now_text)
        elif stored != misses:
            changes["fix_misses"] = misses
    elif misses >= need:
        if stored != misses:
            changes["fix_misses"] = misses
        if h.get("possibly_fixed_at") is None:
            changes["possibly_fixed_at"] = now_text
    elif h.get("possibly_fixed_at") is not None or h.get("previous_status") is not None:  # flagged by the check and seen again: back to what it was
        back = h.get("previous_status")
        changes.update(status=back if back in _RESTORE_STATUSES else "new", previous_status=None,
                       possibly_fixed_at=None, fix_misses=misses, updated_at=now_text)
    elif stored != misses:
        changes["fix_misses"] = misses
    return changes or None
