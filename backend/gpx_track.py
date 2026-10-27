"""GPX parsing and position lookup (task 6).

Interface
---------
    from backend.gpx_track import GpxTrack, GpxError, GpxRangeError, parse_gpx

    track = parse_gpx("track.gpx")      # same as GpxTrack.from_file; also from_points / from_drive / from_drive_id
    pos = track.position_at(12.5, video_start=started_at, clock_offset_seconds=0.0)
    pos.latitude, pos.longitude   # degrees (WGS84)
    pos.speed_mps                 # metres per second
    pos.heading_deg               # 0 <= heading < 360, 0 = north, clockwise
    pos.timestamp                 # timezone-aware UTC datetime of the GPX position

    GpxTrack.position_at(video_seconds, *, video_start=None, clock_offset_seconds=None, clamp=False)
    GpxTrack.positions_at(video_seconds_iterable, *, video_start=None, clock_offset_seconds=None, clamp=False)

Time mapping: GPX time = video_start + clock_offset_seconds + video_seconds. The offset is added to
video time to get GPX time (same convention as the drives table), so `video_start + offset` is
the drive's `effective_started_at`. `video_start` is a datetime (naive values are taken as UTC).
Tracks built with from_drive / from_drive_id remember the drive's started_at and offset, so
`track.position_at(12.5)` works without the keyword arguments; explicit arguments override them.
For any other track, video_start is required and the offset defaults to 0.

Parsing: the GPX is stream-parsed (ElementTree.iterparse), namespace tolerant, and every `trkpt`
of every `trk` and `trkseg` is merged into one track. Times may carry fractional seconds and any
UTC offset (converted to UTC; no offset means UTC). Points without a usable time or with bad
coordinates are skipped. Points are sorted by time; a point with the same timestamp as an earlier
one is dropped (this removes exact duplicates; of two different positions at one instant the
first in file order wins). Gaps between points, including between segments, are interpolated
linearly. A file with no trackpoints, or none with a usable time, raises GpxError.

Interpolation: latitude and longitude are linear in time between the two bracketing points.
Speed (m/s): linear between the bracketing points when both carry a `<speed>` tag (also inside
extensions); otherwise the haversine distance divided by the time of the bracketing segment.
Heading: the bearing of the segment containing the time. Segments shorter than 0.5 m count as
stationary and keep the last heading (before any movement, the first later heading; 0 if the
track never moves). A single-point track returns that point with heading 0.

Out of range: a GPX time before the first or after the last point raises GpxRangeError (a
ValueError) unless clamp=True, which returns the first or last point (timestamp is that point's
time). Non-finite video_seconds always raises ValueError.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Iterable

EARTH_RADIUS_M = 6371008.8
STATIONARY_METERS = 0.5  # segments shorter than this keep the previous heading
_RANGE_EPSILON = 1e-6  # seconds of float slack at the ends of the track


class GpxError(ValueError):
    """The GPX could not be turned into a usable track (message is safe to show)."""


class GpxRangeError(ValueError):
    """The requested time is outside the time span covered by the GPX track."""


@dataclass(frozen=True)
class TrackPoint:
    time: datetime  # timezone-aware, converted to UTC
    latitude: float
    longitude: float
    speed_mps: float | None = None  # from a <speed> tag, if any


@dataclass(frozen=True)
class TrackPosition:
    latitude: float
    longitude: float
    speed_mps: float
    heading_deg: float  # 0 to 360 (exclusive), 0 = north, clockwise
    timestamp: datetime  # UTC time on the GPX clock


_TIME_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[Tt ](\d{2}):(\d{2}):(\d{2})(?:[.,](\d+))?\s*(Z|z|[+-]\d{2}(?::?\d{2})?)?"
)


def _parse_time(value: str | None) -> datetime | None:
    """ISO 8601 to aware UTC; fractional seconds of any length; naive means UTC; None if unusable."""
    match = _TIME_RE.fullmatch((value or "").strip())
    if not match:
        return None
    year, month, day, hour, minute, second, frac, zone = match.groups()
    try:
        micro = int((frac or "0")[:6].ljust(6, "0"))
        tz = timezone.utc
        if zone and zone not in ("Z", "z"):
            digits = zone[1:].replace(":", "")
            delta = timedelta(hours=int(digits[:2]), minutes=int(digits[2:4] or 0))
            tz = timezone(delta if zone[0] == "+" else -delta)
        parsed = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), micro, tzinfo=tz)
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    heading = math.degrees(math.atan2(y, x)) % 360.0
    return 0.0 if heading >= 360.0 else heading


def _read_points(source: str | Path | BinaryIO) -> list[TrackPoint]:
    points: list[TrackPoint] = []
    saw_trkpt = False
    root_checked = False
    try:
        for event, elem in ET.iterparse(source, events=("start", "end")):
            name = _local(elem.tag)
            if event == "start":
                if not root_checked:
                    root_checked = True
                    if name != "gpx":
                        raise GpxError("File is not a GPX document")
                continue
            if name != "trkpt":
                continue
            saw_trkpt = True
            time = None
            speed = None
            for child in elem.iter():
                child_name = _local(child.tag)
                if child_name == "time" and time is None:
                    time = _parse_time(child.text)
                elif child_name == "speed" and speed is None:
                    try:
                        value = float((child.text or "").strip())
                    except ValueError:
                        continue
                    if math.isfinite(value) and value >= 0:
                        speed = value
            try:
                lat, lon = float(elem.get("lat")), float(elem.get("lon"))
            except (TypeError, ValueError):
                lat = lon = math.nan
            if time is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
                points.append(TrackPoint(time, lat, lon, speed))
            elem.clear()
    except ET.ParseError as exc:
        raise GpxError("GPX file is not valid XML") from exc
    if not saw_trkpt:
        raise GpxError("GPX file has no trackpoints")
    if not points:
        raise GpxError("GPX file has no trackpoints with a usable timestamp and coordinates")
    return points


def parse_gpx(source: str | Path | BinaryIO) -> "GpxTrack":
    """Parse a GPX file into a GpxTrack (same as GpxTrack.from_file). Raises GpxError if unusable."""
    return GpxTrack.from_file(source)


def _to_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


class GpxTrack:
    """A time-sorted GPS track with interpolated lookup. See the module docstring."""

    def __init__(self, points: Iterable[TrackPoint], *, video_start: datetime | None = None,
                 clock_offset_seconds: float | None = None):
        ordered = sorted(((_to_utc(p.time), i, p) for i, p in enumerate(points)), key=lambda t: t[:2])
        kept: list[tuple[datetime, TrackPoint]] = []
        for when, _, point in ordered:
            if kept and kept[-1][0] == when:
                continue  # same instant as the previous point: drop (exact duplicates and conflicts)
            kept.append((when, point))
        if not kept:
            raise GpxError("Track has no points")

        self._t0 = kept[0][0]
        self._rel = [(when - self._t0).total_seconds() for when, _ in kept]
        self._lat = [p.latitude for _, p in kept]
        self._lon = [p.longitude for _, p in kept]
        self._tag_speed = [p.speed_mps for _, p in kept]

        count = len(kept)
        self._seg_speed: list[float] = []
        headings: list[float | None] = []
        for i in range(count - 1):
            dist = _haversine(self._lat[i], self._lon[i], self._lat[i + 1], self._lon[i + 1])
            self._seg_speed.append(dist / (self._rel[i + 1] - self._rel[i]))
            moving = dist >= STATIONARY_METERS
            headings.append(
                _bearing(self._lat[i], self._lon[i], self._lat[i + 1], self._lon[i + 1]) if moving else None
            )
        last = None
        for i, h in enumerate(headings):  # carry the last heading through stationary stretches
            if h is None:
                headings[i] = last
            else:
                last = h
        following = next((h for h in headings if h is not None), 0.0)
        for i, h in enumerate(headings):  # back-fill a stationary start with the first real heading
            if h is None:
                headings[i] = following
            else:
                break
        self._seg_heading: list[float] = [h if h is not None else 0.0 for h in headings]

        self.video_start = _to_utc(video_start) if video_start is not None else None
        self.clock_offset_seconds = clock_offset_seconds

    # ----- construction -------------------------------------------------------------------

    @classmethod
    def from_points(cls, points: Iterable[TrackPoint], **defaults) -> "GpxTrack":
        return cls(points, **defaults)

    @classmethod
    def from_file(cls, source: str | Path | BinaryIO, **defaults) -> "GpxTrack":
        """Parse a GPX file (path or binary file object). Raises GpxError if unusable."""
        return cls(_read_points(source), **defaults)

    @classmethod
    def from_drive(cls, drive, storage=None) -> "GpxTrack":
        """Build from a stored drive record (dict or sqlite Row with gpx_path, started_at and
        clock_offset_seconds). The track remembers the drive's start and offset as defaults."""
        record = dict(drive)
        started = record["started_at"]
        if isinstance(started, str):
            started = _parse_time(started)
            if started is None:
                raise GpxError("Drive started_at is not a valid timestamp")
        if storage is None:
            from backend.storage import get_storage

            storage = get_storage()
        offset = record.get("clock_offset_seconds")
        return cls.from_file(
            storage.local_path(record["gpx_path"]),
            video_start=started,
            clock_offset_seconds=float(offset) if offset is not None else 0.0,
        )

    @classmethod
    def from_drive_id(cls, drive_id: str, storage=None) -> "GpxTrack":
        """Look the drive up in the database, then build as from_drive. KeyError if not found."""
        from backend.db import get_connection, init_db

        init_db()
        conn = get_connection()
        try:
            row = conn.execute("SELECT * FROM drives WHERE id = ?", (drive_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(f"drive not found: {drive_id}")
        return cls.from_drive(row, storage)

    # ----- properties ---------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._rel)

    @property
    def start_time(self) -> datetime:
        return self._t0

    @property
    def end_time(self) -> datetime:
        return self._t0 + timedelta(seconds=self._rel[-1])

    @property
    def duration_seconds(self) -> float:
        return self._rel[-1]

    def samples(self) -> list[tuple[float, float, float]]:
        """``(seconds after start_time, latitude, longitude)`` of every kept point, in time order (a copy)."""
        return list(zip(self._rel, self._lat, self._lon))

    # ----- lookup -------------------------------------------------------------------------

    def _gps_offset(self, video_start, clock_offset_seconds) -> float:
        """Seconds from the first point to video time 0 on the GPX clock."""
        start = video_start if video_start is not None else self.video_start
        if start is None:
            raise ValueError("video_start is required (this track has no drive start time)")
        offset = clock_offset_seconds if clock_offset_seconds is not None else self.clock_offset_seconds
        offset = 0.0 if offset is None else float(offset)
        if not math.isfinite(offset):
            raise ValueError("clock_offset_seconds must be finite")
        return (_to_utc(start) - self._t0).total_seconds() + offset

    def _lookup(self, rel: float, clamp: bool, video_seconds: float) -> TrackPosition:
        end = self._rel[-1]
        if rel < -_RANGE_EPSILON or rel > end + _RANGE_EPSILON:
            if not clamp:
                raise GpxRangeError(
                    f"video time {video_seconds}s maps to GPX time "
                    f"{(self._t0 + timedelta(seconds=rel)).isoformat()}, outside the track "
                    f"({self.start_time.isoformat()} to {self.end_time.isoformat()})"
                )
        rel = min(max(rel, 0.0), end)
        stamp = self._t0 + timedelta(seconds=rel)

        if len(self._rel) == 1:
            speed = self._tag_speed[0]
            return TrackPosition(self._lat[0], self._lon[0], 0.0 if speed is None else speed, 0.0, stamp)

        i = min(max(bisect_right(self._rel, rel) - 1, 0), len(self._rel) - 2)
        a, b = self._rel[i], self._rel[i + 1]
        frac = min(max((rel - a) / (b - a), 0.0), 1.0)
        lat = self._lat[i] + (self._lat[i + 1] - self._lat[i]) * frac
        lon = self._lon[i] + (self._lon[i + 1] - self._lon[i]) * frac
        s0, s1 = self._tag_speed[i], self._tag_speed[i + 1]
        speed = s0 + (s1 - s0) * frac if s0 is not None and s1 is not None else self._seg_speed[i]
        return TrackPosition(lat, lon, speed, self._seg_heading[i], stamp)

    def position_at(self, video_seconds: float, *, video_start: datetime | None = None,
                    clock_offset_seconds: float | None = None, clamp: bool = False) -> TrackPosition:
        """Interpolated position at `video_seconds` seconds into the video.

        GPX time = video_start + clock_offset_seconds + video_seconds. Raises GpxRangeError
        (a ValueError) when that falls outside the track, unless clamp=True.
        """
        seconds = float(video_seconds)
        if not math.isfinite(seconds):
            raise ValueError("video_seconds must be a finite number")
        base = self._gps_offset(video_start, clock_offset_seconds)
        return self._lookup(base + seconds, clamp, seconds)

    def positions_at(self, video_seconds: Iterable[float], *, video_start: datetime | None = None,
                     clock_offset_seconds: float | None = None, clamp: bool = False) -> list[TrackPosition]:
        """position_at for many timestamps; results are in the same order as the input."""
        base = self._gps_offset(video_start, clock_offset_seconds)
        out = []
        for value in video_seconds:
            seconds = float(value)
            if not math.isfinite(seconds):
                raise ValueError("video_seconds must be finite numbers")
            out.append(self._lookup(base + seconds, clamp, seconds))
        return out
