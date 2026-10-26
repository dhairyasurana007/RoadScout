"""Input validation and metadata: ffprobe for video, streaming XML parse for GPX."""
import json
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class InvalidMedia(ValueError):
    """The uploaded file is not usable (message is safe to show to the user)."""


class ProbeUnavailable(RuntimeError):
    """ffprobe is not installed or could not run."""


@dataclass
class VideoInfo:
    created_at: datetime | None  # container creation_time in UTC, None if missing or bogus
    duration_seconds: float | None
    codec: str | None
    width: int | None
    height: int | None
    stream_index: int | None = None  # ffprobe index of the main video stream (skips cover art)


@dataclass
class GpxInfo:
    point_count: int
    first_time: datetime | None  # time of the first trackpoint, UTC


def parse_time(value: str | None) -> datetime | None:
    """Parse an ISO 8601 timestamp to aware UTC. Naive values are taken as UTC."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _tag(tags: dict | None, name: str) -> str | None:
    for key, value in (tags or {}).items():
        if key.lower() == name:
            return value
    return None


def probe_video(path: Path) -> VideoInfo:
    """Run ffprobe; raise InvalidMedia unless the file has a decodable video stream."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError as exc:
        raise ProbeUnavailable("ffprobe is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise InvalidMedia("Video could not be read (ffprobe timed out)") from exc
    reason = (result.stderr or "").strip().splitlines()
    detail = f" ({reason[-1][:200]})" if reason else f" (ffprobe exit {result.returncode}, stdout {result.stdout[:120]!r})"
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise InvalidMedia(f"File is not a readable video{detail}") from exc

    # Some ffprobe builds exit non-zero on a trailing demuxer warning (seen with MPEG-TS) while
    # still reporting the streams, so judge the file by what it reports, not by the exit code.
    streams = [
        s for s in data.get("streams", [])
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
    ]
    if not streams:
        if result.returncode != 0:
            raise InvalidMedia(f"File is not a readable video{detail}")
        raise InvalidMedia("File has no video stream")
    stream = streams[0]
    fmt = data.get("format", {})

    created = None
    for tags in [fmt.get("tags")] + [s.get("tags") for s in streams]:
        created = parse_time(_tag(tags, "creation_time"))
        if created and created.year > 1971:  # some recorders write 1970/0000 placeholders
            break
        created = None

    duration = None
    for raw in (fmt.get("duration"), stream.get("duration")):
        try:
            duration = float(raw)
            break
        except (TypeError, ValueError):
            continue

    return VideoInfo(
        created_at=created,
        duration_seconds=duration,
        codec=stream.get("codec_name"),
        width=stream.get("width"),
        height=stream.get("height"),
        stream_index=stream.get("index"),
    )


def parse_gpx(path: Path) -> GpxInfo:
    """Stream-parse a GPX file; raise InvalidMedia unless it is XML with trackpoints."""
    count = 0
    first_time = None
    root_checked = False
    try:
        for event, elem in ET.iterparse(path, events=("start", "end")):
            name = elem.tag.rsplit("}", 1)[-1].lower()
            if event == "start":
                if not root_checked:
                    root_checked = True
                    if name != "gpx":
                        raise InvalidMedia("File is not a GPX document")
                continue
            if name == "trkpt":
                try:
                    lat, lon = float(elem.get("lat")), float(elem.get("lon"))
                except (TypeError, ValueError):
                    raise InvalidMedia("GPX trackpoint is missing a valid lat/lon") from None
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    raise InvalidMedia("GPX trackpoint has out-of-range coordinates")
                count += 1
                if first_time is None:
                    for child in elem:
                        if child.tag.rsplit("}", 1)[-1].lower() == "time":
                            first_time = parse_time(child.text)
                elem.clear()
    except ET.ParseError as exc:
        raise InvalidMedia("GPX file is not valid XML") from exc
    if count == 0:
        raise InvalidMedia("GPX file has no trackpoints")
    return GpxInfo(point_count=count, first_time=first_time)
