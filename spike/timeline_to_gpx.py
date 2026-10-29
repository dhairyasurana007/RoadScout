"""Throwaway spike (task 26): turn a Google Maps Timeline export into a GPX track for one clip.

Reads the clip's creation_time, duration and embedded start location with ffprobe, takes the
Timeline "timelinePath" segments that overlap the clip, and writes a GPX 1.1 track with UTC times.
The Timeline stores whole-minute offsets, so points that share a minute are spread evenly across
it. The clip's own start location (a precise GPS fix) is added as the first point.
Everything stays local: the output belongs in the git-ignored spike/footage folder.
Usage: python spike/timeline_to_gpx.py Timeline.json CLIP.MOV OUT.gpx
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone


def probe(clip: str) -> tuple[datetime, float, tuple[float, float] | None]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:format_tags=creation_time,com.apple.quicktime.location.ISO6709",
         "-of", "json", clip],
        capture_output=True, check=True, text=True,
    ).stdout
    fmt = json.loads(out)["format"]
    tags = fmt.get("tags", {})
    start = datetime.fromisoformat(tags["creation_time"].replace("Z", "+00:00"))
    loc = None
    iso = tags.get("com.apple.quicktime.location.ISO6709")
    if iso:
        m = re.match(r"([+-]\d+\.?\d*)([+-]\d+\.?\d*)", iso)
        if m:
            loc = (float(m.group(1)), float(m.group(2)))
    return start, float(fmt["duration"]), loc


def parse_geo(text: str) -> tuple[float, float]:
    lat, lon = text.removeprefix("geo:").split(",")
    return float(lat), float(lon)


def main() -> None:
    timeline_path, clip, out_path = sys.argv[1:4]
    start, duration, loc = probe(clip)
    end = start + timedelta(seconds=duration)
    window_lo, window_hi = start - timedelta(minutes=10), end + timedelta(minutes=10)
    entries = json.load(open(timeline_path, encoding="utf-8"))

    points: list[tuple[datetime, float, float]] = []
    if loc:
        points.append((start, loc[0], loc[1]))
    for seg in entries:
        path = seg.get("timelinePath")
        if not path:
            continue
        seg_start = datetime.fromisoformat(seg["startTime"].replace("Z", "+00:00"))
        by_minute: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for p in path:
            by_minute[int(p["durationMinutesOffsetFromStartTime"])].append(parse_geo(p["point"]))
        for minute, pts in sorted(by_minute.items()):
            for i, (lat, lon) in enumerate(pts):
                t = seg_start + timedelta(minutes=minute, seconds=60.0 * i / len(pts))
                if window_lo <= t <= window_hi:
                    points.append((t, lat, lon))
    points.sort(key=lambda p: p[0])
    if len(points) < 2:
        raise SystemExit("no Timeline points overlap this clip")

    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<gpx version="1.1" creator="roadscout-spike" xmlns="http://www.topografix.com/GPX/1/1">',
             "  <trk><name>timeline</name><trkseg>"]
    for t, lat, lon in points:
        stamp = t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines.append(f'    <trkpt lat="{lat:.6f}" lon="{lon:.6f}"><time>{stamp}</time></trkpt>')
    lines += ["  </trkseg></trk>", "</gpx>"]
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"clip {start:%Y-%m-%d %H:%M:%S}Z for {duration / 60:.1f} min; wrote {len(points)} points "
          f"from {points[0][0]:%H:%M:%S}Z to {points[-1][0]:%H:%M:%S}Z")


if __name__ == "__main__":
    main()
