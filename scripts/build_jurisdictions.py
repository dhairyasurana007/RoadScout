"""Rebuild config/jurisdictions.json (the boundary polygons behind backend/agency.py).

This is a developer tool; the app never runs it and never downloads anything. The committed
config/jurisdictions.json is the result of running it once on the US Census Bureau TIGER/Line 2023 state
boundaries (public domain):

    https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip

Unzip that file into an empty folder of its own and pass the folder:

    python -I scripts/build_jurisdictions.py <folder with tl_2023_us_state.shp and .dbf> [tolerance_m]

Only the District of Columbia, Virginia and Maryland are kept. Each outline is simplified with the
Douglas-Peucker algorithm (default tolerance 50 m, measured in a local flat projection), rings with an area
under 0.3 square km are dropped, and coordinates are rounded to 5 decimals (about 1 m). The shapefile is read
by the small reader below, so no extra packages are needed.
"""
import json
import math
import struct
import sys
from pathlib import Path

KEEP = {  # order matters: the first polygon that contains a point wins (agency_for_point)
    "DC": {"name": "District of Columbia", "agency": None},
    "VA": {"name": "Virginia", "agency": "VDOT"},
    "MD": {"name": "Maryland", "agency": "MDOT SHA"},
}
MIN_RING_KM2 = 0.3
LAT0 = 38.5  # flat projection for the simplification, good enough for a 3-state area


def read_shp(path: Path) -> list[list[list[tuple[float, float]]]]:
    """The polygon records of a .shp file: per record a list of rings of (lon, lat) points."""
    data = path.read_bytes()
    pos, records = 100, []
    while pos + 8 <= len(data):
        _number, words = struct.unpack(">ii", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + words * 2]
        pos += 8 + words * 2
        shape_type = struct.unpack("<i", body[:4])[0]
        if shape_type == 0:
            records.append([])
            continue
        if shape_type != 5:
            raise ValueError(f"unsupported shape type {shape_type}")
        n_parts, n_points = struct.unpack("<ii", body[36:44])
        parts = list(struct.unpack(f"<{n_parts}i", body[44:44 + 4 * n_parts])) + [n_points]
        start = 44 + 4 * n_parts
        coords = struct.unpack(f"<{2 * n_points}d", body[start:start + 16 * n_points])
        points = [(coords[2 * i], coords[2 * i + 1]) for i in range(n_points)]
        records.append([points[parts[i]:parts[i + 1]] for i in range(n_parts)])
    return records


def read_dbf_column(path: Path, name: str) -> list[str]:
    data = path.read_bytes()
    n_records, header_len, record_len = struct.unpack("<IHH", data[4:12])
    offset, fields, pos = 1, {}, 32
    while data[pos] != 0x0D:
        field = data[pos:pos + 11].split(b"\0")[0].decode("ascii")
        length = data[pos + 16]
        fields[field] = (offset, length)
        offset += length
        pos += 32
    start, length = fields[name]
    out = []
    for i in range(n_records):
        row = data[header_len + i * record_len:header_len + (i + 1) * record_len]
        out.append(row[start:start + length].decode("ascii").strip())
    return out


def project(point: tuple[float, float]) -> tuple[float, float]:
    return point[0] * math.cos(math.radians(LAT0)) * 111_320.0, point[1] * 110_574.0


def _dp(points: list[tuple[float, float]], tolerance: float) -> list[int]:
    """Indexes kept by Douglas-Peucker on an open polyline (iterative)."""
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        lo, hi = stack.pop()
        (x1, y1), (x2, y2) = points[lo], points[hi]
        dx, dy = x2 - x1, y2 - y1
        norm = math.hypot(dx, dy)
        far, far_d = -1, tolerance
        for i in range(lo + 1, hi):
            x, y = points[i]
            d = math.hypot(x - x1, y - y1) if norm == 0 else abs(dx * (y1 - y) - (x1 - x) * dy) / norm
            if d > far_d:
                far, far_d = i, d
        if far >= 0:
            keep.add(far)
            stack += [(lo, far), (far, hi)]
    return sorted(keep)


def simplify_ring(ring: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    if ring[0] == ring[-1]:
        ring = ring[:-1]
    flat = [project(p) for p in ring]
    # A ring is closed, so split it at the point farthest from the first one and simplify both halves.
    far = max(range(len(flat)), key=lambda i: math.hypot(flat[i][0] - flat[0][0], flat[i][1] - flat[0][1]))
    kept = set()
    for lo, hi in ((0, far), (far, len(flat))):
        part = flat[lo:hi] + ([flat[0]] if hi == len(flat) else [])
        kept.update(lo + i for i in _dp(part, tolerance) if lo + i < len(flat))
    return [ring[i] for i in sorted(kept)]


def area_km2(ring: list[tuple[float, float]]) -> float:
    flat = [project(p) for p in ring]
    total = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(flat, flat[1:] + flat[:1]))
    return abs(total) / 2 / 1e6


def main(folder: Path, tolerance: float) -> None:
    shp = next(folder.glob("*.shp"))
    states = read_dbf_column(shp.with_suffix(".dbf"), "STUSPS")
    records = read_shp(shp)
    out = []
    for code, info in KEEP.items():
        rings = records[states.index(code)]
        kept = []
        for ring in rings:
            simple = simplify_ring(ring, tolerance)
            if len(simple) >= 3 and area_km2(simple) >= MIN_RING_KM2:
                kept.append([[round(lon, 5), round(lat, 5)] for lon, lat in simple])
        out.append({"id": code, "name": info["name"], "agency": info["agency"], "rings": kept})
        print(code, sum(len(r) for r in kept), "points in", len(kept), "rings")
    document = {
        "version": 1,
        "description": "Simplified outlines of DC, Virginia and Maryland for backend/agency.py. Points are "
                       "[longitude, latitude]. A point is inside a jurisdiction when it is inside an odd number "
                       "of its rings (even-odd rule). The first jurisdiction that contains the point wins.",
        "source": "US Census Bureau TIGER/Line 2023 state boundaries (public domain)",
        "tolerance_m": tolerance,
        "jurisdictions": out,
    }
    target = Path(__file__).resolve().parent.parent / "config" / "jurisdictions.json"
    target.write_text(json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8")
    print("wrote", target, target.stat().st_size, "bytes")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 50.0)
