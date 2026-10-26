"""Helpers for task 6 hidden tests: synthetic GPX generation, math, and a tolerant adapter
that discovers the coder's GPX parsing / position lookup interface by introspection."""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import os as _os

_cands = [Path(_os.environ["ROADSCOUT_REPO"])] if _os.environ.get("ROADSCOUT_REPO") else []
_cands += [HERE, *HERE.parents, *(p / "RoadScout" for p in HERE.parents)]
for _p in _cands:
    if (_p / "backend").is_dir() and (_p / "backend" / "__init__.py").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

T0 = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
R_EARTH = 6371008.8
M_PER_DEG = R_EARTH * math.pi / 180.0
LAT0, LON0 = 38.8800, -77.3000


# ----------------------------------------------------------------------------- geometry
def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(min(1.0, math.sqrt(a)))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def step(lat, lon, heading_deg, dist_m):
    h = math.radians(heading_deg)
    dn = dist_m * math.cos(h)
    de = dist_m * math.sin(h)
    return lat + dn / M_PER_DEG, lon + de / (M_PER_DEG * math.cos(math.radians(lat)))


class Pt(NamedTuple):
    t: float  # seconds after T0
    lat: float
    lon: float
    speed: float | None = None


def line(heading, speed, n=11, dt=1.0, lat=LAT0, lon=LON0, t0=0.0, tags=False):
    pts = []
    for i in range(n):
        pts.append(Pt(t0 + i * dt, lat, lon, speed if tags else None))
        lat, lon = step(lat, lon, heading, speed * dt)
    return pts


def circle(radius_m=200.0, n=41, dt=1.0, speed=10.0, lat=LAT0, lon=LON0):
    """Clockwise circle starting at the south point heading east."""
    pts = []
    ang_rate = speed / radius_m  # rad/s
    clat, clon = step(lat, lon, 0.0, radius_m)  # centre is north of start
    for i in range(n):
        th = i * dt * ang_rate
        # position relative to centre: starts due south, goes clockwise (west->? no: east first)
        dn = -radius_m * math.cos(th)
        de = radius_m * math.sin(th)
        pts.append(Pt(i * dt, clat + dn / M_PER_DEG, clon + de / (M_PER_DEG * math.cos(math.radians(clat)))))
    return pts


# ----------------------------------------------------------------------------- GPX writing
def fmt_time(t_sec, style="z"):
    dt = T0 + timedelta(seconds=t_sec)
    if style == "z":
        s = dt.strftime("%Y-%m-%dT%H:%M:%S")
        frac = f"{dt.microsecond / 1e6:.6f}"[1:] if dt.microsecond else ""
        return s + frac + "Z"
    if style == "plus0":
        s = dt.strftime("%Y-%m-%dT%H:%M:%S")
        frac = f"{dt.microsecond / 1e6:.6f}"[1:] if dt.microsecond else ""
        return s + frac + "+00:00"
    if style == "minus4":
        d2 = dt.astimezone(timezone(timedelta(hours=-4)))
        s = d2.strftime("%Y-%m-%dT%H:%M:%S")
        frac = f"{d2.microsecond / 1e6:.6f}"[1:] if d2.microsecond else ""
        return s + frac + "-04:00"
    if style == "plus530":
        d2 = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
        s = d2.strftime("%Y-%m-%dT%H:%M:%S")
        frac = f"{d2.microsecond / 1e6:.6f}"[1:] if d2.microsecond else ""
        return s + frac + "+05:30"
    if style == "naive":  # no suffix: should be treated as UTC
        s = dt.strftime("%Y-%m-%dT%H:%M:%S")
        frac = f"{dt.microsecond / 1e6:.6f}"[1:] if dt.microsecond else ""
        return s + frac
    raise ValueError(style)


def gpx_xml(segments, time_style="z", speed_in="speed", with_time=True, trk_count=1):
    """segments: list of lists of Pt (one <trkseg> each). speed_in: 'speed' or 'extensions'."""
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="rs-test" xmlns="http://www.topografix.com/GPX/1/1">',
    ]
    if trk_count == 1:
        out.append("<trk><name>t</name>")
    for seg in segments:
        if trk_count != 1:
            out.append("<trk>")
        out.append("<trkseg>")
        for p in seg:
            out.append(f'<trkpt lat="{p.lat:.9f}" lon="{p.lon:.9f}">')
            out.append("<ele>10</ele>")
            if with_time and p.t is not None:
                out.append(f"<time>{fmt_time(p.t, time_style)}</time>")
            if p.speed is not None:
                if speed_in == "speed":
                    out.append(f"<speed>{p.speed}</speed>")
                else:
                    out.append(f"<extensions><speed>{p.speed}</speed></extensions>")
            out.append("</trkpt>")
        out.append("</trkseg>")
        if trk_count != 1:
            out.append("</trk>")
    if trk_count == 1:
        out.append("</trk>")
    out.append("</gpx>")
    return "\n".join(out)


def write_gpx(path, segments, **kw):
    if segments and isinstance(segments[0], Pt):
        segments = [segments]
    Path(path).write_text(gpx_xml(segments, **kw), encoding="utf-8")
    return Path(path)


# ----------------------------------------------------------------------------- adapter
MODULE_CANDIDATES = [
    "backend.gpx_track", "backend.gpx", "backend.track", "backend.gps_track", "backend.gps",
    "backend.geo", "backend.geotag", "backend.gpx_parser", "backend.gpx_parsing",
    "backend.positions", "backend.position", "backend.location", "backend.gpxtrack",
    "backend.media",
]
PARSE_NAMES = ["parse_gpx", "load_gpx", "read_gpx", "load_track", "parse_track", "read_track",
               "from_gpx", "load_gpx_track", "parse_gpx_track", "read_gpx_track", "gpx_track"]
CLASS_NAMES = ["GpxTrack", "GPXTrack", "GpsTrack", "GPSTrack", "Track", "GpxPath", "TrackLookup",
               "GpxTrackLookup", "PositionTrack"]
CLASSMETHODS = ["from_file", "from_path", "from_bytes", "from_string", "from_xml", "parse",
                "load", "read", "from_gpx", "open"]
LOOKUP_NAMES = ["position_at", "lookup", "at", "get_position", "position_for", "interpolate",
                "locate", "position", "sample", "query", "get", "position_at_video_time",
                "at_video_time", "lookup_video_time", "__call__"]
LOOKUP_FUNCS = ["position_at", "lookup", "lookup_position", "position_at_video_time",
                "interpolate", "get_position", "locate"]

LAT_K = ["lat", "latitude"]
LON_K = ["lon", "lng", "long", "longitude"]
SPD_K = ["speed_mps", "speed_ms", "speed", "speed_m_s", "velocity"]
HDG_K = ["heading", "heading_deg", "bearing", "course", "heading_degrees"]


class Pos(NamedTuple):
    lat: float
    lon: float
    speed: float
    heading: float


class InterfaceError(Exception):
    pass


def _find_module_attrs():
    found = []
    for name in MODULE_CANDIDATES:
        try:
            mod = importlib.import_module(name)
        except ImportError as e:
            if e.name and e.name != name and not name.startswith(e.name):
                # a real missing dependency inside an existing module: surface it
                if importlib.util.find_spec(name) is not None:  # type: ignore[attr-defined]
                    raise
            continue
        found.append(mod)
    return found


importlib.util  # noqa: B018  (ensure submodule loaded)


def _kw_map(fn):
    """Map semantic names to parameter names via signature introspection."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}, None
    m = {}
    for pname in sig.parameters:
        low = pname.lower()
        if "offset" in low and "offset" not in m:
            m["offset"] = pname
        elif ("start" in low or "epoch" in low or "origin" in low) and "start" not in m:
            m["start"] = pname
    return m, sig


def _call(fn, first, start, offset, use_start=True, use_offset=True):
    m, sig = _kw_map(fn)
    kwargs = {}
    if use_start and start is not None and "start" in m:
        kwargs[m["start"]] = start
    if use_offset and "offset" in m:
        kwargs[m["offset"]] = offset
    args = () if first is _NOARG else (first,)
    return fn(*args, **kwargs), set(m)


_NOARG = object()


def _to_pos(res):
    if res is None:
        raise InterfaceError("lookup returned None")
    if isinstance(res, dict):
        get = lambda keys: next((res[k] for k in keys if k in res), None)  # noqa: E731
    elif isinstance(res, (tuple, list)) and not hasattr(res, "_fields"):
        if len(res) != 4:
            raise InterfaceError(f"tuple result of len {len(res)}; expected (lat, lon, speed, heading)")
        return Pos(*[float(x) for x in res])
    else:
        get = lambda keys: next((getattr(res, k) for k in keys if hasattr(res, k)), None)  # noqa: E731
    vals = [get(LAT_K), get(LON_K), get(SPD_K), get(HDG_K)]
    if any(v is None for v in vals):
        raise InterfaceError(f"result {res!r} lacks lat/lon/speed/heading fields")
    return Pos(*[float(v) for v in vals])


class Adapter:
    """Parses a GPX at (path|bytes) and does lookups with a given video_start and offset."""

    def __init__(self):
        self.mods = _find_module_attrs()
        if not self.mods:
            raise InterfaceError("no GPX track module found under backend/ (tried %s)" % MODULE_CANDIDATES)
        self.parse_fn = None
        self.parse_mod = None
        for mod in self.mods:
            for n in PARSE_NAMES:
                f = getattr(mod, n, None)
                if callable(f) and not inspect.isclass(f):
                    self.parse_fn, self.parse_mod = f, mod
                    break
            if self.parse_fn:
                break
        self.cls = None
        if self.parse_fn is None:
            for mod in self.mods:
                for n in CLASS_NAMES:
                    c = getattr(mod, n, None)
                    if inspect.isclass(c):
                        self.cls = c
                        self.parse_mod = mod
                        break
                if self.cls:
                    break
        if self.parse_fn is None and self.cls is None:
            raise InterfaceError("no parse function/class found (tried %s / %s)" % (PARSE_NAMES, CLASS_NAMES))
        # standalone lookup function (e.g. position_at(track, ts, ...))
        self.lookup_func = None
        for mod in self.mods:
            for n in LOOKUP_FUNCS:
                f = getattr(mod, n, None)
                if callable(f) and not inspect.isclass(f) and n != "lookup":
                    self.lookup_func = f
                    break
            if self.lookup_func:
                break

    # --- building -------------------------------------------------------------
    def _parse(self, src, start, offset):
        """Returns (track_obj, parse_param_names)."""
        if self.parse_fn is not None:
            return _call(self.parse_fn, src, start, offset)
        for cm in CLASSMETHODS:
            f = getattr(self.cls, cm, None)
            if callable(f):
                try:
                    return _call(f, src, start, offset)
                except (TypeError, AttributeError):
                    continue
        return _call(self.cls, src, start, offset)

    def _lookup_method(self, tr):
        for n in LOOKUP_NAMES:
            f = getattr(tr, n, None)
            if callable(f):
                return f
        return None

    def build(self, src, start=T0, offset=0.0):
        """Parse and return a callable ts -> Pos bound to (start, offset)."""
        if isinstance(src, Path):
            src_arg = src
        else:
            src_arg = src
        try:
            tr, pnames = self._parse(src_arg, start, offset)
        except TypeError:
            if isinstance(src, Path):
                tr, pnames = self._parse(str(src), start, offset)
            else:
                raise
        meth = self._lookup_method(tr)
        if meth is None and self.lookup_func is not None:
            lf = self.lookup_func

            def meth(ts, **kw):  # noqa: F811
                return lf(tr, ts, **kw)

            mm, _ = _kw_map(lf)
        else:
            mm, _ = _kw_map(meth) if meth is not None else ({}, None)
        if meth is None:
            raise InterfaceError("parsed track %r has no lookup method (tried %s)" % (type(tr), LOOKUP_NAMES))
        # offset must be supplied somewhere
        if "offset" not in pnames and "offset" not in mm:
            # try attribute assignment
            done = False
            for an in ("clock_offset_seconds", "clock_offset", "offset_seconds", "offset"):
                if hasattr(tr, an):
                    try:
                        setattr(tr, an, offset)
                        done = True
                        break
                    except Exception:
                        pass
            if not done and offset != 0:
                raise InterfaceError("no way found to supply clock offset (param on parse/lookup or attribute)")
        self.supports_start = ("start" in pnames) or ("start" in mm)

        def lookup(ts):
            kw = {}
            if "start" in mm and "start" not in pnames and start is not None:
                kw[mm["start"]] = start
            if "offset" in mm and "offset" not in pnames:
                kw[mm["offset"]] = offset
            return _to_pos(meth(ts, **kw))

        return lookup


_ADAPTER = None


def adapter():
    global _ADAPTER
    if _ADAPTER is None:
        _ADAPTER = Adapter()
    return _ADAPTER


def build(src, start=T0, offset=0.0):
    return adapter().build(src, start, offset)


def supports_video_start():
    a = adapter()
    return getattr(a, "supports_start", True)


def lookup_or_oor(lk, ts):
    """Return Pos, or the string 'OOR' if the implementation raised an error for out-of-range."""
    try:
        return lk(ts)
    except InterfaceError:
        raise
    except Exception:
        return "OOR"


def finite(p: Pos):
    return all(math.isfinite(x) for x in p)
