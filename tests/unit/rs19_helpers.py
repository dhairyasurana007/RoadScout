"""Shared helpers and fixtures for the task 19 (possibly-fixed detection, backend) hidden tests.

Interface assumed (everything below is discovered tolerantly at run time; see the report to the main agent):

    the check      a public function in a backend module (preferred: backend/possibly_fixed.py, function
                   check_possibly_fixed(*, db_path=None, storage=None, min_misses=None, radius_m=None,
                   heading_tolerance_deg=None)); it re-evaluates EVERY hazard from scratch against all later
                   'done' drives (never incremental), is idempotent, and never raises for a bad GPX.
                   If no such function exists the tests fall back to backend.dedup.deduplicate_drive (check
                   embedded in the dedup stage) called for every drive in chronological order.
    config         min misses (default 3), radius in m (default 25), heading tolerance in degrees (default 45):
                   check parameters or ROADSCOUT_* env variables (found by scanning backend/*.py).
    hazards table  columns for the miss count, the flag time and the previous status (names discovered).
    hazard JSON    keys 'misses' (int) and 'possibly_fixed_at' (ISO text or null) next to all existing keys.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import math
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

ROOT = None
_cands = [Path(os.environ["ROADSCOUT_REPO"])] if os.environ.get("ROADSCOUT_REPO") else []
_cands += [HERE, *HERE.parents, *(p / "RoadScout" for p in HERE.parents)]
for _p in _cands:
    if (_p / "backend").is_dir() and (_p / "backend" / "__init__.py").exists():
        ROOT = _p
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

FFMPEG = shutil.which("ffmpeg")
LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "::"}
EARTH_R = 6371008.8
M_PER_DEG = EARTH_R * math.pi / 180.0
T0 = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)      # clip creation time (pipeline tests)
BASE = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)      # start of the drive that first saw a hazard
SEED_TS = "2026-10-21T00:00:00+00:00"
VA = (37.5407, -77.4360)
HAZARD_AT = (38.88, -77.30)
DEFAULT_MISSES = 3

OLD_KEYS = ("id", "hazard_class", "lat", "lon", "direction", "severity", "confidence", "sightings",
            "candidate_count", "first_seen", "last_seen", "status", "reviewed_at", "class_edited",
            "severity_edited", "original_hazard_class", "original_severity", "agency", "agency_source",
            "automatic_agency", "photo_url", "crop_url")
UNTOUCHED = ("lat", "lon", "direction", "severity", "confidence", "sightings", "candidate_count", "first_seen",
             "last_seen", "photo_path", "crop_path", "hazard_class", "created_at", "reviewed_at", "class_edited",
             "severity_edited", "original_hazard_class", "original_severity", "agency_override")

CHECK_MODULES = ["backend.possibly_fixed", "backend.fixed", "backend.fix_check", "backend.fixes",
                 "backend.fix_detection", "backend.fixed_detection", "backend.possibly_fixed_check",
                 "backend.hazard_fixes", "backend.fix_hazards", "backend.hazard_fix", "backend.possiblyfixed",
                 "backend.dedup", "backend.hazards", "backend.pipeline"]
CHECK_NAMES = ["check_possibly_fixed", "detect_possibly_fixed", "mark_possibly_fixed", "update_possibly_fixed",
               "refresh_possibly_fixed", "find_possibly_fixed", "flag_possibly_fixed", "evaluate_possibly_fixed",
               "run_possibly_fixed_check", "check_fixed", "detect_fixed", "check_fixes", "detect_fixes",
               "mark_fixed_hazards", "run_fix_check", "check_hazards_fixed", "update_fixed_status",
               "check_possibly_fixed_hazards", "detect_possibly_fixed_hazards", "run_possibly_fixed"]
DBP = ["db_path", "db", "database", "database_path"]


# ------------------------------------------------------------------ small utilities

def offset(lat, lon, north_m=0.0, east_m=0.0):
    return (lat + north_m / M_PER_DEG, lon + east_m / (M_PER_DEG * math.cos(math.radians(lat))))


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value):
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    s = str(value).strip().replace(" ", "T")
    s = re.sub(r"Z$", "+00:00", s)
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def unwrap(body):
    if isinstance(body, dict) and "id" not in body:
        for k in ("hazard", "data", "item", "result"):
            if isinstance(body.get(k), dict):
                return body[k]
    return body


def known_classes():
    cfg = json.loads((ROOT / "config" / "hazard_prompts.json").read_text(encoding="utf-8"))
    return list(cfg["classes"].keys())


def day(k, hour=12):
    """Start of the k-th later drive (k days after BASE)."""
    return BASE + timedelta(days=k, hours=hour - 12)


# ------------------------------------------------------------------ synthetic GPX

def _trkseg(points):
    body = "".join('<trkpt lat="%.8f" lon="%.8f"><ele>100</ele><time>%s</time></trkpt>'
                   % (la, lo, w.strftime("%Y-%m-%dT%H:%M:%SZ")) for w, la, lo in points)
    return "<trkseg>" + body + "</trkseg>"


def gpx_doc(segments):
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="rs19" '
            'xmlns="http://www.topografix.com/GPX/1/1"><trk>' + "".join(_trkseg(s) for s in segments)
            + "</trk></gpx>\n").encode("utf-8")


def pass_points(h_lat, h_lon, heading, *, started, offset_s=0.0, t_pass=30, t_from=-10, t_to=70, speed=10.0,
                lateral=0.0):
    """1 Hz points of a straight drive with bearing `heading` that is closest to (h_lat, h_lon) at video time
    `t_pass` (a vertex of the track) with a sideways distance of `lateral` metres. GPX time of video time t is
    started + offset_s + t."""
    hr = math.radians(heading)
    north_u, east_u = math.cos(hr), math.sin(hr)
    right_n, right_e = math.cos(hr + math.pi / 2), math.sin(hr + math.pi / 2)
    pts = []
    for t in range(int(t_from), int(t_to) + 1):
        along = speed * (t - t_pass)
        la, lo = offset(h_lat, h_lon, along * north_u + lateral * right_n, along * east_u + lateral * right_e)
        pts.append((started + timedelta(seconds=offset_s + t), la, lo))
    return pts


def gpx_pass(h_lat, h_lon, heading, **kw):
    return gpx_doc([pass_points(h_lat, h_lon, heading, **kw)])


# ------------------------------------------------------------------ discovery

def find_check():
    """(function, 'module.name') of the possibly-fixed check, or (None, None)."""
    mods = {}
    for name in CHECK_MODULES:
        try:
            mods[name] = importlib.import_module(name)
        except Exception:
            continue
    for modname in CHECK_MODULES:
        mod = mods.get(modname)
        if mod is None:
            continue
        for n in CHECK_NAMES:
            f = getattr(mod, n, None)
            if callable(f):
                return f, "%s.%s" % (modname, n)
    # regex fallback: any module of backend/ whose file name mentions 'fix', then dedup / hazards / pipeline
    pool = []
    for path in sorted((ROOT / "backend").glob("*.py")):
        if re.search("fix", path.stem, re.I):
            try:
                pool.append(importlib.import_module("backend." + path.stem))
            except Exception:
                pass
    for modname in ("backend.dedup", "backend.hazards", "backend.pipeline"):
        if modname in mods:
            pool.append(mods[modname])
    verbs = re.compile(r"check|detect|mark|flag|refresh|evaluate|scan|apply|update|run", re.I)
    for mod in pool:
        for name, f in inspect.getmembers(mod, inspect.isfunction):
            if f.__module__ != mod.__name__ or name.startswith("_"):
                continue
            if re.search(r"fix", name, re.I) and verbs.search(name) and "get_hazard" not in name:
                return f, "%s.%s" % (mod.__name__, name)
    return None, None


_ENV_NAMES = None


def env_names():
    global _ENV_NAMES
    if _ENV_NAMES is None:
        found = set()
        for path in (ROOT / "backend").glob("*.py"):
            found |= set(re.findall(r"ROADSCOUT_[A-Z0-9_]+", path.read_text(encoding="utf-8", errors="replace")))
        _ENV_NAMES = sorted(found)
    return _ENV_NAMES


def config_env(kind):
    """Name of the env variable for 'misses' | 'radius' | 'tol' (or None)."""
    names = [n for n in env_names() if not re.search(r"DEDUP|STATIONARY|BLUR|PREPROCESS|CHUNK|VERIFY|SPEND|COST", n)]
    if kind == "misses":
        pick = [n for n in names if re.search(r"FIX|POSSIBL|MISS", n) and re.search(r"MISS|DRIVES|PASSES|COUNT|MIN|N$", n)]
    elif kind == "radius":
        pick = [n for n in names if re.search(r"FIX|POSSIBL|PASS", n) and re.search(r"RADIUS|DIST|_M$", n)]
    else:
        pick = [n for n in names if re.search(r"FIX|POSSIBL|PASS", n) and re.search(r"HEADING|TOL|DEG|BEARING", n)]
    return pick[0] if pick else None


PARAM_PATTERNS = {
    "misses": r"miss|consecutive|min_drives|min_passes|n_drives",
    "radius": r"radius|distance|within",
    "tol": r"heading|tolerance|bearing",
}


def param_for(func, kind):
    if func is None:
        return None
    for pname in inspect.signature(func).parameters:
        if re.search(PARAM_PATTERNS[kind], pname, re.I):
            if kind == "radius" and re.search(r"heading|tol", pname, re.I):
                continue
            return pname
    return None


# ------------------------------------------------------------------ environment

class Env:
    def __init__(self, mp, tmp_path, client):
        self.mp = mp
        self.tmp = Path(tmp_path)
        self.db = self.tmp / "db" / "rs19.sqlite"
        self.storage = self.tmp / "storage"
        self.client = client
        self._cn = 0
        self.drives = []          # (started_at, id) in creation order
        self.cfg = {}
        self.check_calls = 0

    # ---- db
    def q(self, sql, params=()):
        con = sqlite3.connect(str(self.db))
        con.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in con.execute(sql, params)]
        finally:
            con.close()

    def x(self, sql, params=()):
        con = sqlite3.connect(str(self.db))
        try:
            con.execute("PRAGMA foreign_keys = ON")
            cur = con.execute(sql, params)
            con.commit()
            return cur.lastrowid
        finally:
            con.close()

    def row(self, hid):
        r = self.q("SELECT * FROM hazards WHERE id = ?", (hid,))
        return r[0] if r else None

    def rows(self):
        return self.q("SELECT * FROM hazards ORDER BY id")

    def columns(self):
        return [r["name"] for r in self.q("PRAGMA table_info(hazards)")]

    def prev_status_cols(self):
        return [c for c in self.columns() if re.search(r"prev|prior|before|restore", c, re.I)]

    # ---- drives and gpx
    def add_drive(self, did, started, *, offset_s=0.0, duration=60.0, status="done", gpx="missing"):
        """gpx: bytes (written to <storage>/<id>/track.gpx), 'missing' (key set, no file), '' (empty path)."""
        key = "%s/track.gpx" % did
        if isinstance(gpx, (bytes, bytearray)):
            p = self.storage / did / "track.gpx"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(bytes(gpx))
        elif gpx == "":
            key = ""
        self.x("INSERT OR REPLACE INTO drives (id, started_at, started_at_source, clock_offset_seconds, video_path, "
               "gpx_path, status, duration_seconds, created_at, updated_at) VALUES (?, ?, 'test', ?, ?, ?, ?, ?, ?, ?)",
               (did, iso(started), offset_s, "%s/video.mp4" % did, key, status, duration, iso(started), iso(started)))
        self.drives.append((started, did))
        return did

    def add_pass_drive(self, did, started, h_lat, h_lon, heading, *, status="done", offset_s=0.0, duration=60.0,
                       **kw):
        g = gpx_pass(h_lat, h_lon, heading, started=started, offset_s=offset_s, **kw)
        return self.add_drive(did, started, offset_s=offset_s, duration=duration, status=status, gpx=g)

    # ---- candidates
    def add_cand(self, drive_id, lat, lon, heading, *, cls="pothole", score=0.5, conf=0.8, severity=2, best_ts=10.0):
        self._cn += 1
        n = self._cn
        frame_path = str(self.db.parent / ("frame_%d.jpg" % n))
        con = sqlite3.connect(str(self.db))
        try:
            con.execute("PRAGMA foreign_keys = ON")
            cur = con.execute(
                "INSERT INTO candidates (drive_id, hazard_id, hazard_class, track_key, track_id, chunk, "
                "first_timestamp, last_timestamp, frames_seen, max_mask_area, best_frame_index, best_timestamp, "
                "best_box, best_frame_path, crop_path, blurred_frame_path, blurred_crop_path, blur_status, "
                "lat, lon, heading, score, geotag_status, geotag_note) "
                "VALUES (?, NULL, ?, ?, ?, 0, 9.0, 11.0, 5, 100.0, 30, ?, '[0, 0, 10, 10]', ?, NULL, ?, NULL, "
                "'ok', ?, ?, ?, ?, 'ok', NULL)",
                (drive_id, cls, "0:%d" % n, n, best_ts, frame_path, frame_path, lat, lon, heading, score))
            cid = cur.lastrowid
            con.execute(
                "INSERT INTO verdicts (candidate_id, drive_id, is_hazard, hazard_class, severity, lane, confidence, "
                "reason, report_description, model, prompt_version, accepted, created_at) "
                "VALUES (?, ?, 1, ?, ?, NULL, ?, 'r', 'd', 'fake', 'v1', 1, ?)",
                (cid, drive_id, cls, severity, conf, iso(T0)))
            con.commit()
            return cid
        finally:
            con.close()

    def dedup(self, drive_id):
        from backend.dedup import deduplicate_drive

        return deduplicate_drive(drive_id, db_path=str(self.db))

    # ---- hazards
    def make_hazard(self, lat=HAZARD_AT[0], lon=HAZARD_AT[1], heading=0.0, *, cls="pothole", status="new",
                    drive="s00", started=BASE, severity=2):
        """A real hazard created by deduplication from one candidate of the old drive `drive` (last_seen =
        started + 10 s), then given `status` through the review route."""
        if not self.q("SELECT id FROM drives WHERE id = ?", (drive,)):
            self.add_drive(drive, started, gpx="missing")
        cid = self.add_cand(drive, lat, lon, heading, cls=cls, severity=severity)
        self.dedup(drive)
        hid = self.q("SELECT hazard_id FROM candidates WHERE id = ?", (cid,))[0]["hazard_id"]
        assert hid is not None, "synthetic candidate was not turned into a hazard"
        if status == "reported":
            self.patch_ok(hid, status="approved")
            self.patch_ok(hid, status="reported")
        elif status != "new":
            self.patch_ok(hid, status=status)
        return hid

    def sight(self, hid, did, started, *, east_m=2.0, heading=None, cls=None):
        """A later drive that sees the hazard again: a candidate next to it, merged by deduplication."""
        h = self.row(hid)
        self.add_drive(did, started, gpx="missing")
        lat, lon = offset(h["lat"], h["lon"], east_m=east_m)
        self.add_cand(did, lat, lon, h["direction"] if heading is None else heading, cls=cls or h["hazard_class"])
        self.dedup(did)

    def link_candidate(self, hid, did, heading=0.0):
        """A member candidate of drive `did` linked to the hazard WITHOUT rebuilding (last_seen stays)."""
        h = self.row(hid)
        cid = self.add_cand(did, h["lat"], h["lon"], heading, cls=h["hazard_class"])
        self.x("UPDATE candidates SET hazard_id = ? WHERE id = ?", (hid, cid))
        return cid

    # ---- configuration of the check
    def configure(self, **cfg):
        self.cfg.update(cfg)

    def _apply_cfg(self, func):
        kwargs = {}
        for kind, val in self.cfg.items():
            pname = param_for(func, kind)
            if pname:
                kwargs[pname] = val
                continue
            name = config_env(kind)
            if name is None:
                pytest.fail("cannot configure %r: the check has no matching parameter and no ROADSCOUT_* env "
                            "variable for it was found in backend/*.py (check = %r)" % (kind, func))
            self.mp.setenv(name, str(val))
        return kwargs

    def check(self):
        """Run the possibly-fixed check once over everything. Never swallows an exception."""
        self.check_calls += 1
        func, name = find_check()
        kwargs = self._apply_cfg(func)
        order = [d for _s, d in sorted(self.drives)]
        if func is None or name.startswith("backend.dedup.deduplicate_drive"):
            # the check is not a separate function: it must run inside deduplication
            for d in order or ["none"]:
                self.dedup(d)
            return None
        sig = inspect.signature(func)
        dbp = next((n for n in DBP if n in sig.parameters), None)
        if dbp:
            kwargs[dbp] = str(self.db)
        required = [p for p in sig.parameters.values()
                    if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    and p.name not in kwargs]
        if required and required[0].name in ("hazard_id", "hazard"):
            for h in self.rows():
                self._call(func, dict(kwargs, **{required[0].name: h["id"]}))
            return None
        if required and re.search(r"drive", required[0].name):
            for d in order:
                self._call(func, dict(kwargs, **{required[0].name: d}))
            return None
        if required:
            pytest.fail("do not know how to call %s%s" % (name, sig))
        return self._call(func, kwargs)

    @staticmethod
    def _call(func, kwargs):
        res = func(**kwargs)
        if inspect.isawaitable(res):
            res = asyncio.run(res)
        return res

    # ---- http
    def hazards(self, **params):
        r = self.client.get("/api/hazards", params=params or None)
        assert r.status_code == 200, r.text[:300]
        body = r.json()
        assert isinstance(body, list)
        return body

    def item(self, hid):
        for i in self.hazards():
            if int(i["id"]) == int(hid):
                return i
        raise AssertionError("hazard %s not in GET /api/hazards" % hid)

    def patch(self, hid, **fields):
        return self.client.patch("/api/hazards/%s" % hid, json=fields)

    def patch_ok(self, hid, **fields):
        r = self.patch(hid, **fields)
        assert r.status_code == 200, "%s -> %s %s" % (fields, r.status_code, r.text[:300])
        return unwrap(r.json())

    # ---- state readers
    def status(self, hid):
        return self.row(hid)["status"]

    def misses(self, hid):
        item = self.item(hid)
        key = "misses" if "misses" in item else next((k for k in item if "miss" in k.lower()), None)
        assert key is not None, "no miss-count key in the hazard JSON: %s" % sorted(item)
        return item[key]

    def flagged_at(self, hid):
        item = self.item(hid)
        key = "possibly_fixed_at" if "possibly_fixed_at" in item else next(
            (k for k in item if re.search(r"fixed.*_at|flagged|fixed_since", k, re.I)), None)
        assert key is not None, "no flag-time key in the hazard JSON: %s" % sorted(item)
        return item[key]

    def snapshot(self, hid):
        return {k: v for k, v in self.row(hid).items() if k in UNTOUCHED}

    # ---- misses drives
    def miss_drives(self, hid, count, *, first=1, prefix="m", heading=None, **kw):
        """`count` later 'done' drives (days first..first+count-1 after BASE) whose GPX passes the hazard."""
        h = self.row(hid)
        hd = h["direction"] if heading is None else heading
        ids = []
        for k in range(first, first + count):
            did = "%s%02d" % (prefix, k)
            self.add_pass_drive(did, day(k), h["lat"], h["lon"], hd, **kw)
            ids.append(did)
        return ids


def make_env(mp, tmp_path):
    import backend.config  # noqa: F401  (loads a local .env before the key is cleared)

    mp.setenv("ROADSCOUT_DB", str(tmp_path / "db" / "rs19.sqlite"))
    mp.setenv("ROADSCOUT_STORAGE_DIR", str(tmp_path / "storage"))
    mp.setenv("ROADSCOUT_PROVIDER", "fake")
    mp.setenv("ROADSCOUT_PREPROCESS_FPS", "3")
    mp.setenv("ROADSCOUT_CHUNK_SECONDS", "4")
    mp.setenv("ROADSCOUT_COST_LOG", "off")
    for k in list(os.environ):
        if k == "META_API_KEY" or (k.startswith("ROADSCOUT") and re.search(r"FIX|MISS", k)):
            mp.delenv(k, raising=False)
    for k in ("ROADSCOUT_FAKE_SCENARIO", "ROADSCOUT_MIN_TRACK_FRAMES", "ROADSCOUT_VERIFY_MIN_CONFIDENCE",
              "ROADSCOUT_PREPROCESS_MAX_HEIGHT", "ROADSCOUT_DEDUP_DISTANCE_M",
              "ROADSCOUT_DEDUP_HEADING_TOLERANCE_DEG", "ROADSCOUT_STATIONARY_SPEED_MPS", "ROADSCOUT_SKIP_STATIONARY"):
        mp.delenv(k, raising=False)
    real_connect = socket.socket.connect

    def _guard(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if host is not None and str(host) not in LOOPBACK:
            raise AssertionError("network access attempted during a task 19 test: %r" % (address,))
        return real_connect(self, address)

    mp.setattr(socket.socket, "connect", _guard)
    from fastapi.testclient import TestClient

    from backend.db import init_db
    from backend.main import app

    init_db()
    return TestClient(app)


@pytest.fixture()
def rs19_env(monkeypatch, tmp_path):
    client = make_env(monkeypatch, tmp_path)
    with client:
        yield Env(monkeypatch, tmp_path, client)


# ------------------------------------------------------------------ synthetic clip for pipeline runs

_CLIP = {}


def clip_bytes():
    if "b" not in _CLIP:
        if not FFMPEG:
            pytest.skip("ffmpeg not on PATH")
        d = Path(tempfile.mkdtemp(prefix="rs19media"))
        out = d / "clip.mp4"
        p = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10",
             "-t", "24", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-metadata",
             "creation_time=" + T0.strftime("%Y-%m-%dT%H:%M:%SZ"), str(out)], capture_output=True, timeout=240)
        assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
        _CLIP["b"] = out.read_bytes()
    return _CLIP["b"]


def northbound_gpx(seconds=60, speed=12.0):
    """Car driving north along lon -77.30 from lat 38.88 at 12 m/s, GPX time = T0 + s."""
    pts = [(T0 + timedelta(seconds=s), 38.88 + s * speed / M_PER_DEG, -77.30) for s in range(seconds + 1)]
    return gpx_doc([pts])


def upload(client):
    r = client.post("/api/drives", files={"video": ("clip.mp4", clip_bytes(), "video/mp4"),
                                          "gpx": ("t.gpx", northbound_gpx(), "application/gpx+xml")})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def run_pipeline_for(did, db):
    from backend.pipeline import run_pipeline
    from backend.providers import FakeProvider

    return run_pipeline(did, provider=FakeProvider("default"), db_path=str(db))
