"""Shared helpers and fixtures for the task 16 (map view backend) hidden tests.

Interface assumed (see the report to the main agent):

    GET /api/hazards                       -> 200, bare JSON list of the deduplicated hazards (all drives)
        item keys: id, hazard_class, lat, lon, direction, severity, confidence, sightings, first_seen, last_seen,
                   status, photo_url (relative "/api/..." URL or null), crop_url (same, optional)
        optional query filters (tested only when the OpenAPI schema lists the parameter): status, drive_id
    GET /api/hazards/{id}/photo[?variant=crop]   -> 200 image/jpeg, bytes == the BLURRED file, else 404
"""
from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
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
FFPROBE = shutil.which("ffprobe")

T0 = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
M_PER_DEG = 6371008.8 * 3.141592653589793 / 180.0
LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "::"}

DONE = {"done", "complete", "completed", "processed", "finished", "ready", "succeeded", "success"}
FAILED = {"failed", "error", "errored", "failure"}
TERMINAL = DONE | FAILED
STATUSES = ("new", "approved", "rejected", "reported", "possibly_fixed")
REQUIRED_KEYS = ("id", "hazard_class", "lat", "lon", "direction", "severity", "confidence", "sightings",
                 "first_seen", "last_seen", "status")
RAW_PATH_KEYS = {"best_frame_path", "crop_path", "blurred_frame_path", "blurred_crop_path", "photo_path",
                 "video_path", "gpx_path", "path", "file", "filename"}


def status_of(d):
    return str(d.get("status", "")).strip().lower()


def fwd(p):
    return str(p).replace("\\", "/")


# ------------------------------------------------------------------ synthetic images / media

def write_jpeg(path, seed, size=(48, 32)):
    """A tiny valid JPEG whose bytes depend on `seed` (so different files never compare equal)."""
    import numpy as np
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, size=(size[1], size[0], 3)).astype("uint8")
    Image.fromarray(arr, "RGB").save(str(path), "JPEG", quality=90)
    return Path(path)


_CLIP = None


def clip_bytes():
    """24 s 640x360 synthetic clip (moving checkerboard)."""
    global _CLIP
    if _CLIP is None:
        if not (FFMPEG and FFPROBE):
            pytest.skip("ffmpeg/ffprobe not on PATH")
        d = Path(tempfile.mkdtemp(prefix="rs16media"))
        out = d / "clip.mp4"
        expr = "255*mod(floor(X/8)+floor(Y/8)+floor(T*3),2)"
        p = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "nullsrc=s=640x360:r=30,format=yuv420p,geq=lum='%s':cb=128:cr=128" % expr,
             "-t", "24", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-metadata", "creation_time=" + T0.strftime("%Y-%m-%dT%H:%M:%SZ"), str(out)],
            capture_output=True, timeout=300)
        assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
        _CLIP = out.read_bytes()
    return _CLIP


def gpx_bytes(start=T0, seconds=60):
    pts, lat = [], 38.88
    for s in range(seconds + 1):
        t = (start + timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%SZ")
        pts.append('<trkpt lat="%.7f" lon="-77.3000000"><ele>100</ele><time>%s</time></trkpt>' % (lat, t))
        lat += 10.0 / M_PER_DEG
    body = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<gpx version="1.1" creator="roadscout-tests" xmlns="http://www.topografix.com/GPX/1/1">'
            "<trk><name>synthetic</name><trkseg>" + "".join(pts) + "</trkseg></trk></gpx>\n")
    return body.encode("utf-8")


# ------------------------------------------------------------------ environment

class Env:
    def __init__(self, mp, tmp_path, client):
        self.mp = mp
        self.tmp = Path(tmp_path)
        self.db = self.tmp / "db" / "rs16.sqlite"
        self.storage = self.tmp / "storage"
        self.client = client
        self._n = 0
        self._seed = 1000

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
            cur = con.execute(sql, params)
            con.commit()
            return cur.lastrowid
        finally:
            con.close()

    def hazards_rows(self):
        return self.q("SELECT * FROM hazards ORDER BY id")

    def cands(self, did):
        return self.q("SELECT * FROM candidates WHERE drive_id = ? ORDER BY id", (did,))

    # ---- synthetic data
    def cand_dir(self, drive):
        d = self.storage / drive / "candidates"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def jpeg(self, path):
        self._seed += 1
        return write_jpeg(path, self._seed)

    def add_hazard(self, *, hazard_class="pothole", lat=38.88, lon=-77.30, direction=90.0, severity=2,
                   confidence=0.8, sightings=1, candidate_count=1, first_seen="2026-10-20T12:00:05+00:00",
                   last_seen="2026-10-20T12:00:05+00:00", photo_path=None, crop_path=None, status="new"):
        now = "2026-10-21T00:00:00+00:00"
        return self.x(
            "INSERT INTO hazards (hazard_class, lat, lon, direction, severity, confidence, sightings, "
            "candidate_count, first_seen, last_seen, photo_path, crop_path, status, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (hazard_class, lat, lon, direction, severity, confidence, sightings, candidate_count, first_seen,
             last_seen, None if photo_path is None else str(photo_path),
             None if crop_path is None else str(crop_path), status, now, now))

    def add_candidate(self, drive, *, hazard_id=None, best, crop=None, blurred=None, blurred_crop=None,
                      hazard_class="pothole"):
        """A candidates row (all required NOT NULL columns filled) with the given files."""
        self._n += 1
        blur_status = "ok" if blurred else None
        cols = {
            "drive_id": drive, "hazard_id": hazard_id, "hazard_class": hazard_class,
            "track_key": "0:%d" % self._n, "track_id": self._n, "chunk": 0, "first_timestamp": 1.0,
            "last_timestamp": 2.0, "frames_seen": 5, "max_mask_area": 100.0, "best_frame_index": 3,
            "best_timestamp": 1.5, "best_box": "[0,0,10,10]", "best_frame_path": str(best),
            "crop_path": None if crop is None else str(crop), "lat": 38.88, "lon": -77.30, "heading": 90.0,
            "score": 1.0, "geotag_status": "ok", "geotag_note": None,
        }
        extra = {"blurred_frame_path": None if blurred is None else str(blurred),
                 "blurred_crop_path": None if blurred_crop is None else str(blurred_crop),
                 "blur_status": blur_status}
        have = {r["name"] for r in self.q("PRAGMA table_info(candidates)")}
        cols.update({k: v for k, v in extra.items() if k in have})
        names = list(cols)
        return self.x("INSERT INTO candidates (%s) VALUES (%s)" % (", ".join(names), ", ".join("?" * len(names))),
                      [cols[n] for n in names])

    def make_pair(self, drive, name, *, with_crop=False):
        """Original + blurred (+ crop pair) JPEGs in <storage>/<drive>/candidates/ and a candidates row.
        Returns a dict of paths plus the candidate id (hazard_id still NULL)."""
        d = self.cand_dir(drive)
        o = self.jpeg(d / ("%s_best.jpg" % name))
        b = self.jpeg(d / ("%s_best_blurred.jpg" % name))
        oc = bc = None
        if with_crop:
            oc = self.jpeg(d / ("%s_crop.jpg" % name))
            bc = self.jpeg(d / ("%s_crop_blurred.jpg" % name))
        cid = self.add_candidate(drive, best=o, crop=oc, blurred=b, blurred_crop=bc)
        return {"orig": o, "blur": b, "orig_crop": oc, "blur_crop": bc, "cid": cid}

    def link(self, cid, hid):
        self.x("UPDATE candidates SET hazard_id = ? WHERE id = ?", (hid, cid))

    # ---- routes (discovered from OpenAPI)
    def paths(self):
        from backend.main import app

        return app.openapi().get("paths", {})

    def list_route(self):
        paths = self.paths()
        if "get" in paths.get("/api/hazards", {}):
            return "/api/hazards"
        pat = re.compile(r"^/api/(hazards|map|map/hazards|hazards/all|hazards/map)/?$")
        for p, ops in paths.items():
            if pat.match(p) and "get" in ops:
                return p
        pytest.fail("no GET /api/hazards route in the OpenAPI schema")

    def list_params(self):
        ops = self.paths()[self.list_route()]["get"]
        return {p["name"] for p in ops.get("parameters", []) if p.get("in") == "query"}

    def photo_route(self):
        paths = self.paths()
        pat = re.compile(r"^/api/hazards/\{[^}/]+\}/photo/?$")
        for p, ops in paths.items():
            if pat.match(p) and "get" in ops:
                return p
        pat2 = re.compile(r"^/api/(?!drives)\S*\{[^}/]+\}\S*/photo/?$")
        for p, ops in paths.items():
            if pat2.match(p) and "get" in ops:
                return p
        pytest.fail("no GET /api/hazards/{id}/photo route in the OpenAPI schema")

    def photo_url_for(self, hid, crop=False):
        url = re.sub(r"\{[^}/]+\}", str(hid), self.photo_route())
        return url + ("?variant=crop" if crop else "")

    # ---- http
    def hazards(self, **params):
        r = self.client.get(self.list_route(), params=params or None)
        assert r.status_code == 200, r.text[:300]
        body = r.json()
        assert isinstance(body, list), "expected a bare JSON list, got %s" % type(body).__name__
        return body

    def by_id(self, items=None):
        items = self.hazards() if items is None else items
        return {int(i["id"]): i for i in items}

    def photo(self, hid, crop=False):
        return self.client.get(self.photo_url_for(hid, crop))

    # ---- drives / pipeline
    def upload(self):
        files = {"video": ("clip.mp4", clip_bytes(), "video/mp4"),
                 "gpx": ("track.gpx", gpx_bytes(), "application/gpx+xml")}
        r = self.client.post("/api/drives", files=files)
        assert r.status_code == 201, r.text
        return r.json()["id"]

    def get_drive(self, did):
        r = self.client.get("/api/drives/%s" % did)
        assert r.status_code == 200, r.text
        return r.json()

    def process_route(self):
        pat = re.compile(r"^/api/drives/\{[^}/]+\}/(process|run|pipeline|start|analy[sz]e|reprocess)/?$")
        for path, ops in self.paths().items():
            if pat.match(path) and "post" in ops:
                return path
        pytest.fail("no POST /api/drives/{id}/process route")

    def wait(self, did, timeout=240):
        deadline = time.time() + timeout
        d = None
        while time.time() < deadline:
            d = self.get_drive(did)
            if status_of(d) in TERMINAL:
                return d
            time.sleep(0.05)
        pytest.fail("drive %s never reached a terminal status; last: %s" % (did, d))

    def run(self, did):
        from backend.pipeline import run_pipeline

        return run_pipeline(did, db_path=str(self.db))

    def process_ok(self):
        did = self.upload()
        self.run(did)
        assert status_of(self.get_drive(did)) in DONE
        return did

    def delete_video_route(self):
        found = []
        pat = re.compile(r"^/api/drives/\{[^}/]+\}/(video|raw-video|raw_video|delete-video|delete_video|"
                         r"video/delete|video-delete)/?$")
        for path, ops in self.paths().items():
            if pat.match(path):
                for m in ("delete", "post"):
                    if m in ops:
                        found.append((0 if m == "delete" else 1, m, path))
        if not found:
            return None
        found.sort()
        return found[0][1], found[0][2]

    def resolve(self, stored):
        if not stored:
            return None
        p = Path(stored)
        for c in (p, self.storage / stored, self.tmp / stored, (ROOT / stored) if ROOT else p):
            try:
                if c.is_file():
                    return c
            except OSError:
                pass
        return None


def make_env(mp, tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    import backend.config  # noqa: F401  (loads a local .env before the key is cleared)

    mp.setenv("ROADSCOUT_DB", str(tmp_path / "db" / "rs16.sqlite"))
    mp.setenv("ROADSCOUT_STORAGE_DIR", str(tmp_path / "storage"))
    mp.setenv("ROADSCOUT_PROVIDER", "fake")
    mp.setenv("ROADSCOUT_PREPROCESS_FPS", "3")
    mp.setenv("ROADSCOUT_CHUNK_SECONDS", "4")
    mp.setenv("ROADSCOUT_COST_LOG", "off")
    for k in ("META_API_KEY", "ROADSCOUT_FAKE_SCENARIO", "ROADSCOUT_MIN_TRACK_FRAMES",
              "ROADSCOUT_VERIFY_MIN_CONFIDENCE", "ROADSCOUT_PREPROCESS_MAX_HEIGHT",
              "ROADSCOUT_DEDUP_DISTANCE_M", "ROADSCOUT_DEDUP_HEADING_TOLERANCE_DEG"):
        mp.delenv(k, raising=False)

    real_connect = socket.socket.connect

    def _guard(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if host is not None and str(host) not in LOOPBACK:
            raise AssertionError("network access attempted during a task 16 test: %r" % (address,))
        return real_connect(self, address)

    mp.setattr(socket.socket, "connect", _guard)

    from fastapi.testclient import TestClient

    from backend.db import init_db
    from backend.main import app

    init_db()
    return TestClient(app)


@pytest.fixture()
def rs16_env(monkeypatch, tmp_path):
    client = make_env(monkeypatch, tmp_path)
    with client:
        yield Env(monkeypatch, tmp_path, client)


@pytest.fixture()
def rs16_syn(rs16_env):
    """Five synthetic hazards (one per status) over two drives, with blurred/original JPEG pairs.

    Returns (env, data) with data[name] = {"id": hazard id, "pair": make_pair dict or None, ...}.
    """
    env = rs16_env
    data = {}

    def mk(name, drive, status, *, crop=False, photo=True, **kw):
        pair = env.make_pair(drive, name, with_crop=crop) if photo else None
        hid = env.add_hazard(status=status,
                             photo_path=pair["blur"] if pair else None,
                             crop_path=pair["blur_crop"] if pair and crop else None, **kw)
        if pair:
            env.link(pair["cid"], hid)
        data[name] = {"id": hid, "pair": pair, "status": status, "drive": drive, **kw}

    mk("h1", "synA", "new", crop=True, hazard_class="pothole", lat=38.8801, lon=-77.3001, direction=90.0,
       severity=2, confidence=0.81, sightings=1, first_seen="2026-10-20T12:00:05+00:00",
       last_seen="2026-10-20T12:00:05+00:00")
    mk("h2", "synA", "approved", hazard_class="debris_in_lane", lat=38.8812, lon=-77.3012, direction=270.0,
       severity=3, confidence=0.93, sightings=3, candidate_count=3, first_seen="2026-10-18T08:00:00+00:00",
       last_seen="2026-10-20T12:00:09+00:00")
    mk("h3", "synB", "rejected", hazard_class="damaged_guardrail", lat=38.8823, lon=-77.3023, direction=None,
       severity=1, confidence=0.55, sightings=1, first_seen="2026-10-19T09:00:00+00:00",
       last_seen="2026-10-19T09:00:00+00:00")
    mk("h4", "synB", "reported", crop=True, hazard_class="faded_lane_markings", lat=38.8834, lon=-77.3034,
       direction=0.0, severity=1, confidence=0.7, sightings=2, candidate_count=2,
       first_seen="2026-10-17T07:00:00+00:00", last_seen="2026-10-19T09:30:00+00:00")
    mk("h5", "synB", "possibly_fixed", photo=False, hazard_class="pothole", lat=38.8845, lon=-77.3045,
       direction=180.0, severity=2, confidence=0.6, sightings=2, candidate_count=2,
       first_seen="2026-10-10T07:00:00+00:00", last_seen="2026-10-12T07:00:00+00:00")
    return env, data
