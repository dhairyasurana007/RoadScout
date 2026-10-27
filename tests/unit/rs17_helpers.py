"""Shared helpers and fixtures for the task 17 (review actions backend) hidden tests.

Interface assumed (tolerantly discovered through the OpenAPI schema; see the report to the main agent):

    PATCH /api/hazards/{id}   JSON body, every field optional but at least one required:
                                 status        "new" | "approved" | "rejected"   (reported/possibly_fixed: either behaviour)
                                 hazard_class  one of the keys of config/hazard_prompts.json["classes"]
                                 severity      integer 1..3
        -> 200 + the updated hazard as JSON (same shape as an item of GET /api/hazards)
        -> 404 unknown id; 422 (or 400) for invalid / unknown-field / empty bodies, nothing changes
      (PUT or POST on the same path, or POST /api/hazards/{id}/approve and /reject, are also accepted)
    hazards table: updated_at refreshed on every change; a column whose name contains "review" (for example
        reviewed_at) is NULL until the first review and a timestamp afterwards; an edit of class / severity is
        remembered (column names free) and survives backend.dedup.deduplicate_drive re-runs.
"""
from __future__ import annotations

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
T0 = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
SEED_TS = "2026-10-21T00:00:00+00:00"

BASE_COLS = {"id", "hazard_class", "lat", "lon", "direction", "severity", "confidence", "sightings",
             "candidate_count", "first_seen", "last_seen", "photo_path", "crop_path", "status", "created_at",
             "updated_at"}
UNTOUCHED = ("lat", "lon", "direction", "confidence", "sightings", "candidate_count", "first_seen", "last_seen",
             "photo_path", "crop_path", "created_at")
LIST_KEYS = ("id", "hazard_class", "lat", "lon", "direction", "severity", "confidence", "sightings", "first_seen",
             "last_seen", "status")
VA = (37.5407, -77.4360)
DBP = ["db_path", "db", "database", "database_path"]
FUNC_NAMES = ["deduplicate_drive", "dedupe_drive", "dedup_drive", "deduplicate", "dedupe", "run_dedup",
              "run_deduplication", "merge_drive", "deduplicate_hazards"]


def known_classes():
    cfg = json.loads((ROOT / "config" / "hazard_prompts.json").read_text(encoding="utf-8"))
    return list(cfg["classes"].keys())


def parse_time(value):
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    s = str(value).strip().replace(" ", "T")
    s = re.sub(r"Z$", "+00:00", s)
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def offset(lat, lon, north_m=0.0, east_m=0.0):
    return (lat + north_m / M_PER_DEG, lon + east_m / (M_PER_DEG * math.cos(math.radians(lat))))


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def unwrap(body):
    if isinstance(body, dict) and "id" not in body:
        for k in ("hazard", "data", "item", "result"):
            if isinstance(body.get(k), dict):
                return body[k]
    return body


# ------------------------------------------------------------------ environment

class Env:
    def __init__(self, mp, tmp_path, client):
        self.mp = mp
        self.tmp = Path(tmp_path)
        self.db = self.tmp / "db" / "rs17.sqlite"
        self.client = client
        self._cn = 0
        self._notes = []

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

    def review_cols(self):
        return [c for c in self.columns() if "review" in c.lower()]

    def add_hazard(self, *, hazard_class="pothole", lat=38.88, lon=-77.30, direction=90.0, severity=2,
                   confidence=0.8, sightings=1, candidate_count=1, first_seen="2026-10-20T12:00:05+00:00",
                   last_seen="2026-10-20T12:00:05+00:00", photo_path=None, crop_path=None, status="new"):
        return self.x(
            "INSERT INTO hazards (hazard_class, lat, lon, direction, severity, confidence, sightings, "
            "candidate_count, first_seen, last_seen, photo_path, crop_path, status, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (hazard_class, lat, lon, direction, severity, confidence, sightings, candidate_count, first_seen,
             last_seen, photo_path, crop_path, status, SEED_TS, SEED_TS))

    def trio(self):
        """Three synthetic hazards with real (tiny) photo files. Returns [(id, photo Path), ...]."""
        out = []
        for i, (cls, st) in enumerate([("pothole", "new"), ("debris in lane", "new"), ("damaged guardrail", "approved")]):
            p = self.tmp / "photos" / ("h%d.jpg" % i)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\xff\xd8\xff\xe0synthetic-photo-%d\xff\xd9" % i)
            hid = self.add_hazard(hazard_class=cls, status=st, severity=1 + i % 3, lat=38.88 + i * 0.001,
                                  photo_path=str(p), confidence=0.6 + i * 0.1)
            out.append((hid, p))
        return out

    # ---- routes
    def paths(self):
        from backend.main import app

        return app.openapi().get("paths", {})

    def spec(self):
        from backend.main import app

        return app.openapi()

    def generic_route(self):
        """(method, path) of the generic update route, or None."""
        paths = self.paths()
        pat = re.compile(r"^/api/hazards/\{[^}/]+\}/?$")
        for pref in (r"^/api/hazards/\{[^}/]+\}/?$",
                     r"^/api/hazards/\{[^}/]+\}/(review|status|update|edit|set-status|set_status|decision|triage)/?$"):
            pat = re.compile(pref)
            for m in ("patch", "put", "post"):
                for p, ops in paths.items():
                    if pat.match(p) and m in ops:
                        return m, p
        return None

    def action_route(self, name):
        pat = re.compile(r"^/api/hazards/\{[^}/]+\}/%s/?$" % name)
        for p, ops in self.paths().items():
            if pat.match(p) and ("post" in ops or "patch" in ops or "put" in ops):
                m = "post" if "post" in ops else ("patch" if "patch" in ops else "put")
                return m, p
        return None

    def body_props(self):
        """Property names of the request body of the generic route (empty set when not resolvable)."""
        g = self.generic_route()
        if not g:
            return set()
        spec = self.spec()
        op = spec["paths"][g[1]][g[0]]
        schema = (((op.get("requestBody") or {}).get("content") or {}).get("application/json") or {}).get("schema") or {}

        def resolve(s):
            for _ in range(5):
                if "$ref" in s:
                    node = spec
                    for part in s["$ref"].lstrip("#/").split("/"):
                        node = node[part]
                    s = node
                elif s.get("allOf"):
                    s = s["allOf"][0]
                elif s.get("anyOf"):
                    s = next((x for x in s["anyOf"] if x.get("type") != "null"), s["anyOf"][0])
                else:
                    break
            return s

        try:
            return set((resolve(schema).get("properties") or {}).keys())
        except Exception:
            return set()

    def class_key(self):
        props = self.body_props()
        if "hazard_class" in props or not props:
            return "hazard_class"
        for k in ("class", "class_name", "hazardClass"):
            if k in props:
                return k
        return "hazard_class"

    def url(self, hid, path=None):
        g = self.generic_route()
        path = path or (g[1] if g else "/api/hazards/{id}")
        return re.sub(r"\{[^}/]+\}", str(hid), path)

    def _map_body(self, body):
        ck = self.class_key()
        return {(ck if k == "hazard_class" else k): v for k, v in body.items()}

    def send(self, hid, json=None, content=None, headers=None, client=None):
        """Raw request to the generic update route (skips the test when there is none)."""
        g = self.generic_route()
        if g is None:
            pytest.skip("no generic update route (PATCH/PUT/POST /api/hazards/{id}) in the OpenAPI schema")
        c = client or self.client
        kw = {}
        if content is not None:
            kw["content"] = content
            kw["headers"] = headers or {"content-type": "application/json"}
        elif json is not None:
            kw["json"] = self._map_body(json) if isinstance(json, dict) else json
        return getattr(c, g[0])(self.url(hid, g[1]), **kw)

    def update(self, hid, client=None, **fields):
        """Apply fields (status / hazard_class / severity) through whatever route exists. Returns the response."""
        g = self.generic_route()
        exact = g is not None and re.match(r"^/api/hazards/\{[^}/]+\}/?$", g[1])
        if g is not None and (exact or set(fields) != {"status"}
                              or not any(self.action_route(n) for n in ("approve", "reject"))):
            return self.send(hid, json=fields, client=client)
        c = client or self.client
        if set(fields) == {"status"}:
            name = {"approved": "approve", "rejected": "reject", "new": "reset"}.get(fields["status"])
            for n in ([name] if name else []) + (["unreview", "reopen", "undo", "new"] if fields["status"] == "new" else []):
                r = self.action_route(n)
                if r:
                    return getattr(c, r[0])(self.url(hid, r[1]))
        pytest.skip("no usable update route for %r" % (fields,))

    def ok(self, hid, **fields):
        r = self.update(hid, **fields)
        assert r.status_code == 200, "%s -> %s %s" % (fields, r.status_code, r.text[:300])
        return unwrap(r.json())

    # ---- http
    def list_route(self):
        paths = self.paths()
        assert "get" in paths.get("/api/hazards", {}), "no GET /api/hazards"
        return "/api/hazards"

    def hazards(self, **params):
        r = self.client.get("/api/hazards", params=params or None)
        assert r.status_code == 200, r.text[:300]
        body = r.json()
        assert isinstance(body, list)
        return body

    def by_id(self, items=None):
        return {int(i["id"]): i for i in (self.hazards() if items is None else items)}

    def list_params(self):
        ops = self.paths()["/api/hazards"]["get"]
        return {p["name"] for p in ops.get("parameters", []) if p.get("in") == "query"}

    # ---- drives / candidates (for re-run cases)
    def add_drive(self, drive_id, started=T0, db=None):
        if db is not None:
            con = sqlite3.connect(str(db))
            con.execute("INSERT OR REPLACE INTO drives (id, started_at, started_at_source, clock_offset_seconds, "
                        "video_path, gpx_path, status, created_at) VALUES (?, ?, 'test', 0, ?, ?, 'done', ?)",
                        (drive_id, iso(started), "v/%s.mp4" % drive_id, "g/%s.gpx" % drive_id, iso(started)))
            con.commit()
            con.close()
            return
        self.x("INSERT OR REPLACE INTO drives (id, started_at, started_at_source, clock_offset_seconds, video_path, "
               "gpx_path, status, created_at) VALUES (?, ?, 'test', 0, ?, ?, 'done', ?)",
               (drive_id, iso(started), "v/%s.mp4" % drive_id, "g/%s.gpx" % drive_id, iso(started)))

    def add_cand(self, drive_id, lat, lon, heading, *, cls="pothole", score=0.5, conf=0.8, severity=2,
                 frame_path=None, db=None):
        self._cn += 1
        n = self._cn
        db = db or self.db
        frame_path = frame_path or str(Path(db).parent / ("frame_%d.jpg" % n))
        con = sqlite3.connect(str(db))
        try:
            con.execute("PRAGMA foreign_keys = ON")
            cur = con.execute(
                "INSERT INTO candidates (drive_id, hazard_id, hazard_class, track_key, track_id, chunk, "
                "first_timestamp, last_timestamp, frames_seen, max_mask_area, best_frame_index, best_timestamp, "
                "best_box, best_frame_path, crop_path, blurred_frame_path, blurred_crop_path, blur_status, "
                "lat, lon, heading, score, geotag_status, geotag_note) "
                "VALUES (?, NULL, ?, ?, ?, 0, 9.0, 11.0, 5, 100.0, 30, 10.0, '[0, 0, 10, 10]', ?, NULL, ?, NULL, "
                "'ok', ?, ?, ?, ?, 'ok', NULL)",
                (drive_id, cls, "0:%d" % n, n, frame_path, frame_path, lat, lon, heading, score))
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

    def dedup(self, drive_id, db=None):
        import importlib

        mod = importlib.import_module("backend.dedup")
        f = next((getattr(mod, n) for n in FUNC_NAMES if callable(getattr(mod, n, None))), None)
        assert f is not None, "no dedup function"
        sig = inspect.signature(f)
        p = next((n for n in DBP if n in sig.parameters), "db_path")
        res = f(drive_id, **{p: str(db or self.db)})
        if inspect.isawaitable(res):
            import asyncio

            asyncio.run(res)

    def lat_lon(self, i=0):
        return offset(VA[0], VA[1], north_m=i * 500.0)


def make_env(mp, tmp_path):
    import backend.config  # noqa: F401  (loads a local .env before the key is cleared)

    mp.setenv("ROADSCOUT_DB", str(tmp_path / "db" / "rs17.sqlite"))
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
            raise AssertionError("network access attempted during a task 17 test: %r" % (address,))
        return real_connect(self, address)

    mp.setattr(socket.socket, "connect", _guard)
    from fastapi.testclient import TestClient

    from backend.db import init_db
    from backend.main import app

    init_db()
    return TestClient(app)


@pytest.fixture()
def rs17_env(monkeypatch, tmp_path):
    client = make_env(monkeypatch, tmp_path)
    with client:
        yield Env(monkeypatch, tmp_path, client)


# ------------------------------------------------------------------ synthetic clip / gpx for pipeline re-runs

_CLIP = {}


def clip_bytes():
    if "b" not in _CLIP:
        if not FFMPEG:
            pytest.skip("ffmpeg not on PATH")
        d = Path(tempfile.mkdtemp(prefix="rs17media"))
        out = d / "clip.mp4"
        p = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10",
             "-t", "24", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-metadata",
             "creation_time=" + T0.strftime("%Y-%m-%dT%H:%M:%SZ"), str(out)], capture_output=True, timeout=240)
        assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
        _CLIP["b"] = out.read_bytes()
    return _CLIP["b"]


def gpx_bytes(seconds=60, speed=12.0):
    pts = []
    for s in range(seconds + 1):
        t = (T0 + timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%SZ")
        pts.append('<trkpt lat="%.8f" lon="%.8f"><ele>100</ele><time>%s</time></trkpt>'
                   % (38.88 + s * speed / M_PER_DEG, -77.30, t))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="rs17" '
            'xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>' + "".join(pts)
            + "</trkseg></trk></gpx>\n").encode("utf-8")


def upload(client):
    r = client.post("/api/drives", files={"video": ("clip.mp4", clip_bytes(), "video/mp4"),
                                          "gpx": ("t.gpx", gpx_bytes(), "application/gpx+xml")})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def run_pipeline_for(did, db):
    from backend.pipeline import run_pipeline
    from backend.providers import FakeProvider

    return run_pipeline(did, provider=FakeProvider("default"), db_path=str(db))
