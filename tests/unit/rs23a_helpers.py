"""Helpers for the task 23a hidden tests (read-only demo mode, static frontend, deploy files)."""
from __future__ import annotations

import asyncio
import importlib
import os
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = None
_cands = [Path(os.environ["ROADSCOUT_REPO"])] if os.environ.get("ROADSCOUT_REPO") else []
_cands += [HERE, *HERE.parents, *(p / "RoadScout" for p in HERE.parents)]
for _p in _cands:
    if (_p / "backend").is_dir() and (_p / "backend" / "__init__.py").exists():
        ROOT = _p
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
assert ROOT is not None, "cannot find the RoadScout repo (set ROADSCOUT_REPO)"

DEMO_BODY = {"detail": "This is a read-only demo."}
MUTATING = ("POST", "PUT", "PATCH", "DELETE")
CLASSES = ["pothole", "debris in lane", "damaged guardrail", "fallen or damaged sign", "faded lane markings"]
SPA_MARKER = "SPA-INDEX-MARKER-8841"
SECRET = "TOP-SECRET-OUTSIDE-THE-STATIC-DIR-5530"


# ------------------------------------------------------------------ building the app for a given environment

def build_app(mp, tmp_path, demo=None, static=None, extra_env=None):
    """Set the environment, (re)import backend.main so a build-time read of the settings is honoured, return app."""
    mp.setenv("ROADSCOUT_DB", str(Path(tmp_path) / "demo.sqlite"))
    mp.setenv("ROADSCOUT_STORAGE_DIR", str(Path(tmp_path) / "storage"))
    mp.setenv("ROADSCOUT_COST_LOG", "off")
    mp.setenv("ROADSCOUT_PROVIDER", "fake")
    mp.delenv("META_API_KEY", raising=False)
    for name, val in (("ROADSCOUT_DEMO", demo), ("ROADSCOUT_STATIC_DIR", static)):
        if val is None:
            mp.delenv(name, raising=False)
        else:
            mp.setenv(name, str(val))
    for k, v in (extra_env or {}).items():
        if v is None:
            mp.delenv(k, raising=False)
        else:
            mp.setenv(k, str(v))
    import backend.main as main

    main = importlib.reload(main)
    return main.app


def restore_app():
    for name in ("ROADSCOUT_DEMO", "ROADSCOUT_STATIC_DIR"):
        os.environ.pop(name, None)
    import backend.main as main

    importlib.reload(main)


def client_for(app, lifespan=False):
    from fastapi.testclient import TestClient

    return TestClient(app, raise_server_exceptions=True)


def asgi_get(app, raw_path: str, method="GET"):
    """Call the ASGI app with an exact (undecoded-by-the-client) request path. Returns (status, headers, body)."""
    from urllib.parse import unquote

    path = unquote(raw_path)
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
        "path": path, "raw_path": raw_path.encode("latin-1"), "query_string": b"", "root_path": "",
        "headers": [(b"host", b"testserver")], "client": ("127.0.0.1", 5000), "server": ("testserver", 80),
    }
    out = {"status": None, "headers": {}, "body": b""}

    async def run():
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await asyncio.sleep(3600)

        async def send(msg):
            if msg["type"] == "http.response.start":
                out["status"] = msg["status"]
                out["headers"] = {k.decode().lower(): v.decode() for k, v in msg["headers"]}
            elif msg["type"] == "http.response.body":
                out["body"] += msg.get("body", b"")

        await asyncio.wait_for(app(scope, receive, send), timeout=20)

    asyncio.run(run())
    return out["status"], out["headers"], out["body"]


# ------------------------------------------------------------------ a tiny static frontend

def make_static(tmp_path):
    """<tmp>/site/dist (the served folder) and <tmp>/site/secret.txt (outside it)."""
    site = Path(tmp_path) / "site"
    dist = site / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(f"<!doctype html><html><body>{SPA_MARKER}</body></html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log('asset-js-marker-1234');", encoding="utf-8")
    (dist / "assets" / "style.css").write_text("body{color:red}/*asset-css-marker-1234*/", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'><!--svg-marker-1234--></svg>",
                                      encoding="utf-8")
    (site / "secret.txt").write_text(SECRET, encoding="utf-8")
    (Path(tmp_path) / "secret.txt").write_text(SECRET, encoding="utf-8")
    return dist


# ------------------------------------------------------------------ seeding

class Seed:
    def __init__(self, tmp_path):
        self.tmp = Path(tmp_path)
        self.db = self.tmp / "demo.sqlite"
        self.storage = self.tmp / "storage"
        from backend.db import init_db

        init_db(self.db)
        self.n = 0

    def conn(self):
        c = sqlite3.connect(str(self.db))
        c.row_factory = sqlite3.Row
        return c

    def drive(self, drive_id="d1", status="done"):
        con = self.conn()
        try:
            with con:
                con.execute(
                    "INSERT INTO drives (id, started_at, started_at_source, clock_offset_seconds, video_path, "
                    "gpx_path, status, duration_seconds, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (drive_id, "2026-10-20T12:00:00+00:00", "metadata", 0.0, f"{drive_id}/video.mov",
                     f"{drive_id}/track.gpx", status, 120.0, "2026-10-20T12:00:00+00:00"))
        finally:
            con.close()
        return drive_id

    def image(self, drive_id, name, data=None):
        p = self.storage / drive_id / "candidates" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if data is not None else b"\xff\xd8\xff\xe0blurred-" + name.encode() + b"\xff\xd9")
        return p

    def candidate(self, drive_id="d1", blur="ok"):
        """A candidate with an unblurred original and (when blur == 'ok') a blurred copy. Returns (id, orig, blurred)."""
        self.n += 1
        cid = 100 + self.n
        orig = self.image(drive_id, f"c{cid}_best.jpg", b"\xff\xd8\xff\xe0UNBLURRED-ORIGINAL-%d\xff\xd9" % cid)
        blurred = self.image(drive_id, f"c{cid}_best_blur.jpg") if blur == "ok" else None
        con = self.conn()
        try:
            with con:
                con.execute(
                    "INSERT INTO candidates (id, drive_id, hazard_class, track_key, track_id, chunk, first_timestamp, "
                    "last_timestamp, frames_seen, max_mask_area, best_frame_index, best_timestamp, best_box, "
                    "best_frame_path, lat, lon, heading, score, geotag_status, blurred_frame_path, blur_status) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (cid, drive_id, "pothole", f"0:{cid}", cid, 0, 1.0, 5.0, 5, 400.0, 3, 2.0, "[1,2,10,10]",
                     str(orig), 38.88, -77.3, 0.0, 0.7, "ok", str(blurred) if blurred else None,
                     blur if blur else None))
                con.execute(
                    "INSERT INTO verdicts (candidate_id, drive_id, is_hazard, hazard_class, severity, lane, "
                    "confidence, reason, report_description, model, prompt_version, accepted, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (cid, drive_id, 1, "pothole", 2, "center", 0.9, "r", "d", "m", "v1", 1,
                     "2026-10-20T12:00:00+00:00"))
        finally:
            con.close()
        return cid, orig, blurred

    def hazard(self, photo=None, status="new", drive_id="d1"):
        con = self.conn()
        try:
            with con:
                cur = con.execute(
                    "INSERT INTO hazards (hazard_class, lat, lon, direction, severity, confidence, sightings, "
                    "candidate_count, first_seen, last_seen, photo_path, crop_path, status, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    ("pothole", 38.88, -77.3, 0.0, 2, 0.9, 1, 1, "2026-10-20T12:01:00+00:00",
                     "2026-10-20T12:01:00+00:00", str(photo) if photo else None, None, status,
                     "2026-10-20T12:00:00+00:00", "2026-10-20T12:00:00+00:00"))
                return cur.lastrowid
        finally:
            con.close()

    def hazard_status(self, hid):
        con = self.conn()
        try:
            return con.execute("SELECT status FROM hazards WHERE id = ?", (hid,)).fetchone()[0]
        finally:
            con.close()

    def drive_count(self):
        con = self.conn()
        try:
            return con.execute("SELECT COUNT(*) FROM drives").fetchone()[0]
        finally:
            con.close()


def mutating_routes(app):
    """[(method, path_template)] of every real route that changes state."""
    out = []
    for r in app.routes:
        path = getattr(r, "path", "")
        methods = getattr(r, "methods", None) or set()
        if not path.startswith("/api"):
            continue
        for m in sorted(methods):
            if m in MUTATING:
                out.append((m, path))
    return out


def fill(path, **ids):
    import re

    return re.sub(r"\{([^}:]+)(?::[^}]*)?\}", lambda m: str(ids.get(m.group(1), "1")), path)
