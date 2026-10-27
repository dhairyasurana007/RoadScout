"""Shared helpers and fixtures for the task 10 (pipeline runner) hidden tests.

Interface assumed (see the report to the main agent):

    backend/pipeline.py
        run_pipeline(drive_id, *, provider=None, db_path=None)
            (alias names tolerated: process_drive, run_drive_pipeline, run_drive, process, run)
            runs preprocess -> detection -> aggregation (+geotag) -> verification for one drive,
            updates drives.status (and optionally a stage/progress field), records failures.
    POST /api/drives/{id}/process      -> 200/202 JSON with the drive id and its status, job runs in background
    GET  /api/drives/{id}              -> drive JSON with `status` and, after a failure, `error`
    status values                      -> uploaded -> (any in-progress values) -> done | failed
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
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

DONE = {"done", "complete", "completed", "processed", "finished", "ready", "succeeded", "success"}
FAILED = {"failed", "error", "errored", "failure"}
INITIAL = {"uploaded", "queued", "pending", "new"}
TERMINAL = DONE | FAILED

RUNNER_NAMES = ["run_pipeline", "process_drive", "run_drive_pipeline", "run_drive", "process", "run"]
DB_PARAMS = ["db_path", "db", "database", "database_path"]
PROVIDER_PARAMS = ["provider", "model_provider"]
ERROR_KEYS = ["error", "error_message", "last_error", "failure", "failure_message", "error_text"]
STAGE_KEY_RE = re.compile(r"^(stage|current_stage|step|current_step|progress|phase)", re.I)

CHUNKS = 3  # 12 s clip, 4 s chunks, 3 fps -> 3 chunks of 12 frames; fake default gives 1 track per chunk
DETECTIONS = 36


def status_of(d):
    return str(d.get("status", "")).strip().lower()


def err_of(d):
    for k in ERROR_KEYS:
        v = d.get(k)
        if v:
            return str(v)
    return ""


def stage_pair(d):
    """(status, stage-like fields) used to prove that two moments are different stages."""
    extra = tuple(sorted((k, str(v)) for k, v in d.items() if STAGE_KEY_RE.match(k) and v is not None))
    return (status_of(d), extra)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ synthetic media

_MEDIA_DIR = None
_CLIP = None


def clip_bytes():
    """12 s 640x360 synthetic clip (ffmpeg testsrc2) with creation_time = T0. Cached per process."""
    global _MEDIA_DIR, _CLIP
    if _CLIP is None:
        if not (FFMPEG and FFPROBE):
            pytest.skip("ffmpeg/ffprobe not on PATH")
        _MEDIA_DIR = Path(tempfile.mkdtemp(prefix="rs10media"))
        out = _MEDIA_DIR / "clip.mp4"
        p = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "testsrc2=size=640x360:rate=30", "-t", "12", "-c:v", "libx264", "-preset", "ultrafast",
             "-pix_fmt", "yuv420p", "-metadata", "creation_time=" + T0.strftime("%Y-%m-%dT%H:%M:%SZ"),
             str(out)], capture_output=True, timeout=240)
        assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
        _CLIP = out.read_bytes()
    return _CLIP


def gpx_bytes(start=T0, seconds=60):
    """Synthetic GPX: made-up coordinates, 1 point per second, ~10 m/s north."""
    pts, lat = [], 38.88
    for s in range(seconds + 1):
        t = (start + timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%SZ")
        pts.append('<trkpt lat="%.7f" lon="-77.3000000"><ele>100</ele><time>%s</time></trkpt>' % (lat, t))
        lat += 10.0 / M_PER_DEG
    body = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<gpx version="1.1" creator="roadscout-tests" xmlns="http://www.topografix.com/GPX/1/1">'
            "<trk><name>synthetic</name><trkseg>" + "".join(pts) + "</trkseg></trk></gpx>\n")
    return body.encode("utf-8")


# ------------------------------------------------------------------ patching

def patch_everywhere(mp, mod_name, attr, replacement):
    """Replace mod_name.attr and every backend module that imported the same object by name."""
    importlib.import_module("backend.main")
    find_module()
    mod = importlib.import_module(mod_name)
    orig = getattr(mod, attr)
    for name, m in list(sys.modules.items()):
        if m is not None and name.startswith("backend") and getattr(m, attr, None) is orig:
            mp.setattr(m, attr, replacement)
    return orig


# ------------------------------------------------------------------ interface adapter

def find_module():
    return importlib.import_module("backend.pipeline")


def find_runner():
    mod = find_module()
    for n in RUNNER_NAMES:
        f = getattr(mod, n, None)
        if callable(f):
            return f
    pytest.fail("backend.pipeline has none of the expected functions %s" % RUNNER_NAMES)


def _pick(sig, names):
    for n in names:
        if n in sig.parameters:
            return n
    return None


# ------------------------------------------------------------------ providers

def _make_spy_class():
    from backend.providers import FakeProvider, ModelProvider

    class Spy(ModelProvider):
        """Fake provider that records calls, can fail on chosen calls and run hooks."""

        name = "spy"

        def __init__(self, scenario="default", *, fail_detect_on=None, fail_verify_on=None,
                     on_detect=None, on_verify=None):
            self.inner = FakeProvider(scenario)
            self.fail_detect_on = fail_detect_on
            self.fail_verify_on = fail_verify_on
            self.on_detect = on_detect
            self.on_verify = on_verify
            self.detect_calls = []
            self.verify_calls = []
            self.events = []

        def detect(self, video_path, prompts):
            n = len(self.detect_calls)
            self.detect_calls.append(str(video_path))
            self.events.append("detect")
            if self.on_detect:
                self.on_detect(self, n, video_path)
            if self.fail_detect_on is not None and n == self.fail_detect_on:
                raise RuntimeError("boom-detect-xyz")
            return self.inner.detect(video_path, prompts)

        def verify(self, images, *, class_prompt=None, prompt_version=None, prompt_text=None):
            n = len(self.verify_calls)
            self.verify_calls.append({"images": list(images), "class_prompt": class_prompt})
            self.events.append("verify")
            if self.on_verify:
                self.on_verify(self, n)
            if self.fail_verify_on is not None and n == self.fail_verify_on:
                raise RuntimeError("boom-verify-xyz")
            return self.inner.verify(images, class_prompt=class_prompt, prompt_version=prompt_version)

    return Spy


def spy(scenario="default", **kw):
    return _make_spy_class()(scenario, **kw)


# ------------------------------------------------------------------ environment

LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "::"}


class Env:
    def __init__(self, mp, tmp_path, client):
        self.mp = mp
        self.tmp = Path(tmp_path)
        self.db = self.tmp / "db" / "rs10.sqlite"
        self.storage = self.tmp / "storage"
        self.client = client
        self.drives = []

    # --- drives
    def upload(self, *, offset=None, gpx=None, video=None):
        files = {"video": ("clip.mp4", video if video is not None else clip_bytes(), "video/mp4"),
                 "gpx": ("track.gpx", gpx if gpx is not None else gpx_bytes(), "application/gpx+xml")}
        data = {"clock_offset_seconds": str(offset)} if offset is not None else None
        r = self.client.post("/api/drives", files=files, data=data)
        assert r.status_code == 201, r.text
        d = r.json()
        self.drives.append(d["id"])
        return d["id"]

    def get(self, drive_id):
        r = self.client.get("/api/drives/%s" % drive_id)
        assert r.status_code == 200, r.text
        return r.json()

    def row(self, drive_id):
        con = sqlite3.connect(str(self.db))
        con.row_factory = sqlite3.Row
        try:
            r = con.execute("SELECT * FROM drives WHERE id = ?", (drive_id,)).fetchone()
            return dict(r) if r else None
        finally:
            con.close()

    def video_file(self, drive_id):
        from backend.storage import get_storage

        return Path(get_storage().local_path(self.row(drive_id)["video_path"]))

    def gpx_file(self, drive_id):
        from backend.storage import get_storage

        return Path(get_storage().local_path(self.row(drive_id)["gpx_path"]))

    # --- db reads
    def q(self, sql, params=()):
        con = sqlite3.connect(str(self.db))
        con.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in con.execute(sql, params)]
        finally:
            con.close()

    def counts(self, drive_id):
        return {
            "detections": self.q("SELECT COUNT(*) n FROM detections WHERE drive_id = ?", (drive_id,))[0]["n"],
            "candidates": self.q("SELECT COUNT(*) n FROM candidates WHERE drive_id = ?", (drive_id,))[0]["n"],
            "verdicts": self.q("SELECT COUNT(*) n FROM verdicts v JOIN candidates c ON c.id = v.candidate_id "
                               "WHERE c.drive_id = ?", (drive_id,))[0]["n"],
            "all_verdicts": self.q("SELECT COUNT(*) n FROM verdicts WHERE drive_id = ?", (drive_id,))[0]["n"],
        }

    def candidates(self, drive_id):
        return self.q("SELECT * FROM candidates WHERE drive_id = ? ORDER BY id", (drive_id,))

    def verdicts(self, drive_id):
        return self.q("SELECT * FROM verdicts WHERE drive_id = ? ORDER BY candidate_id", (drive_id,))

    def orphan_verdicts(self):
        return self.q("SELECT v.candidate_id FROM verdicts v LEFT JOIN candidates c ON c.id = v.candidate_id "
                      "WHERE c.id IS NULL")

    def files_for(self, drive_id):
        out = []
        did = re.sub(r"[^A-Za-z0-9_-]", "_", drive_id)
        for p in self.tmp.rglob("*"):
            if p.is_file():
                parts = [x for x in p.relative_to(self.tmp).parts]
                if any(did in part for part in parts):
                    out.append(p)
        return out

    # --- running
    def run(self, drive_id, provider=None):
        """Call the runner directly. Returns (result, exception_or_None)."""
        f = find_runner()
        sig = inspect.signature(f)
        kw = {}
        db_p = _pick(sig, DB_PARAMS)
        if db_p:
            kw[db_p] = str(self.db)
        if provider is not None:
            p = _pick(sig, PROVIDER_PARAMS)
            if p is None:
                pytest.fail("runner has no provider parameter; signature %s" % (sig,))
            kw[p] = provider
        try:
            res = f(drive_id, **kw)
            if inspect.isawaitable(res):
                res = asyncio.run(res)
            return res, None
        except Exception as e:  # noqa: BLE001
            return None, e

    def wait(self, drive_id, timeout=180):
        deadline = time.time() + timeout
        d = None
        while time.time() < deadline:
            d = self.get(drive_id)
            if status_of(d) in TERMINAL:
                return d
            time.sleep(0.05)
        pytest.fail("drive %s never reached a terminal status; last: %s" % (drive_id, d))

    def process_route(self):
        from backend.main import app

        pat = re.compile(r"^/api/drives/\{[^}/]+\}/(process|run|pipeline|start|analy[sz]e|reprocess)/?$")
        # OpenAPI flattens included routers on every FastAPI version; app.routes does not.
        paths = app.openapi().get("paths", {})
        for path, ops in paths.items():
            if pat.match(path) and "post" in ops:
                return path
        pytest.fail("no POST /api/drives/{id}/process route; routes: %s" % sorted(paths))

    def post_process(self, drive_id):
        path = re.sub(r"\{[^}/]+\}", drive_id, self.process_route())
        return self.client.post(path)

    def process_via_http(self, drive_id):
        r = self.post_process(drive_id)
        assert r.status_code in (200, 201, 202), "process returned %s: %s" % (r.status_code, r.text)
        return r

    def use_provider(self, provider):
        """Make get_provider() return `provider` everywhere in backend (for HTTP-started jobs)."""
        patch_everywhere(self.mp, "backend.providers", "get_provider", lambda *a, **k: provider)


@pytest.fixture()
def rs10_env(monkeypatch, tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    if not (FFMPEG and FFPROBE):
        pytest.skip("ffmpeg/ffprobe not on PATH")
    import backend.config  # noqa: F401  (loads a local .env before we clear the key)

    mp = monkeypatch
    mp.setenv("ROADSCOUT_DB", str(tmp_path / "db" / "rs10.sqlite"))
    mp.setenv("ROADSCOUT_STORAGE_DIR", str(tmp_path / "storage"))
    mp.setenv("ROADSCOUT_PROVIDER", "fake")
    mp.setenv("ROADSCOUT_PREPROCESS_FPS", "3")
    mp.setenv("ROADSCOUT_CHUNK_SECONDS", "4")
    mp.setenv("ROADSCOUT_COST_LOG", "off")
    for k in ("META_API_KEY", "ROADSCOUT_FAKE_SCENARIO", "ROADSCOUT_MIN_TRACK_FRAMES",
              "ROADSCOUT_VERIFY_MIN_CONFIDENCE", "ROADSCOUT_PREPROCESS_MAX_HEIGHT"):
        mp.delenv(k, raising=False)

    real_connect = socket.socket.connect

    def _guard(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if host is not None and str(host) not in LOOPBACK:
            raise AssertionError("network access attempted during a task 10 test: %r" % (address,))
        return real_connect(self, address)

    mp.setattr(socket.socket, "connect", _guard)

    from fastapi.testclient import TestClient

    from backend.db import init_db
    from backend.main import app

    init_db()
    find_module()
    with TestClient(app) as client:
        yield Env(mp, tmp_path, client)
