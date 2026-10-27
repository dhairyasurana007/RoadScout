"""Shared helpers and fixtures for the task 11 (results page backend) hidden tests.

Interface assumed (see the report to the main agent):

    GET /api/drives/{id}/hazards   -> 200 JSON list (a dict wrapping a list under hazards/items/results is tolerated)
                                      of the ACCEPTED verdicts of the drive. Each item:
                                          candidate_id, hazard_class, severity (1-3), confidence,
                                          lat, lon (null when not geotagged), reason, report_description,
                                          photo_url (relative "/api/..." URL), crop_url (optional, may be null)
                                      unknown drive -> 404; drive without accepted hazards or not processed -> 200 []
    GET <photo_url>                -> 200 image/jpeg of the candidate's best frame
    GET <crop_url>                 -> 200 image/jpeg of the candidate's crop
    unknown candidate / candidate of another drive / traversal ids -> 404 (4xx, never 200 or 5xx)
    GET /api/drives/{id}           -> drive JSON with status, stage, error, failed_stage
"""
from __future__ import annotations

import importlib
import io
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
from urllib.parse import urlparse

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import os  # noqa: E402

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
TERMINAL = DONE | FAILED

LIST_SEGMENTS = ["hazards", "verified-hazards", "verified_hazards", "results", "verified", "findings"]
LIST_KEYS = ["hazards", "items", "results", "data", "verified", "candidates"]
PHOTO_KEYS = ["photo_url", "image_url", "frame_url", "best_frame_url", "photo", "image"]
CROP_KEYS = ["crop_url", "crop_photo_url", "crop_image_url", "crop"]
ID_KEYS = ["candidate_id", "id"]
CLASS_KEYS = ["hazard_class", "class", "class_name", "hazard_type"]
LAT_KEYS = ["lat", "latitude"]
LON_KEYS = ["lon", "lng", "longitude"]
REASON_KEYS = ["reason", "report_description", "description"]


def status_of(d):
    return str(d.get("status", "")).strip().lower()


def first_key(item, names):
    for n in names:
        if n in item:
            return n
    return None


def val(item, names, default=None):
    k = first_key(item, names)
    return item[k] if k else default


# ------------------------------------------------------------------ synthetic media

_CLIP = None


def clip_bytes():
    global _CLIP
    if _CLIP is None:
        if not (FFMPEG and FFPROBE):
            pytest.skip("ffmpeg/ffprobe not on PATH")
        d = Path(tempfile.mkdtemp(prefix="rs11media"))
        out = d / "clip.mp4"
        p = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "testsrc2=size=640x360:rate=30", "-t", "12", "-c:v", "libx264", "-preset", "ultrafast",
             "-pix_fmt", "yuv420p", "-metadata", "creation_time=" + T0.strftime("%Y-%m-%dT%H:%M:%SZ"),
             str(out)], capture_output=True, timeout=240)
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


# ------------------------------------------------------------------ provider injection

def patch_everywhere(mp, mod_name, attr, replacement):
    importlib.import_module("backend.main")
    importlib.import_module("backend.pipeline")
    mod = importlib.import_module(mod_name)
    orig = getattr(mod, attr)
    for name, m in list(sys.modules.items()):
        if m is not None and name.startswith("backend") and getattr(m, attr, None) is orig:
            mp.setattr(m, attr, replacement)
    return orig


def make_hook_provider(*, on_detect=None, fail_detect_on=None):
    from backend.providers import FakeProvider, ModelProvider

    class Hook(ModelProvider):
        name = "hook"

        def __init__(self):
            self.inner = FakeProvider("default")
            self.n = 0

        def detect(self, video_path, prompts):
            # task 15: the privacy blur calls detect() on saved JPEGs; those are not chunk calls
            if Path(str(video_path)).suffix.lower() in (".jpg", ".jpeg", ".png"):
                return self.inner.detect(video_path, prompts)
            n = self.n
            self.n += 1
            if on_detect:
                on_detect(n)
            if fail_detect_on is not None and n == fail_detect_on:
                raise RuntimeError("boom-detect-xyz")
            return self.inner.detect(video_path, prompts)

        def verify(self, images, *, class_prompt=None, prompt_version=None, prompt_text=None):
            return self.inner.verify(images, class_prompt=class_prompt, prompt_version=prompt_version)

    return Hook()


# ------------------------------------------------------------------ environment

LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "::"}


class Env:
    def __init__(self, mp, tmp_path, client):
        self.mp = mp
        self.tmp = Path(tmp_path)
        self.db = self.tmp / "db" / "rs11.sqlite"
        self.storage = self.tmp / "storage"
        self.client = client

    # --- drives
    def upload(self):
        files = {"video": ("clip.mp4", clip_bytes(), "video/mp4"),
                 "gpx": ("track.gpx", gpx_bytes(), "application/gpx+xml")}
        r = self.client.post("/api/drives", files=files)
        assert r.status_code == 201, r.text
        return r.json()["id"]

    def get(self, drive_id):
        r = self.client.get("/api/drives/%s" % drive_id)
        assert r.status_code == 200, r.text
        return r.json()

    def process_route(self):
        from backend.main import app

        pat = re.compile(r"^/api/drives/\{[^}/]+\}/(process|run|pipeline|start|analy[sz]e|reprocess)/?$")
        for path, ops in app.openapi().get("paths", {}).items():
            if pat.match(path) and "post" in ops:
                return path
        pytest.fail("no POST /api/drives/{id}/process route")

    def start(self, drive_id):
        path = re.sub(r"\{[^}/]+\}", drive_id, self.process_route())
        r = self.client.post(path)
        assert r.status_code in (200, 201, 202), r.text
        return r

    def wait(self, drive_id, timeout=180):
        deadline = time.time() + timeout
        d = None
        while time.time() < deadline:
            d = self.get(drive_id)
            if status_of(d) in TERMINAL:
                return d
            time.sleep(0.05)
        pytest.fail("drive %s never reached a terminal status; last: %s" % (drive_id, d))

    def process(self, drive_id):
        self.start(drive_id)
        return self.wait(drive_id)

    def upload_and_process(self):
        did = self.upload()
        d = self.process(did)
        assert status_of(d) in DONE, d
        return did

    # --- db
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
            con.execute(sql, params)
            con.commit()
        finally:
            con.close()

    def verdict_columns(self):
        return [r["name"] for r in self.q("PRAGMA table_info(verdicts)")]

    def candidate_columns(self):
        return [r["name"] for r in self.q("PRAGMA table_info(candidates)")]

    def accepted_candidates(self, drive_id):
        """Rows of candidates (+ verdict fields prefixed v_) whose verdict is an accepted hazard."""
        if "accepted" not in self.verdict_columns():
            pytest.skip("verdicts.accepted column not found")
        rows = self.q(
            "SELECT c.*, v.is_hazard AS v_is_hazard, v.severity AS v_severity, v.confidence AS v_confidence, "
            "v.hazard_class AS v_class, v.reason AS v_reason, v.report_description AS v_desc, "
            "v.accepted AS v_accepted FROM candidates c JOIN verdicts v ON v.candidate_id = c.id "
            "WHERE c.drive_id = ? ORDER BY c.id", (drive_id,))
        return [r for r in rows if r["v_accepted"] and r["v_is_hazard"]]

    def all_candidates(self, drive_id):
        return self.q("SELECT * FROM candidates WHERE drive_id = ? ORDER BY id", (drive_id,))

    # --- files
    def resolve_file(self, stored):
        if not stored:
            return None
        cands = [Path(stored)]
        try:
            from backend.storage import get_storage

            cands.append(Path(get_storage().local_path(stored)))
        except Exception:  # noqa: BLE001
            pass
        cands += [self.storage / stored, self.tmp / stored]
        if ROOT:
            cands.append(ROOT / stored)
        for c in cands:
            try:
                if c.is_file():
                    return c
            except OSError:
                pass
        return None

    # --- api
    def list_route(self):
        from backend.main import app

        found = {}
        for path, ops in app.openapi().get("paths", {}).items():
            m = re.match(r"^/api/drives/\{[^}/]+\}/([A-Za-z_-]+)/?$", path)
            if m and "get" in ops:
                found[m.group(1)] = path
        for seg in LIST_SEGMENTS:
            if seg in found:
                return found[seg]
        pytest.fail("no GET /api/drives/{id}/(%s) route; GET routes under a drive: %s"
                    % ("|".join(LIST_SEGMENTS), sorted(found)))

    def list_url(self, drive_id):
        return re.sub(r"\{[^}/]+\}", drive_id, self.list_route())

    def hazards_response(self, drive_id):
        return self.client.get(self.list_url(drive_id))

    def hazards(self, drive_id):
        r = self.hazards_response(drive_id)
        assert r.status_code == 200, "%s -> %s %s" % (self.list_url(drive_id), r.status_code, r.text)
        return as_list(r.json())

    def photo_url(self, item, crop=False):
        keys = CROP_KEYS if crop else PHOTO_KEYS
        k = first_key(item, keys)
        if k is None or not item[k]:
            return None
        u = item[k]
        assert isinstance(u, str), "%s must be a URL string, got %r" % (k, u)
        p = urlparse(u)
        return p.path + ("?" + p.query if p.query else "")


def as_list(body):
    if isinstance(body, list):
        return body
    assert isinstance(body, dict), "expected a JSON list (or an object wrapping one), got %r" % (type(body),)
    for k in LIST_KEYS:
        if isinstance(body.get(k), list):
            return body[k]
    pytest.fail("no list found in response object keys %s" % sorted(body))


def cid_of(item):
    k = first_key(item, ID_KEYS)
    assert k, "item has no candidate id key (candidate_id or id): %s" % sorted(item)
    return item[k]


def is_jpeg(data):
    if not data.startswith(b"\xff\xd8"):
        return False
    from PIL import Image

    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        return im.format == "JPEG"
    except Exception:  # noqa: BLE001
        return False


def make_env(mp, tmp_path, scenario=None):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    if not (FFMPEG and FFPROBE):
        pytest.skip("ffmpeg/ffprobe not on PATH")
    import backend.config  # noqa: F401  (loads a local .env before the key is cleared)

    mp.setenv("ROADSCOUT_DB", str(tmp_path / "db" / "rs11.sqlite"))
    mp.setenv("ROADSCOUT_STORAGE_DIR", str(tmp_path / "storage"))
    mp.setenv("ROADSCOUT_PROVIDER", "fake")
    mp.setenv("ROADSCOUT_PREPROCESS_FPS", "3")
    mp.setenv("ROADSCOUT_CHUNK_SECONDS", "4")
    mp.setenv("ROADSCOUT_COST_LOG", "off")
    for k in ("META_API_KEY", "ROADSCOUT_FAKE_SCENARIO", "ROADSCOUT_MIN_TRACK_FRAMES",
              "ROADSCOUT_VERIFY_MIN_CONFIDENCE", "ROADSCOUT_PREPROCESS_MAX_HEIGHT"):
        mp.delenv(k, raising=False)
    if scenario:
        mp.setenv("ROADSCOUT_FAKE_SCENARIO", scenario)

    real_connect = socket.socket.connect

    def _guard(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if host is not None and str(host) not in LOOPBACK:
            raise AssertionError("network access attempted during a task 11 test: %r" % (address,))
        return real_connect(self, address)

    mp.setattr(socket.socket, "connect", _guard)

    from fastapi.testclient import TestClient

    from backend.db import init_db
    from backend.main import app

    init_db()
    return TestClient(app), Env


@pytest.fixture()
def rs11_env(monkeypatch, tmp_path):
    client, Env_ = make_env(monkeypatch, tmp_path)
    with client:
        yield Env_(monkeypatch, tmp_path, client)


@pytest.fixture()
def rs11_env_fp(monkeypatch, tmp_path):
    """Same, with the fake provider's false_positive scenario (verifier rejects / non-hazards)."""
    client, Env_ = make_env(monkeypatch, tmp_path, scenario="false_positive")
    with client:
        yield Env_(monkeypatch, tmp_path, client)


@pytest.fixture()
def rs11_env_empty(monkeypatch, tmp_path):
    client, Env_ = make_env(monkeypatch, tmp_path, scenario="empty")
    with client:
        yield Env_(monkeypatch, tmp_path, client)


@pytest.fixture()
def rs11_done(rs11_env):
    """Env with one default-scenario drive processed end to end. Returns (env, drive_id)."""
    did = rs11_env.upload_and_process()
    return rs11_env, did
