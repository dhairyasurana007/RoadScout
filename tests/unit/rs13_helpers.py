"""Shared helpers for the task 13 (skip stationary stretches) hidden tests.

Interface assumed (see the report to the main agent):

    backend/stationary.py   (the function may also live in backend.preprocess / detection / pipeline;
                             those are searched too)
        flag_stationary_frames(timestamps, gpx_track, *, video_start=None, clock_offset_seconds=None,
                               threshold_mps=None) -> list[bool]
            timestamps   video seconds (global frame timestamps), any order
            result       one bool per timestamp, same order; True = stationary (drop)
            speed < threshold -> True; speed >= threshold -> False (exactly at the threshold is moving)
            gpx_track None, or timestamp outside the track -> False (fail open)
            threshold default: env ROADSCOUT_STATIONARY_SPEED_MPS read at CALL time, else 0.5 m/s

    pipeline (run_pipeline, POST /api/drives/{id}/process)
        env ROADSCOUT_SKIP_STATIONARY       default on; "0", "false", "off", "no" turn the feature off
        env ROADSCOUT_STATIONARY_SPEED_MPS  threshold in m/s, default 0.5
        recorded counts: integers whose key contains "stationary" or "skipped" and "frame" / "chunk",
            visible in GET /api/drives/{id} or the drives row (e.g. stationary_frames_skipped,
            stationary_chunks_skipped); frames = frames flagged stationary, chunks = chunks never sent
            to the provider. 0 when nothing was skipped.
"""
from __future__ import annotations

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
M_PER_DEG = 6371008.8 * math.pi / 180.0
LAT0, LON0 = 38.8800, -77.3000

CLIP_SECONDS = 48
FPS = 3
CHUNK_SECONDS = 4
N_CHUNKS = CLIP_SECONDS // CHUNK_SECONDS  # 12
FRAMES_PER_CHUNK = FPS * CHUNK_SECONDS  # 12
N_FRAMES = N_CHUNKS * FRAMES_PER_CHUNK  # 144

# Standard GPS pattern (GPX time == video time when the offset is 0):
#   0 s to 12 s  moving at 15 m/s
#   12 s to 36 s stationary
#   36 s onwards moving at 15 m/s
MOVE, STOP = 15.0, 0.0
STD_SEGMENTS = [(12, MOVE), (36, STOP), (10_000, MOVE)]
MARGIN = 1.0  # seconds either side of a speed change in which either answer is accepted

DONE = {"done", "complete", "completed", "processed", "finished", "ready", "succeeded", "success"}
FAILED = {"failed", "error", "errored", "failure"}
TERMINAL = DONE | FAILED

FUNC_NAMES = ["flag_stationary_frames", "stationary_frame_flags", "stationary_flags", "find_stationary_frames",
              "compute_stationary_flags", "detect_stationary_frames", "flag_stationary", "stationary_mask"]
FUNC_MODULES = ["backend.stationary", "backend.preprocess", "backend.detection", "backend.pipeline",
                "backend.gpx_track"]
THRESH_PARAMS = ["threshold_mps", "speed_threshold_mps", "threshold", "min_speed_mps", "stationary_speed_mps",
                 "speed_threshold", "stationary_threshold_mps"]
START_PARAMS = ["video_start", "started_at", "video_started_at"]
OFFSET_PARAMS = ["clock_offset_seconds", "clock_offset", "offset_seconds", "clock_offset_s"]

ENV_SWITCH = "ROADSCOUT_SKIP_STATIONARY"
ENV_THRESHOLD = "ROADSCOUT_STATIONARY_SPEED_MPS"


# ------------------------------------------------------------------ expected classification

def definite_moving(t, segments=STD_SEGMENTS, shift=0.0):
    """True when video time t is at least MARGIN away from any speed change and moving."""
    return _definite(t, segments, shift, moving=True)


def definite_stationary(t, segments=STD_SEGMENTS, shift=0.0):
    return _definite(t, segments, shift, moving=False)


def _definite(t, segments, shift, moving):
    r = t + shift
    edges = [0.0] + [s for s, _ in segments]
    for i, (until, speed) in enumerate(segments):
        lo = edges[i]
        if lo <= r < until:
            if (speed >= 0.5) != moving:
                return False
            if r - lo < MARGIN and i > 0:
                return False
            if until - r <= MARGIN and until < 9_999:
                return False
            return True
    return False


# ------------------------------------------------------------------ GPX / video synthesis

def lats_for(segments, seconds):
    """Latitude at each whole second 0..seconds; segment speed applies to [s, s+1)."""
    lats = [LAT0]
    for s in range(seconds):
        speed = next(sp for until, sp in segments if s < until)
        lats.append(lats[-1] + speed / M_PER_DEG)
    return lats


def lat_at(lats, r):
    r = min(max(r, 0.0), len(lats) - 1)
    i = min(int(math.floor(r)), len(lats) - 2)
    return lats[i] + (lats[i + 1] - lats[i]) * (r - i)


def gpx_xml(segments=STD_SEGMENTS, seconds=60, start=T0, lon=LON0):
    lats = lats_for(segments, seconds)
    pts = []
    for s, lat in enumerate(lats):
        t = (start + timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%SZ")
        pts.append('<trkpt lat="%.8f" lon="%.8f"><ele>100</ele><time>%s</time></trkpt>' % (lat, lon, t))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<gpx version="1.1" creator="rs13" xmlns="http://www.topografix.com/GPX/1/1">'
            "<trk><name>synthetic</name><trkseg>" + "".join(pts) + "</trkseg></trk></gpx>\n").encode("utf-8")


_MEDIA_DIR = None
_CLIP = None


def clip_bytes():
    """48 s tiny synthetic clip (ffmpeg testsrc2) with creation_time = T0. Cached per process."""
    global _MEDIA_DIR, _CLIP
    if _CLIP is None:
        if not (FFMPEG and FFPROBE):
            pytest.skip("ffmpeg/ffprobe not on PATH")
        _MEDIA_DIR = Path(tempfile.mkdtemp(prefix="rs13media"))
        out = _MEDIA_DIR / "clip.mp4"
        p = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "testsrc2=size=320x180:rate=10", "-t", str(CLIP_SECONDS), "-c:v", "libx264", "-preset", "ultrafast",
             "-pix_fmt", "yuv420p", "-metadata", "creation_time=" + T0.strftime("%Y-%m-%dT%H:%M:%SZ"),
             str(out)], capture_output=True, timeout=240)
        assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
        _CLIP = out.read_bytes()
    return _CLIP


# ------------------------------------------------------------------ providers

def _make_spy_class():
    from backend.providers import FakeProvider, ModelProvider

    class Spy(ModelProvider):
        name = "spy"

        def __init__(self, scenario="default"):
            self.inner = FakeProvider(scenario)
            self.detect_calls = []
            self.verify_calls = []

        def detect(self, video_path, prompts):
            self.detect_calls.append(str(video_path))
            return self.inner.detect(video_path, prompts)

        def verify(self, images, *, class_prompt=None, prompt_version=None, prompt_text=None):
            self.verify_calls.append(list(images))
            return self.inner.verify(images, class_prompt=class_prompt, prompt_version=prompt_version)

    return Spy


def chunk_index_of(path):
    m = re.search(r"chunk_(\d+)", Path(path).name)
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------ scenario runner

LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "::"}
_CACHE = {}

_RUNNER_NAMES = ["run_pipeline", "process_drive", "run_drive_pipeline", "run_drive", "process", "run"]


def _find_runner():
    mod = importlib.import_module("backend.pipeline")
    for n in _RUNNER_NAMES:
        f = getattr(mod, n, None)
        if callable(f):
            return f
    pytest.fail("backend.pipeline has none of %s" % _RUNNER_NAMES)


def _pick(sig, names):
    for n in names:
        if n in sig.parameters:
            return n
    return None


def _rows(db, sql, params=()):
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params)]
    finally:
        con.close()


def _sha(p):
    import hashlib

    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_scenario(key, *, env=None, gpx=None, offset=None, tamper=None, via_http=False, spy=True, twice=False):
    """Upload the synthetic clip + GPX (default: the standard pattern), run the pipeline once, and return a
    plain-data record. Cached by ``key``.

    env     extra environment variables (value None = make sure it is unset)
    gpx     bytes of the GPX to upload (default gpx_xml())
    offset  clock_offset_seconds form field
    tamper  callable(drive_id, storage_gpx_path, db_path) run after upload and before processing
    """
    if key in _CACHE:
        return _CACHE[key]
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    if not (FFMPEG and FFPROBE):
        pytest.skip("ffmpeg/ffprobe not on PATH")
    clip = clip_bytes()
    env = dict(env or {})
    with tempfile.TemporaryDirectory(prefix="rs13run", ignore_cleanup_errors=True) as tmpdir:
        tmp = Path(tmpdir)
        db = tmp / "db" / "rs13.sqlite"
        with pytest.MonkeyPatch.context() as mp:
            import backend.config  # noqa: F401  (loads a local .env before the key is cleared)

            mp.setenv("ROADSCOUT_DB", str(db))
            mp.setenv("ROADSCOUT_STORAGE_DIR", str(tmp / "storage"))
            mp.setenv("ROADSCOUT_PROVIDER", "fake")
            mp.setenv("ROADSCOUT_PREPROCESS_FPS", str(FPS))
            mp.setenv("ROADSCOUT_CHUNK_SECONDS", str(CHUNK_SECONDS))
            mp.setenv("ROADSCOUT_COST_LOG", "off")
            for k in ("META_API_KEY", "ROADSCOUT_FAKE_SCENARIO", "ROADSCOUT_MIN_TRACK_FRAMES",
                      "ROADSCOUT_VERIFY_MIN_CONFIDENCE", "ROADSCOUT_PREPROCESS_MAX_HEIGHT", ENV_SWITCH,
                      ENV_THRESHOLD):
                mp.delenv(k, raising=False)
            for k, v in env.items():
                if v is None:
                    mp.delenv(k, raising=False)
                else:
                    mp.setenv(k, str(v))

            real_connect = socket.socket.connect

            def _guard(self, address):
                host = address[0] if isinstance(address, tuple) else None
                if host is not None and str(host) not in LOOPBACK:
                    raise AssertionError("network access attempted during a task 13 test: %r" % (address,))
                return real_connect(self, address)

            mp.setattr(socket.socket, "connect", _guard)

            from fastapi.testclient import TestClient

            from backend.db import init_db
            from backend.main import app
            from backend.storage import get_storage

            init_db()
            rec = {"key": key, "exception": None, "calls": [], "verify_calls": 0}
            with TestClient(app) as client:
                files = {"video": ("clip.mp4", clip, "video/mp4"),
                         "gpx": ("track.gpx", gpx if gpx is not None else gpx_xml(), "application/gpx+xml")}
                data = {"clock_offset_seconds": str(offset)} if offset is not None else None
                r = client.post("/api/drives", files=files, data=data)
                assert r.status_code == 201, r.text
                did = r.json()["id"]
                rec["drive_id"] = did
                row = _rows(db, "SELECT * FROM drives WHERE id = ?", (did,))[0]
                gpx_path = Path(get_storage().local_path(row["gpx_path"])) if row.get("gpx_path") else None
                video_path = Path(get_storage().local_path(row["video_path"]))
                rec["video_sha_before"] = _sha(video_path)
                if tamper is not None:
                    tamper(did, gpx_path, db)

                provider = _make_spy_class()() if spy else None
                if via_http:
                    _process_http(client, did)
                    rec["final"] = _wait(client, did)
                else:
                    f = _find_runner()
                    sig = inspect.signature(f)
                    kw = {}
                    dbp = _pick(sig, ["db_path", "db", "database", "database_path"])
                    if dbp:
                        kw[dbp] = str(db)
                    if provider is not None:
                        pp = _pick(sig, ["provider", "model_provider"])
                        if pp is None:
                            pytest.fail("runner has no provider parameter; signature %s" % (sig,))
                        kw[pp] = provider
                    try:
                        res = f(did, **kw)
                        if inspect.isawaitable(res):
                            import asyncio

                            res = asyncio.run(res)
                        rec["result_repr"] = repr(res)
                    except Exception as e:  # noqa: BLE001
                        rec["exception"] = repr(e)
                    rec["after_first"] = _snapshot(db, did)
                    if twice and rec["exception"] is None:
                        try:
                            f(did, **kw)
                            rec["second_ok"] = True
                        except Exception as e:  # noqa: BLE001
                            rec["second_ok"] = False
                            rec["second_exception"] = repr(e)
                rec["drive_json"] = client.get("/api/drives/%s" % did).json()
                rec["status"] = str(rec["drive_json"].get("status", "")).strip().lower()
                if provider is not None:
                    rec["calls"] = list(provider.detect_calls)
                    rec["verify_calls"] = len(provider.verify_calls)
                rec["row"] = _rows(db, "SELECT * FROM drives WHERE id = ?", (did,))[0]
                rec["detections"] = _rows(db, "SELECT * FROM detections WHERE drive_id = ? ORDER BY id", (did,))
                rec["candidates"] = _rows(db, "SELECT * FROM candidates WHERE drive_id = ? ORDER BY id", (did,))
                rec["verdicts"] = _rows(db, "SELECT * FROM verdicts WHERE drive_id = ?", (did,))
                rec["video_sha_after"] = _sha(video_path)
                rec["ft_docs"] = []
                for p in tmp.rglob("frame_timestamps.json"):
                    try:
                        rec["ft_docs"].append(json.loads(p.read_text(encoding="utf-8")))
                    except Exception:  # noqa: BLE001
                        pass
    _CACHE[key] = rec
    return rec


def _snapshot(db, did):
    return {"detections": len(_rows(db, "SELECT id FROM detections WHERE drive_id = ?", (did,))),
            "drive": _rows(db, "SELECT * FROM drives WHERE id = ?", (did,))[0]}


def _process_path(app):
    pat = re.compile(r"^/api/drives/\{[^}/]+\}/(process|run|pipeline|start|analy[sz]e)/?$")
    for path, ops in app.openapi().get("paths", {}).items():
        if pat.match(path) and "post" in ops:
            return path
    pytest.fail("no POST /api/drives/{id}/process route")


def _process_http(client, did):
    from backend.main import app

    path = re.sub(r"\{[^}/]+\}", did, _process_path(app))
    r = client.post(path)
    assert r.status_code in (200, 201, 202), "process returned %s: %s" % (r.status_code, r.text)


def _wait(client, did, timeout=180):
    deadline = time.time() + timeout
    d = None
    while time.time() < deadline:
        d = client.get("/api/drives/%s" % did).json()
        if str(d.get("status", "")).strip().lower() in TERMINAL:
            return d
        time.sleep(0.05)
    pytest.fail("drive never reached a terminal status; last: %s" % (d,))


# ------------------------------------------------------------------ derived facts

def ts_set(rec):
    return {round(float(d["frame_timestamp"]), 4) for d in rec["detections"]}


def chunk_set(rec):
    return {int(d["chunk"]) for d in rec["detections"]}


def is_done(rec):
    return rec["exception"] is None and rec["status"] in DONE


def all_frame_times():
    return [round(i / FPS, 4) for i in range(N_FRAMES)]


def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, prefix + "." + str(k) if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_flatten(v, "%s[%d]" % (prefix, i)))
    elif isinstance(obj, (str, bytes)):
        s = obj.decode("utf-8", "replace") if isinstance(obj, bytes) else obj
        st = s.strip()
        if st[:1] in "{[" and st[-1:] in "}]":
            try:
                out.update(_flatten(json.loads(st), prefix))
                return out
            except Exception:  # noqa: BLE001
                pass
        out[prefix] = obj
    else:
        out[prefix] = obj
    return out


def recorded_counts(rec):
    """{key: int} for integer fields whose name mentions stationary/skipped, from the drive JSON,
    the drives row, and the run_pipeline return value (when it is JSON-like)."""
    found = {}
    for label, src in (("json", rec.get("drive_json")), ("row", rec.get("row"))):
        for k, v in _flatten(src or {}).items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if re.search(r"threshold|mps|speed|enabled|switch|min_|max_", k, re.I):
                continue
            if re.search(r"station|skip", k, re.I) and float(v) == int(v):
                found["%s:%s" % (label, k)] = int(v)
    return found


def frame_counts(counts):
    return {k: v for k, v in counts.items() if re.search(r"frame", k, re.I)}


def chunk_counts(counts):
    return {k: v for k, v in counts.items() if re.search(r"chunk", k, re.I)}


# ------------------------------------------------------------------ function-level adapter

def find_flag_function():
    for modname in FUNC_MODULES:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        for n in FUNC_NAMES:
            f = getattr(mod, n, None)
            if callable(f):
                return f
    pytest.fail("none of %s found in %s" % (FUNC_NAMES, FUNC_MODULES))


def call_flags(timestamps, track, *, threshold=None, video_start=T0, offset=0.0):
    f = find_flag_function()
    sig = inspect.signature(f)
    kw = {}
    if video_start is not None:
        p = _pick(sig, START_PARAMS)
        if p is None:
            pytest.fail("flag function has no video start parameter; signature %s" % (sig,))
        kw[p] = video_start
    if offset is not None:
        p = _pick(sig, OFFSET_PARAMS)
        if p is None:
            pytest.fail("flag function has no clock offset parameter; signature %s" % (sig,))
        kw[p] = offset
    if threshold is not None:
        p = _pick(sig, THRESH_PARAMS)
        if p is None:
            pytest.fail("flag function has no threshold parameter; signature %s" % (sig,))
        kw[p] = threshold
    res = f(list(timestamps), track, **kw)
    return normalise(res, len(timestamps))


def normalise(res, n):
    if isinstance(res, (set, frozenset)):
        return [i in res for i in range(n)]
    try:
        out = [bool(x) for x in list(res)]
    except TypeError:
        pytest.fail("flag function returned %r, expected one bool per timestamp" % (res,))
    assert len(out) == n, "expected %d flags, got %d" % (n, len(out))
    return out


def tag_track(speed, seconds=30, start=T0):
    """Track whose <speed> tags are all `speed` (positions are irrelevant when tags are present)."""
    from backend.gpx_track import GpxTrack, TrackPoint

    pts = [TrackPoint(start + timedelta(seconds=s), LAT0 + s * 1e-6, LON0, speed) for s in range(seconds + 1)]
    return GpxTrack.from_points(pts)


def pattern_track(segments=STD_SEGMENTS, seconds=60, start=T0):
    from backend.gpx_track import GpxTrack, TrackPoint

    lats = lats_for(segments, seconds)
    return GpxTrack.from_points([TrackPoint(start + timedelta(seconds=s), lat, LON0) for s, lat in enumerate(lats)])
