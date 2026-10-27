"""Shared helpers and fixtures for the task 8 (track aggregation, best frame, geotag) hidden tests.

Interface assumed (see the report to the main agent):

    backend/candidates.py
        aggregate_drive(drive_id, chunks, *, gpx=None, video_start=None, clock_offset_seconds=None,
                        min_frames=None, output_dir=None, db_path=None)
        load_candidates(drive_id, *, db_path=None) -> list[dict]
    chunks       same kinds as run_detection: PreprocessResult, frame_timestamps.json path, or folder
    gpx          GpxTrack (or a GPX path); None -> built from the drive row (drives.gpx_path via storage)
    video_start / clock_offset_seconds   default to the drives row (started_at, clock_offset_seconds);
                 explicit values override
    min_frames   default env ROADSCOUT_MIN_TRACK_FRAMES, else a built-in default (> 2 and <= 8)
    output_dir   where best frame + crop images go; default <chunks folder>/candidates

    table candidates: id, drive_id, hazard_id (NULL), hazard_class, track_id, track_key,
        first_timestamp, last_timestamp, frames_seen, max_mask_area, score, best_frame_path,
        crop_path, latitude, longitude, heading
"""
from __future__ import annotations

import importlib
import inspect
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
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

W, H = 160, 96  # synthetic chunk frame size
FPS = 3
FPC = 8  # frames per synthetic chunk
T0 = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
LAT0, LON0 = 38.8800, -77.3000
M_PER_DEG = 6371008.8 * math.pi / 180.0

FUNC_NAMES = ["aggregate_drive", "build_candidates", "aggregate_candidates", "aggregate_tracks",
              "run_aggregation", "aggregate"]
GPX_PARAMS = ["gpx_track", "gpx", "track", "gpx_track"]
START_PARAMS = ["video_start", "started_at", "video_started_at"]
OFFSET_PARAMS = ["clock_offset_seconds", "clock_offset", "offset_seconds", "clock_offset_s"]
MIN_PARAMS = ["min_frames", "min_track_frames", "min_track_length", "min_frames_seen", "min_length"]
OUT_PARAMS = ["output_dir", "out_dir", "images_dir", "frames_dir", "image_dir"]
DB_PARAMS = ["db_path", "db", "database", "database_path"]
MIN_ENV = "ROADSCOUT_MIN_TRACK_FRAMES"


# ------------------------------------------------------------------ synthetic chunk videos


def _ffmpeg(args):
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
                       capture_output=True, timeout=240)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")


def make_spec(n_chunks=3, fpc=FPC, blur=None, levels=None):
    """spec[chunk][frame] = (level, blur_sigma). Levels identify frames; sigma 0 is sharp."""
    blur = blur or {}
    levels = levels or {}
    spec = []
    for c in range(n_chunks):
        chunk = []
        for f in range(fpc):
            g = c * fpc + f
            chunk.append((levels.get((c, f), 30 + 10 * (g % 16)), float(blur.get((c, f), 0.0))))
        spec.append(chunk)
    return spec


def _frame_array(level, sigma, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    noise = rng.integers(-25, 26, size=(H, W)).astype("float64")
    if sigma > 0:
        from scipy.ndimage import gaussian_filter

        noise = gaussian_filter(noise, sigma=sigma)
    return np.clip(level + noise, 0, 255).astype("uint8")


def build_chunks(root, spec):
    """Write chunk_NNNN.mp4 files plus frame_timestamps.json for ``spec`` under ``root``."""
    from PIL import Image

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    chunks, frames = [], []
    start = 0
    for c, chunk in enumerate(spec):
        png = root / ("_png%d" % c)
        png.mkdir(exist_ok=True)
        for f, (level, sigma) in enumerate(chunk):
            Image.fromarray(_frame_array(level, sigma, seed=1234), "L").save(png / ("%03d.png" % f))
        name = "chunk_%04d.mp4" % c
        _ffmpeg(["-framerate", FPS, "-i", png / "%03d.png", "-c:v", "libx264", "-crf", 8, "-preset", "veryfast",
                 "-pix_fmt", "yuv420p", "-r", FPS, root / name])
        shutil.rmtree(png)
        n = len(chunk)
        chunks.append({"index": c, "file": name, "start_frame": start, "frame_count": n,
                       "start_time": round(start / FPS, 6), "end_time": round((start + n) / FPS, 6),
                       "duration_seconds": round(n / FPS, 6)})
        for f in range(n):
            g = start + f
            frames.append({"index": g, "chunk": c, "frame_in_chunk": f, "timestamp": round(g / FPS, 6),
                           "chunk_timestamp": round(f / FPS, 6)})
        start += n
    doc = {"version": 1, "source_name": "synthetic.mp4", "fps": float(FPS), "width": W, "height": H,
           "chunk_seconds": FPC / FPS, "chunk_frames": FPC, "frame_count": start,
           "duration_seconds": start / FPS, "chunks": chunks, "frames": frames}
    (root / "frame_timestamps.json").write_text(json.dumps(doc), encoding="utf-8")
    return root


_WORK_CACHE: dict[str, Path] = {}


@pytest.fixture(scope="session")
def rs8_cache(tmp_path_factory):
    if not (FFMPEG and FFPROBE):
        pytest.skip("ffmpeg/ffprobe not on PATH")
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    pytest.importorskip("scipy")
    return tmp_path_factory.mktemp("rs8cache")


# ------------------------------------------------------------------ GPS


def make_track(seconds=120, turn_at=30, lat0=LAT0, lon0=LON0, speed=10.0, start=T0):
    """North at ``speed`` m/s until ``turn_at`` s, then east. One point per second."""
    from backend.gpx_track import GpxTrack, TrackPoint

    pts = []
    lat, lon = lat0, lon0
    for s in range(seconds + 1):
        pts.append(TrackPoint(start + timedelta(seconds=s), lat, lon))
        if s < turn_at:
            lat += speed / M_PER_DEG
        else:
            lon += speed / (M_PER_DEG * math.cos(math.radians(lat)))
    return GpxTrack.from_points(pts)


def write_gpx_file(path, seconds=120, turn_at=30, speed=10.0):
    lat, lon = LAT0, LON0
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<gpx version="1.1" creator="rs8" xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>']
    for s in range(seconds + 1):
        t = (T0 + timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines.append('<trkpt lat="%.8f" lon="%.8f"><time>%s</time></trkpt>' % (lat, lon, t))
        if s < turn_at:
            lat += speed / M_PER_DEG
        else:
            lon += speed / (M_PER_DEG * math.cos(math.radians(lat)))
    lines.append("</trkseg></trk></gpx>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return Path(path)


def angdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


# ------------------------------------------------------------------ interface adapter


def load_module():
    return importlib.import_module("backend.candidates")


def find_func():
    mod = load_module()
    for n in FUNC_NAMES:
        f = getattr(mod, n, None)
        if callable(f):
            return f
    pytest.fail("backend.candidates has none of the expected functions %s" % FUNC_NAMES)


def _pick(sig, names):
    for n in names:
        if n in sig.parameters:
            return n
    return None


def ts_of(chunk, fic):
    return round((chunk * FPC + fic) / FPS, 6)


def _insert_drive(db, drive_id, started_at, offset, gpx_key):
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT OR REPLACE INTO drives (id, started_at, started_at_source, clock_offset_seconds, "
            "video_path, gpx_path, status, duration_seconds, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (drive_id, started_at, "metadata", offset, "v/%s.mp4" % drive_id, gpx_key, "detected", 30.0,
             "2026-10-20T12:00:00+00:00"))
        conn.commit()
    finally:
        conn.close()


def _json(v):
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        v = bytes(v).decode("utf-8")
    if isinstance(v, str):
        return json.loads(v)
    return v


def _first(row, names):
    for n in names:
        if n in row:
            return row[n]
    raise AssertionError("candidates table has none of the columns %s (has %s)" % (names, list(row)))


def _has(row, names):
    return any(n in row for n in names)


class Env:
    def __init__(self, mp, tmp_path, cache):
        self.mp = mp
        self.tmp = Path(tmp_path)
        self.cache = cache
        self.db = self.tmp / "rs8.sqlite"
        self.storage = self.tmp / "storage"
        self.storage.mkdir()
        self.out = self.tmp / "images"
        mp.setenv("ROADSCOUT_DB", str(self.db))
        mp.setenv("ROADSCOUT_STORAGE_DIR", str(self.storage))
        for k in ("META_API_KEY", "ROADSCOUT_PROVIDER", "ROADSCOUT_FAKE_SCENARIO", MIN_ENV):
            mp.delenv(k, raising=False)
        from backend.db import init_db

        init_db()
        self.work = None
        self.spec = None
        self.add_drive("drive-a")

    # --- fixtures
    def use_spec(self, spec):
        key = json.dumps(spec)
        if key not in _WORK_CACHE or not _WORK_CACHE[key].exists():
            _WORK_CACHE[key] = build_chunks(self.cache / ("w%d" % len(_WORK_CACHE)), spec)
        self.work = self.tmp / "work"
        shutil.copytree(_WORK_CACHE[key], self.work)
        self.spec = spec
        return self.work

    @property
    def chunks_arg(self):
        return self.work / "frame_timestamps.json"

    def add_drive(self, drive_id, started_at="2026-10-20T12:00:00+00:00", offset=0.0, gpx_key="none/none.gpx"):
        _insert_drive(self.db, drive_id, started_at, offset, gpx_key)

    def add_det(self, chunk, fic, track_id, box, *, drive="drive-a", hazard_class="pothole", prompt=None,
                score=0.9, mask="box"):
        from backend.providers import rle_from_box

        box = [float(x) for x in box]
        if mask == "box":
            m = json.dumps(rle_from_box(W, H, box), separators=(",", ":"))
        elif mask is None:
            m = None
        else:
            m = json.dumps(rle_from_box(W, H, mask), separators=(",", ":"))
        g = chunk * FPC + fic
        conn = sqlite3.connect(str(self.db))
        try:
            conn.execute(
                "INSERT INTO detections (drive_id, chunk, frame_index, frame_in_chunk, frame_timestamp, "
                "hazard_class, class_prompt, track_id, track_key, box, mask, score) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (drive, chunk, g, fic, ts_of(chunk, fic), hazard_class, prompt or hazard_class, track_id,
                 "%d:%d" % (chunk, track_id), json.dumps(box), m, score))
            conn.commit()
        finally:
            conn.close()

    def add_track(self, chunk, frames, track_id, box, **kw):
        """Same box on every frame of ``frames`` (iterable of frame_in_chunk)."""
        for f in frames:
            self.add_det(chunk, f, track_id, box, **kw)

    def delete_dets(self, where="1=1", args=()):
        conn = sqlite3.connect(str(self.db))
        try:
            conn.execute("DELETE FROM detections WHERE " + where, args)
            conn.commit()
        finally:
            conn.close()

    # --- running the code under test
    def run(self, drive_id="drive-a", chunks=None, *, gpx=None, video_start=None, offset=None,
            min_frames=None, output_dir="default", **extra):
        f = find_func()
        sig = inspect.signature(f)
        params = list(sig.parameters)
        kw = dict(extra)
        chunks = self.chunks_arg if chunks is None else chunks
        positional = []
        drive_p = next((p for p in params if "drive" in p.lower()), None)
        rest = [p for p in params if p != drive_p and sig.parameters[p].kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)]
        chunk_p = next((p for p in rest if any(s in p.lower() for s in ("chunk", "work", "preproc", "dir"))
                        and p not in OUT_PARAMS), rest[0] if rest else None)
        if drive_p and chunk_p:
            kw[drive_p] = drive_id
            kw[chunk_p] = chunks
        else:
            positional = [drive_id, chunks]
        db_p = _pick(sig, DB_PARAMS)
        if db_p:
            kw[db_p] = str(self.db)

        def put(names, value, label):
            if value is None:
                return
            p = _pick(sig, names)
            if p is None:
                pytest.fail("aggregate function has no parameter for %s (looked for %s); signature %s"
                            % (label, names, sig))
            kw[p] = value

        put(GPX_PARAMS, gpx, "the GPX track")
        put(START_PARAMS, video_start, "video start")
        put(OFFSET_PARAMS, offset, "clock offset")
        put(MIN_PARAMS, min_frames, "the minimum track length")
        if output_dir == "default":
            output_dir = self.out
        put(OUT_PARAMS, output_dir, "the output directory")
        return f(*positional, **kw)

    # --- reading back
    def raw(self, drive_id="drive-a"):
        conn = sqlite3.connect(str(self.db))
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM candidates WHERE drive_id = ? ORDER BY id", (drive_id,))]
        finally:
            conn.close()

    def cands(self, drive_id="drive-a"):
        return [norm(r, self) for r in self.raw(drive_id)]

    def resolve(self, p):
        assert p, "path column is empty"
        path = Path(p)
        if path.is_absolute():
            return path
        for base in (self.out, self.work, Path.cwd(), ROOT):
            if base is not None and (Path(base) / path).exists():
                return Path(base) / path
        return Path(p)


def norm(r, env=None):
    out = {"raw": r, "id": r.get("id")}
    out["hazard_id"] = r.get("hazard_id")
    out["cls"] = _first(r, ["hazard_class", "class", "class_name", "label"])
    out["track_key"] = r.get("track_key")
    out["track_id"] = r.get("track_id")
    out["first_ts"] = float(_first(r, ["first_timestamp", "first_ts", "first_seen", "first_time",
                                       "start_timestamp"]))
    out["last_ts"] = float(_first(r, ["last_timestamp", "last_ts", "last_seen", "last_time", "end_timestamp"]))
    out["frames"] = int(_first(r, ["frames_seen", "frame_count", "n_frames", "num_frames"]))
    out["best_path"] = _first(r, ["best_frame_path", "best_frame", "best_frame_image", "best_image_path"])
    out["crop_path"] = _first(r, ["crop_path", "crop", "best_crop_path"])
    out["lat"] = _first(r, ["latitude", "lat"])
    out["lon"] = _first(r, ["longitude", "lon", "lng"])
    out["heading"] = _first(r, ["heading", "heading_deg"])
    mk = [k for k in r if "mask" in k.lower() and "area" in k.lower()]
    out["mask_area"] = r[mk[0]] if mk else None
    return out


@pytest.fixture()
def rs8_env(monkeypatch, tmp_path, rs8_cache):
    import socket

    def _no_net(*a, **k):
        raise AssertionError("network access attempted during a task 8 test")

    monkeypatch.setattr(socket.socket, "connect", _no_net)
    monkeypatch.setattr(socket, "create_connection", _no_net)
    return Env(monkeypatch, tmp_path, rs8_cache)


# ------------------------------------------------------------------ image helpers


def open_gray(path):
    from PIL import Image

    with Image.open(path) as im:
        im.load()
        return im.convert("L")


def mean_level(path):
    import numpy as np

    return float(np.asarray(open_gray(path), dtype="float64").mean())


def image_size(path):
    from PIL import Image

    with Image.open(path) as im:
        return im.size  # (w, h)


def identify_frame(env, path, candidates):
    """Which (chunk, frame) of ``candidates`` does the saved image at ``path`` show? Nearest level."""
    m = mean_level(env.resolve(path))
    best = min(candidates, key=lambda cf: abs(env.spec[cf[0]][cf[1]][0] - m))
    return best


def laplacian_var(path):
    import numpy as np

    a = np.asarray(open_gray(path), dtype="float64")
    lap = a[1:-1, 1:-1] * -4 + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    return float(lap.var())
