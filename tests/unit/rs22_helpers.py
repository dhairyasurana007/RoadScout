"""Shared helpers and fixtures for the task 22 (annotated video export, backend only) hidden tests.

Interface assumed (everything is discovered tolerantly where possible; see the report to the main agent):

    backend/export.py
        export_hazard_clip(hazard_id, *, provider=None, db_path=None, ...) -> dict   (the export record)
    window config   env ROADSCOUT_EXPORT_BEFORE_SECONDS (default 4), ROADSCOUT_EXPORT_AFTER_SECONDS (default 4),
                    ROADSCOUT_EXPORT_MAX_SECONDS (default 20); read at call time.
    routes          POST /api/hazards/{id}/export  -> 201/202 + export JSON {id, hazard_id, drive_id, status,
                    start_seconds, end_seconds, created_at, ...} (no filesystem paths)
                    GET /api/hazards/{id}/exports  or  GET /api/exports   (list)
                    GET /api/exports/{export_id}/video  -> video/mp4 (Range supported), 404 unknown
    table           `exports` (id, drive_id, hazard_id, start/end seconds, path/key, created_at, status)
    file layout     <storage>/<drive_id>/exports/<name>.mp4 (only the final, blurred file; plus optional .json)
    pixels          single view = source resolution (no scaling for sources <= 720p); output = [raw | annotated]
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
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
from types import SimpleNamespace

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
LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "::"}

SRC_W, SRC_H, SRC_FPS, SRC_DUR = 320, 180, 10, 16
BASE = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
VA = (38.88, -77.30)
OVERLAY_BOX = (110, 50, 190, 100)       # pothole overlay box (clear of the blur masks and of the bottom strip)
OTHER_BOX = (20, 100, 60, 140)          # a box that must NOT be drawn (other drive / outside the window)
FAR_BOX = (250, 90, 300, 130)
BANNER_ROWS = 26

OK_STATUS = {"ok", "done", "complete", "completed", "ready", "succeeded", "success", "exported", "finished"}
FAILED_STATUS = {"failed", "error", "errored", "failure"}
PENDING_STATUS = {"queued", "pending", "running", "processing", "started", "exporting", "in_progress", "created"}
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts"}
IMG_EXT = {".jpg", ".jpeg", ".png"}

EXPORT_ENV_EXACT = {"before": "ROADSCOUT_EXPORT_BEFORE_SECONDS", "after": "ROADSCOUT_EXPORT_AFTER_SECONDS",
                    "max": "ROADSCOUT_EXPORT_MAX_SECONDS"}
EXPORT_ENV_RE = {"before": r"BEFORE|LEAD|PRE_|_PRE|BACK", "after": r"AFTER|POST|TRAIL|FORWARD",
                 "max": r"MAX|CAP|LIMIT|LONGEST"}


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def unwrap(body):
    if isinstance(body, dict) and "id" not in body:
        for k in ("export", "data", "item", "result"):
            if isinstance(body.get(k), dict):
                return body[k]
    return body


def as_list(body):
    if isinstance(body, list):
        return body
    assert isinstance(body, dict), "expected a JSON list, got %r" % (type(body),)
    for k in ("exports", "items", "results", "data"):
        if isinstance(body.get(k), list):
            return body[k]
    pytest.fail("no list found in response object keys %s" % sorted(body))


def all_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from all_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from all_strings(v)


def all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from all_keys(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from all_keys(v)


def known_classes():
    cfg = json.loads((ROOT / "config" / "hazard_prompts.json").read_text(encoding="utf-8"))
    return list(cfg["classes"].keys())


# ------------------------------------------------------------------ ffmpeg / ffprobe / numpy

def need_media():
    pytest.importorskip("numpy")
    if not (FFMPEG and FFPROBE):
        pytest.skip("ffmpeg/ffprobe not on PATH")


def probe(path):
    p = subprocess.run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                       capture_output=True, timeout=120)
    assert p.returncode == 0, "ffprobe cannot read %s: %s" % (Path(path).name, p.stderr.decode("utf-8", "replace"))
    return json.loads(p.stdout.decode("utf-8", "replace"))


def vstreams(info):
    return [s for s in info.get("streams", []) if s.get("codec_type") == "video"]


def astreams(info):
    return [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]


def _frac(v):
    if not v or v == "0/0":
        return 0.0
    if "/" in str(v):
        a, b = str(v).split("/")
        return float(a) / float(b) if float(b) else 0.0
    return float(v)


def fps_of(stream):
    return _frac(stream.get("avg_frame_rate")) or _frac(stream.get("r_frame_rate"))


def duration_of(info):
    d = info.get("format", {}).get("duration")
    if d not in (None, "N/A"):
        return float(d)
    for s in vstreams(info):
        if s.get("duration") not in (None, "N/A"):
            return float(s["duration"])
    return 0.0


def decode_frames(path):
    """All frames of the first video stream as a uint8 array (n, h, w, 3), RGB."""
    import numpy as np

    info = probe(path)
    s = vstreams(info)[0]
    w, h = int(s["width"]), int(s["height"])
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:v:0", "-f",
                        "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, timeout=300)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    buf = np.frombuffer(p.stdout, dtype=np.uint8)
    n = buf.size // (w * h * 3)
    return buf[: n * w * h * 3].reshape(n, h, w, 3)


_A = None
_B = None
_A_FRAMES = None


def clip_a_path():
    """16 s 320x180 10 fps clip with audio. R = time ramp (255*T/16), G = 12 px checkerboard flipping every 0.2 s."""
    global _A
    if _A is None:
        need_media()
        d = Path(tempfile.mkdtemp(prefix="rs22media"))
        out = d / "a.mp4"
        vf = ("nullsrc=s=320x180:r=10,format=gbrp,"
              "geq=r='255*T/16':g='255*mod(floor(X/12)+floor(Y/12)+floor(T*5),2)':b='128',format=yuv420p")
        p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", vf, "-f", "lavfi",
                            "-i", "sine=frequency=440:sample_rate=22050", "-t", "16", "-c:v", "libx264", "-preset",
                            "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p", "-g", "10", "-c:a", "aac", "-b:a",
                            "32k", "-shortest", "-metadata", "creation_time=" + iso(BASE), str(out)],
                           capture_output=True, timeout=300)
        assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
        _A = out.read_bytes()
    return _A


def clip_b_bytes():
    """40 s 160x90 5 fps clip (cap tests only)."""
    global _B
    if _B is None:
        need_media()
        d = Path(tempfile.mkdtemp(prefix="rs22mediab"))
        out = d / "b.mp4"
        p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                            "testsrc2=size=160x90:rate=5", "-t", "40", "-c:v", "libx264", "-preset", "ultrafast",
                            "-pix_fmt", "yuv420p", str(out)], capture_output=True, timeout=300)
        assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
        _B = out.read_bytes()
    return _B


def src_frames():
    global _A_FRAMES
    if _A_FRAMES is None:
        d = Path(tempfile.mkdtemp(prefix="rs22src"))
        f = d / "a.mp4"
        f.write_bytes(clip_a_path())
        _A_FRAMES = decode_frames(f)
    return _A_FRAMES


def _strip_r(frame):
    return float(frame[SRC_H - 12:, :, 0].mean())


_CURVE = None


def est_index(left_frame):
    """Index of the source frame whose bottom-strip red mean is closest (the red channel is a time ramp)."""
    import numpy as np

    global _CURVE
    if _CURVE is None:
        _CURVE = np.array([_strip_r(f) for f in src_frames()])
    return int(np.argmin(np.abs(_CURVE - _strip_r(left_frame))))


def luma(a):
    import numpy as np

    a = a.astype("float64")
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def lap_var(gray, box, erode=4):
    import numpy as np

    g = gray.astype("float64")
    lap = np.zeros_like(g)
    lap[1:-1, 1:-1] = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = x1 + erode, y1 + erode, x2 - erode, y2 - erode
    assert x2 - x1 > 4 and y2 - y1 > 4, "test box too small"
    return float(lap[y1:y2, x1:x2].var())


FACE = (16, 20, 60, 70)
PLATE = (240, 120, 300, 160)
REF = (110, 118, 154, 168)       # an unblurred, un-annotated region of the same size class as the masks (left half)


def mask_boxes(w, h):
    sx, sy = w / float(SRC_W), h / float(SRC_H)
    sc = lambda b: (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)  # noqa: E731
    return {"face": sc(FACE), "license plate": sc(PLATE)}


def diff_map(frame):
    """(h, w) max-channel absolute difference between the right (annotated) and the left (raw) half."""
    import numpy as np

    w = frame.shape[1] // 2
    d = np.abs(frame[:, w:].astype("int16") - frame[:, :w].astype("int16")).max(axis=2)
    d[:BANNER_ROWS, :] = 0     # calibration: the annotated half carries a static header banner in the top rows
    return d


def region_count(d, box, thr=40, margin=0, exclude=None, exclude_margin=0):
    """Number of pixels of the diff map d above thr inside box (+margin), minus the ones inside `exclude`."""
    import numpy as np

    h, w = d.shape
    m = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = [int(v) for v in box]
    m[max(0, y1 - margin):min(h, y2 + margin), max(0, x1 - margin):min(w, x2 + margin)] = True
    if exclude is not None:
        ex = [int(v) for v in exclude]
        m[max(0, ex[1] - exclude_margin):min(h, ex[3] + exclude_margin),
          max(0, ex[0] - exclude_margin):min(w, ex[2] + exclude_margin)] = False
    return int(np.count_nonzero((d > thr) & m))


def total_count(d, thr=40):
    import numpy as np

    return int(np.count_nonzero(d > thr))


# ------------------------------------------------------------------ providers

def _norm(s):
    return str(s).strip().lower()


try:
    from backend.providers import ModelProvider as _Base
except Exception:  # noqa: BLE001
    _Base = object


def spend_exc():
    import pkgutil

    pkg = importlib.import_module("backend.providers")
    if hasattr(pkg, "SpendCapExceeded"):
        return pkg.SpendCapExceeded
    for info in pkgutil.walk_packages(pkg.__path__, "backend.providers."):
        m = importlib.import_module(info.name)
        if hasattr(m, "SpendCapExceeded"):
            return m.SpendCapExceeded
    pytest.fail("SpendCapExceeded not found in backend.providers")


class StubProvider(_Base):
    """Records every detect() call (on a video segment) and returns face / plate masks per `mode`.

    modes: dense (every frame), first_half (frames of the first half of the segment only), empty, half (dense, masks
    at half resolution), raise, none, string, bad_counts, bad_size, spend.
    """

    name = "stub22"

    def __init__(self, mode="dense"):
        self.mode = mode
        self.calls = []

    def verify(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def detect_image(self, image_path, prompts):  # pragma: no cover
        return self.detect(image_path, prompts)

    def detect(self, video_path, prompts):
        from backend.providers.types import Detection, DetectionResult, FrameRecord, rle_from_box

        p = str(video_path)
        rec = {"path": p, "prompts": [_norm(x) for x in prompts], "exists": Path(p).exists(),
               "suffix": Path(p).suffix.lower()}
        try:
            info = probe(p)
            s = vstreams(info)[0]
            rec.update(w=int(s["width"]), h=int(s["height"]), fps=fps_of(s), dur=duration_of(info))
        except Exception as e:  # noqa: BLE001
            rec["probe_error"] = repr(e)
        self.calls.append(rec)
        m = self.mode
        if m == "raise":
            raise RuntimeError("boom-export-xyz")
        if m == "spend":
            raise spend_exc()("spend cap reached for this run")
        if m == "none":
            return None
        if m == "string":
            return "not a detection result"
        w, h, fps = rec.get("w", SRC_W), rec.get("h", SRC_H), rec.get("fps") or SRC_FPS
        dur = rec.get("dur", 8.0)
        if m in ("bad_counts", "bad_size"):
            mask = {"size": [h, w], "counts": [3, 5, 7]} if m == "bad_counts" else {"size": [0, -5], "counts": []}
            box = mask_boxes(w, h)["face"]
            try:
                det = Detection(track_id=1, box=box, score=0.9, prompt="face", mask=mask)
            except Exception:  # noqa: BLE001
                det = SimpleNamespace(track_id=1, box=box, score=0.9, prompt="face", mask=mask)
            try:
                fr = FrameRecord(0, 0.0, (det,))
            except Exception:  # noqa: BLE001
                fr = SimpleNamespace(frame_index=0, timestamp=0.0, detections=[det])
            try:
                return DetectionResult((fr,))
            except Exception:  # noqa: BLE001
                return SimpleNamespace(frames=[fr])
        if m == "empty":
            return DetectionResult(())
        n = max(1, int(round(dur * fps)))
        wanted = {_norm(x) for x in prompts}
        scale = 0.5 if m == "half" else 1.0
        mw, mh = max(1, int(w * scale)), max(1, int(h * scale))
        frames = []
        for i in range(n):
            t = i / float(fps)
            if m == "first_half" and t >= dur / 2.0:
                continue
            if m == "sparse" and i % max(1, int(round(fps / 2.0))) != 0:     # a record every 0.5 s only
                continue
            dets = []
            for k, (name, box) in enumerate(mask_boxes(w, h).items()):
                if name not in wanted:
                    continue
                b = tuple(v * scale for v in box)
                dets.append(Detection(track_id=k + 1, box=b, score=0.9, prompt=name, mask=rle_from_box(mw, mh, b)))
            frames.append(FrameRecord(i, t, tuple(dets)))
        return DetectionResult(tuple(frames))


# ------------------------------------------------------------------ environment

def export_env(kind):
    exact = EXPORT_ENV_EXACT[kind]
    found = set()
    for path in (ROOT / "backend").rglob("*.py"):
        found |= set(re.findall(r"ROADSCOUT_[A-Z0-9_]+", path.read_text(encoding="utf-8", errors="replace")))
    if exact in found:
        return exact
    pick = [n for n in sorted(found) if "EXPORT" in n and re.search(EXPORT_ENV_RE[kind], n)]
    if pick:
        return pick[0]
    pytest.fail("no env variable for the export %r setting found in backend/*.py (expected %s)" % (kind, exact))


class Env:
    def __init__(self, mp, tmp_path, client):
        self.mp = mp
        self.tmp = Path(tmp_path)
        self.db = self.tmp / "db" / "rs22.sqlite"
        self.storage = self.tmp / "storage"
        self.client = client
        self._cn = 0
        self.stub = None

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

    # ---- drives
    def video_file(self, did):
        return self.storage / did / "video.mp4"

    def add_drive(self, did="d22a", *, clip=None, status="done", duration=float(SRC_DUR), started=BASE, place=True):
        data = clip_a_path() if clip is None else clip
        if place:
            p = self.video_file(did)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        self.x("INSERT OR REPLACE INTO drives (id, started_at, started_at_source, clock_offset_seconds, video_path, "
               "gpx_path, status, duration_seconds, created_at, updated_at) VALUES (?, ?, 'test', 0, ?, ?, ?, ?, ?, ?)",
               (did, iso(started), "%s/video.mp4" % did, "%s/track.gpx" % did, status, duration, iso(started),
                iso(started)))
        return did

    # ---- candidates, hazards, detections
    def add_cand(self, drive_id, *, score, best_ts, cls="pothole", lat=VA[0], lon=VA[1], heading=0.0):
        self._cn += 1
        n = self._cn
        fp = str(self.db.parent / ("frame_%d.jpg" % n))
        con = sqlite3.connect(str(self.db))
        try:
            con.execute("PRAGMA foreign_keys = ON")
            cur = con.execute(
                "INSERT INTO candidates (drive_id, hazard_id, hazard_class, track_key, track_id, chunk, "
                "first_timestamp, last_timestamp, frames_seen, max_mask_area, best_frame_index, best_timestamp, "
                "best_box, best_frame_path, crop_path, blurred_frame_path, blurred_crop_path, blur_status, "
                "lat, lon, heading, score, geotag_status, geotag_note) "
                "VALUES (?, NULL, ?, ?, ?, 0, ?, ?, 5, 100.0, 30, ?, '[0, 0, 10, 10]', ?, NULL, ?, NULL, "
                "'ok', ?, ?, ?, ?, 'ok', NULL)",
                (drive_id, cls, "0:%d" % n, n, best_ts - 1.0, best_ts + 1.0, best_ts, fp, fp, lat, lon, heading,
                 score))
            cid = cur.lastrowid
            con.execute(
                "INSERT INTO verdicts (candidate_id, drive_id, is_hazard, hazard_class, severity, lane, confidence, "
                "reason, report_description, model, prompt_version, accepted, created_at) "
                "VALUES (?, ?, 1, ?, 2, NULL, 0.8, 'r', 'd', 'fake', 'v1', 1, ?)",
                (cid, drive_id, cls, iso(BASE)))
            con.commit()
            return cid
        finally:
            con.close()

    def seed(self, drive="d22a", *, best_t=9.0, others=((0.3, 3.0),), cls="pothole", others_drive=None):
        """A hazard (built by deduplication from the best candidate, score 0.9 at best_t of `drive`) with extra,
        lower scoring member candidates (score, t) linked to it, in `others_drive` (default: the same drive)."""
        from backend.dedup import deduplicate_drive

        cid = self.add_cand(drive, score=0.9, best_ts=best_t, cls=cls)
        deduplicate_drive(drive, db_path=str(self.db))
        hid = self.q("SELECT hazard_id FROM candidates WHERE id = ?", (cid,))[0]["hazard_id"]
        assert hid is not None, "synthetic candidate was not turned into a hazard"
        for score, t in others:
            oc = self.add_cand(others_drive or drive, score=score, best_ts=t, cls=cls)
            self.x("UPDATE candidates SET hazard_id = ? WHERE id = ?", (hid, oc))
        return hid

    def seed_other_best(self, best_drive="d22b", other_drive="d22a", *, best_t=9.0, other_t=3.0, cls="pothole"):
        """Hazard whose best-scoring member candidate is in `best_drive` (score 0.9) and a weaker one in other_drive."""
        from backend.dedup import deduplicate_drive

        cid = self.add_cand(other_drive, score=0.3, best_ts=other_t, cls=cls)
        deduplicate_drive(other_drive, db_path=str(self.db))
        hid = self.q("SELECT hazard_id FROM candidates WHERE id = ?", (cid,))[0]["hazard_id"]
        oc = self.add_cand(best_drive, score=0.9, best_ts=best_t, cls=cls)
        self.x("UPDATE candidates SET hazard_id = ? WHERE id = ?", (hid, oc))
        return hid

    def bare_hazard(self, cls="pothole"):
        """A hazards row without any member candidate."""
        return self.x("INSERT INTO hazards (hazard_class, lat, lon, direction, severity, confidence, sightings, "
                      "candidate_count, first_seen, last_seen, status, created_at, updated_at) "
                      "VALUES (?, ?, ?, 0, 2, 0.8, 1, 1, ?, ?, 'new', ?, ?)",
                      (cls, VA[0], VA[1], iso(BASE), iso(BASE), iso(BASE), iso(BASE)))

    def add_det(self, drive, t, box, *, cls="pothole", track=1, mask=None, score=0.8):
        self.x("INSERT INTO detections (drive_id, chunk, frame_index, frame_in_chunk, frame_timestamp, hazard_class, "
               "class_prompt, track_id, track_key, box, mask, score) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
               (drive, int(t // 4), int(round(t * 3)), 0, float(t), cls, cls, track, "0:%d" % track,
                json.dumps(list(box)), json.dumps(mask) if mask is not None else None, score))

    # ---- provider injection
    def use_provider(self, provider):
        self.stub = provider
        importlib.import_module("backend.main")
        mod = importlib.import_module("backend.providers")
        if not hasattr(self, "_patched"):
            orig = getattr(mod, "get_provider")
            self._patched = [m for name, m in list(sys.modules.items())
                             if m is not None and name.startswith("backend")
                             and getattr(m, "get_provider", None) is orig]
        for m in self._patched:
            self.mp.setattr(m, "get_provider", lambda *a, **k: provider)
        return provider

    # ---- routes
    def paths(self):
        from backend.main import app

        return app.openapi().get("paths", {})

    def _find(self, method, patterns):
        for pat in patterns:
            for path, ops in self.paths().items():
                if re.match(pat, path) and method in ops:
                    return path
        return None

    def post_route(self):
        p = self._find("post", [r"^/api/hazards/\{[^}/]+\}/exports?/?$"])
        if p:
            return "path", p
        p = self._find("post", [r"^/api/exports/?$"])
        if p:
            return "body", p
        pytest.fail("no POST /api/hazards/{id}/export route in the OpenAPI schema; routes: %s" % sorted(self.paths()))

    def list_route(self):
        p = self._find("get", [r"^/api/hazards/\{[^}/]+\}/exports/?$"])
        if p:
            return "hazard", p
        p = self._find("get", [r"^/api/exports/?$"])
        if p:
            return "all", p
        pytest.fail("no GET /api/exports (or /api/hazards/{id}/exports) route")

    def video_route(self):
        p = self._find("get", [r"^/api/exports/\{[^}/]+\}/(video|file|download|media)/?$"])
        if p:
            return p
        pytest.fail("no GET /api/exports/{export_id}/video route")

    def detail_route(self):
        return self._find("get", [r"^/api/exports/\{[^}/]+\}/?$"])

    # ---- http helpers
    def post_export(self, hid, **body):
        kind, path = self.post_route()
        if kind == "path":
            return self.client.post(re.sub(r"\{[^}/]+\}", str(hid), path), json=body)
        return self.client.post(path, json=dict(body, hazard_id=hid))

    def list_for(self, hid):
        kind, path = self.list_route()
        if kind == "hazard":
            r = self.client.get(re.sub(r"\{[^}/]+\}", str(hid), path))
            assert r.status_code == 200, r.text[:300]
            return as_list(r.json())
        r = self.client.get(path)
        assert r.status_code == 200, r.text[:300]
        return [e for e in as_list(r.json()) if str(e.get("hazard_id")) == str(hid)]

    def all_exports(self):
        kind, path = self.list_route()
        if kind == "all":
            r = self.client.get(path)
            assert r.status_code == 200, r.text[:300]
            return as_list(r.json())
        out = []
        for h in self.q("SELECT id FROM hazards"):
            out += self.list_for(h["id"])
        return out

    def video_url(self, rec):
        if isinstance(rec.get("video_url"), str) and rec["video_url"].startswith("/"):
            return rec["video_url"]
        return re.sub(r"\{[^}/]+\}", str(rec["id"]), self.video_route())

    def settle(self, rec, hid, timeout=120):
        """Poll until the export is no longer pending. Returns the final record."""
        deadline = time.time() + timeout
        while str(rec.get("status", "")).strip().lower() in PENDING_STATUS and time.time() < deadline:
            time.sleep(0.2)
            fresh = None
            d = self.detail_route()
            if d:
                r = self.client.get(re.sub(r"\{[^}/]+\}", str(rec["id"]), d))
                if r.status_code == 200:
                    fresh = unwrap(r.json())
            if fresh is None:
                for e in self.list_for(hid):
                    if str(e.get("id")) == str(rec["id"]):
                        fresh = e
            if fresh is not None:
                rec = fresh
        return rec

    def export_resp(self, hid, **body):
        """POST and, if the answer is a record, poll it to a terminal status. Returns (response, record|None)."""
        r = self.post_export(hid, **body)
        rec = None
        if r.status_code in (200, 201, 202):
            rec = self.settle(unwrap(r.json()), hid)
        return r, rec

    def export_ok(self, hid, **body):
        r, rec = self.export_resp(hid, **body)
        assert r.status_code in (201, 202), "POST export -> %s %s" % (r.status_code, r.text[:400])
        assert rec is not None and str(rec.get("status", "")).strip().lower() in OK_STATUS, rec
        return rec

    # ---- files
    def exports_dir(self, did):
        return self.storage / did / "exports"

    def export_files(self, did):
        d = self.exports_dir(did)
        return sorted(p for p in d.rglob("*") if p.is_file()) if d.exists() else []

    def export_videos(self, did):
        return [p for p in self.export_files(did) if p.suffix.lower() in VIDEO_EXT]

    def storage_files(self):
        return sorted(p for p in self.storage.rglob("*") if p.is_file()) if self.storage.exists() else []

    def stray_videos(self, allowed_uploads):
        """Video files anywhere under storage that are neither an upload nor in an exports folder."""
        out = []
        for p in self.storage_files():
            if p.suffix.lower() in VIDEO_EXT and "exports" not in p.relative_to(self.storage).parts[:-1]:
                if str(p) not in {str(a) for a in allowed_uploads}:
                    out.append(p)
        return out

    def uploads_sha(self, dids):
        return {d: sha(self.video_file(d)) for d in dids if self.video_file(d).exists()}


def make_env(mp, tmp_path):
    need_media()
    import backend.config  # noqa: F401  (loads a local .env before the key is cleared)

    mp.setenv("ROADSCOUT_DB", str(tmp_path / "db" / "rs22.sqlite"))
    mp.setenv("ROADSCOUT_STORAGE_DIR", str(tmp_path / "storage"))
    mp.setenv("ROADSCOUT_PROVIDER", "fake")
    mp.setenv("ROADSCOUT_COST_LOG", "off")
    for k in list(os.environ):
        if k == "META_API_KEY" or (k.startswith("ROADSCOUT") and "EXPORT" in k):
            mp.delenv(k, raising=False)
    for k in ("ROADSCOUT_FAKE_SCENARIO",):
        mp.delenv(k, raising=False)
    real_connect = socket.socket.connect

    def _guard(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if host is not None and str(host) not in LOOPBACK:
            raise AssertionError("network access attempted during a task 22 test: %r" % (address,))
        return real_connect(self, address)

    mp.setattr(socket.socket, "connect", _guard)
    from fastapi.testclient import TestClient

    from backend.db import init_db
    from backend.main import app

    init_db()
    return TestClient(app)


@pytest.fixture()
def rs22_env(monkeypatch, tmp_path):
    client = make_env(monkeypatch, tmp_path)
    with client:
        yield Env(monkeypatch, tmp_path, client)


class Std:
    """The standard scenario: drive d22a (16 s clip), hazard with best candidate at t=9 s (window [5, 13]),
    pothole detections at 7.0 .. 8.5 s, dense face + plate masks, one finished export."""

    def __init__(self, env, did, hid, stub, rec):
        self.env, self.did, self.hid, self.stub, self.rec = env, did, hid, stub, rec
        self._frames = None

    @property
    def video(self):
        vids = self.env.export_videos(self.did)
        assert len(vids) == 1, "expected exactly one exported video in %s, got %s" % (
            self.env.exports_dir(self.did), [p.name for p in vids])
        return vids[0]

    @property
    def frames(self):
        if self._frames is None:
            self._frames = decode_frames(self.video)
        return self._frames

    @property
    def start(self):
        return rec_start(self.rec)


@pytest.fixture()
def rs22_std(rs22_env):
    env = rs22_env
    stub = env.use_provider(StubProvider("dense"))
    did = env.add_drive("d22a")
    hid = env.seed(did, best_t=9.0, others=((0.3, 3.0),))
    for t in (7.0, 7.5, 8.0, 8.5):
        env.add_det(did, t, OVERLAY_BOX)
    rec = env.export_ok(hid)
    return Std(env, did, hid, stub, rec)


def rec_key(rec, *names, pattern=None):
    for n in names:
        if n in rec:
            return rec[n]
    if pattern:
        for k, v in rec.items():
            if re.search(pattern, k, re.I) and isinstance(v, (int, float)):
                return v
    raise AssertionError("export JSON has none of %s (keys: %s)" % (names, sorted(rec)))


def rec_start(rec):
    return float(rec_key(rec, "start_seconds", "start", "start_time", "start_s", pattern=r"start"))


def rec_end(rec):
    return float(rec_key(rec, "end_seconds", "end", "end_time", "end_s", pattern=r"(^|_)end"))


def assert_no_paths(obj, env, *extra):
    blob_strings = list(all_strings(obj))
    forbidden = [str(env.tmp), str(env.tmp).replace("\\", "/"), str(env.storage), ".mp4", "\\"] + list(extra)
    for s in blob_strings:
        for f in forbidden:
            assert f not in s, "JSON leaks %r in %r" % (f, s[:120])
        assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", s.replace("https://", "").replace("http://", "")), s
        assert not re.search(r"d22[ab]/(exports|video)", s), s
    for k in all_keys(obj):
        low = k.lower()
        assert "path" not in low, "key %r looks like a filesystem path" % k
        assert low not in ("file", "filename", "storage_key", "video_key", "key"), "key %r exposes storage" % k
