"""Shared helpers and fixtures for the task 15 (privacy blur + Delete video) hidden tests.

Interface assumed (see the report to the main agent):

    backend/blur.py   (aliases tolerated for the module: privacy.py, privacy_blur.py)
        blur_image(src_path, dst_path, masks) -> any
            masks: sequence of RLE dicts {"size": [h, w], "counts": [...]} (may be empty)
            writes a JPEG copy of src to dst with a strong blur inside the union of the masks only;
            masks of another size than the image are scaled; src is never modified;
            invalid masks raise (fail closed) and leave no dst.
        blur_drive(drive_id, *, provider=None, db_path=None)
            asks the provider for ["face", "license plate"] on the saved best-frame and crop JPEGs of the drive's
            ACCEPTED candidates (provider.detect(<jpeg path>, prompts)), blurs, stores the new paths.
    candidates columns: blurred_frame_path, blurred_crop_path (NULL when absent), blur_status
        ('ok' | 'failed' | NULL = not run / not applicable)
    backend.pipeline.STAGES: ..., 'verifying', 'blurring', 'deduplicating'
    hazards.photo_path (and crop_path when that column exists) point at BLURRED images.
    GET /api/drives/{id}/hazards/{candidate_id}/photo[?variant=crop] serves ONLY the blurred JPEG, else 404.
    DELETE /api/drives/{id}/video (discovered tolerantly) -> 200/204; drive JSON has `video_deleted` (bool).
"""
from __future__ import annotations

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
TERMINAL = DONE | FAILED
OK_STATUS = {"ok", "done", "blurred", "success", "succeeded", "complete", "completed"}
BAD_STATUS = {"failed", "error", "errored", "failure"}
IMG_EXT = {".jpg", ".jpeg", ".png"}
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
CHUNKS = 3

BLUR_MODULES = ["backend.blur", "backend.privacy", "backend.privacy_blur"]
BLUR_IMAGE_NAMES = ["blur_image", "blur_image_file", "blur_file", "apply_blur", "blur_masks", "blur_regions"]
BLUR_DRIVE_NAMES = ["blur_drive", "blur_candidates", "blur_drive_candidates", "run_blur", "privacy_blur", "blur_all"]

LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "::"}


def status_of(d):
    return str(d.get("status", "")).strip().lower()


def is_image_path(p):
    return Path(str(p)).suffix.lower() in IMG_EXT


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm_prompt(s):
    return str(s).strip().lower()


# ------------------------------------------------------------------ synthetic images

def checker_array(w, h, cell=8):
    import numpy as np

    yy, xx = np.mgrid[0:h, 0:w]
    return ((((yy // cell) + (xx // cell)) % 2) * 255).astype("uint8")


def save_gray_jpeg(path, arr, quality=95):
    import numpy as np
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rgb = np.stack([arr, arr, arr], axis=-1)
    Image.fromarray(rgb, "RGB").save(str(path), "JPEG", quality=quality, subsampling=0)


def load_gray(path):
    import numpy as np
    from PIL import Image

    return np.array(Image.open(str(path)).convert("L")).astype("float64")


def laplacian(a):
    import numpy as np

    a = a.astype("float64")
    out = np.zeros_like(a)
    out[1:-1, 1:-1] = a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1]
    return out


def lap_var(a, box, erode=0):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = x1 + erode, y1 + erode, x2 - erode, y2 - erode
    assert x2 - x1 > 4 and y2 - y1 > 4, "test box too small for this erosion"
    return float(laplacian(a)[y1:y2, x1:x2].var())


def outside_boxes(shape, boxes, margin):
    """Boolean array: True for pixels farther than `margin` from every box."""
    import numpy as np

    h, w = shape
    m = np.ones((h, w), dtype=bool)
    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = int(x1 - margin), int(y1 - margin), int(x2 + margin), int(y2 + margin)
        m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = False
    return m


def rle_box(w, h, box):
    from backend.providers.types import rle_from_box

    return rle_from_box(w, h, box)


# ------------------------------------------------------------------ discovering the blur API

def blur_module():
    for n in BLUR_MODULES:
        try:
            return importlib.import_module(n)
        except ModuleNotFoundError as e:
            if e.name != n:
                raise
    pytest.fail("none of the modules %s exists (expected backend/blur.py)" % BLUR_MODULES)


def blur_image_fn():
    mod = blur_module()
    for n in BLUR_IMAGE_NAMES:
        f = getattr(mod, n, None)
        if callable(f):
            return f
    pytest.fail("%s has none of %s" % (mod.__name__, BLUR_IMAGE_NAMES))


def blur_drive_fn():
    mod = blur_module()
    for n in BLUR_DRIVE_NAMES:
        f = getattr(mod, n, None)
        if callable(f):
            return f
    pytest.fail("%s has none of %s" % (mod.__name__, BLUR_DRIVE_NAMES))


# ------------------------------------------------------------------ synthetic video / gpx

_CLIP = None


def clip_bytes():
    """12 s 640x360 synthetic clip: an 8 px moving checkerboard over the whole frame (lots of detail everywhere)."""
    global _CLIP
    if _CLIP is None:
        if not (FFMPEG and FFPROBE):
            pytest.skip("ffmpeg/ffprobe not on PATH")
        d = Path(tempfile.mkdtemp(prefix="rs15media"))
        out = d / "clip.mp4"
        expr = "255*mod(floor(X/8)+floor(Y/8)+floor(T*3),2)"
        p = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             "nullsrc=s=640x360:r=30,format=yuv420p,geq=lum='%s':cb=128:cr=128" % expr,
             "-t", "12", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-metadata", "creation_time=" + T0.strftime("%Y-%m-%dT%H:%M:%SZ"), str(out)],
            capture_output=True, timeout=240)
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


# ------------------------------------------------------------------ providers

def face_plate_boxes(w, h):
    return {"face": (0.10 * w, 0.15 * h, 0.30 * w, 0.45 * h),
            "license plate": (0.60 * w, 0.70 * h, 0.85 * w, 0.90 * h)}


def default_image_result(stub, path, prompts):
    """Face and plate masks (image-sized RLE) at fixed fractions of the image, only for requested prompts."""
    from PIL import Image

    from backend.providers.types import Detection, DetectionResult, FrameRecord

    with Image.open(path) as im:
        w, h = im.size
    wanted = {norm_prompt(p) for p in prompts}
    dets = []
    for i, (name, box) in enumerate(face_plate_boxes(w, h).items()):
        if name in wanted:
            dets.append(Detection(track_id=i + 1, box=box, score=0.9, prompt=name, mask=rle_box(w, h, box)))
    return DetectionResult((FrameRecord(0, 0.0, tuple(dets)),))


def empty_image_result(stub, path, prompts):
    from backend.providers.types import DetectionResult

    return DetectionResult(())


def make_stub(*, scenario="default", reject_verify=(), image_result=None, fail_image_on=None, fail_video_on=None,
              on_image=None, on_video=None, on_verify=None):
    """A provider recording every call. detect() on a JPEG/PNG path is the blur call; on anything else a chunk call."""
    from backend.providers import FakeProvider, ModelProvider

    class Stub(ModelProvider):
        name = "stub15"

        def __init__(self):
            self.inner = FakeProvider(scenario)
            self.fp = FakeProvider("false_positive")
            self.video_calls = []
            self.image_calls = []
            self.verify_calls = []
            self.events = []

        def detect(self, video_path, prompts):
            p = str(video_path)
            if is_image_path(p):
                return self._image(p, prompts)
            n = len(self.video_calls)
            self.video_calls.append(p)
            self.events.append("video")
            if on_video:
                on_video(self, n, p)
            if fail_video_on is not None and n == fail_video_on:
                raise RuntimeError("boom-video-xyz")
            return self.inner.detect(video_path, prompts)

        def detect_image(self, image_path, prompts):  # in case the coder adds a dedicated method
            return self.detect(image_path, prompts)

        def _image(self, p, prompts):
            n = len(self.image_calls)
            rec = {"n": n, "path": p, "prompts": list(prompts),
                   "sha": sha(p) if Path(p).is_file() else None}
            self.image_calls.append(rec)
            self.events.append("image")
            if on_image:
                on_image(self, n, p)
            if fail_image_on is not None and n == fail_image_on:
                raise RuntimeError("boom-blur-xyz")
            fn = image_result or default_image_result
            return fn(self, p, prompts)

        def verify(self, images, *, class_prompt=None, prompt_version=None, prompt_text=None):
            n = len(self.verify_calls)
            self.verify_calls.append(list(images))
            self.events.append("verify")
            if on_verify:
                on_verify(self, n)
            src = self.fp if n in set(reject_verify) else self.inner
            return src.verify(images, class_prompt=class_prompt, prompt_version=prompt_version)

    return Stub()


# ------------------------------------------------------------------ environment

class Env:
    def __init__(self, mp, tmp_path, client):
        self.mp = mp
        self.tmp = Path(tmp_path)
        self.db = self.tmp / "db" / "rs15.sqlite"
        self.storage = self.tmp / "storage"
        self.client = client

    # --- drives / http
    def upload(self):
        files = {"video": ("clip.mp4", clip_bytes(), "video/mp4"),
                 "gpx": ("track.gpx", gpx_bytes(), "application/gpx+xml")}
        r = self.client.post("/api/drives", files=files)
        assert r.status_code == 201, r.text
        return r.json()["id"]

    def get(self, did):
        r = self.client.get("/api/drives/%s" % did)
        assert r.status_code == 200, r.text
        return r.json()

    def drive_list(self):
        r = self.client.get("/api/drives")
        assert r.status_code == 200, r.text
        body = r.json()
        return body if isinstance(body, list) else next(v for v in body.values() if isinstance(v, list))

    def process_route(self):
        from backend.main import app

        pat = re.compile(r"^/api/drives/\{[^}/]+\}/(process|run|pipeline|start|analy[sz]e|reprocess)/?$")
        for path, ops in app.openapi().get("paths", {}).items():
            if pat.match(path) and "post" in ops:
                return path
        pytest.fail("no POST /api/drives/{id}/process route")

    def post_process(self, did):
        return self.client.post(re.sub(r"\{[^}/]+\}", did, self.process_route()))

    def use_provider(self, provider):
        importlib.import_module("backend.main")
        importlib.import_module("backend.pipeline")
        mod = importlib.import_module("backend.providers")
        orig = getattr(mod, "get_provider")
        for name, m in list(sys.modules.items()):
            if m is not None and name.startswith("backend") and getattr(m, "get_provider", None) is orig:
                self.mp.setattr(m, "get_provider", lambda *a, **k: provider)

    def wait(self, did, timeout=180):
        deadline = time.time() + timeout
        d = None
        while time.time() < deadline:
            d = self.get(did)
            if status_of(d) in TERMINAL:
                return d
            time.sleep(0.05)
        pytest.fail("drive %s never reached a terminal status; last: %s" % (did, d))

    def run(self, did, provider=None):
        """Call the runner directly. Returns (result, exception_or_None)."""
        from backend.pipeline import run_pipeline

        kw = {"db_path": str(self.db)}
        if provider is not None:
            kw["provider"] = provider
        try:
            return run_pipeline(did, **kw), None
        except Exception as e:  # noqa: BLE001
            return None, e

    def process(self, stub=None, did=None):
        """Upload (unless did is given), run the pipeline directly, return the drive id."""
        did = did or self.upload()
        res, exc = self.run(did, stub)
        assert exc is None, repr(exc)
        return did

    def process_ok(self, stub=None):
        did = self.process(stub)
        d = self.get(did)
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

    def columns(self, table):
        return [r["name"] for r in self.q("PRAGMA table_info(%s)" % table)]

    def cands(self, did):
        return self.q("SELECT * FROM candidates WHERE drive_id = ? ORDER BY id", (did,))

    def verdict_rows(self, did):
        return self.q("SELECT c.id AS cid, v.is_hazard AS is_hazard, v.accepted AS accepted FROM candidates c "
                      "JOIN verdicts v ON v.candidate_id = c.id WHERE c.drive_id = ? ORDER BY c.id", (did,))

    def accepted(self, did):
        ok = {r["cid"] for r in self.verdict_rows(did) if r["accepted"] and r["is_hazard"]}
        return [c for c in self.cands(did) if c["id"] in ok]

    def rejected(self, did):
        ok = {r["cid"] for r in self.verdict_rows(did) if r["accepted"] and r["is_hazard"]}
        return [c for c in self.cands(did) if c["id"] not in ok]

    def hazards_rows(self):
        return self.q("SELECT * FROM hazards ORDER BY id")

    def drive_row(self, did):
        return self.q("SELECT * FROM drives WHERE id = ?", (did,))[0]

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

    def path_of(self, stored):
        """Best guess of the absolute Path of a stored path/key (may not exist)."""
        f = self.resolve_file(stored)
        if f is not None:
            return f
        p = Path(stored)
        return p if p.is_absolute() else self.storage / stored

    def video_file(self, did):
        from backend.storage import get_storage

        return Path(get_storage().local_path(self.drive_row(did)["video_path"]))

    def gpx_file(self, did):
        from backend.storage import get_storage

        return Path(get_storage().local_path(self.drive_row(did)["gpx_path"]))

    def drive_files(self, did):
        out = []
        key = re.sub(r"[^A-Za-z0-9_-]", "_", did)
        for p in self.tmp.rglob("*"):
            if p.is_file() and p != self.db and not p.name.startswith(self.db.name):
                rel = p.relative_to(self.tmp).parts
                if any(did in part or key in part for part in rel):
                    out.append(p)
        return out

    def blurred_paths(self, did):
        out = []
        for c in self.cands(did):
            for k in ("blurred_frame_path", "blurred_crop_path"):
                if c.get(k):
                    out.append(self.path_of(c[k]))
        return out

    def unblurred_paths(self, did):
        out = []
        for c in self.cands(did):
            for k in ("best_frame_path", "crop_path"):
                if c.get(k):
                    out.append(self.path_of(c[k]))
        return out

    # --- results api
    def photo(self, did, cid, crop=False):
        url = "/api/drives/%s/hazards/%s/photo" % (did, cid)
        if crop:
            url += "?variant=crop"
        return self.client.get(url)

    def hazards_list(self, did):
        r = self.client.get("/api/drives/%s/hazards" % did)
        assert r.status_code == 200, r.text
        return as_list(r.json())

    # --- delete video
    def delete_route(self):
        from backend.main import app

        pat = re.compile(r"^/api/drives/\{[^}/]+\}/(video|raw-video|raw_video|delete-video|delete_video|video/delete|"
                         r"video-delete)/?$")
        found = []
        for path, ops in app.openapi().get("paths", {}).items():
            if pat.match(path):
                for m in ("delete", "post"):
                    if m in ops:
                        found.append((0 if m == "delete" else 1, m, path))
        if not found:
            pytest.fail("no DELETE /api/drives/{id}/video (or POST .../delete-video) route in the OpenAPI schema")
        found.sort()
        return found[0][1], found[0][2]

    def delete_video(self, did):
        m, path = self.delete_route()
        return getattr(self.client, m)(re.sub(r"\{[^}/]+\}", did, path))


def as_list(body):
    if isinstance(body, list):
        return body
    assert isinstance(body, dict), "expected a JSON list, got %r" % (type(body),)
    for k in ("hazards", "items", "results", "data", "verified", "candidates"):
        if isinstance(body.get(k), list):
            return body[k]
    pytest.fail("no list found in response object keys %s" % sorted(body))


def cid_of(item):
    for k in ("candidate_id", "id"):
        if k in item:
            return item[k]
    pytest.fail("item has no candidate id: %s" % sorted(item))


def same_file(a, b):
    try:
        return os.path.samefile(str(a), str(b))
    except OSError:
        return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def make_env(mp, tmp_path, scenario=None):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    if not (FFMPEG and FFPROBE):
        pytest.skip("ffmpeg/ffprobe not on PATH")
    import backend.config  # noqa: F401  (loads a local .env before the key is cleared)

    mp.setenv("ROADSCOUT_DB", str(tmp_path / "db" / "rs15.sqlite"))
    mp.setenv("ROADSCOUT_STORAGE_DIR", str(tmp_path / "storage"))
    mp.setenv("ROADSCOUT_PROVIDER", "fake")
    mp.setenv("ROADSCOUT_PREPROCESS_FPS", "3")
    mp.setenv("ROADSCOUT_CHUNK_SECONDS", "4")
    mp.setenv("ROADSCOUT_COST_LOG", "off")
    for k in ("META_API_KEY", "ROADSCOUT_FAKE_SCENARIO", "ROADSCOUT_MIN_TRACK_FRAMES",
              "ROADSCOUT_VERIFY_MIN_CONFIDENCE", "ROADSCOUT_PREPROCESS_MAX_HEIGHT",
              "ROADSCOUT_DEDUP_DISTANCE_M", "ROADSCOUT_DEDUP_HEADING_TOLERANCE_DEG"):
        mp.delenv(k, raising=False)
    if scenario:
        mp.setenv("ROADSCOUT_FAKE_SCENARIO", scenario)

    real_connect = socket.socket.connect

    def _guard(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if host is not None and str(host) not in LOOPBACK:
            raise AssertionError("network access attempted during a task 15 test: %r" % (address,))
        return real_connect(self, address)

    mp.setattr(socket.socket, "connect", _guard)

    from fastapi.testclient import TestClient

    from backend.db import init_db
    from backend.main import app

    init_db()
    return TestClient(app)


import pytest as _pytest  # noqa: E402


@_pytest.fixture()
def rs15_env(monkeypatch, tmp_path):
    client = make_env(monkeypatch, tmp_path)
    with client:
        yield Env(monkeypatch, tmp_path, client)


@_pytest.fixture()
def rs15_done(rs15_env):
    """(env, drive_id, stub): one drive processed end to end with the boxes stub (face + plate masks)."""
    stub = make_stub()
    did = rs15_env.process_ok(stub)
    return rs15_env, did, stub
