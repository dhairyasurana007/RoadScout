"""Shared helpers and fixtures for the task 7 (SAM 3.1 detection per chunk) hidden tests.

Interface assumed (see the report to the main agent):

    backend/detection.py
        detect_drive(drive_id, work_dir, *, provider=None, prompts_path=None, db_path=None)
    work_dir      the preprocess output directory (chunk_NNNN.mp4 + frame_timestamps.json)
    provider      a ModelProvider; None means backend.providers.get_provider()
    prompts_path  prompt config file; None means ROADSCOUT_PROMPTS_FILE, else config/prompts.json
    db_path       SQLite file; None means backend.config.db_path() (ROADSCOUT_DB)

    config/prompts.json  {"classes": {"pothole": ["pothole", ...], "debris in lane": [...], ...}}

    table detections: id, drive_id, chunk, frame_index, frame_timestamp, class_prompt,
                      track_id (raw provider id), track_key (namespaced per chunk), box (JSON
                      [x1,y1,x2,y2]), mask (JSON RLE dict or NULL), score
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
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

FUNC_NAMES = ["run_detection", "detect_drive", "detect_chunks", "detect_all", "run_detect", "detect"]
PROVIDER_PARAMS = ["provider", "model_provider", "detector"]
PROMPTS_PARAMS = ["prompts_path", "prompts_file", "prompt_config", "config_path", "prompt_config_path", "config"]
DB_PARAMS = ["db_path", "db", "database", "database_path"]
WORK_PARAMS = ["work_dir", "output_dir", "chunks_dir", "chunk_dir", "preprocess_dir", "preprocessed_dir",
               "preprocess_output", "workdir"]

PROMPTS_ENV = "ROADSCOUT_HAZARD_PROMPTS"

CHUNK_SECONDS = 4
FPS = 3
EXPECTED_CHUNKS = 3  # 12 s clip, 4 s chunks, 3 fps -> 12 frames per chunk

CLASS_KEYWORDS = {
    "pothole": ["pothole"],
    "debris": ["debris"],
    "guardrail": ["guardrail", "guard rail"],
    "sign": ["sign"],
    "lane": ["lane marking", "lane mark"],
}


# ------------------------------------------------------------------ clips and preprocessing


def _ffmpeg(args):
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
                       capture_output=True, timeout=180)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")


def make_clip(path, seconds=12):
    _ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=160x90:rate=30", "-t", seconds,
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", path])
    return Path(path)


@pytest.fixture(scope="session")
def rs7_work(tmp_path_factory):
    """Preprocessed synthetic clip: (work_dir, timestamps dict). 3 chunks of 12 frames at 3 fps."""
    if not (FFMPEG and FFPROBE):
        pytest.skip("ffmpeg/ffprobe not on PATH")
    base = tmp_path_factory.mktemp("rs7work")
    clip = make_clip(base / "clip.mp4")
    from backend.preprocess import preprocess_video

    out = base / "out"
    preprocess_video(clip, out, fps=FPS, chunk_seconds=CHUNK_SECONDS)
    ts = json.loads((out / "frame_timestamps.json").read_text(encoding="utf-8"))
    assert len(ts["chunks"]) == EXPECTED_CHUNKS, "synthetic clip did not split into 3 chunks"
    assert all(c["frame_count"] == 12 for c in ts["chunks"])
    return out, ts


# ------------------------------------------------------------------ providers


def chunk_index_of(video_path) -> int:
    m = re.search(r"chunk_(\d+)", str(video_path))
    assert m, "detect() was not given a chunk file path: %r" % (video_path,)
    return int(m.group(1))


class RecordingProvider:
    """Records every detect() call. ``script(call_no, chunk_idx, prompts)`` may return a
    DetectionResult (or None to delegate to ``inner``), or raise."""

    def __init__(self, script=None, inner=None):
        from backend.providers import FakeProvider, ModelProvider

        outer = self

        class _P(ModelProvider):
            name = "recording"

            def detect(self, video_path, prompts):
                call_no = len(outer.calls)
                outer.calls.append({
                    "path": video_path,
                    "exists": os.path.isfile(str(video_path)),
                    "prompts": list(prompts),
                    "chunk": chunk_index_of(video_path),
                })
                if outer.script is not None:
                    res = outer.script(call_no, chunk_index_of(video_path), list(prompts))
                    if res is not None:
                        return res
                return outer.inner.detect(video_path, prompts)

            def verify(self, images, *, class_prompt=None, prompt_version=None):
                raise NotImplementedError("task 7 must not call verification")

        self.calls: list[dict] = []
        self.script = script
        self.inner = inner or FakeProvider("default")
        self.provider = _P()

    def prompts_for_chunk(self, chunk):
        out = []
        for c in self.calls:
            if c["chunk"] == chunk:
                out.extend(c["prompts"])
        return out


def det(track_id, box, score, prompt, mask="auto"):
    from backend.providers import Detection, rle_from_box

    if mask == "auto":
        mask = rle_from_box(160, 90, box)
    return Detection(track_id=track_id, box=tuple(box), score=score, prompt=prompt, mask=mask)


def frame(index, detections, fps=FPS):
    from backend.providers import FrameRecord

    return FrameRecord(frame_index=index, timestamp=round(index / fps, 6), detections=tuple(detections))


def result(*frames):
    from backend.providers import DetectionResult

    return DetectionResult(tuple(frames))


# ------------------------------------------------------------------ interface adapter


def load_module():
    return importlib.import_module("backend.detection")


def find_func():
    mod = load_module()
    for n in FUNC_NAMES:
        f = getattr(mod, n, None)
        if callable(f):
            return f
    pytest.fail("backend.detection has none of the expected functions %s" % FUNC_NAMES)


def _pick(sig, names):
    for n in names:
        if n in sig.parameters:
            return n
    return None


def _insert_drive(db, drive_id):
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT OR REPLACE INTO drives (id, started_at, started_at_source, clock_offset_seconds, "
            "video_path, gpx_path, status, duration_seconds, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (drive_id, "2026-10-20T12:00:00+00:00", "metadata", 0.0, "v/%s.mp4" % drive_id,
             "g/%s.gpx" % drive_id, "preprocessed", 12.0, "2026-10-20T12:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()


class Env:
    def __init__(self, mp, tmp_path, work):
        self.mp = mp
        self.tmp = Path(tmp_path)
        self.work_dir, self.ts = work
        self.db = self.tmp / "rs7.sqlite"
        mp.setenv("ROADSCOUT_DB", str(self.db))
        for k in ("META_API_KEY", "ROADSCOUT_PROVIDER", "ROADSCOUT_FAKE_SCENARIO", PROMPTS_ENV):
            mp.delenv(k, raising=False)
        from backend.db import init_db

        init_db()
        self.drives = set()
        self.add_drive("drive-a")

    def add_drive(self, drive_id):
        _insert_drive(self.db, drive_id)
        self.drives.add(drive_id)

    # --- running the code under test
    def run(self, drive_id="drive-a", provider=None, prompts_path=None, work_dir=None, **extra):
        f = find_func()
        sig = inspect.signature(f)
        params = list(sig.parameters)
        work_dir = self.work_dir if work_dir is None else work_dir
        kw = dict(extra)
        positional = []
        drive_p = next((p for p in params if "drive" in p.lower()), None)
        work_p = _pick(sig, WORK_PARAMS) or next(
            (p for p in params if p not in (drive_p,) and any(s in p.lower() for s in ("dir", "work", "chunk", "preproc"))),
            None)
        if drive_p and work_p:
            kw[drive_p] = drive_id
            kw[work_p] = work_dir
        else:
            positional = [drive_id, work_dir]
        db_p = _pick(sig, DB_PARAMS)
        if db_p and db_p not in kw:
            kw[db_p] = str(self.db)
        if provider is not None:
            pp = _pick(sig, PROVIDER_PARAMS)
            prov = getattr(provider, "provider", provider)
            if pp:
                kw[pp] = prov
            else:
                self._patch_get_provider(prov)
        if prompts_path is not None:
            cp = _pick(sig, PROMPTS_PARAMS)
            if cp is not None:
                kw[cp] = str(prompts_path)
            elif "prompts" in sig.parameters:
                from backend.hazard_prompts import load_hazard_prompts

                kw["prompts"] = load_hazard_prompts(prompts_path)
            else:
                pytest.skip("no prompt-config parameter (env var path tested elsewhere)")
        return f(*positional, **kw)

    def _patch_get_provider(self, prov):
        mod = load_module()
        for target in (mod, importlib.import_module("backend.providers"),
                       importlib.import_module("backend.providers.factory")):
            if hasattr(target, "get_provider"):
                self.mp.setattr(target, "get_provider", lambda *a, **k: prov)

    def has_prompts_param(self):
        sig = inspect.signature(find_func())
        return _pick(sig, PROMPTS_PARAMS) is not None or "prompts" in sig.parameters

    # --- reading the table back
    def raw_rows(self, drive_id="drive-a"):
        conn = sqlite3.connect(str(self.db))
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM detections WHERE drive_id = ?", (drive_id,))]
        finally:
            conn.close()

    def all_count(self):
        conn = sqlite3.connect(str(self.db))
        try:
            return conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        finally:
            conn.close()

    def columns(self):
        conn = sqlite3.connect(str(self.db))
        try:
            return [r[1] for r in conn.execute("PRAGMA table_info(detections)")]
        finally:
            conn.close()

    def rows(self, drive_id="drive-a"):
        return [normalize(r) for r in self.raw_rows(drive_id)]


@pytest.fixture()
def rs7_env(monkeypatch, tmp_path, rs7_work):
    # No network at all in these tests.
    import socket

    def _no_net(*a, **k):
        raise AssertionError("network access attempted during a task 7 test")

    monkeypatch.setattr(socket.socket, "connect", _no_net)
    monkeypatch.setattr(socket, "create_connection", _no_net)
    return Env(monkeypatch, tmp_path, rs7_work)


# ------------------------------------------------------------------ row normalisation


def _pick_key(row, names):
    for n in names:
        if n in row:
            return n
    return None


def _json(v):
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        v = bytes(v).decode("utf-8")
    if isinstance(v, str):
        return json.loads(v)
    return v


def _last_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v == int(v):
        return int(v)
    if isinstance(v, str):
        m = re.findall(r"-?\d+", v)
        return int(m[-1]) if m else None
    return None


def normalize(r):
    out = {"id": r.get("id"), "raw": r}
    k = _pick_key(r, ["chunk", "chunk_index", "chunk_id", "chunk_file"])
    assert k, "detections table has no chunk column: %s" % list(r)
    out["chunk"] = _last_int(r[k])
    k = _pick_key(r, ["frame_index", "frame", "frame_idx", "frame_number"])
    assert k, "detections table has no frame index column: %s" % list(r)
    out["frame_index"] = int(r[k])
    k = _pick_key(r, ["frame_timestamp", "timestamp", "global_timestamp", "frame_time", "time_s"])
    assert k, "detections table has no frame timestamp column: %s" % list(r)
    out["ts"] = float(r[k])
    k = _pick_key(r, ["class_prompt", "prompt", "class", "hazard_class"])
    assert k, "detections table has no class prompt column: %s" % list(r)
    out["prompt"] = r[k]
    out["tracks"] = {c: v for c, v in r.items() if "track" in c.lower()}
    assert out["tracks"], "detections table has no track id column: %s" % list(r)
    if "box" in r:
        box = _json(r["box"])
    else:
        names = [("x1", "y1", "x2", "y2"), ("box_x1", "box_y1", "box_x2", "box_y2")]
        box = None
        for n in names:
            if all(c in r for c in n):
                box = [r[c] for c in n]
        assert box is not None, "detections table has no box column: %s" % list(r)
    out["box"] = tuple(float(x) for x in box)
    out["mask"] = _json(r.get("mask"))
    k = _pick_key(r, ["score", "confidence"])
    out["score"] = None if k is None or r[k] is None else float(r[k])
    return out


def signature_rows(rows):
    """Order-independent, id-independent comparable form of a list of normalised rows."""
    return sorted(
        (
            r["chunk"], r["frame_index"], round(r["ts"], 6), r["prompt"], r["box"],
            json.dumps(r["mask"], sort_keys=True), r["score"],
            tuple(sorted((c, str(v)) for c, v in r["tracks"].items())),
        )
        for r in rows
    )


# ------------------------------------------------------------------ prompt config


def find_default_config():
    for rel in ("config/prompts.json", "config/prompts.yaml", "config/prompts.yml",
                "config/hazard_prompts.json", "config/hazard_prompts.yaml", "config/hazard_prompts.yml"):
        p = ROOT / rel
        if p.is_file():
            return p
    cfg = ROOT / "config"
    if cfg.is_dir():
        for p in sorted(cfg.iterdir()):
            if "prompt" in p.name.lower() and p.suffix.lower() in (".json", ".yaml", ".yml"):
                return p
    return None


def parse_config(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(text)


def dump_config(obj, path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    else:
        yaml = pytest.importorskip("yaml")
        path.write_text(yaml.safe_dump(obj), encoding="utf-8")
    return path


def strings(obj):
    """All string leaf values (dict keys excluded)."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in strings(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in strings(v)]
    return []


def class_groups(cfg):
    """{class name: [phrasings]} from a parsed config (dict of classes, or list of class objects)."""
    root = cfg.get("classes", cfg) if isinstance(cfg, dict) else cfg
    groups = {}
    if isinstance(root, dict):
        for k, v in root.items():
            if isinstance(v, (list, dict, str)) and strings(v):
                groups[str(k)] = strings(v)
    elif isinstance(root, list):
        for item in root:
            if isinstance(item, dict):
                name = next((item[n] for n in ("class", "name", "label", "id") if isinstance(item.get(n), str)), None)
                phr = [s for k, v in item.items() if k not in ("class", "name", "label", "id") for s in strings(v)]
                if name and phr:
                    groups[name] = phr
    return groups


def required_groups(groups):
    """Map each required class keyword group to the config class name that covers it."""
    found = {}
    for key, variants in CLASS_KEYWORDS.items():
        for name in groups:
            n = name.lower().replace("_", " ").replace("-", " ")
            if any(v in n for v in variants):
                found[key] = name
                break
    return found


def swapped_copy(tmp_path, prefix="swapped: "):
    """Copy of the default config with every phrasing value prefixed (keys untouched)."""
    src = find_default_config()
    assert src is not None, "no default prompt config under config/"
    cfg = parse_config(src)

    def walk(o):
        if isinstance(o, str):
            return prefix + o
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o

    return dump_config(walk(cfg), Path(tmp_path) / ("swapped" + src.suffix)), prefix
