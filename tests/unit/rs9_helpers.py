"""Shared helpers and fixtures for the task 9 (Muse Spark verification) hidden tests.

Interface assumed (see the report to the main agent):

    backend/verification.py
        verify_drive(drive_id, *, provider=None, min_confidence=None, db_path=None)
            (alias names tolerated: verify_candidates, run_verification)
    provider=None            -> backend.providers.get_provider() (fake unless configured)
    min_confidence           -> overrides env ROADSCOUT_VERIFY_MIN_CONFIDENCE (default 0.5..0.8)
    verification prompt      -> JSON file {"version": "<str>", "prompt": "<text>"}, path from env
                                ROADSCOUT_VERIFY_PROMPT (default config/verification_prompt.json)
    table verdicts           -> candidate_id, is_hazard, class (or hazard_class), severity, lane,
                                confidence, reason, report_description, model, prompt_version,
                                accepted (0/1; or status; or a helper returning accepted verdicts)
"""
from __future__ import annotations

import copy
import importlib
import inspect
import io
import json
import os
import shutil
import sqlite3
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

PROMPT_ENV = "ROADSCOUT_VERIFY_PROMPT"
THRESH_ENV = "ROADSCOUT_VERIFY_MIN_CONFIDENCE"
PV = "t9-v1"

FUNC_NAMES = ["verify_drive", "verify_candidates", "run_verification", "verify_all"]
THRESH_PARAMS = ["min_confidence", "confidence_threshold", "threshold", "min_conf"]
DB_PARAMS = ["db_path", "db", "database", "database_path"]
PROVIDER_PARAMS = ["provider", "model_provider"]

BEST_SIZE = (64, 48)
CROP_SIZE = (16, 12)


# ------------------------------------------------------------------ misc


def class_names():
    from backend.hazard_prompts import load_hazard_prompts

    return load_hazard_prompts().class_names


def write_prompt_config(path, version=PV, prompt="Judge the hazard in the images and answer in JSON."):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": version, "prompt": prompt}), encoding="utf-8")
    return path


def make_verdict(is_hazard=True, cls=None, severity=2, lane="center", confidence=0.9, reason="A clear hazard.",
                 report_description="A hazard in the lane.", model="stub-muse", prompt_version=""):
    from backend.providers import Verdict

    return Verdict(is_hazard=is_hazard, hazard_class=cls or class_names()[0], severity=severity, lane=lane,
                   confidence=confidence, reason=reason, report_description=report_description,
                   model=model, prompt_version=prompt_version)


def corrupt(verdict, **fields):
    """A Verdict that bypasses its own validation (frozen dataclass, so object.__setattr__)."""
    for k, v in fields.items():
        object.__setattr__(verdict, k, v)
    return verdict


def _jpeg(path, size, rgb):
    from PIL import Image

    Image.new("RGB", size, rgb).save(path, "JPEG", quality=95)


def rgb_of(item):
    """Mean RGB of an image given as a path or as raw bytes."""
    import numpy as np
    from PIL import Image

    if isinstance(item, (bytes, bytearray)):
        im = Image.open(io.BytesIO(bytes(item)))
    else:
        im = Image.open(str(item))
    with im:
        im.load()
        arr = np.asarray(im.convert("RGB"), dtype="float64")
    return tuple(float(x) for x in arr.reshape(-1, 3).mean(axis=0))


def size_of(item):
    from PIL import Image

    if isinstance(item, (bytes, bytearray)):
        im = Image.open(io.BytesIO(bytes(item)))
    else:
        im = Image.open(str(item))
    with im:
        return im.size


def close(a, b, tol=22):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# ------------------------------------------------------------------ interface adapter


def load_module():
    return importlib.import_module("backend.verification")


def find_func():
    mod = load_module()
    for n in FUNC_NAMES:
        f = getattr(mod, n, None)
        if callable(f):
            return f
    pytest.fail("backend.verification has none of the expected functions %s" % FUNC_NAMES)


def _pick(sig, names):
    for n in names:
        if n in sig.parameters:
            return n
    return None


# ------------------------------------------------------------------ recording stub provider


def _make_stub_class():
    from backend.providers import ModelProvider

    class RecordingProvider(ModelProvider):
        """Records every verify call; verdicts are scripted per candidate (by registry index)."""

        name = "recording"

        def __init__(self, env, script=None, default=None, echo_pv=True):
            self.env = env
            self.script = dict(script or {})
            self.default = default
            self.echo_pv = echo_pv
            self.calls = []

        def detect(self, video_path, prompts):
            raise AssertionError("verification must not call detect")

        def verify(self, images, *, class_prompt=None, prompt_version=None):
            images = list(images)
            cand = self.env.identify(images[0]) if images else None
            self.calls.append({"images": images, "class_prompt": class_prompt,
                               "prompt_version": prompt_version, "cand": cand})
            if cand is not None and cand["k"] in self.script:
                item = self.script[cand["k"]]
            else:
                item = self.default
            if callable(item) and not isinstance(item, BaseException):
                item = item(cand, class_prompt, prompt_version)
            elif item is None:
                item = make_verdict(cls=class_prompt)
            if isinstance(item, BaseException):
                raise item
            if self.echo_pv and hasattr(item, "prompt_version") and not item.prompt_version:
                item = copy.copy(item)
                object.__setattr__(item, "prompt_version", prompt_version or "")
            return item

    return RecordingProvider


def stub(env, **kw):
    return _make_stub_class()(env, **kw)


# ------------------------------------------------------------------ environment


class Env:
    def __init__(self, mp, tmp_path):
        self.mp = mp
        self.tmp = Path(tmp_path)
        self.db = self.tmp / "rs9.sqlite"
        self.storage = self.tmp / "storage"
        self.storage.mkdir()
        self.prompt_cfg = write_prompt_config(self.tmp / "verify_prompt.json")
        mp.setenv("ROADSCOUT_DB", str(self.db))
        mp.setenv("ROADSCOUT_STORAGE_DIR", str(self.storage))
        mp.setenv(PROMPT_ENV, str(self.prompt_cfg))
        for k in ("META_API_KEY", "ROADSCOUT_PROVIDER", "ROADSCOUT_FAKE_SCENARIO", THRESH_ENV):
            mp.delenv(k, raising=False)
        from backend.db import init_db

        init_db()
        self.registry = []  # candidates added by add_candidates
        self.img_dir = self.tmp / "imgs"
        self.img_dir.mkdir()

    # --- config
    def set_version(self, version):
        write_prompt_config(self.prompt_cfg, version=version)

    def set_threshold(self, value):
        if value is None:
            self.mp.delenv(THRESH_ENV, raising=False)
        else:
            self.mp.setenv(THRESH_ENV, str(value))

    # --- synthetic candidates
    def add_candidates(self, n=3, drive="drive-a", base_id=9101, crop=True, classes=None):
        names = classes or class_names()
        added = []
        conn = sqlite3.connect(str(self.db))
        try:
            for i in range(n):
                k = len(self.registry)
                best_rgb = (200, 20 + 25 * k, 30)
                crop_rgb = (30, 20 + 25 * k, 200)
                best = self.img_dir / ("best_%d.jpg" % (base_id + i))
                cp = self.img_dir / ("crop_%d.jpg" % (base_id + i))
                _jpeg(best, BEST_SIZE, best_rgb)
                if crop:
                    _jpeg(cp, CROP_SIZE, crop_rgb)
                cid = base_id + i
                conn.execute(
                    "INSERT INTO candidates (id, drive_id, hazard_id, hazard_class, track_key, track_id, chunk, "
                    "first_timestamp, last_timestamp, frames_seen, max_mask_area, best_frame_index, "
                    "best_timestamp, best_box, best_frame_path, crop_path, lat, lon, heading, score, "
                    "geotag_status, geotag_note) VALUES (?,?,NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (cid, drive, names[i % len(names)], "0:%d" % (i + 1), i + 1, 0, float(i), float(i) + 3.0, 6,
                     500.0, i * 3, float(i) + 1.0, "[1,2,10,10]", str(best), str(cp) if crop else None,
                     38.88 + 0.001 * i, -77.3, 0.0, 0.7, "ok", None))
                entry = {"k": k, "id": cid, "drive": drive, "cls": names[i % len(names)],
                         "best": best, "crop": cp if crop else None, "best_rgb": best_rgb, "crop_rgb": crop_rgb}
                self.registry.append(entry)
                added.append(entry)
            conn.commit()
        finally:
            conn.close()
        return added

    def delete_candidates(self, drive="drive-a"):
        """Replace like aggregate_drive does: DELETE through the project's own connection (FKs on)."""
        from backend.db import get_connection

        conn = get_connection(self.db)
        try:
            with conn:
                conn.execute("DELETE FROM candidates WHERE drive_id = ?", (drive,))
        finally:
            conn.close()

    def identify(self, item):
        rgb = rgb_of(item)
        best = min(self.registry, key=lambda e: sum((a - b) ** 2 for a, b in zip(rgb, e["best_rgb"])))
        if not close(rgb, best["best_rgb"]):
            return None
        return best

    # --- running the code under test
    def run(self, drive_id="drive-a", provider=None, *, min_confidence=None, **extra):
        f = find_func()
        sig = inspect.signature(f)
        params = list(sig.parameters)
        kw = dict(extra)
        positional = []
        drive_p = next((p for p in params if "drive" in p.lower()), None)
        if drive_p:
            kw[drive_p] = drive_id
        else:
            positional = [drive_id]
        db_p = _pick(sig, DB_PARAMS)
        if db_p:
            kw[db_p] = str(self.db)
        if provider is not None:
            p = _pick(sig, PROVIDER_PARAMS)
            if p is None:
                pytest.fail("verification function has no provider parameter; signature %s" % (sig,))
            kw[p] = provider
        if min_confidence is not None:
            p = _pick(sig, THRESH_PARAMS)
            if p is None:
                pytest.fail("verification function has no threshold parameter (looked for %s); signature %s"
                            % (THRESH_PARAMS, sig))
            kw[p] = min_confidence
        return f(*positional, **kw)

    # --- reading back
    def raw_all(self):
        conn = sqlite3.connect(str(self.db))
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM verdicts ORDER BY candidate_id")]
        finally:
            conn.close()

    def rows(self, drive="drive-a"):
        conn = sqlite3.connect(str(self.db))
        conn.row_factory = sqlite3.Row
        try:
            q = ("SELECT v.* FROM verdicts v JOIN candidates c ON c.id = v.candidate_id "
                 "WHERE c.drive_id = ? ORDER BY v.candidate_id")
            return [dict(r) for r in conn.execute(q, (drive,))]
        finally:
            conn.close()

    def by_candidate(self, drive="drive-a"):
        out = {}
        for r in self.rows(drive):
            assert r["candidate_id"] not in out, "two verdict rows for candidate %s" % r["candidate_id"]
            out[r["candidate_id"]] = r
        return out

    def columns(self):
        conn = sqlite3.connect(str(self.db))
        try:
            return [r[1] for r in conn.execute("PRAGMA table_info(verdicts)")]
        finally:
            conn.close()

    def accepted(self, drive="drive-a"):
        """{candidate_id: bool}. Discovers the representation: accepted column, status column, or helper."""
        rows = self.rows(drive)
        out = {}
        if rows and "accepted" in rows[0]:
            for r in rows:
                assert r["accepted"] in (0, 1, True, False), "accepted must be 0/1, got %r" % (r["accepted"],)
                out[r["candidate_id"]] = bool(r["accepted"])
            return out
        if rows and "status" in rows[0]:
            for r in rows:
                s = str(r["status"]).strip().lower()
                assert s in ("accepted", "rejected"), "status must be accepted/rejected, got %r" % s
                out[r["candidate_id"]] = s == "accepted"
            return out
        mod = load_module()
        for n in ("accepted_verdicts", "load_accepted_verdicts", "get_accepted_verdicts", "list_accepted"):
            h = getattr(mod, n, None)
            if callable(h):
                sig = inspect.signature(h)
                kw = {}
                db_p = _pick(sig, DB_PARAMS)
                if db_p:
                    kw[db_p] = str(self.db)
                res = h(drive, **kw)
                ids = set()
                for x in res:
                    ids.add(x["candidate_id"] if isinstance(x, dict) or hasattr(x, "keys")
                            else getattr(x, "candidate_id"))
                return {r["candidate_id"]: r["candidate_id"] in ids for r in rows}
        pytest.fail("no accepted column, no status column and no accepted-verdicts helper (columns: %s)"
                    % (list(rows[0]) if rows else self.columns()))


def cls_of(row):
    for k in ("class", "hazard_class", "class_name"):
        if k in row:
            return row[k]
    raise AssertionError("verdicts row has no class column: %s" % list(row))


@pytest.fixture()
def rs9_env(monkeypatch, tmp_path):
    import socket

    def _no_net(*a, **k):
        raise AssertionError("network access attempted during a task 9 test")

    monkeypatch.setattr(socket.socket, "connect", _no_net)
    monkeypatch.setattr(socket, "create_connection", _no_net)
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")
    return Env(monkeypatch, tmp_path)
