"""Helpers for the task 23b hidden tests (demo dataset builder): a seeded source database and storage tree."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
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

IMG_EXT = (".jpg", ".jpeg", ".png")
DRIVE_LETTER = re.compile(r"[A-Za-z]:[\\/]")
PATH_COLS = ("best_frame_path", "crop_path", "blurred_frame_path", "blurred_crop_path")


def jpeg(label: str) -> bytes:
    return b"\xff\xd8\xff\xe0" + label.encode() + b"-" + (label.encode() * 8) + b"\xff\xd9"


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree_hash(root) -> dict:
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


class World:
    """Source database + storage tree made with the project's own schema (see the module docstring of the tests)."""

    def __init__(self, tmp_path):
        self.tmp = Path(tmp_path)
        self.db = self.tmp / "src" / "roadscout.sqlite"
        self.storage = self.tmp / "src" / "storage"
        self.outside = self.tmp / "elsewhere"
        self.db.parent.mkdir(parents=True)
        self.storage.mkdir(parents=True)
        self.outside.mkdir(parents=True)
        from backend.db import init_db

        init_db(self.db)
        self.ids = {}
        self.files = {}
        self._build()

    # ---- raw helpers
    def conn(self):
        c = sqlite3.connect(str(self.db))
        c.row_factory = sqlite3.Row
        return c

    def cand_dir(self, drive):
        d = self.storage / drive / "candidates"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write(self, drive, name, label=None):
        p = self.cand_dir(drive) / name
        p.write_bytes(jpeg(label or f"{drive}-{name}"))
        return p

    def fingerprint(self):
        data = self.db.read_bytes()
        return hashlib.sha256(data).hexdigest(), self.db.stat().st_mtime_ns

    # ---- the world
    def _drive(self, con, did, filename):
        con.execute(
            "INSERT INTO drives (id, started_at, started_at_source, clock_offset_seconds, video_path, gpx_path, "
            "status, duration_seconds, created_at, stage, error, failed_stage, updated_at, finished_at, "
            "warning, video_filename) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (did, "2026-10-20T12:00:00+00:00", "metadata", 1.5, str(self.storage / did / "video.mov"),
             str(self.storage / did / "track.gpx"), "done", 120.0, "2026-10-20T12:00:00+00:00", "done",
             f"old error in {self.storage / did}", None, "2026-10-20T12:30:00+00:00",
             "2026-10-20T12:30:00+00:00", f"a warning at {self.storage}", filename))
        d = self.storage / did
        d.mkdir(parents=True, exist_ok=True)
        (d / "video.mov").write_bytes(b"RAW-VIDEO-" + did.encode())
        (d / "track.gpx").write_bytes(b"<gpx>RAW-GPX</gpx>")
        (d / "work").mkdir(exist_ok=True)
        (d / "work" / "chunk_0000.mp4").write_bytes(b"CHUNK")
        (d / "exports").mkdir(exist_ok=True)
        (d / "exports" / "export_1.mp4").write_bytes(b"EXPORTED-VIDEO")
        self.write(did, "stray_unreferenced.jpg")

    def _candidate(self, con, cid, did, cls, hazard_id, *, crop=True, blurred=True, blurred_crop=True):
        best = self.write(did, f"c{cid}_best.jpg", f"ORIGINAL-BEST-{cid}")
        crop_p = self.write(did, f"c{cid}_crop.jpg", f"ORIGINAL-CROP-{cid}") if crop else None
        bl = self.write(did, f"c{cid}_best_blur.jpg", f"BLURRED-BEST-{cid}") if blurred else None
        bc = self.write(did, f"c{cid}_crop_blur.jpg", f"BLURRED-CROP-{cid}") if (crop and blurred_crop) else None
        con.execute(
            "INSERT INTO candidates (id, drive_id, hazard_id, hazard_class, track_key, track_id, chunk, "
            "first_timestamp, last_timestamp, frames_seen, max_mask_area, best_frame_index, best_timestamp, "
            "best_box, best_frame_path, crop_path, lat, lon, heading, score, geotag_status, geotag_note, "
            "blurred_frame_path, blurred_crop_path, blur_status, blur_note) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (cid, did, hazard_id, cls, f"0:{cid}", cid, 0, 1.0, 5.0, 5, 400.0, 3, 2.0, "[1,2,10,10]", str(best),
             str(crop_p) if crop_p else None, 38.88, -77.3, 10.0, 0.7, "ok", None,
             str(bl) if bl else None, str(bc) if bc else None, "ok" if bl else None, None))
        con.execute(
            "INSERT INTO verdicts (candidate_id, drive_id, is_hazard, hazard_class, severity, lane, confidence, "
            "reason, report_description, model, prompt_version, accepted, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (cid, did, 1, cls, 2, "center", 0.9, f"verdict reason {cid}", f"description {cid}", "muse-spark-1.3", "v1",
             1, "2026-10-20T12:10:00+00:00"))
        self.files[cid] = {"best": best, "crop": crop_p, "blurred": bl, "blurred_crop": bc}
        return cid

    def _hazard(self, con, hid, cls, status, photo, crop=None):
        con.execute(
            "INSERT INTO hazards (id, hazard_class, lat, lon, direction, severity, confidence, sightings, "
            "candidate_count, first_seen, last_seen, photo_path, crop_path, status, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (hid, cls, 38.88 + hid * 0.001, -77.3, 90.0, 2, 0.85, 1, 1, "2026-10-20T12:01:00+00:00",
             "2026-10-20T12:02:00+00:00", str(photo) if photo else None, str(crop) if crop else None, status,
             "2026-10-20T12:00:00+00:00", "2026-10-20T12:05:00+00:00"))

    def _build(self):
        con = self.conn()
        try:
            with con:
                con.execute("INSERT OR REPLACE INTO app_meta (key, value) VALUES ('demo_note', 'copied as is')")
                self._drive(con, "d1", "IMG_1.MOV")
                self._drive(con, "d2", "IMG_2.MOV")
                self._drive(con, "d3", "IMG_3.MOV")
                # kept: approved hazard of d1 with two sightings (c101 gives the photo and crop)
                c101 = self._candidate(con, 101, "d1", "pothole", 1)
                c102 = self._candidate(con, 102, "d1", "pothole", 1, crop=False)
                c103 = self._candidate(con, 103, "d1", "debris in lane", None)          # unlinked
                c104 = self._candidate(con, 104, "d1", "debris in lane", 4)             # hazard 4 is 'new'
                # kept: reported hazard of d2, no crop
                c201 = self._candidate(con, 201, "d2", "damaged guardrail", 2, crop=False)
                c202 = self._candidate(con, 202, "d2", "pothole", 5)                    # hazard 5 is rejected
                # d3: nothing servable
                c301 = self._candidate(con, 301, "d3", "pothole", 6)
                c302 = self._candidate(con, 302, "d3", "pothole", 7)
                c303 = self._candidate(con, 303, "d3", "pothole", 8)
                c304 = self._candidate(con, 304, "d3", "pothole", 9)
                c305 = self._candidate(con, 305, "d3", "pothole", 10)
                f = self.files
                self._hazard(con, 1, "pothole", "approved", f[101]["blurred"], f[101]["blurred_crop"])
                self._hazard(con, 2, "damaged guardrail", "reported", f[201]["blurred"])
                self._hazard(con, 4, "debris in lane", "new", f[104]["blurred"], f[104]["blurred_crop"])
                self._hazard(con, 5, "pothole", "rejected", f[202]["blurred"], f[202]["blurred_crop"])
                self._hazard(con, 6, "pothole", "approved", f[301]["best"])                         # an ORIGINAL
                self._hazard(con, 7, "pothole", "approved", self.cand_dir("d3") / "missing_blur.jpg")
                outside = self.outside / "photo_blur.jpg"
                outside.write_bytes(jpeg("OUTSIDE-STORAGE"))
                self._hazard(con, 8, "pothole", "approved", outside)
                nested = self.cand_dir("d3") / "sub" / "nested_blur.jpg"
                nested.parent.mkdir()
                nested.write_bytes(jpeg("NESTED"))
                self._hazard(con, 9, "pothole", "approved", nested)
                self._hazard(con, 10, "pothole", "approved", None)
                # tables the demo must not carry
                con.execute("INSERT INTO detections (drive_id, chunk, frame_index, frame_in_chunk, frame_timestamp, "
                            "hazard_class, class_prompt, track_id, track_key, box, mask, score) VALUES "
                            "('d1',0,1,1,0.3,'pothole','pothole',1,'0:1','[1,2,3,4]',NULL,0.9)")
                con.execute("INSERT INTO detection_chunks (drive_id, chunk, frames, dropped) VALUES ('d1',0,12,0)")
                con.execute("INSERT INTO pipeline_runs (drive_id, fingerprint, complete, stages, updated_at) "
                            "VALUES ('d1','abc',1,'{}','2026-10-20T12:00:00+00:00')")
                con.execute("INSERT INTO exports (drive_id, hazard_id, start_seconds, end_seconds, key, status, error, "
                            "created_at) VALUES ('d1',1,0,5,'d1/exports/export_1.mp4','ok',NULL,'2026-10-20T12:00:00+00:00')")
        finally:
            con.close()
        self.kept = {1, 2}
        self.skipped_unservable = {6, 7, 8, 9, 10}
        self.kept_candidates = {101, 102, 201}
        self.kept_drives = {"d1", "d2"}
        self.photo_files = [self.files[101]["blurred"], self.files[101]["blurred_crop"], self.files[201]["blurred"]]
        self.originals = [p for c in self.files.values() for p in (c["best"], c["crop"]) if p]

    def build(self, out, **kw):
        from backend.demo_dataset import build_demo_dataset

        return build_demo_dataset(str(self.db), str(self.storage), str(out), **kw)


# ------------------------------------------------------------------ reading an output folder

def rows(db_path, sql, params=()):
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params)]
    finally:
        con.close()


def all_files(out):
    out = Path(out)
    return sorted(str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*") if p.is_file())


def tables(db_path):
    return [r["name"] for r in rows(db_path, "SELECT name FROM sqlite_master WHERE type='table' AND "
                                              "name NOT LIKE 'sqlite_%' ORDER BY name")]


def all_text_values(db_path):
    con = sqlite3.connect(str(db_path))
    try:
        for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall():
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
            for row in con.execute(f"SELECT * FROM {t}"):
                for c, v in zip(cols, row):
                    if isinstance(v, str):
                        yield t, c, v
    finally:
        con.close()


def dump(db_path):
    return {t: rows(db_path, f"SELECT * FROM {t} ORDER BY rowid") for t in tables(db_path)}


def run_cli(args, cwd=None, env_over=None, timeout=120):
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "META_API_KEY": "", "ROADSCOUT_PROVIDER": "live",
                "ROADSCOUT_COST_LOG": "off"})
    for k, v in (env_over or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "build_demo_dataset.py"), *map(str, args)],
                          cwd=str(cwd or ROOT), env=env, capture_output=True, text=True, timeout=timeout)
