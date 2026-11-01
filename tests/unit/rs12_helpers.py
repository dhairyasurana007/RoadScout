"""Helpers for the task 12 (eval script v0) hidden tests: seeded databases and label files, no network."""
from __future__ import annotations

import hashlib
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

CLASSES = ["pothole", "debris in lane", "damaged guardrail", "fallen or damaged sign", "faded lane markings"]
HEADER = "row_type,segment_id,start,end,class,severity,hazard_id,note"


# ------------------------------------------------------------------ label files

def write_labels(dirpath, stem, rows, header=HEADER, newline="\n", encoding="utf-8"):
    """rows: list of CSV lines (strings) or tuples (written comma-joined)."""
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    lines = [header] if header is not None else []
    for r in rows:
        lines.append(r if isinstance(r, str) else ",".join(str(x) for x in r))
    p = dirpath / f"{stem}.csv"
    p.write_bytes((newline.join(lines) + newline).encode(encoding))
    return p


def seg(sid, start, end, note=""):
    return f"segment,{sid},{start},{end},,,,{note}"


def haz(sid, start, end, cls, sev="2", hid="", note=""):
    return f"hazard,{sid},{start},{end},{cls},{sev},{hid},{note}"


# ------------------------------------------------------------------ databases

class Db:
    def __init__(self, path):
        self.path = Path(path)
        from backend.db import init_db

        init_db(self.path)
        self._cid = 0
        self._hid = 0

    def conn(self):
        c = sqlite3.connect(str(self.path))
        c.row_factory = sqlite3.Row
        return c

    def drive(self, drive_id, filename=None, status="done"):
        con = self.conn()
        try:
            with con:
                con.execute(
                    "INSERT INTO drives (id, started_at, started_at_source, clock_offset_seconds, video_path, "
                    "gpx_path, status, duration_seconds, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (drive_id, "2026-10-20T12:00:00+00:00", "metadata", 0.0, f"{drive_id}/video.mov",
                     f"{drive_id}/track.gpx", status, 600.0, "2026-10-20T12:00:00+00:00"))
                if filename is not None:
                    con.execute("UPDATE drives SET video_filename = ? WHERE id = ?", (filename, drive_id))
        finally:
            con.close()
        return drive_id

    def hazard(self, cls, status="new", hazard_id=None):
        if hazard_id is None:
            self._hid += 1
            hazard_id = 5000 + self._hid
        con = self.conn()
        try:
            with con:
                con.execute(
                    "INSERT INTO hazards (id, hazard_class, lat, lon, direction, severity, confidence, sightings, "
                    "candidate_count, first_seen, last_seen, photo_path, crop_path, status, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (hazard_id, cls, 38.88, -77.3, 0.0, 2, 0.9, 1, 1, "2026-10-20T12:01:00+00:00",
                     "2026-10-20T12:01:00+00:00", None, None, status, "2026-10-20T12:00:00+00:00",
                     "2026-10-20T12:00:00+00:00"))
        finally:
            con.close()
        return hazard_id

    def candidate(self, drive_id, cls, t, accepted=1, hazard_id=None, verdict=True):
        """A candidate at video time ``t`` seconds. accepted: 1/0 (verdict row), verdict=False: no verdict row."""
        self._cid += 1
        cid = 7000 + self._cid
        con = self.conn()
        try:
            with con:
                con.execute(
                    "INSERT INTO candidates (id, drive_id, hazard_id, hazard_class, track_key, track_id, chunk, "
                    "first_timestamp, last_timestamp, frames_seen, max_mask_area, best_frame_index, best_timestamp, "
                    "best_box, best_frame_path, crop_path, lat, lon, heading, score, geotag_status, geotag_note) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (cid, drive_id, hazard_id, cls, f"0:{cid}", cid, 0, t - 1, t + 1, 5, 400.0, int(t * 3), float(t),
                     "[1,2,10,10]", f"{drive_id}/c{cid}.jpg", None, 38.88, -77.3, 0.0, 0.7, "ok", None))
                if verdict:
                    con.execute(
                        "INSERT INTO verdicts (candidate_id, drive_id, is_hazard, hazard_class, severity, lane, "
                        "confidence, reason, report_description, model, prompt_version, accepted, created_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (cid, drive_id, 1, cls, 2, "center", 0.9 if accepted else 0.3, "r", "d", "stub", "v1",
                         1 if accepted else 0, "2026-10-20T12:00:00+00:00"))
        finally:
            con.close()
        return cid

    def fingerprint(self):
        data = self.path.read_bytes()
        return hashlib.sha256(data).hexdigest(), self.path.stat().st_mtime_ns, self.path.stat().st_size


def make_db(tmp_path, name="eval.sqlite"):
    return Db(Path(tmp_path) / name)


# ------------------------------------------------------------------ report helpers

def evaluate(db, labels_dir, **kw):
    from backend.evaluation import evaluate as ev, load_labels

    labels = load_labels(labels_dir) if not hasattr(labels_dir, "drives") else labels_dir
    path = db.path if hasattr(db, "path") else db
    return ev(str(path), labels, **kw)


def metrics(report, cls):
    return report.per_class[cls]


def rate_tokens(line):
    """Numbers in a report line as fractions where they look like rates (0.5, 50%, 50.0%)."""
    out = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)(%?)", line):
        v = float(m.group(1))
        out.append(v / 100.0 if m.group(2) else v)
        if not m.group(2):
            out.append(v / 100.0)
    return out


def line_with(text, *needles):
    for ln in text.splitlines():
        low = ln.lower()
        if all(n.lower() in low for n in needles):
            return ln
    return None


# ------------------------------------------------------------------ CLI

def run_cli(args, cwd=None, env_over=None, timeout=120):
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "META_API_KEY": "", "ROADSCOUT_PROVIDER": "live",
                "ROADSCOUT_COST_LOG": "off"})
    env.pop("PYTHONIOENCODING", None)
    for k, v in (env_over or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "eval.py"), *map(str, args)],
                          cwd=str(cwd or ROOT), env=env, capture_output=True, text=True, timeout=timeout)
