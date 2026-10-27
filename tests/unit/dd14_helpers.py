"""Shared helpers for the task 14 (deduplicate hazards) hidden tests.

Interface assumed (see the report to the main agent):

    backend/dedup.py
        deduplicate_drive(drive_id, *, db_path=None, distance_m=None, heading_tolerance_deg=None)
            Recomputes the hazards that the drive's accepted candidates belong to. Return value is not
            inspected. distance_m / heading_tolerance_deg override the env vars below; None means env, then
            the defaults (25 m, 45 degrees). Env vars are read at CALL time.
        load_hazards(*, db_path=None) -> list[dict]   (optional helper, used only if present)
    env ROADSCOUT_DEDUP_DISTANCE_M              default 25
    env ROADSCOUT_DEDUP_HEADING_TOLERANCE_DEG   default 45

    SQLite table hazards: id INTEGER PK, hazard_class TEXT, lat REAL, lon REAL, direction REAL (degrees 0..360),
        severity INTEGER, confidence REAL, sightings INTEGER, first_seen TEXT, last_seen TEXT,
        photo_path TEXT, status TEXT DEFAULT 'new'   (a column named "class" is accepted instead of hazard_class)
    candidates.hazard_id -> hazards.id for every merged candidate, NULL otherwise.
"""
from __future__ import annotations

import importlib
import inspect
import math
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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

EARTH_R = 6371008.8
M_PER_DEG = EARTH_R * math.pi / 180.0

# Virginia (Richmond) and Maryland (Baltimore) reference points
VA = (37.5407, -77.4360)
MD = (39.2904, -76.6122)
NOVA = (38.8800, -77.3000)

T_A = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)
T_B = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)
T_C = datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)

# smallest valid JPEG-ish payload is enough: dedup must not decode images
TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912130f14"
    "1d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432ffc0000b08000100010101"
    "1100ffc400140001000000000000000000000000000000ffc40014100100000000000000000000000000000000ffda0008"
    "010100003f00d2cf20ffd9"
)

FUNC_NAMES = ["deduplicate_drive", "dedupe_drive", "dedup_drive", "deduplicate", "dedupe", "run_dedup",
              "run_deduplication", "merge_drive", "deduplicate_hazards"]
DIST_PARAMS = ["distance_m", "max_distance_m", "radius_m", "merge_distance_m", "dedup_distance_m",
               "distance_threshold_m", "threshold_m", "distance"]
HEAD_PARAMS = ["heading_tolerance_deg", "heading_tolerance", "max_heading_diff_deg", "heading_tol_deg",
               "direction_tolerance_deg", "heading_threshold_deg", "angle_tolerance_deg", "max_heading_diff"]
DBP = ["db_path", "db", "database", "database_path"]

ENV_DIST = "ROADSCOUT_DEDUP_DISTANCE_M"
ENV_HEAD = "ROADSCOUT_DEDUP_HEADING_TOLERANCE_DEG"


# ------------------------------------------------------------------ geometry

def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def offset(lat, lon, north_m=0.0, east_m=0.0):
    """(lat, lon) moved by metres north / east (local tangent plane, exact enough for < 1 km)."""
    dlat = north_m / M_PER_DEG
    dlon = east_m / (M_PER_DEG * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def ang_diff(a, b):
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


# ------------------------------------------------------------------ db

def make_db(tmp_path, monkeypatch=None):
    from backend.db import init_db

    db = tmp_path / "db" / "dd14.sqlite"
    if monkeypatch is not None:
        monkeypatch.setenv("ROADSCOUT_DB", str(db))
        monkeypatch.delenv(ENV_DIST, raising=False)
        monkeypatch.delenv(ENV_HEAD, raising=False)
        monkeypatch.setenv("ROADSCOUT_COST_LOG", "off")
    init_db(db)
    return db


def rows(db, sql, params=()):
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params)]
    finally:
        con.close()


def execute(db, sql, params=()):
    con = sqlite3.connect(str(db))
    try:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute(sql, params)
        con.commit()
    finally:
        con.close()


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def add_drive(db, drive_id, started=T_A, status="done"):
    execute(db,
            "INSERT OR REPLACE INTO drives (id, started_at, started_at_source, clock_offset_seconds, video_path, "
            "gpx_path, status, created_at) VALUES (?, ?, 'test', 0, ?, ?, ?, ?)",
            (drive_id, iso(started), "v/%s.mp4" % drive_id, "g/%s.gpx" % drive_id, status, iso(started)))


_NEXT = {"n": 0}


def add_cand(db, drive_id, lat, lon, heading, *, cls="pothole", score=0.5, conf=0.8, severity=2,
             is_hazard=True, accepted=True, geotag="ok", best_ts=10.0, frame_path=None, crop_path=None,
             with_verdict=True, verdict_cls=None):
    """Insert one candidate (+ verdict) and return the candidate id."""
    _NEXT["n"] += 1
    n = _NEXT["n"]
    frame_path = frame_path if frame_path is not None else str(Path(db).parent / ("frame_%d.jpg" % n))
    con = sqlite3.connect(str(db))
    try:
        con.execute("PRAGMA foreign_keys = ON")
        cur = con.execute(
            "INSERT INTO candidates (drive_id, hazard_id, hazard_class, track_key, track_id, chunk, first_timestamp, "
            "last_timestamp, frames_seen, max_mask_area, best_frame_index, best_timestamp, best_box, "
            "best_frame_path, crop_path, blurred_frame_path, blurred_crop_path, blur_status, "
            "lat, lon, heading, score, geotag_status, geotag_note) "
            "VALUES (?, NULL, ?, ?, ?, 0, ?, ?, 5, 100.0, 30, ?, '[0, 0, 10, 10]', ?, ?, ?, ?, 'ok', ?, ?, ?, ?, ?, NULL)",
            (drive_id, cls, "0:%d" % n, n, best_ts - 1, best_ts + 1, best_ts, frame_path, crop_path,
             # task 15: dedup takes the hazard photo from the BLURRED paths; the test rows use the same paths
             frame_path, crop_path,
             lat if geotag == "ok" else None, lon if geotag == "ok" else None,
             heading if geotag == "ok" else None, score, geotag))
        cid = cur.lastrowid
        if with_verdict:
            con.execute(
                "INSERT INTO verdicts (candidate_id, drive_id, is_hazard, hazard_class, severity, lane, confidence, "
                "reason, report_description, model, prompt_version, accepted, created_at) "
                "VALUES (?, ?, ?, ?, ?, NULL, ?, 'r', 'd', 'fake', 'v1', ?, ?)",
                (cid, drive_id, 1 if is_hazard else 0, verdict_cls or cls, severity if is_hazard else 0, conf,
                 1 if accepted else 0, iso(T_A)))
        con.commit()
        return cid
    finally:
        con.close()


def drop_drive_candidates(db, drive_id):
    """What re-running aggregation does: delete the drive's candidates (verdicts cascade)."""
    execute(db, "DELETE FROM verdicts WHERE drive_id = ?", (drive_id,))
    execute(db, "DELETE FROM candidates WHERE drive_id = ?", (drive_id,))


# ------------------------------------------------------------------ calling dedup

def find_dedup():
    mod = importlib.import_module("backend.dedup")
    for n in FUNC_NAMES:
        f = getattr(mod, n, None)
        if callable(f):
            return f
    pytest.fail("backend.dedup has none of %s" % FUNC_NAMES)


def _pick(sig, names):
    for n in names:
        if n in sig.parameters:
            return n
    return None


def dedup(db, drive_id, *, distance_m=None, heading_tolerance_deg=None):
    f = find_dedup()
    sig = inspect.signature(f)
    kw = {}
    p = _pick(sig, DBP)
    if p is None and not any(x.kind == x.VAR_KEYWORD for x in sig.parameters.values()):
        pytest.fail("dedup function has no db_path parameter; signature %s" % (sig,))
    kw[p or "db_path"] = str(db)
    if distance_m is not None:
        p = _pick(sig, DIST_PARAMS)
        if p is None:
            pytest.fail("dedup function has no distance parameter; signature %s" % (sig,))
        kw[p] = distance_m
    if heading_tolerance_deg is not None:
        p = _pick(sig, HEAD_PARAMS)
        if p is None:
            pytest.fail("dedup function has no heading tolerance parameter; signature %s" % (sig,))
        kw[p] = heading_tolerance_deg
    res = f(drive_id, **kw)
    if inspect.isawaitable(res):
        import asyncio

        res = asyncio.run(res)
    return res


def dedup_all(db, drive_ids, **kw):
    for d in drive_ids:
        dedup(db, d, **kw)


# ------------------------------------------------------------------ reading results

def class_col(db):
    cols = {r["name"] for r in rows(db, "PRAGMA table_info(hazards)")}
    for c in ("hazard_class", "class"):
        if c in cols:
            return c
    pytest.fail("hazards table has neither hazard_class nor class column; columns: %s" % sorted(cols))


def hazards(db):
    cc = class_col(db)
    out = rows(db, "SELECT * FROM hazards ORDER BY id")
    for h in out:
        h["cls"] = h[cc]
    return out


def cand_rows(db, drive_id=None):
    if drive_id is None:
        return rows(db, "SELECT * FROM candidates ORDER BY id")
    return rows(db, "SELECT * FROM candidates WHERE drive_id = ? ORDER BY id", (drive_id,))


def hazard_of(db, cid):
    r = rows(db, "SELECT hazard_id FROM candidates WHERE id = ?", (cid,))
    return r[0]["hazard_id"] if r else None


def partition(db, ids):
    """Set of frozensets of candidate ids that share a hazard (unlinked candidates are singletons-with-None)."""
    groups = {}
    for cid in ids:
        groups.setdefault(hazard_of(db, cid), set()).add(cid)
    out = set()
    for hid, members in groups.items():
        if hid is None:
            out.update(frozenset([m]) for m in members)  # treat unlinked as separate
        else:
            out.add(frozenset(members))
    return out


def same_hazard(db, a, b):
    ha, hb = hazard_of(db, a), hazard_of(db, b)
    return ha is not None and ha == hb


def parse_time(value):
    """Parse an ISO-ish string (or number of epoch seconds) into an aware datetime."""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    s = str(value).strip().replace(" ", "T")
    s = re.sub(r"Z$", "+00:00", s)
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def circ_mean_ok(direction, headings, tol=46.0):
    return all(ang_diff(direction, h) <= tol for h in headings)
