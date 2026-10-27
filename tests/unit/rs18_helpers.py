"""Shared helpers and fixtures for the task 18 (prefilled agency report) hidden tests.

Interface assumed (tolerantly discovered; see the report to the main agent):

    backend/<module>.agency_for_point(lat, lon) -> 'VDOT' | 'MDOT SHA' | None      (pure, deterministic)
    GET   /api/hazards/{id}/report            -> JSON report for an approved (or reported) hazard
    GET   /api/hazards/{id}/report.txt (or ?format=txt, /report/download ...) -> text/plain attachment
    GET   /api/hazards/{id}/photo?download=1  -> blurred JPEG as an attachment
    PATCH /api/hazards/{id}  {"agency": "VDOT" | "MDOT SHA" | "none" | null | "auto"}  (or PUT /api/hazards/{id}/agency)
    PATCH /api/hazards/{id}  {"status": "reported"}  allowed for approved hazards
"""
from __future__ import annotations

import enum
import importlib
import inspect
import json
import math
import os
import pkgutil
import re
import shutil
import socket
import sqlite3
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

LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "::", "testserver"}
EARTH_R = 6371008.8
M_PER_DEG = EARTH_R * math.pi / 180.0
T0 = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
SEED_TS = "2026-10-21T00:00:00+00:00"
MISSING = object()

VDOT_HOST = "my.vdot.virginia.gov"
MDOT_URL = "mdotsha.my.salesforce-sites.com/customercare"

# Hazard point used for most tests (Fairfax City, Virginia) and a Maryland one (Rockville).
VA_PT = (38.85123, -77.30345)
MD_PT = (39.0840, -77.1530)
DC_PT = (38.90, -77.03)
PHILLY_PT = (39.95, -75.17)

# class (normalized with spaces) -> MDOT SHA Topic text
MDOT_TOPICS = {
    "pothole": "Potholes / Holes / Sinkholes",
    "debris in lane": "Debris, Graffiti, & Litter",
    "damaged guardrail": "Guardrail/Roadway Safety Devices",
    "fallen or damaged sign": "Sign / Digital Message Sign",
    "faded lane markings": "Line Striping / Roadway & Cross Walk Markings",
}


def known_classes():
    cfg = json.loads((ROOT / "config" / "hazard_prompts.json").read_text(encoding="utf-8"))
    return list(cfg["classes"].keys())


def cls_norm(c):
    return str(c).replace("_", " ").strip().lower()


def topic_for(cls):
    return MDOT_TOPICS[cls_norm(cls)]


def nospace(s):
    return re.sub(r"\s+", "", str(s)).lower()


def offset(lat, lon, north_m=0.0, east_m=0.0):
    return (lat + north_m / M_PER_DEG, lon + east_m / (M_PER_DEG * math.cos(math.radians(lat))))


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def unwrap(body):
    if isinstance(body, dict) and "id" not in body:
        for k in ("hazard", "data", "item", "result"):
            if isinstance(body.get(k), dict):
                return body[k]
    return body


# ------------------------------------------------------------------ agency normalisation

def norm_agency(v):
    """'VDOT' | 'MDOT SHA' | None (no agency) | '?...' (unrecognised)."""
    if v is None:
        return None
    if isinstance(v, enum.Enum):
        for cand in (v.value, v.name):
            r = norm_agency(cand)
            if r is None or not str(r).startswith("?"):
                return r
        return "?" + repr(v)
    if isinstance(v, dict):
        for k in ("code", "id", "key", "name", "agency", "value", "label"):
            if k in v:
                r = norm_agency(v[k])
                if r is None or not str(r).startswith("?"):
                    return r
        return "?" + repr(v)
    if isinstance(v, bool):
        return "?" + repr(v)
    s = re.sub(r"[^a-z]", "", str(v).lower())
    if s == "vdot":
        return "VDOT"
    if s in ("mdotsha", "mdot", "sha", "mdotstatehighwayadministration"):
        return "MDOT SHA"
    if s in ("", "none", "noagency", "na", "null", "nil", "unknown", "other", "outofjurisdiction", "nojurisdiction"):
        return None
    return "?" + str(v)


# ------------------------------------------------------------------ JSON walking helpers

def walk(obj, path=()):
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, path + (str(i),))


def find_key(body, *names, default=MISSING):
    """Shallowest value whose key (case-insensitive) is one of names."""
    names = {n.lower() for n in names}
    best = None
    for path, v in walk(body):
        if path and path[-1].lower() in names:
            if best is None or len(path) < len(best[0]):
                best = (path, v)
    return default if best is None else best[1]


def all_strings(body, keys=True):
    out = []
    for path, v in walk(body):
        if isinstance(v, str):
            out.append(v)
        if keys and path and not path[-1].isdigit():
            out.append(path[-1])
    return out


def dump(body):
    return json.dumps(body, ensure_ascii=False, default=str)


def rep_agency(body):
    v = find_key(body, "agency", "agency_code", "agency_name")
    assert v is not MISSING, "report has no 'agency' key: %s" % sorted(body)
    return norm_agency(v)


def rep_user_chosen(body):
    """True / False when the report tells whether the agency was chosen by the user, None when it cannot be found."""
    for path, v in walk(body):
        if not path:
            continue
        k = path[-1].lower()
        if "agency" not in k and not any(w in k for w in ("source", "chosen", "override", "default", "auto", "manual")):
            continue
        if not any(w in k for w in ("source", "chosen", "override", "default", "auto", "manual", "origin", "mode", "selection")):
            continue
        if isinstance(v, bool):
            if any(w in k for w in ("user", "chosen", "override", "manual")):
                return v
            if any(w in k for w in ("auto", "default")):
                return not v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("user", "manual", "override", "overridden", "user-chosen", "user_chosen", "chosen", "user chosen",
                     "user_override", "custom", "selected"):
                return True
            if s in ("auto", "automatic", "default", "location", "gps", "derived", "auto-detected", "inferred",
                     "auto_detected", "auto-default"):
                return False
    return None


def category_blob(body):
    v = find_key(body, "category", "agency_category", "topic")
    assert v is not MISSING, "report has no 'category' key: %s" % sorted(body)
    if isinstance(v, dict):
        return " ".join(str(x) for x in v.values() if x is not None)
    return str(v)


def rep_text(body):
    v = find_key(body, "text", "plain_text", "report_text", "copy_text", "text_report", "clipboard", "plain")
    assert isinstance(v, str) and v.strip(), "report has no plain-text version (key 'text'): %s" % sorted(body)
    return v


def rep_description(body):
    v = find_key(body, "description")
    assert isinstance(v, str) and v.strip(), "report has no 'description' string: %s" % sorted(body)
    return v


def rep_photo_url(body):
    v = find_key(body, "photo_url", "image_url", "photo")
    return None if v is MISSING else v


def map_links(body):
    return [s for s in all_strings(body, keys=False)
            if re.search(r"https?://\S*(openstreetmap\.org|google\.[a-z.]+/maps|maps\.google|maps\.apple|bing\.com/maps)", s)]


def form_urls(body):
    return [s for s in all_strings(body, keys=False) if VDOT_HOST in s or MDOT_URL in s]


COMPASS_WORDS = {"northeast": "NE", "northwest": "NW", "southeast": "SE", "southwest": "SW", "north": "N",
                 "south": "S", "east": "E", "west": "W"}


def compass_tokens(s):
    toks = set()
    low = str(s).lower()
    for w, a in COMPASS_WORDS.items():
        if re.search(r"\b%s(bound)?\b" % w, low):
            toks.add(a)
    for m in re.findall(r"\b(NE|NW|SE|SW|N|S|E|W)\b", str(s)):
        toks.add(m)
    # "southeast" also contains south/east words: keep only the compound when present
    for comp in ("NE", "NW", "SE", "SW"):
        if comp in toks:
            toks -= set(comp)
    return toks


def expected_compass(deg):
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return names[int(((deg % 360) + 22.5) // 45) % 8]


def rep_compass(body):
    """String compass value found in the report (key containing 'compass', or a non-numeric direction/heading string)."""
    for path, v in walk(body):
        if path and isinstance(v, str) and "compass" in path[-1].lower():
            return v
    for path, v in walk(body):
        if path and isinstance(v, str) and any(w in path[-1].lower() for w in ("direction", "heading", "bearing")):
            if compass_tokens(v):
                return v
    return None


def has_submitted_note(body):
    """A clear 'nothing is submitted' marker: a falsy submit-ish flag or a text that says so."""
    for path, v in walk(body):
        if path and "submit" in path[-1].lower():
            if v is False or v is None:
                return True
            if isinstance(v, str) and re.search(r"\b(not|no|never|nothing|manual|yourself|you)\b", v.lower()):
                return True
    for s in all_strings(body, keys=False):
        low = s.lower()
        if "submit" in low and re.search(r"\b(not|no|never|nothing|manual|manually|yourself)\b", low):
            return True
    return False


# ------------------------------------------------------------------ environment

class Env:
    def __init__(self, mp, tmp_path, client):
        self.mp = mp
        self.tmp = Path(tmp_path)
        self.db = self.tmp / "db" / "rs18.sqlite"
        self.storage = self.tmp / "storage"
        self.client = client
        self._seed = 500
        self._cn = 0

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

    def row(self, hid):
        r = self.q("SELECT * FROM hazards WHERE id = ?", (hid,))
        return r[0] if r else None

    def rows(self):
        return self.q("SELECT * FROM hazards ORDER BY id")

    def columns(self):
        return [r["name"] for r in self.q("PRAGMA table_info(hazards)")]

    # ---- synthetic data
    def jpeg(self, path):
        import numpy as np
        from PIL import Image

        self._seed += 1
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.RandomState(self._seed)
        arr = rng.randint(0, 256, size=(32, 48, 3)).astype("uint8")
        Image.fromarray(arr, "RGB").save(str(path), "JPEG", quality=90)
        return Path(path)

    def add_drive(self, drive_id, started=T0):
        self.x("INSERT OR REPLACE INTO drives (id, started_at, started_at_source, clock_offset_seconds, video_path, "
               "gpx_path, status, created_at) VALUES (?, ?, 'test', 0, ?, ?, 'done', ?)",
               (drive_id, iso(started), "v/%s.mp4" % drive_id, "g/%s.gpx" % drive_id, iso(started)))

    def add_hazard(self, *, hazard_class="pothole", lat=VA_PT[0], lon=VA_PT[1], direction=135.0, severity=3,
                   confidence=0.8, sightings=4, candidate_count=1, first_seen="2026-10-18T09:15:00+00:00",
                   last_seen="2026-10-22T17:40:00+00:00", photo_path=None, crop_path=None, status="approved"):
        return self.x(
            "INSERT INTO hazards (hazard_class, lat, lon, direction, severity, confidence, sightings, "
            "candidate_count, first_seen, last_seen, photo_path, crop_path, status, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (hazard_class, lat, lon, direction, severity, confidence, sightings, candidate_count, first_seen,
             last_seen, None if photo_path is None else str(photo_path),
             None if crop_path is None else str(crop_path), status, SEED_TS, SEED_TS))

    def add_candidate_files(self, hid, drive, blur, orig):
        """A candidates row pointing at an ORIGINAL and a BLURRED frame, linked to hazard hid."""
        self._cn += 1
        cols = {
            "drive_id": drive, "hazard_id": hid, "hazard_class": "pothole", "track_key": "0:%d" % self._cn,
            "track_id": self._cn, "chunk": 0, "first_timestamp": 1.0, "last_timestamp": 2.0, "frames_seen": 5,
            "max_mask_area": 100.0, "best_frame_index": 3, "best_timestamp": 1.5, "best_box": "[0,0,10,10]",
            "best_frame_path": str(orig), "crop_path": None, "lat": 38.85, "lon": -77.30, "heading": 90.0,
            "score": 1.0, "geotag_status": "ok", "geotag_note": None,
        }
        extra = {"blurred_frame_path": str(blur), "blurred_crop_path": None, "blur_status": "ok"}
        have = {r["name"] for r in self.q("PRAGMA table_info(candidates)")}
        cols.update({k: v for k, v in extra.items() if k in have})
        names = list(cols)
        return self.x("INSERT INTO candidates (%s) VALUES (%s)" % (", ".join(names), ", ".join("?" * len(names))),
                      [cols[n] for n in names])

    def seed(self, name="a", status="approved", with_photo=True, **kw):
        """Hazard with a blurred JPEG (photo_path) and, next to it, an unblurred original that must never be served.
        Returns (id, info) where info has 'blur' and 'orig' paths."""
        info = {"blur": None, "orig": None}
        photo = None
        if with_photo:
            d = self.storage / "drv18" / "candidates"
            info["orig"] = self.jpeg(d / ("zqorig_%s_best.jpg" % name))
            info["blur"] = self.jpeg(d / ("zqblur_%s_best_blurred.jpg" % name))
            photo = info["blur"]
        hid = self.add_hazard(status=status, photo_path=photo, **kw)
        if with_photo:
            self.add_drive("drv18")
            self.add_candidate_files(hid, "drv18", info["blur"], info["orig"])
        return hid, info

    # ---- routes
    def paths(self):
        from backend.main import app

        return app.openapi().get("paths", {})

    def spec(self):
        from backend.main import app

        return app.openapi()

    def sub(self, path, hid):
        return re.sub(r"\{[^}/]+\}", str(hid), path)

    def generic_route(self):
        paths = self.paths()
        pat = re.compile(r"^/api/hazards/\{[^}/]+\}/?$")
        for m in ("patch", "put", "post"):
            for p, ops in paths.items():
                if pat.match(p) and m in ops:
                    return m, p
        return None

    def body_props(self, route):
        spec = self.spec()
        op = spec["paths"][route[1]][route[0]]
        schema = (((op.get("requestBody") or {}).get("content") or {}).get("application/json") or {}).get("schema") or {}

        def resolve(s):
            for _ in range(5):
                if "$ref" in s:
                    node = spec
                    for part in s["$ref"].lstrip("#/").split("/"):
                        node = node[part]
                    s = node
                elif s.get("allOf"):
                    s = s["allOf"][0]
                elif s.get("anyOf"):
                    s = next((x for x in s["anyOf"] if x.get("type") != "null"), s["anyOf"][0])
                else:
                    break
            return s

        try:
            return set((resolve(schema).get("properties") or {}).keys())
        except Exception:
            return set()

    def report_path(self):
        paths = self.paths()
        for p, ops in paths.items():
            if "get" in ops and re.match(r"^/api/hazards/\{[^}/]+\}/report/?$", p):
                return p
        for p, ops in paths.items():
            if "get" in ops and "report" in p and p.count("{") == 1 and not any(
                    s in p for s in (".txt", "download", "text", "photo", "txt")):
                return p
        return None

    def get_report(self, hid, **params):
        rp = self.report_path()
        assert rp is not None, "no GET report route in the OpenAPI schema (expected GET /api/hazards/{id}/report)"
        return self.client.get(self.sub(rp, hid), params=params or None)

    def report(self, hid, **params):
        r = self.get_report(hid, **params)
        assert r.status_code == 200, "GET report for hazard %s -> %s %s" % (hid, r.status_code, r.text[:300])
        body = r.json()
        assert isinstance(body, dict)
        return body

    def photo_path(self):
        for p, ops in self.paths().items():
            if "get" in ops and re.match(r"^/api/hazards/\{[^}/]+\}/photo/?$", p):
                return p
        return None

    def agency_route(self):
        g = self.generic_route()
        if g and "agency" in self.body_props(g):
            return g
        paths = self.paths()
        pat = re.compile(r"^/api/hazards/\{[^}/]+\}/agency/?$")
        for m in ("patch", "put", "post"):
            for p, ops in paths.items():
                if pat.match(p) and m in ops:
                    return m, p
        for m in ("patch", "put", "post"):
            for p, ops in paths.items():
                if "agency" in p and p.count("{") == 1 and m in ops:
                    return m, p
        if g:
            return g  # hope the generic PATCH accepts it (the body model may be permissive)
        return None

    def set_agency(self, hid, value, **extra):
        r = self.agency_route()
        assert r is not None, "no route to change the agency (expected PATCH /api/hazards/{id} with 'agency')"
        body = {"agency": value}
        body.update(extra)
        return getattr(self.client, r[0])(self.sub(r[1], hid), json=body)

    def ok_agency(self, hid, value):
        r = self.set_agency(hid, value)
        assert r.status_code == 200, "set agency %r -> %s %s" % (value, r.status_code, r.text[:300])
        return r

    def set_status(self, hid, status):
        g = self.generic_route()
        assert g is not None, "no generic update route"
        return getattr(self.client, g[0])(self.sub(g[1], hid), json={"status": status})

    def hazards(self):
        r = self.client.get("/api/hazards")
        assert r.status_code == 200, r.text[:300]
        return r.json()

    # ---- dedup
    def add_cand(self, drive_id, lat, lon, heading, *, cls="pothole", score=0.5, conf=0.8, severity=2,
                 frame_path=None):
        self._cn += 1
        n = self._cn
        frame_path = frame_path or str(self.db.parent / ("frame18_%d.jpg" % n))
        con = sqlite3.connect(str(self.db))
        try:
            con.execute("PRAGMA foreign_keys = ON")
            cur = con.execute(
                "INSERT INTO candidates (drive_id, hazard_id, hazard_class, track_key, track_id, chunk, "
                "first_timestamp, last_timestamp, frames_seen, max_mask_area, best_frame_index, best_timestamp, "
                "best_box, best_frame_path, crop_path, blurred_frame_path, blurred_crop_path, blur_status, "
                "lat, lon, heading, score, geotag_status, geotag_note) "
                "VALUES (?, NULL, ?, ?, ?, 0, 9.0, 11.0, 5, 100.0, 30, 10.0, '[0, 0, 10, 10]', ?, NULL, ?, NULL, "
                "'ok', ?, ?, ?, ?, 'ok', NULL)",
                (drive_id, cls, "0:%d" % n, n, frame_path, frame_path, lat, lon, heading, score))
            cid = cur.lastrowid
            con.execute(
                "INSERT INTO verdicts (candidate_id, drive_id, is_hazard, hazard_class, severity, lane, confidence, "
                "reason, report_description, model, prompt_version, accepted, created_at) "
                "VALUES (?, ?, 1, ?, ?, NULL, ?, 'r', 'd', 'fake', 'v1', 1, ?)",
                (cid, drive_id, cls, severity, conf, iso(T0)))
            con.commit()
            return cid
        finally:
            con.close()

    def dedup(self, drive_id):
        mod = importlib.import_module("backend.dedup")
        names = ["deduplicate_drive", "dedupe_drive", "dedup_drive", "deduplicate", "dedupe", "run_dedup",
                 "run_deduplication", "merge_drive", "deduplicate_hazards"]
        f = next((getattr(mod, n) for n in names if callable(getattr(mod, n, None))), None)
        assert f is not None, "no dedup function"
        sig = inspect.signature(f)
        p = next((n for n in ("db_path", "db", "database", "database_path") if n in sig.parameters), "db_path")
        res = f(drive_id, **{p: str(self.db)})
        if inspect.isawaitable(res):
            import asyncio

            asyncio.run(res)


def make_env(mp, tmp_path):
    import backend.config  # noqa: F401  (loads a local .env before the key is cleared)

    mp.setenv("ROADSCOUT_DB", str(tmp_path / "db" / "rs18.sqlite"))
    mp.setenv("ROADSCOUT_STORAGE_DIR", str(tmp_path / "storage"))
    mp.setenv("ROADSCOUT_PROVIDER", "fake")
    mp.setenv("ROADSCOUT_COST_LOG", "off")
    for k in ("META_API_KEY", "ROADSCOUT_FAKE_SCENARIO", "ROADSCOUT_DEDUP_DISTANCE_M",
              "ROADSCOUT_DEDUP_HEADING_TOLERANCE_DEG"):
        mp.delenv(k, raising=False)
    real_connect = socket.socket.connect

    def _guard(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if host is not None and str(host) not in LOOPBACK:
            raise AssertionError("network access attempted during a task 18 test: %r" % (address,))
        return real_connect(self, address)

    mp.setattr(socket.socket, "connect", _guard)
    from fastapi.testclient import TestClient

    from backend.db import init_db
    from backend.main import app

    init_db()
    return TestClient(app)


@pytest.fixture()
def rs18_env(monkeypatch, tmp_path):
    client = make_env(monkeypatch, tmp_path)
    with client:
        yield Env(monkeypatch, tmp_path, client)


# ------------------------------------------------------------------ agency function discovery

_PROBES = ((38.85, -77.30), (39.08, -77.15))
_NAME_PREF = ["agency_for_point", "agency_for_location", "agency_for_coordinates", "agency_for_latlon",
              "agency_for_lat_lon", "default_agency", "default_agency_for", "agency_from_location",
              "agency_from_point", "agency_at", "agency_for", "lookup_agency", "get_agency", "find_agency",
              "jurisdiction_for_point", "jurisdiction_for"]


def _try(f, lat, lon):
    try:
        res = f(lat, lon)
        if inspect.isawaitable(res):
            return "?await"
        return norm_agency(res)
    except Exception:
        return "?error"


def discover_agency_function():
    import backend

    found = []
    for mi in pkgutil.iter_modules(backend.__path__):
        if not re.search(r"agenc|jurisdic|report|region|boundar|state|geo|locat", mi.name):
            continue
        try:
            mod = importlib.import_module("backend." + mi.name)
        except Exception:
            continue
        for name, f in inspect.getmembers(mod, callable):
            if name.startswith("_") or not inspect.isfunction(f) or f.__module__ != mod.__name__:
                continue
            if not re.search(r"agency|jurisdiction", name):
                continue
            try:
                params = [p for p in inspect.signature(f).parameters.values()
                          if p.default is inspect.Parameter.empty
                          and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            except (TypeError, ValueError):
                continue
            if len(params) != 2:
                continue
            if _try(f, *_PROBES[0]) == "VDOT" and _try(f, *_PROBES[1]) == "MDOT SHA":
                found.append((_NAME_PREF.index(name) if name in _NAME_PREF else 99, name, f))
    found.sort(key=lambda t: (t[0], t[1]))
    return found[0][2] if found else None


@pytest.fixture(scope="session")
def agency_fn():
    f = discover_agency_function()
    if f is None:
        pytest.fail("no pure agency-from-location function found in backend/ (expected e.g. "
                    "backend/agency.py agency_for_point(lat, lon) -> 'VDOT' | 'MDOT SHA' | None)")

    def call(lat, lon):
        return norm_agency(f(lat, lon))

    return call
