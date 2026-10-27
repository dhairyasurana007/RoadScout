"""Task 14: the pipeline runs deduplication after verification (stage 'deduplicating')."""
import importlib
import inspect
import math
import shutil
import socket
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dd14_helpers import FUNC_NAMES, ROOT, class_col

FFMPEG = shutil.which("ffmpeg")
T0 = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
M_PER_DEG = 6371008.8 * math.pi / 180.0
LAT0, LON0 = 38.8800, -77.3000
LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "::"}

_CLIP = {}


def clip_bytes():
    if "b" not in _CLIP:
        if not FFMPEG:
            pytest.skip("ffmpeg not on PATH")
        d = Path(tempfile.mkdtemp(prefix="dd14media"))
        out = d / "clip.mp4"
        p = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10",
             "-t", "24", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-metadata",
             "creation_time=" + T0.strftime("%Y-%m-%dT%H:%M:%SZ"), str(out)], capture_output=True, timeout=240)
        assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
        _CLIP["b"] = out.read_bytes()
    return _CLIP["b"]


def gpx_bytes(seconds=60, speed=12.0):
    pts = []
    for s in range(seconds + 1):
        t = (T0 + timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%SZ")
        pts.append('<trkpt lat="%.8f" lon="%.8f"><ele>100</ele><time>%s</time></trkpt>'
                   % (LAT0 + s * speed / M_PER_DEG, LON0, t))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="dd14" '
            'xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>' + "".join(pts)
            + "</trkseg></trk></gpx>\n").encode("utf-8")


def q(db, sql, params=()):
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params)]
    finally:
        con.close()


@pytest.fixture
def env(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    if not FFMPEG:
        pytest.skip("ffmpeg not on PATH")
    import backend.config  # noqa: F401

    db = tmp_path / "db" / "dd14p.sqlite"
    monkeypatch.setenv("ROADSCOUT_DB", str(db))
    monkeypatch.setenv("ROADSCOUT_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("ROADSCOUT_PROVIDER", "fake")
    monkeypatch.setenv("ROADSCOUT_PREPROCESS_FPS", "3")
    monkeypatch.setenv("ROADSCOUT_CHUNK_SECONDS", "4")
    monkeypatch.setenv("ROADSCOUT_COST_LOG", "off")
    for k in ("META_API_KEY", "ROADSCOUT_FAKE_SCENARIO", "ROADSCOUT_MIN_TRACK_FRAMES",
              "ROADSCOUT_VERIFY_MIN_CONFIDENCE", "ROADSCOUT_DEDUP_DISTANCE_M",
              "ROADSCOUT_DEDUP_HEADING_TOLERANCE_DEG"):
        monkeypatch.delenv(k, raising=False)
    real_connect = socket.socket.connect

    def guard(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if host is not None and str(host) not in LOOPBACK:
            raise AssertionError("network access attempted: %r" % (address,))
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guard)
    from backend.db import init_db

    init_db()
    return db


def upload(client):
    r = client.post("/api/drives", files={"video": ("clip.mp4", clip_bytes(), "video/mp4"),
                                          "gpx": ("t.gpx", gpx_bytes(), "application/gpx+xml")})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def run(did, db, provider=None):
    from backend.pipeline import run_pipeline
    from backend.providers import FakeProvider

    return run_pipeline(did, provider=provider or FakeProvider("default"), db_path=str(db))


def find_dedup_name():
    mod = importlib.import_module("backend.dedup")
    for n in FUNC_NAMES:
        if callable(getattr(mod, n, None)):
            return n
    pytest.fail("no dedup function in backend.dedup")


# ------------------------------------------------------------------ tests

def test_stage_list_has_deduplicating_after_verifying():
    import backend.pipeline as p

    stages = list(getattr(p, "STAGES", ()))
    assert "verifying" in stages
    assert "deduplicating" in stages, "STAGES = %r" % (stages,)
    assert stages.index("deduplicating") == stages.index("verifying") + 1
    assert stages[-1] == "deduplicating"


def test_processed_drive_ends_done_with_hazards(env):
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        did = upload(client)
        res = run(did, env)
        assert res.status == "done", getattr(res, "error", res)
        drive = client.get("/api/drives/%s" % did).json()
        assert drive["status"] == "done"
        assert drive.get("stage") == "done"
    cands = q(env, "SELECT c.*, v.accepted, v.is_hazard FROM candidates c LEFT JOIN verdicts v "
                   "ON v.candidate_id = c.id WHERE c.drive_id = ?", (did,))
    eligible = [c for c in cands if c["accepted"] == 1 and c["is_hazard"] == 1 and c["geotag_status"] == "ok"
                and c["lat"] is not None]
    if not eligible:
        pytest.skip("the fake provider produced no accepted, geotagged candidate; cannot check hazards")
    hs = q(env, "SELECT * FROM hazards")
    assert hs, "no hazards after a successful run with %d eligible candidates" % len(eligible)
    ids = {h["id"] for h in hs}
    for c in cands:
        if c in eligible:
            assert c["hazard_id"] in ids, "eligible candidate %s not linked" % c["id"]
        else:
            assert c["hazard_id"] is None, "ineligible candidate %s was linked" % c["id"]
    assert all(h["status"] == "new" for h in hs)
    assert all(h["sightings"] >= 1 for h in hs)
    cc = class_col(env)
    assert {h[cc] for h in hs} <= {c["hazard_class"] for c in eligible}


def test_rerunning_the_pipeline_does_not_duplicate_hazards(env):
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        did = upload(client)
        assert run(did, env).status == "done"
        first = q(env, "SELECT * FROM hazards ORDER BY lat, lon, id")
        if not first:
            pytest.skip("no hazards produced by the fake provider")
        assert run(did, env).status == "done"
        second = q(env, "SELECT * FROM hazards ORDER BY lat, lon, id")
    assert len(second) == len(first)
    assert [h["sightings"] for h in second] == [h["sightings"] for h in first]
    for h in second:
        n = q(env, "SELECT COUNT(*) AS n FROM candidates WHERE hazard_id = ?", (h["id"],))[0]["n"]
        assert n >= 1, "orphan hazard %s" % h["id"]


def test_two_runs_of_two_identical_drives_merge(env):
    """The same synthetic clip + GPX uploaded twice is two drives seeing the same hazards: they merge."""
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        d1 = upload(client)
        assert run(d1, env).status == "done"
        n1 = len(q(env, "SELECT id FROM hazards"))
        if n1 == 0:
            pytest.skip("no hazards produced by the fake provider")
        d2 = upload(client)
        assert run(d2, env).status == "done"
    hs = q(env, "SELECT * FROM hazards")
    assert len(hs) == n1, "second identical drive created new hazards instead of merging"
    assert any(h["sightings"] >= 2 for h in hs)


def test_dedup_runs_after_verification_in_stage_deduplicating(env, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.main import app

    name = find_dedup_name()
    seen = []
    import backend.dedup as dd
    import backend.pipeline as pl

    original = getattr(dd, name)

    def spy(drive_id, *a, **kw):
        dbp = kw.get("db_path") or env
        row = q(dbp, "SELECT stage, status FROM drives WHERE id = ?", (drive_id,))[0]
        nverd = q(dbp, "SELECT COUNT(*) AS n FROM verdicts WHERE drive_id = ?", (drive_id,))[0]["n"]
        ncand = q(dbp, "SELECT COUNT(*) AS n FROM candidates WHERE drive_id = ?", (drive_id,))[0]["n"]
        seen.append((row["stage"], row["status"], nverd, ncand))
        return original(drive_id, *a, **kw)

    monkeypatch.setattr(dd, name, spy)
    if hasattr(pl, name):
        monkeypatch.setattr(pl, name, spy)
    with TestClient(app) as client:
        did = upload(client)
        res = run(did, env)
    assert res.status == "done"
    if not seen:
        pytest.skip("pipeline does not call backend.dedup.%s through a patchable name" % name)
    stage, status, nverd, ncand = seen[-1]
    assert stage == "deduplicating"
    assert status == "processing"
    assert nverd == ncand, "deduplication ran before every candidate had a verdict"


def test_dedup_failure_is_recorded_against_its_stage(env, monkeypatch):
    from fastapi.testclient import TestClient

    from backend.main import app

    name = find_dedup_name()
    import backend.dedup as dd
    import backend.pipeline as pl

    called = []

    def boom(*a, **kw):
        called.append(1)
        raise RuntimeError("synthetic dedup failure")

    monkeypatch.setattr(dd, name, boom)
    if hasattr(pl, name):
        monkeypatch.setattr(pl, name, boom)
    with TestClient(app) as client:
        did = upload(client)
        res = run(did, env)
        if not called:
            pytest.skip("pipeline does not call backend.dedup.%s through a patchable name" % name)
        assert res.status == "failed"
        assert res.failed_stage == "deduplicating"
        drive = client.get("/api/drives/%s" % did).json()
    assert drive["status"] == "failed"
    assert drive.get("failed_stage") == "deduplicating"
    assert "deduplicating" in (drive.get("error") or "")
    assert "synthetic dedup failure" in (drive.get("error") or "")


def test_failed_rerun_leaves_no_orphan_hazards(env):
    """A re-run clears the drive's old candidates; if it then fails early, hazards that only that drive
    supported must not linger without any candidate."""
    from fastapi.testclient import TestClient

    from backend.main import app
    from backend.providers import FakeProvider

    class Boom(FakeProvider):
        def detect(self, *a, **kw):
            raise RuntimeError("detector down")

    with TestClient(app) as client:
        did = upload(client)
        assert run(did, env).status == "done"
        if not q(env, "SELECT id FROM hazards"):
            pytest.skip("no hazards produced by the fake provider")
        res = run(did, env, provider=Boom("default"))
        assert res.status == "failed"
    for h in q(env, "SELECT id FROM hazards"):
        n = q(env, "SELECT COUNT(*) AS n FROM candidates WHERE hazard_id = ?", (h["id"],))[0]["n"]
        assert n >= 1, "hazard %s has no candidates after a failed re-run" % h["id"]
