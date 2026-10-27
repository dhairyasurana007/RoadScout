"""Task 22: export routes, registry, serving, error statuses, re-export, module function."""
from __future__ import annotations

import importlib
import inspect
import re
from datetime import datetime

import pytest

import rs22_helpers as H
from rs22_helpers import (StubProvider, assert_no_paths, probe, rs22_env, rs22_std, unwrap)  # noqa: F401

pytestmark = pytest.mark.filterwarnings("ignore")


# ------------------------------------------------------------------ routes and JSON

def test_routes_exist_in_the_openapi_schema(rs22_env):
    env = rs22_env
    assert env.post_route()
    assert env.list_route()
    assert env.video_route()


def test_post_returns_201_or_202_with_the_export_json(rs22_env):
    env = rs22_env
    env.use_provider(StubProvider("dense"))
    did = env.add_drive("d22a")
    hid = env.seed(did)
    r = env.post_export(hid)
    assert r.status_code in (201, 202), (r.status_code, r.text[:300])
    rec = unwrap(r.json())
    assert "id" in rec and str(rec["id"]).strip() != ""
    assert str(rec["hazard_id"]) == str(hid)
    assert str(rec["drive_id"]) == did
    assert str(rec["status"]).lower() in H.OK_STATUS | H.PENDING_STATUS
    final = env.settle(rec, hid)
    assert str(final["status"]).lower() in H.OK_STATUS, final
    start, end = H.rec_start(final), H.rec_end(final)
    assert 0 <= start < end <= H.SRC_DUR + 0.05
    assert final.get("created_at"), final
    ts = str(final["created_at"]).replace("Z", "+00:00")
    datetime.fromisoformat(ts)


def test_list_returns_the_export_and_never_exposes_paths(rs22_std):
    s = rs22_std
    items = s.env.list_for(s.hid)
    assert len(items) == 1
    it = items[0]
    assert str(it["id"]) == str(s.rec["id"])
    assert str(it["status"]).lower() in H.OK_STATUS
    assert str(it["hazard_id"]) == str(s.hid) and str(it["drive_id"]) == s.did
    assert_no_paths(items, s.env)
    assert_no_paths(s.rec, s.env)
    everything = s.env.all_exports()
    assert any(str(e["id"]) == str(s.rec["id"]) for e in everything)
    assert_no_paths(everything, s.env)


def test_post_response_exposes_no_paths(rs22_env):
    env = rs22_env
    env.use_provider(StubProvider("dense"))
    did = env.add_drive("d22a")
    hid = env.seed(did)
    r = env.post_export(hid)
    assert r.status_code in (201, 202)
    assert_no_paths(r.json(), env)
    d = env.detail_route()
    if d:
        rr = env.client.get(re.sub(r"\{[^}/]+\}", str(unwrap(r.json())["id"]), d))
        assert rr.status_code == 200
        assert_no_paths(rr.json(), env)


def test_exports_are_registered_in_a_table(rs22_std):
    s = rs22_std
    tables = [t["name"] for t in s.env.q("SELECT name FROM sqlite_master WHERE type='table'")]
    ex = [t for t in tables if "export" in t.lower()]
    assert ex, "no exports table in %s" % tables
    rows = s.env.q("SELECT * FROM %s" % ex[0])
    assert len(rows) == 1, rows
    cols = set(rows[0])
    assert {"id", "drive_id", "hazard_id", "status", "created_at"} <= cols, sorted(cols)
    assert any(re.search("start", c) for c in cols) and any(re.search(r"(^|_)end", c) for c in cols), sorted(cols)
    assert any(re.search("path|key|file", c) for c in cols), sorted(cols)
    assert str(rows[0]["drive_id"]) == s.did and str(rows[0]["hazard_id"]) == str(s.hid)


# ------------------------------------------------------------------ serving

def test_video_is_served_as_mp4_and_is_the_stored_file(rs22_std):
    s = rs22_std
    r = s.env.client.get(s.env.video_url(s.rec))
    assert r.status_code == 200, r.status_code
    assert r.headers["content-type"].split(";")[0].strip() == "video/mp4", r.headers["content-type"]
    assert r.content == s.video.read_bytes()
    assert r.content[4:8] == b"ftyp"


def test_video_supports_a_range_request(rs22_std):
    s = rs22_std
    r = s.env.client.get(s.env.video_url(s.rec), headers={"Range": "bytes=0-99"})
    assert r.status_code in (200, 206), r.status_code
    if r.status_code == 206:
        assert len(r.content) == 100
        assert re.match(r"bytes 0-99/\d+", r.headers.get("content-range", "")), r.headers
        assert r.content == s.video.read_bytes()[:100]
    r2 = s.env.client.get(s.env.video_url(s.rec), headers={"Range": "bytes=100-199"})
    assert r2.status_code in (200, 206)
    if r2.status_code == 206:
        assert r2.content == s.video.read_bytes()[100:200]


def test_unknown_export_video_is_404(rs22_env):
    env = rs22_env
    path = env.video_route()
    assert env.client.get(re.sub(r"\{[^}/]+\}", "999999", path)).status_code == 404
    assert env.client.get(re.sub(r"\{[^}/]+\}", "nope", path)).status_code in (404, 422)


def test_unknown_hazard_is_404_and_the_provider_is_not_called(rs22_env):
    env = rs22_env
    stub = env.use_provider(StubProvider("dense"))
    r = env.post_export(424242)
    assert r.status_code == 404, (r.status_code, r.text[:200])
    assert stub.calls == []
    kind, path = env.list_route()
    if kind == "hazard":
        assert env.client.get(re.sub(r"\{[^}/]+\}", "424242", path)).status_code == 404


# ------------------------------------------------------------------ 409 cases

def _conflict(env, stub, hid, did=None):
    r = env.post_export(hid)
    assert r.status_code in (409, 422), "expected 409, got %s %s" % (r.status_code, r.text[:300])
    msg = " ".join(H.all_strings(r.json())) if r.headers.get("content-type", "").startswith("application/json") else r.text
    assert len(msg.strip()) >= 5, "no message"
    assert_no_paths({"detail": r.json().get("detail") if isinstance(r.json(), dict) else r.json()}, env)
    assert stub.calls == [], "the provider must not be called"
    if did:
        assert env.export_videos(did) == []
    ok = [e for e in env.all_exports() if str(e.get("status", "")).lower() in H.OK_STATUS]
    assert ok == []
    return r


def test_video_deleted_flag_is_409_and_does_not_call_the_provider(rs22_env):
    env = rs22_env
    stub = env.use_provider(StubProvider("dense"))
    did = env.add_drive("d22a")
    hid = env.seed(did)
    env.x("UPDATE drives SET video_deleted_at = ? WHERE id = ?", ("2026-10-21T00:00:00Z", did))
    r = _conflict(env, stub, hid, did)
    assert r.status_code == 409


def test_video_file_missing_is_409_and_does_not_call_the_provider(rs22_env):
    env = rs22_env
    stub = env.use_provider(StubProvider("dense"))
    did = env.add_drive("d22a")
    hid = env.seed(did)
    env.video_file(did).unlink()
    r = _conflict(env, stub, hid, did)
    assert r.status_code == 409


def test_video_deleted_through_the_api_is_409(rs22_env):
    env = rs22_env
    stub = env.use_provider(StubProvider("dense"))
    did = env.add_drive("d22a")
    hid = env.seed(did)
    pat = re.compile(r"^/api/drives/\{[^}/]+\}/(video|raw-video|raw_video|delete-video|delete_video)/?$")
    found = [(m, p) for p, ops in env.paths().items() if pat.match(p) for m in ("delete", "post") if m in ops]
    if not found:
        pytest.skip("no delete-video route")
    m, p = sorted(found, key=lambda x: x[0] != "delete")[0]
    r = getattr(env.client, m)(re.sub(r"\{[^}/]+\}", did, p))
    assert r.status_code in (200, 204), r.text
    r = _conflict(env, stub, hid, did)
    assert r.status_code == 409


def test_hazard_without_member_candidates_is_rejected(rs22_env):
    env = rs22_env
    stub = env.use_provider(StubProvider("dense"))
    env.add_drive("d22a")
    hid = env.bare_hazard()
    r = _conflict(env, stub, hid)
    assert r.status_code in (409, 422)


def test_drive_being_processed_is_409(rs22_env):
    env = rs22_env
    stub = env.use_provider(StubProvider("dense"))
    did = env.add_drive("d22a", status="processing")
    hid = env.seed(did)
    r = _conflict(env, stub, hid, did)
    assert r.status_code == 409


# ------------------------------------------------------------------ re-export

def test_re_export_adds_or_replaces_without_stale_files(rs22_env):
    env = rs22_env
    stub = env.use_provider(StubProvider("dense"))
    did = env.add_drive("d22a")
    hid = env.seed(did)
    r1 = env.export_ok(hid)
    r2 = env.export_ok(hid)
    assert len(stub.calls) == 2, "one provider call per export"
    items = env.list_for(hid)
    ok = [e for e in items if str(e["status"]).lower() in H.OK_STATUS]
    assert ok, items
    assert any(str(e["id"]) == str(r2["id"]) for e in items), "the new export must be listed"
    vids = env.export_videos(did)
    assert len(vids) == len(ok), "%d mp4 files for %d registered exports: stale or missing files" % (len(vids), len(ok))
    assert [p for p in env.export_files(did) if p.suffix.lower() not in (".mp4", ".json")] == []
    for e in ok:
        resp = env.client.get(env.video_url(e))
        assert resp.status_code == 200 and resp.content[4:8] == b"ftyp"
    # every registered file is a decodable mp4
    for p in vids:
        assert H.vstreams(probe(p))
    assert env.stray_videos([env.video_file(did)]) == []


def test_each_export_gets_its_own_id(rs22_env):
    env = rs22_env
    env.use_provider(StubProvider("empty"))
    did = env.add_drive("d22a")
    from backend.dedup import deduplicate_drive

    c1 = env.add_cand(did, score=0.8, best_ts=6.0, cls="pothole")
    c2 = env.add_cand(did, score=0.8, best_ts=12.0, cls="pothole", lat=H.VA[0] + 0.01)
    deduplicate_drive(did, db_path=str(env.db))
    h1 = env.q("SELECT hazard_id FROM candidates WHERE id = ?", (c1,))[0]["hazard_id"]
    h2 = env.q("SELECT hazard_id FROM candidates WHERE id = ?", (c2,))[0]["hazard_id"]
    if h1 is None or h2 is None or h2 == h1:
        pytest.skip("synthetic second hazard was not created")
    a = env.export_ok(h1)
    b = env.export_ok(h2)
    assert a["id"] != b["id"]
    assert str(b["hazard_id"]) == str(h2)
    assert len(env.export_videos(did)) == 2
    assert [str(e["id"]) for e in env.list_for(h1)] == [str(a["id"])]


# ------------------------------------------------------------------ module function

FUNC_NAMES = ["export_hazard_clip", "export_hazard", "export_hazard_segment", "export_clip", "create_export"]


def test_export_function_in_backend_export_module(rs22_env):
    env = rs22_env
    try:
        mod = importlib.import_module("backend.export")
    except ModuleNotFoundError as e:
        if e.name == "backend.export":
            pytest.fail("backend/export.py is missing")
        raise
    func = next((getattr(mod, n) for n in FUNC_NAMES if callable(getattr(mod, n, None))), None)
    assert func is not None, "backend.export has none of %s" % FUNC_NAMES
    stub = env.use_provider(StubProvider("dense"))
    did = env.add_drive("d22a")
    hid = env.seed(did)
    sig = inspect.signature(func)
    kwargs = {}
    for n in ("provider",):
        if n in sig.parameters:
            kwargs[n] = stub
    for n in ("db_path", "db", "database_path"):
        if n in sig.parameters:
            kwargs[n] = str(env.db)
            break
    res = func(hid, **kwargs)
    if inspect.isawaitable(res):
        import asyncio

        res = asyncio.run(res)
    rec = res if isinstance(res, dict) else getattr(res, "__dict__", {})
    assert len(env.export_videos(did)) == 1
    if "status" in rec:
        assert str(rec["status"]).lower() in H.OK_STATUS
    assert len(stub.calls) == 1
