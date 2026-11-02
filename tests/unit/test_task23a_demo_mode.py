"""Task 23a: read-only demo mode (ROADSCOUT_DEMO) - every state change under /api is a 403, GETs keep working."""
from __future__ import annotations

import os

import pytest

import rs23a_helpers as H
from rs23a_fixtures import demo_world, make_app  # noqa: F401
from rs23a_helpers import DEMO_BODY, MUTATING, fill, mutating_routes

TRUTHY = ["1", "true", "TRUE", "True", "on", "ON", "On", "yes", "YES", "Yes"]
FALSEY = [None, "", "0", "false", "FALSE", "off", "no", "NO"]


def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)


# ------------------------------------------------------------------ the switch

@pytest.mark.parametrize("val", TRUTHY)
def test_truthy_values_turn_demo_on(make_app, val):
    app, seed = make_app(demo=val)
    c = client(app)
    assert c.get("/api/demo").json() == {"demo": True}
    r = c.post("/api/drives/x/process")
    assert r.status_code == 403 and r.json() == DEMO_BODY


@pytest.mark.parametrize("val", FALSEY)
def test_falsey_values_leave_demo_off(make_app, val):
    app, seed = make_app(demo=val)
    c = client(app)
    r = c.get("/api/demo")
    assert r.status_code == 200 and r.json() == {"demo": False}
    r = c.post("/api/drives/no-such-drive/process")
    assert r.status_code == 404, "demo off: the request reaches the handler, exactly as before"


def test_unset_env_means_demo_false(make_app):
    app, seed = make_app(demo=None)
    assert "ROADSCOUT_DEMO" not in os.environ
    assert client(app).get("/api/demo").json() == {"demo": False}


def test_demo_endpoint_is_json_with_a_boolean(make_app):
    app, seed = make_app(demo="1")
    r = client(app).get("/api/demo")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    assert r.json()["demo"] is True
    assert set(r.json()) == {"demo"}


def test_demo_endpoint_does_not_need_data_or_a_database_row(make_app):
    app, seed = make_app(demo="1")
    assert client(app).get("/api/demo").status_code == 200


# ------------------------------------------------------------------ blocking every state change

def test_the_app_has_mutating_routes_to_test(make_app):
    app, seed = make_app(demo="1")
    got = set(mutating_routes(app))
    methods = {m for m, _ in got}
    assert {"POST", "PATCH", "DELETE"} <= methods
    assert any(p == "/api/drives" for m, p in got)
    assert any(p.endswith("/process") for m, p in got)
    assert any(p.endswith("/video") and m == "DELETE" for m, p in got)
    assert any(p.endswith("/export") for m, p in got)
    assert any(m == "PATCH" and p.startswith("/api/hazards/") for m, p in got)


def test_every_real_mutating_route_returns_the_demo_403(make_app):
    app, seed = make_app(demo="1")
    c = client(app)
    routes = mutating_routes(app)
    assert len(routes) >= 5
    for method, path in routes:
        r = c.request(method, fill(path))
        assert r.status_code == 403, (method, path, r.status_code, r.text)
        assert r.json() == DEMO_BODY, (method, path)
        assert r.headers["content-type"].startswith("application/json")


def test_the_403_body_is_exactly_the_documented_detail(make_app):
    app, seed = make_app(demo="1")
    r = client(app).post("/api/drives")
    assert r.status_code == 403
    assert r.json() == {"detail": "This is a read-only demo."}
    assert r.text.replace(" ", "") == '{"detail":"Thisisaread-onlydemo."}'


@pytest.mark.parametrize("method", MUTATING)
def test_each_method_on_a_nonexistent_api_path_is_forbidden_not_404(make_app, method):
    app, seed = make_app(demo="1")
    r = client(app).request(method, "/api/this/route/does/not/exist")
    assert r.status_code == 403 and r.json() == DEMO_BODY


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE", "POST"])
def test_wrong_method_on_a_read_only_route_is_forbidden_not_405(make_app, method):
    app, seed = make_app(demo="1")
    seed.drive("d1")
    for path in ("/api/drives", "/api/hazards", "/api/drives/d1", "/api/demo", "/api/exports"):
        r = client(app).request(method, path)
        assert r.status_code == 403 and r.json() == DEMO_BODY, (method, path, r.status_code)


def test_403_comes_before_validation_errors(demo_world):
    c = demo_world["client"]
    hid = demo_world["hazard_ok"]
    r = c.patch(f"/api/hazards/{hid}", content=b"{this is not json", headers={"content-type": "application/json"})
    assert r.status_code == 403 and r.json() == DEMO_BODY
    r = c.patch(f"/api/hazards/{hid}", json={"status": "definitely-not-a-status", "severity": 99})
    assert r.status_code == 403 and r.json() == DEMO_BODY
    r = c.patch("/api/hazards/not-a-number", json={})
    assert r.status_code == 403 and r.json() == DEMO_BODY
    r = c.post("/api/drives", data={"clock_offset_seconds": "banana"})          # no files at all
    assert r.status_code == 403 and r.json() == DEMO_BODY
    r = c.post("/api/drives", files={"video": ("a.mp4", b"not a video", "video/mp4")})
    assert r.status_code == 403 and r.json() == DEMO_BODY
    assert demo_world["seed"].drive_count() == 1


def test_403_comes_before_unknown_ids(demo_world):
    c = demo_world["client"]
    for method, path in (("POST", "/api/drives/ghost/process"), ("DELETE", "/api/drives/ghost/video"),
                         ("PATCH", "/api/hazards/987654321"), ("POST", "/api/hazards/987654321/export"),
                         ("POST", "/api/hazards/987654321/report")):
        r = c.request(method, path)
        assert r.status_code == 403 and r.json() == DEMO_BODY, (method, path, r.status_code)


def test_the_database_is_not_changed_by_blocked_requests(demo_world):
    seed, c = demo_world["seed"], demo_world["client"]
    hid = demo_world["hazard_ok"]
    before = seed.db.read_bytes()
    c.patch(f"/api/hazards/{hid}", json={"status": "approved"})
    c.post(f"/api/hazards/{hid}/export")
    c.delete("/api/drives/d1/video")
    c.post("/api/drives/d1/process")
    c.post("/api/drives", files={"video": ("a.mp4", b"x", "video/mp4"), "gpx": ("t.gpx", b"<gpx/>", "text/xml")})
    assert seed.hazard_status(hid) == "new"
    assert seed.db.read_bytes() == before
    assert not list((seed.storage).glob("*/video.*"))


def test_review_actions_and_mark_as_reported_are_blocked(demo_world):
    c, seed = demo_world["client"], demo_world["seed"]
    hid = demo_world["hazard_ok"]
    for status in ("approved", "rejected", "new", "reported"):
        r = c.patch(f"/api/hazards/{hid}", json={"status": status})
        assert r.status_code == 403 and r.json() == DEMO_BODY
    r = c.patch(f"/api/hazards/{hid}", json={"hazard_class": "debris in lane", "severity": 3})
    assert r.status_code == 403 and r.json() == DEMO_BODY
    assert seed.hazard_status(hid) == "new"


def test_lowercase_and_odd_methods_are_treated_as_state_changes(make_app):
    app, seed = make_app(demo="1")
    c = client(app)
    r = c.request("PUT", "/api/hazards/1", json={})
    assert r.status_code == 403 and r.json() == DEMO_BODY


def test_path_prefix_boundary_only_api_is_guarded(make_app):
    app, seed = make_app(demo="1")
    r = client(app).post("/apiary/thing")
    assert r.status_code != 403 or r.json() != DEMO_BODY


# ------------------------------------------------------------------ reads keep working

def test_get_routes_work_in_demo_mode(demo_world):
    c = demo_world["client"]
    assert c.get("/health").status_code == 200
    assert c.get("/api/demo").json() == {"demo": True}
    assert c.get("/api/drives").status_code == 200
    r = c.get("/api/drives/d1")
    assert r.status_code == 200 and r.json()["id"] == "d1"
    assert c.get("/api/hazards").status_code == 200
    assert c.get(f"/api/hazards/{demo_world['hazard_ok']}/report").status_code in (200, 404, 409, 422)
    assert c.get("/api/drives/d1/hazards").status_code == 200
    assert c.get("/api/drives/ghost").status_code == 404, "an unknown id is still a plain 404 for a GET"


def test_health_stays_open(make_app):
    app, seed = make_app(demo="1")
    r = client(app).get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_health_is_not_turned_into_a_demo_403(make_app):
    app, seed = make_app(demo="1")
    r = client(app).post("/health")
    assert not (r.status_code == 403 and r.json() == DEMO_BODY), "/health is outside /api"


def test_head_requests_under_api_are_not_blocked(demo_world):
    r = demo_world["client"].head("/api/drives")
    assert r.status_code != 403


def test_cors_preflight_is_not_blocked(demo_world):
    r = demo_world["client"].options("/api/hazards/1", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "PATCH"})
    assert r.status_code == 200
    r = demo_world["client"].options("/api/drives")
    assert r.status_code != 403


def test_get_with_a_body_or_odd_query_is_still_a_get(demo_world):
    r = demo_world["client"].get("/api/hazards?status=approved&limit=5")
    assert r.status_code in (200, 422)
    assert not (r.status_code == 403 and r.json() == DEMO_BODY)


# ------------------------------------------------------------------ privacy: only blurred images, fail closed

def test_blurred_photo_routes_keep_working(demo_world):
    c, w = demo_world["client"], demo_world
    r = c.get(f"/api/hazards/{w['hazard_ok']}/photo")
    assert r.status_code == 200 and r.content == w["blurred_ok"].read_bytes()
    r = c.get(f"/api/drives/d1/hazards/{w['cid_ok']}/photo")
    assert r.status_code == 200 and r.content == w["blurred_ok"].read_bytes()


def test_photo_routes_stay_fail_closed_in_demo_mode(demo_world):
    c, w = demo_world["client"], demo_world
    assert c.get(f"/api/hazards/{w['hazard_orig']}/photo").status_code == 404, "an unblurred original is never served"
    assert c.get(f"/api/hazards/{w['hazard_none']}/photo").status_code == 404
    assert c.get(f"/api/hazards/987654/photo").status_code == 404
    r = c.get(f"/api/drives/d1/hazards/{w['cid_failed']}/photo")
    assert r.status_code == 404, "blur failed: nothing is served"
    for r in (c.get(f"/api/hazards/{w['hazard_orig']}/photo"), c.get(f"/api/drives/d1/hazards/{w['cid_failed']}/photo")):
        assert b"UNBLURRED-ORIGINAL" not in r.content


def test_no_get_route_serves_the_raw_video_or_the_unblurred_original(demo_world):
    c, w = demo_world["client"], demo_world
    for path in ("/api/drives/d1/video", "/api/drives/d1/video.mp4", "/api/drives/d1/raw", "/api/drives/d1/frames/1",
                 "/api/drives/d1/hazards/%d/original" % w["cid_ok"], "/api/drives/d1/hazards/%d/frame" % w["cid_ok"],
                 "/api/drives/d1/hazards/%d/photo?variant=original" % w["cid_ok"]):
        r = c.get(path)
        assert r.status_code in (403, 404, 405, 422), (path, r.status_code)
        assert b"UNBLURRED-ORIGINAL" not in r.content


def test_the_stored_unblurred_file_cannot_be_fetched_through_any_listed_url(demo_world):
    """Follow every photo/crop URL the API hands out: none of them returns the original's bytes."""
    c, w = demo_world["client"], demo_world
    urls = []
    body = c.get("/api/hazards").json()
    items = body if isinstance(body, list) else body.get("hazards", body.get("items", []))
    for it in items:
        urls += [it.get("photo_url"), it.get("crop_url")]
    body = c.get("/api/drives/d1/hazards").json()
    items = body if isinstance(body, list) else body.get("hazards", body.get("items", []))
    for it in items:
        urls += [it.get("photo_url"), it.get("crop_url")]
    for u in [u for u in urls if u]:
        assert b"UNBLURRED-ORIGINAL" not in c.get(u).content, u


# ------------------------------------------------------------------ demo off behaves as before

def test_demo_off_review_actions_work(make_app):
    app, seed = make_app(demo=None)
    seed.drive("d1")
    hid = seed.hazard()
    r = client(app).patch(f"/api/hazards/{hid}", json={"status": "approved"})
    assert r.status_code == 200, r.text
    assert seed.hazard_status(hid) == "approved"


def test_demo_off_validation_and_404_are_unchanged(make_app):
    app, seed = make_app(demo="0")
    c = client(app)
    assert c.post("/api/drives").status_code == 422
    assert c.post("/api/drives/ghost/process").status_code == 404
    assert c.delete("/api/drives/ghost/video").status_code == 404
    assert c.patch("/api/hazards/987654", json={"status": "approved"}).status_code == 404
    r = c.post("/api/this/does/not/exist")
    assert r.status_code == 404


def test_toggling_the_env_toggles_the_behaviour(make_app):
    app, seed = make_app(demo="1")
    assert client(app).post("/api/drives/x/process").status_code == 403
    app2, _ = make_app(demo=None)
    assert client(app2).post("/api/drives/x/process").status_code == 404


# ------------------------------------------------------------------ no key, no live provider

def test_the_demo_never_reads_the_api_key_or_builds_the_live_provider(make_app, monkeypatch, tmp_path):
    import backend.providers.live as live

    asked = []

    class Spy(dict):
        def get(self, k, *a):
            asked.append(k)
            return dict.get(self, k, *a)

        def __getitem__(self, k):
            asked.append(k)
            return dict.__getitem__(self, k)

        def __contains__(self, k):
            asked.append(k)
            return dict.__contains__(self, k)

    def boom(*a, **k):
        raise AssertionError("the live provider must not be constructed in demo mode")

    app, seed = make_app(demo="1", extra_env={"ROADSCOUT_PROVIDER": "live", "META_API_KEY": "dummy-key-23a"})
    seed.drive("d1")
    cid, orig, blurred = seed.candidate("d1")
    hid = seed.hazard(photo=blurred)
    monkeypatch.setattr(live.LiveProvider, "__init__", boom)
    monkeypatch.setattr(os, "environ", Spy(os.environ))
    from fastapi.testclient import TestClient

    with TestClient(app) as c:                       # runs the lifespan too
        assert c.get("/health").status_code == 200
        assert c.get("/api/demo").json() == {"demo": True}
        assert c.get("/api/drives").status_code == 200
        assert c.get(f"/api/hazards/{hid}/photo").status_code == 200
        assert c.post("/api/drives/d1/process").status_code == 403
        assert c.post(f"/api/hazards/{hid}/export").status_code == 403
        assert c.post("/api/drives", files={"video": ("a.mp4", b"x", "video/mp4")}).status_code == 403
    assert "META_API_KEY" not in asked, "the key must not be read in demo mode"


def test_demo_with_live_provider_and_no_key_still_starts(make_app):
    app, seed = make_app(demo="1", extra_env={"ROADSCOUT_PROVIDER": "live"})
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/api/drives").status_code == 200
