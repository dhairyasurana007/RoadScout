"""Task 23a: the backend serves the built frontend (ROADSCOUT_STATIC_DIR) with an SPA fallback, safely."""
from __future__ import annotations

import pytest

import rs23a_helpers as H
from rs23a_fixtures import make_app  # noqa: F401
from rs23a_helpers import DEMO_BODY, SECRET, SPA_MARKER, asgi_get, make_static


def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)


@pytest.fixture(params=["1", None], ids=["demo", "not-demo"])
def served(request, make_app, tmp_path):
    dist = make_static(tmp_path)
    app, seed = make_app(demo=request.param, static=dist)
    seed.drive("d1")
    return {"app": app, "client": client(app), "dist": dist, "seed": seed, "demo": request.param}


# ------------------------------------------------------------------ serving

def test_root_serves_index_html(served):
    r = served["client"].get("/")
    assert r.status_code == 200 and SPA_MARKER in r.text
    assert r.headers["content-type"].startswith("text/html")


def test_index_html_by_name(served):
    r = served["client"].get("/index.html")
    assert r.status_code == 200 and SPA_MARKER in r.text


def test_assets_are_served_with_their_content(served):
    c = served["client"]
    r = c.get("/assets/app.js")
    assert r.status_code == 200 and "asset-js-marker-1234" in r.text
    assert "javascript" in r.headers["content-type"] or "ecmascript" in r.headers["content-type"]
    r = c.get("/assets/style.css")
    assert r.status_code == 200 and "asset-css-marker-1234" in r.text and "css" in r.headers["content-type"]
    r = c.get("/favicon.svg")
    assert r.status_code == 200 and "svg-marker-1234" in r.text
    assert "svg" in r.headers["content-type"]


@pytest.mark.parametrize("path", ["/hazards", "/drives/abc", "/some/deep/spa/route", "/hazards/12/report",
                                  "/not-a-file.txt", "/x.y.z"])
def test_unknown_non_api_paths_return_index_html(served, path):
    r = served["client"].get(path)
    assert r.status_code == 200 and SPA_MARKER in r.text, (path, r.status_code)
    assert r.headers["content-type"].startswith("text/html")


def test_missing_asset_does_not_crash(served):
    r = served["client"].get("/assets/missing.js")
    assert r.status_code in (200, 404)
    assert r.status_code == 404 or SPA_MARKER in r.text


def test_head_for_the_index(served):
    r = served["client"].head("/")
    assert r.status_code == 200


# ------------------------------------------------------------------ the API and /health are unaffected

def test_health_is_not_shadowed(served):
    r = served["client"].get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_api_routes_still_answer(served):
    c = served["client"]
    assert c.get("/api/demo").json() == {"demo": served["demo"] is not None}
    assert c.get("/api/drives").status_code == 200
    r = c.get("/api/drives/d1")
    assert r.status_code == 200 and r.json()["id"] == "d1"
    assert c.get("/api/hazards").status_code == 200


def test_spa_fallback_does_not_shadow_api_404s(served):
    c = served["client"]
    for path in ("/api/nothing-here", "/api/drives/ghost", "/api/hazards/987654/photo", "/api/exports/987654",
                 "/api/definitely/not/a/route"):
        r = c.get(path)
        assert r.status_code == 404, (path, r.status_code)
        assert SPA_MARKER not in r.text
        assert r.headers["content-type"].startswith("application/json")
        assert "detail" in r.json()


def test_unknown_api_id_error_bodies_are_unchanged(served):
    r = served["client"].get("/api/drives/ghost")
    assert r.status_code == 404 and SPA_MARKER not in r.text


def test_api_validation_errors_are_not_replaced_by_the_spa(served):
    r = served["client"].get("/api/hazards/not-a-number/photo")
    assert r.status_code == 422 and SPA_MARKER not in r.text


def test_mutating_requests_still_hit_the_api_rules(served):
    c = served["client"]
    r = c.post("/api/drives/ghost/process")
    assert r.status_code == (403 if served["demo"] else 404)
    if served["demo"]:
        assert r.json() == DEMO_BODY
    assert SPA_MARKER not in r.text


def test_the_spa_does_not_answer_api_prefixed_paths_with_html(served):
    for path in ("/api/", "/api/x"):
        r = served["client"].get(path)
        assert SPA_MARKER not in r.text, path


# ------------------------------------------------------------------ path traversal

TRAVERSAL = ["/../secret.txt", "/%2e%2e/secret.txt", "/%2E%2E/secret.txt", "/..%2fsecret.txt", "/..%2Fsecret.txt",
             "/assets/../../secret.txt", "/assets/%2e%2e/%2e%2e/secret.txt", "/..%5csecret.txt",
             "/%2e%2e%5csecret.txt", "/assets/..%2f..%2fsecret.txt", "/../../../../../../etc/passwd",
             "/%2e%2e/%2e%2e/%2e%2e/windows/win.ini", "//../secret.txt", "/./../secret.txt"]


@pytest.mark.parametrize("path", TRAVERSAL)
def test_traversal_never_escapes_the_directory_raw_asgi(served, path):
    status, headers, body = asgi_get(served["app"], path)
    assert SECRET.encode() not in body, (path, status)
    assert b"root:" not in body and b"[fonts]" not in body.lower()
    assert status in (200, 400, 403, 404, 422), (path, status)
    if status == 200:
        assert SPA_MARKER.encode() in body, "a 200 for a traversal path may only be the SPA index"


@pytest.mark.parametrize("path", ["/%2e%2e/secret.txt", "/..%2fsecret.txt", "/assets/..%2f..%2fsecret.txt",
                                  "/%2e%2e%5csecret.txt"])
def test_traversal_never_escapes_the_directory_via_the_test_client(served, path):
    r = served["client"].get(path)
    assert SECRET not in r.text
    assert r.status_code in (200, 400, 403, 404, 422)


def test_a_file_next_to_the_static_dir_is_not_reachable_by_name(served):
    r = served["client"].get("/secret.txt")
    assert SECRET not in r.text, "secret.txt lives outside the static directory"


def test_dotfiles_and_the_directory_listing_are_not_exposed(served):
    r = served["client"].get("/assets/")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert SPA_MARKER in r.text or "app.js" not in r.text


# ------------------------------------------------------------------ without the env nothing changes

def test_without_the_env_the_root_is_not_served(make_app, tmp_path):
    make_static(tmp_path)
    app, seed = make_app(demo=None, static=None)
    c = client(app)
    r = c.get("/")
    assert SPA_MARKER not in r.text and r.status_code == 404
    assert c.get("/some/spa/route").status_code == 404
    assert c.get("/health").json() == {"status": "ok"}
    assert c.get("/api/drives").status_code == 200


def test_static_env_does_not_enable_demo_mode(make_app, tmp_path):
    dist = make_static(tmp_path)
    app, seed = make_app(demo=None, static=dist)
    seed.drive("d1")
    c = client(app)
    assert c.get("/api/demo").json() == {"demo": False}
    assert c.post("/api/drives/ghost/process").status_code == 404


def test_demo_env_does_not_enable_static_serving(make_app, tmp_path):
    make_static(tmp_path)
    app, seed = make_app(demo="1", static=None)
    r = client(app).get("/")
    assert SPA_MARKER not in r.text


def test_the_real_frontend_build_can_be_served_if_present(make_app):
    dist = H.ROOT / "frontend" / "dist"
    if not (dist / "index.html").is_file():
        pytest.skip("frontend/dist is not built")
    app, seed = make_app(demo="1", static=dist)
    r = client(app).get("/")
    assert r.status_code == 200 and "<div id=\"root\"" in r.text or "<script" in r.text
    assets = list((dist / "assets").glob("*.js"))
    if assets:
        assert client(app).get("/assets/" + assets[0].name).status_code == 200
