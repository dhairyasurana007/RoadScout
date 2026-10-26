"""FastAPI backend: imports, answers a health route, no API key required."""
import os

import pytest

HEALTH_PATHS = ("/health", "/api/health", "/healthz")


def test_fastapi_app_is_importable(backend):
    from fastapi import FastAPI

    app, modname, _ = backend
    assert isinstance(app, FastAPI), modname


def test_health_endpoint_returns_200_json(client):
    ok = []
    for path in HEALTH_PATHS:
        r = client.get(path)
        if r.status_code == 200:
            ok.append((path, r))
    assert ok, f"none of {HEALTH_PATHS} returned 200"
    path, r = ok[0]
    assert r.headers["content-type"].startswith("application/json"), path
    body = r.json()
    assert isinstance(body, dict), path
    status = str(body.get("status", "")).lower()
    assert status in {"ok", "healthy", "up"}, f"{path} -> {body}"


def test_health_endpoint_is_stable_across_calls(client):
    path = next(p for p in HEALTH_PATHS if client.get(p).status_code == 200)
    first = client.get(path).json()
    second = client.get(path).json()
    assert first == second


def test_health_endpoint_rejects_post(client):
    path = next(p for p in HEALTH_PATHS if client.get(p).status_code == 200)
    assert client.post(path).status_code == 405


def test_unknown_route_is_404(client):
    r = client.get("/definitely/not/a/route/xyz")
    assert r.status_code == 404


def test_openapi_schema_lists_a_health_path(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json().get("paths", {})
    assert any(p in paths for p in HEALTH_PATHS), list(paths)


def test_app_serves_without_meta_api_key(client):
    assert "META_API_KEY" not in os.environ
    assert any(client.get(p).status_code == 200 for p in HEALTH_PATHS)


def test_app_startup_and_shutdown_succeed(backend, tmp_path, monkeypatch):
    """Lifespan/startup hooks must work with no key, from an arbitrary cwd."""
    from fastapi.testclient import TestClient

    app, _, _ = backend
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("META_API_KEY", raising=False)
    with TestClient(app) as c:
        assert any(c.get(p).status_code == 200 for p in HEALTH_PATHS)


def test_health_does_not_leak_secrets(client, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "sentinel-not-a-real-key-123456")
    for p in HEALTH_PATHS:
        r = client.get(p)
        assert "sentinel-not-a-real-key-123456" not in r.text
