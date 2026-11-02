"""Fixtures for the task 23a hidden tests (imported by each test module; deliberately not a conftest)."""
from __future__ import annotations

import pytest

import rs23a_helpers as H


@pytest.fixture()
def make_app(monkeypatch, tmp_path):
    """make_app(demo=..., static=..., extra_env=...) -> (app, Seed). The app module is reloaded for each call and
    restored to the default configuration at teardown."""

    def make(demo=None, static=None, extra_env=None):
        app = H.build_app(monkeypatch, tmp_path, demo=demo, static=static, extra_env=extra_env)
        return app, H.Seed(tmp_path)

    yield make
    H.restore_app()


@pytest.fixture()
def demo_world(make_app):
    """A demo-mode app with one done drive, one hazard that has a blurred photo, and one candidate per blur state."""
    app, seed = make_app(demo="1")
    seed.drive("d1")
    cid_ok, orig_ok, blurred_ok = seed.candidate("d1", blur="ok")
    cid_failed, orig_failed, _ = seed.candidate("d1", blur="failed")
    hazard_ok = seed.hazard(photo=blurred_ok)
    hazard_orig = seed.hazard(photo=orig_ok)                      # a (bad) row pointing at an unblurred original
    hazard_none = seed.hazard(photo=None)
    from fastapi.testclient import TestClient

    client = TestClient(app)
    yield {"app": app, "seed": seed, "client": client, "cid_ok": cid_ok, "cid_failed": cid_failed,
           "orig_ok": orig_ok, "blurred_ok": blurred_ok, "hazard_ok": hazard_ok, "hazard_orig": hazard_orig,
           "hazard_none": hazard_none}
