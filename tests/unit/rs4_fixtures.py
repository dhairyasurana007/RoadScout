"""Fixtures for the task 04 tests. Import them into each test module:

    from rs4_fixtures import *  # noqa: F401,F403

(no conftest.py, so nothing clashes with tests/unit/conftest.py).
"""
import pytest

from rs4_helpers import (
    apply_env,
    discover_base,
    discover_upload,
    load_app,
    make_clip,
    make_gpx,
)

__all__ = ["rs4_env", "rs4_app", "rs4_client", "rs4_api", "rs4_clip", "rs4_gpx", "clip_dir"]


@pytest.fixture(scope="session")
def clip_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("rs4-clips")


@pytest.fixture(scope="session")
def rs4_app():
    return load_app()


@pytest.fixture()
def rs4_env(tmp_path, monkeypatch):
    """Temp DB + temp storage dir selected via environment variables."""
    return apply_env(monkeypatch, tmp_path)


@pytest.fixture()
def rs4_client(rs4_app, rs4_env):
    from fastapi.testclient import TestClient

    # The context manager runs the app lifespan (init_db) against the temp DB.
    with TestClient(rs4_app) as c:
        yield c


@pytest.fixture()
def rs4_api(rs4_app, rs4_client):
    api = discover_upload(rs4_app)
    api.base = discover_base(rs4_client, api)
    return api


@pytest.fixture()
def rs4_clip(clip_dir):
    """Factory: rs4_clip(kind, color, creation_time) -> bytes."""

    def factory(kind="mp4-h264", color="red", creation_time="2026-03-14T12:34:56Z"):
        return make_clip(clip_dir, kind, color, creation_time)

    return factory


@pytest.fixture()
def rs4_gpx():
    return make_gpx
