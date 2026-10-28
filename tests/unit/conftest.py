import os

import pytest

from rs_helpers import load_backend


@pytest.fixture(autouse=True)
def _isolate_local_state(monkeypatch, tmp_path):
    """Point the database, the storage folder and the cost log at a per-test temp folder.

    Without this, a test that starts the app (its lifespan runs recover_interrupted()) or calls init_db()
    without setting ROADSCOUT_DB would use the real data/roadscout.sqlite, which a live run may be using.
    This fixture runs before the test's own fixtures, so any test that sets these variables itself still
    wins (monkeypatch.setenv later overrides, and delenv still works). The shared spend tracker is
    reset so a tracker built for one test (with its cost-log path) never leaks into the next.
    """
    root = tmp_path / "_isolated_state"
    monkeypatch.setenv("ROADSCOUT_DB", str(root / "db" / "roadscout.sqlite"))
    monkeypatch.setenv("ROADSCOUT_STORAGE_DIR", str(root / "storage"))
    monkeypatch.setenv("ROADSCOUT_COST_LOG", str(root / "cost_log.jsonl"))
    import sys

    def reset():  # only if the factory is already imported: importing the backend here would load .env early
        mod = sys.modules.get("backend.providers.factory")
        if mod is not None:
            mod.reset_spend_tracker()

    reset()
    yield
    reset()


@pytest.fixture(scope="session")
def backend():
    """(app, module_name, root) for the discovered FastAPI app.

    META_API_KEY is removed first: the scaffold must import and serve without it.
    """
    old = os.environ.pop("META_API_KEY", None)
    try:
        yield load_backend()
    finally:
        if old is not None:
            os.environ["META_API_KEY"] = old


@pytest.fixture()
def client(backend):
    from fastapi.testclient import TestClient

    app, _, _ = backend
    return TestClient(app)  # no context manager: no lifespan/startup side effects
