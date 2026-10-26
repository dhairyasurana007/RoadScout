import os

import pytest

from rs_helpers import load_backend


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
