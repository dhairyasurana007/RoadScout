"""SQLite: an init function and a connection helper work against a temp file.

Expected interface (see report to the main agent): a module named db or
database next to the FastAPI app exposing init_db(path) and get_connection(path)
where path is a file path string. The tests degrade to environment-variable
configuration when the functions take no argument.
"""
import contextlib
import inspect
import os
import sqlite3

import pytest

from rs_helpers import CONN_NAMES, INIT_NAMES, load_db_modules, pick

ENV_VARS = ("DATABASE_URL", "DATABASE_PATH", "DB_PATH", "SQLITE_PATH",
            "ROADSCOUT_DB", "ROADSCOUT_DB_PATH", "ROADSCOUT_DATABASE_URL")


@pytest.fixture()
def db_api(backend):
    _, modname, _ = backend
    mods = load_db_modules(modname)
    _, init_name, init_fn = pick(mods, INIT_NAMES)
    _, conn_name, conn_fn = pick(mods, CONN_NAMES)
    assert init_fn is not None, f"no init function among {INIT_NAMES}"
    return init_fn, conn_fn


def _takes_arg(fn) -> bool:
    try:
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    except (TypeError, ValueError):
        return False
    return len(params) > 0


def _call(fn, path, monkeypatch):
    for var in ENV_VARS:
        monkeypatch.setenv(var, f"sqlite:///{path}" if "URL" in var else str(path))
    if _takes_arg(fn):
        try:
            return fn(str(path))
        except Exception:
            return fn(f"sqlite:///{path}")
    return fn()


def _as_connection(obj):
    """Normalise whatever the helper returns to something with execute()."""
    if inspect.isgenerator(obj):
        return next(obj)
    return obj


def test_init_db_creates_a_usable_sqlite_file(db_api, tmp_path, monkeypatch):
    init_fn, _ = db_api
    path = tmp_path / "roadscout_test.sqlite"
    _call(init_fn, path, monkeypatch)
    assert path.exists(), "init_db did not create the database file at the given path"
    con = sqlite3.connect(path)
    try:
        assert con.execute("SELECT 1").fetchone() == (1,)
        con.execute("CREATE TABLE IF NOT EXISTS _probe (x INTEGER)")
        con.execute("INSERT INTO _probe VALUES (7)")
        assert con.execute("SELECT x FROM _probe").fetchone() == (7,)
    finally:
        con.close()


def test_init_db_is_idempotent(db_api, tmp_path, monkeypatch):
    init_fn, _ = db_api
    path = tmp_path / "idem.sqlite"
    _call(init_fn, path, monkeypatch)
    _call(init_fn, path, monkeypatch)  # must not raise on an existing db
    assert path.exists()


def test_init_db_creates_missing_parent_directory_or_uses_existing(db_api, tmp_path, monkeypatch):
    init_fn, _ = db_api
    path = tmp_path / "sub" / "dir" / "x.sqlite"
    path.parent.mkdir(parents=True)
    _call(init_fn, path, monkeypatch)
    assert path.exists()


def test_init_db_writes_valid_sqlite_header_when_nonempty(db_api, tmp_path, monkeypatch):
    init_fn, _ = db_api
    path = tmp_path / "hdr.sqlite"
    _call(init_fn, path, monkeypatch)
    data = path.read_bytes()
    assert data == b"" or data.startswith(b"SQLite format 3\x00")


def test_connection_helper_returns_working_connection(db_api, tmp_path, monkeypatch):
    init_fn, conn_fn = db_api
    assert conn_fn is not None, f"no connection helper among {CONN_NAMES}"
    path = tmp_path / "conn.sqlite"
    _call(init_fn, path, monkeypatch)
    obj = _as_connection(_call(conn_fn, path, monkeypatch))
    try:
        if hasattr(obj, "execute") and not hasattr(obj, "connect"):
            res = obj.execute("SELECT 1")
            assert res.fetchone()[0] == 1
        elif hasattr(obj, "connect"):  # SQLAlchemy engine
            from sqlalchemy import text

            with obj.connect() as c:
                assert c.execute(text("SELECT 1")).scalar() == 1
        elif hasattr(obj, "__enter__"):
            with obj as c:
                assert c.execute("SELECT 1").fetchone()[0] == 1
        else:
            pytest.fail(f"unrecognised connection object: {type(obj)}")
    finally:
        with contextlib.suppress(Exception):
            obj.close()


def test_two_connections_see_the_same_data(db_api, tmp_path, monkeypatch):
    init_fn, conn_fn = db_api
    if conn_fn is None:
        pytest.fail("no connection helper")
    path = tmp_path / "shared.sqlite"
    _call(init_fn, path, monkeypatch)
    a = _as_connection(_call(conn_fn, path, monkeypatch))
    if not isinstance(a, sqlite3.Connection):
        pytest.skip("connection helper does not return a plain sqlite3 connection")
    a.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")
    a.execute("INSERT INTO t VALUES (42)")
    a.commit()
    b = _as_connection(_call(conn_fn, path, monkeypatch))
    assert b.execute("SELECT v FROM t").fetchone()[0] == 42
    a.close()
    b.close()
