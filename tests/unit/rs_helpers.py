"""Shared helpers for the task 01 (scaffold) tests.

The task text does not fix file names, so these helpers discover the layout
(backend dir, frontend dir, FastAPI app object, DB module) instead of
hard-coding it.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SKIP_DIRS = {
    "node_modules", ".git", ".venv", "venv", "env", "__pycache__", "dist",
    "build", "playwright-report", "test-results", "blob-report", ".ruff_cache",
    ".pytest_cache", "roadscout-hidden-tests", ".mypy_cache", "site-packages",
}


def find_repo_root() -> Path:
    override = os.environ.get("ROADSCOUT_REPO")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "CLAUDE.md").exists() and (parent / "docs").exists():
            return parent
    sibling = here.parents[2] / "RoadScout"
    if sibling.exists():
        return sibling
    raise RuntimeError("cannot locate the RoadScout repo root; set ROADSCOUT_REPO")


REPO = find_repo_root()


def walk_files(root: Path, suffixes: tuple[str, ...] = ()):
    """Yield files under root, skipping vendored/build dirs."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if not suffixes or name.endswith(suffixes):
                yield Path(dirpath) / name


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60
    )


def is_ignored(rel: str) -> bool:
    return git("check-ignore", "-q", "--no-index", rel).returncode == 0


# ---------------------------------------------------------------- frontend

FRONTEND_NAMES = ("frontend", "web", "client", "ui", "app")


def find_frontend_dir() -> Path | None:
    for name in FRONTEND_NAMES:
        d = REPO / name
        if (d / "package.json").exists():
            pj = json.loads(read(d / "package.json"))
            deps = {**pj.get("dependencies", {}), **pj.get("devDependencies", {})}
            if "react" in deps:
                return d
    return None


def package_json(d: Path) -> dict:
    return json.loads(read(d / "package.json"))


def all_deps(pj: dict) -> dict:
    return {**pj.get("dependencies", {}), **pj.get("devDependencies", {})}


def node_package_jsons() -> list[Path]:
    """package.json files at the repo root and one level down."""
    out = []
    if (REPO / "package.json").exists():
        out.append(REPO / "package.json")
    for child in REPO.iterdir():
        if child.is_dir() and child.name not in SKIP_DIRS:
            if (child / "package.json").exists():
                out.append(child / "package.json")
    return out


# ----------------------------------------------------------------- backend

def _py_files_with_fastapi() -> list[Path]:
    hits = []
    for p in walk_files(REPO, (".py",)):
        rel_parts = p.relative_to(REPO).parts
        if rel_parts[0] in {"tests", "scripts", "test", "docs"}:
            continue
        if re.search(r"\bFastAPI\s*\(", read(p)):
            hits.append(p)
    return hits


def _module_attempts(py: Path):
    """Yield (sys.path root, dotted module name) pairs, repo root first."""
    parts = list(py.relative_to(REPO).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    for i in range(len(parts)):
        yield REPO.joinpath(*parts[:i]), ".".join(parts[i:])


def _try_import(root: Path, modname: str):
    before = set(sys.modules)
    sys.path.insert(0, str(root))
    try:
        mod = importlib.import_module(modname)
        return mod
    except Exception as exc:  # noqa: BLE001
        for name in set(sys.modules) - before:
            sys.modules.pop(name, None)
        sys.path.remove(str(root))
        return exc


def _extract_app(mod):
    from fastapi import FastAPI

    for attr in ("app", "application", "api"):
        obj = getattr(mod, attr, None)
        if isinstance(obj, FastAPI):
            return obj
    for obj in vars(mod).values():
        if isinstance(obj, FastAPI):
            return obj
    for factory in ("create_app", "get_app", "make_app"):
        fn = getattr(mod, factory, None)
        if callable(fn):
            try:
                obj = fn()
            except TypeError:
                continue
            if isinstance(obj, FastAPI):
                return obj
    return None


def load_backend():
    """Return (app, module_name, root) for the FastAPI app, or raise."""
    errors = []
    files = _py_files_with_fastapi()
    if not files:
        raise AssertionError("no python file in the repo constructs FastAPI(...)")
    for py in files:
        for root, modname in _module_attempts(py):
            res = _try_import(root, modname)
            if isinstance(res, Exception):
                errors.append(f"{modname} (root {root.name}): {res!r}")
                continue
            app = _extract_app(res)
            if app is not None:
                return app, modname, root
    raise AssertionError("could not import a FastAPI app; tried:\n" + "\n".join(errors))


DB_MODULE_NAMES = ("db", "database", "sqlite", "storage", "persistence", "sql")
INIT_NAMES = ("init_db", "init_database", "initialize_db", "initialize_database",
              "create_tables", "create_db", "setup_db", "init_schema", "migrate")
CONN_NAMES = ("get_connection", "connect", "get_db", "get_conn", "open_db",
              "get_engine", "get_session")


def load_db_modules(app_modname: str):
    prefix = app_modname.rpartition(".")[0]
    prefixes = []
    if prefix:
        prefixes.append(prefix)
        parent = prefix.rpartition(".")[0]
        if parent:
            prefixes.append(parent)
        prefixes.append("")
    else:
        prefixes.append("")
    mods = [sys.modules[app_modname]]
    for pre in prefixes:
        for name in DB_MODULE_NAMES:
            full = f"{pre}.{name}" if pre else name
            try:
                mods.append(importlib.import_module(full))
            except ModuleNotFoundError:
                continue
            except Exception:  # noqa: BLE001
                continue
    return mods


def pick(mods, names):
    for m in mods:
        for n in names:
            fn = getattr(m, n, None)
            if callable(fn):
                return m, n, fn
    return None, None, None
