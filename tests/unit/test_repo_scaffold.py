"""Repo-level scaffold checks: pytest config, python deps, .env.example,
.gitignore, README."""
import configparser
import re
import tomllib

import pytest

from rs_helpers import REPO, git, is_ignored, read, walk_files


# ------------------------------------------------------------ pytest config

def _has_pytest_config() -> bool:
    if (REPO / "pytest.ini").exists():
        return True
    pp = REPO / "pyproject.toml"
    if pp.exists():
        data = tomllib.loads(read(pp))
        if "pytest" in data.get("tool", {}):
            return True
    for name in ("tox.ini", "setup.cfg"):
        f = REPO / name
        if f.exists():
            cp = configparser.ConfigParser()
            cp.read_string(read(f))
            if cp.has_section("pytest") or cp.has_section("tool:pytest"):
                return True
    return False


def test_pytest_is_configured():
    assert _has_pytest_config(), "no pytest.ini / [tool.pytest.ini_options] / tox.ini [pytest]"


def test_pytest_config_does_not_collect_node_modules_or_playwright_specs():
    """testpaths or norecursedirs should keep pytest out of the frontend tree."""
    text = ""
    for name in ("pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg"):
        f = REPO / name
        if f.exists():
            text += read(f)
    # either testpaths is set (explicit tests dir) or node_modules is excluded
    assert re.search(r"testpaths\s*=", text) or "node_modules" in text or "norecursedirs" in text


def test_python_dependencies_declared():
    sources = []
    for f in walk_files(REPO, (".txt", ".toml")):
        if f.name.startswith("requirements") or f.name == "pyproject.toml":
            sources.append(read(f).lower())
    assert sources, "no requirements*.txt or pyproject.toml"
    blob = "\n".join(sources)
    for dep in ("fastapi", "uvicorn", "pytest", "httpx"):
        assert dep in blob, f"{dep} not declared as a python dependency"


# --------------------------------------------------------------- .env files

def test_env_example_exists_and_names_meta_api_key():
    f = REPO / ".env.example"
    assert f.exists(), ".env.example missing"
    assert re.search(r"^\s*META_API_KEY\s*=", read(f), re.M), "META_API_KEY= line missing"


def test_env_example_has_no_real_looking_secret():
    text = read(REPO / ".env.example")
    m = re.search(r"^\s*META_API_KEY\s*=\s*(.*)$", text, re.M)
    assert m
    value = m.group(1).split("#")[0].strip().strip("'\"")
    placeholder_words = ("your", "change", "xxx", "here", "placeholder", "replace",
                         "example", "<", "todo", "...")
    ok = value == "" or any(w in value.lower() for w in placeholder_words)
    assert ok, "META_API_KEY in .env.example must be empty or an obvious placeholder"
    assert not re.search(r"[A-Za-z0-9_\-]{32,}", value)


def test_env_example_is_not_gitignored():
    assert not is_ignored(".env.example")


def test_dotenv_is_gitignored():
    assert is_ignored(".env")
    assert is_ignored(".env.local")


def test_dotenv_is_not_tracked_and_not_staged():
    out = git("ls-files", ".env").stdout.strip()
    assert out == ""


@pytest.mark.parametrize("rel", [
    "frontend/node_modules/react/index.js",
    "node_modules/x/y.js",
    "frontend/dist/index.html",
    "playwright-report/index.html",
    "test-results/x.txt",
    "backend/__pycache__/x.pyc",
    ".venv/Scripts/python.exe",
    "data/roadscout.db",
    "roadscout.sqlite",
    "drive.mp4",
    "track.gpx",
])
def test_generated_and_sensitive_paths_are_gitignored(rel):
    assert is_ignored(rel), f"{rel} should be git-ignored"


def test_no_api_key_value_committed_anywhere():
    pat = re.compile(r"META_API_KEY\s*[=:]\s*['\"]?([A-Za-z0-9_\-]{20,})")
    offenders = []
    for f in walk_files(REPO):
        if f.name in {".env"} or f.suffix in {".png", ".jpg", ".ico", ".woff", ".woff2", ".lock"}:
            continue
        if f.name == "package-lock.json":
            continue
        try:
            text = read(f)
        except Exception:  # noqa: BLE001
            continue
        for m in pat.finditer(text):
            val = m.group(1).lower()
            if not any(w in val for w in ("your", "change", "xxx", "here", "placeholder", "replace", "example")):
                offenders.append(str(f.relative_to(REPO)))
    assert not offenders, offenders


def test_no_real_footage_or_gpx_in_repo():
    found = [p.relative_to(REPO).as_posix() for p in walk_files(REPO, (".mov", ".mp4", ".m4v", ".hevc", ".gpx"))]
    # Git-ignored files (data/storage, spike/footage on a developer machine) are never committed; only the rest count.
    bad = [rel for rel in found if not is_ignored(rel)]
    assert not bad, bad


# ------------------------------------------------------------------- README

@pytest.fixture(scope="module")
def readme() -> str:
    for name in ("README.md", "Readme.md", "readme.md", "README.rst", "README"):
        if (REPO / name).exists():
            return read(REPO / name)
    pytest.fail("README missing")


def test_readme_documents_backend_run_command(readme):
    low = readme.lower()
    assert "uvicorn" in low or "fastapi dev" in low or "fastapi run" in low


def test_readme_documents_frontend_run_commands(readme):
    low = readme.lower()
    assert "npm install" in low or "npm ci" in low
    assert "npm run dev" in low or "npm start" in low


def test_readme_documents_running_tests(readme):
    low = readme.lower()
    assert "pytest" in low
    assert "playwright" in low
    assert "playwright install" in low, "README should say how to install Playwright browsers"


def test_readme_documents_python_dependency_install(readme):
    low = readme.lower()
    assert "pip install" in low or "uv " in low or "poetry install" in low


def test_readme_mentions_env_setup(readme):
    assert ".env.example" in readme
    assert "META_API_KEY" in readme


def test_readme_explains_ffmpeg_install_on_windows(readme):
    low = readme.lower()
    assert "ffmpeg" in low
    assert "windows" in low
    # a concrete installer command, not just a mention
    assert re.search(r"(winget|choco(latey)?|scoop)\s+install[^\n]*ffmpeg", low) or \
        "gyan.dev" in low or "ffmpeg.org/download" in low or "btbn" in low, \
        "no concrete Windows ffmpeg install instructions"
    assert "ffmpeg -version" in low or "ffprobe" in low or "path" in low, \
        "README should say how to verify ffmpeg / PATH"


def test_readme_names_the_project(readme):
    assert "roadscout" in readme.lower()


def test_readme_code_blocks_are_balanced(readme):
    assert readme.count("```") % 2 == 0
