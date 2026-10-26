"""Frontend (React + Vite hello world) and Playwright smoke-test configuration.

No browser is launched here; this only checks files and configuration.
"""
import json
import re
import shutil
import subprocess

import pytest

from rs_helpers import (REPO, all_deps, find_frontend_dir, node_package_jsons,
                        package_json, read, walk_files)


@pytest.fixture(scope="module")
def fe():
    d = find_frontend_dir()
    assert d is not None, "no frontend dir (frontend/ web/ client/ ui/) with a React package.json"
    return d


# ---------------------------------------------------------------- React/Vite

def test_package_json_is_valid_json_with_name(fe):
    pj = package_json(fe)
    assert isinstance(pj.get("name"), str) and pj["name"]


def test_frontend_has_react_and_vite(fe):
    deps = all_deps(package_json(fe))
    assert "react" in deps
    assert "react-dom" in deps
    assert "vite" in deps
    assert any(k in deps for k in ("@vitejs/plugin-react", "@vitejs/plugin-react-swc"))


def test_frontend_scripts_dev_and_build(fe):
    scripts = package_json(fe).get("scripts", {})
    assert "vite" in scripts.get("dev", "") or "dev" in scripts
    assert "vite build" in scripts.get("build", "")


def test_vite_config_present(fe):
    assert any((fe / n).exists() for n in
               ("vite.config.js", "vite.config.ts", "vite.config.mjs", "vite.config.mts"))


def test_index_html_mounts_react_root(fe):
    html = read(fe / "index.html")
    assert re.search(r'id=["\']root["\']', html)
    assert re.search(r'<script[^>]+type=["\']module["\'][^>]+src=["\'][^"\']*src/', html)


def test_entrypoint_renders_app(fe):
    entries = [p for p in (fe / "src").iterdir() if re.fullmatch(r"main\.(jsx|tsx|js|ts)", p.name)]
    assert entries, "src/main.(jsx|tsx|js|ts) missing"
    text = read(entries[0])
    assert "createRoot" in text or "ReactDOM" in text
    assert "App" in text


def test_hello_world_page_exists(fe):
    srcs = [p for p in walk_files(fe / "src", (".jsx", ".tsx", ".js", ".ts"))]
    assert any(re.search(r"hello", read(p), re.I) for p in srcs), "no 'hello' text in src/"
    apps = [p for p in srcs if re.fullmatch(r"App\.(jsx|tsx|js|ts)", p.name)]
    assert apps, "src/App.(jsx|tsx) missing"


def test_package_lock_committed_for_ci(fe):
    locks = [fe / "package-lock.json", REPO / "package-lock.json",
             fe / "pnpm-lock.yaml", fe / "yarn.lock"]
    assert any(p.exists() for p in locks), "lockfile missing (CI will want npm ci)"


@pytest.mark.skipif(shutil.which("npm") is None, reason="npm not installed")
def test_frontend_production_build_succeeds(fe):
    if not (fe / "node_modules").exists():
        pytest.skip("frontend dependencies not installed")
    r = subprocess.run("npm run build", cwd=fe, shell=True, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (fe / "dist" / "index.html").exists()


# ---------------------------------------------------------------- Playwright

def _playwright_configs():
    found = []
    for p in walk_files(REPO):
        if re.fullmatch(r"playwright\.config\.(ts|js|mjs|cjs|mts)", p.name):
            found.append(p)
    return found


def _specs():
    return [p for p in walk_files(REPO, (".spec.ts", ".spec.js", ".spec.mjs", ".spec.tsx", ".spec.jsx", ".spec.cjs"))]


def _smoke_spec():
    smoke = [p for p in _specs() if p.name.startswith("smoke.")]
    assert smoke, "the smoke spec (smoke.spec.*) must keep existing"
    return smoke[0]


def test_playwright_package_installed_as_dependency():
    has = False
    for pj in node_package_jsons():
        deps = all_deps(json.loads(read(pj)))
        if "@playwright/test" in deps:
            has = True
    assert has, "@playwright/test not in any package.json"


def test_exactly_one_playwright_config():
    assert len(_playwright_configs()) == 1


def test_playwright_config_starts_the_app_and_sets_base_url():
    cfg = read(_playwright_configs()[0])
    assert "webServer" in cfg, "CI needs Playwright to start the dev server itself"
    assert "baseURL" in cfg or "localhost" in cfg or "127.0.0.1" in cfg


def test_playwright_config_points_at_a_test_dir_with_the_spec():
    cfg_path = _playwright_configs()[0]
    cfg = read(cfg_path)
    m = re.search(r"testDir\s*:\s*['\"]([^'\"]+)['\"]", cfg)
    specs = _specs()
    assert specs
    if m:
        test_dir = (cfg_path.parent / m.group(1)).resolve()
        assert any(test_dir in s.resolve().parents for s in specs), \
            f"spec file(s) not under testDir {test_dir}"


def test_there_is_a_package_script_to_run_playwright():
    scripts = {}
    for pj in node_package_jsons():
        scripts.update(json.loads(read(pj)).get("scripts", {}))
    assert any("playwright test" in v for v in scripts.values()), scripts


def test_at_least_one_playwright_spec_file_and_smoke_spec_exists():
    assert len(_specs()) >= 1
    _smoke_spec()


def test_exactly_one_playwright_test_case():
    text = read(_smoke_spec())
    cases = re.findall(r"(?<![\w.])test\s*\(", text)
    assert len(cases) == 1, f"expected one test(), found {len(cases)}"


def test_smoke_test_visits_page_and_asserts_hello(fe):
    text = read(_smoke_spec())
    assert "goto" in text
    assert "expect" in text
    assert re.search(r"hello", text, re.I)


def test_smoke_test_text_matches_the_page(fe):
    """Any hello-ish literal asserted by the spec must exist in the frontend source."""
    spec = read(_smoke_spec())
    literals = re.findall(r"['\"`]([^'\"`\n]*hello[^'\"`\n]*)['\"`]", spec, re.I)
    if not literals:
        literals = ["hello"]
    src_blob = "\n".join(read(p) for p in walk_files(fe / "src")).lower()
    html = read(fe / "index.html").lower()
    assert any(l.lower() in src_blob or l.lower() in html for l in literals), literals


def test_playwright_spec_is_not_in_python_unit_test_dir():
    assert not [s for s in _specs() if "unit" in s.relative_to(REPO).parts]


def test_playwright_spec_does_not_touch_pipeline_internals():
    text = read(_smoke_spec()).lower()
    for word in ("detect", "geotag", "dedup", "verify", "sam 3", "muse"):
        assert word not in text, f"smoke test should be UI-only, mentions {word}"
