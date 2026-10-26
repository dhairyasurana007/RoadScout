"""Task 02: GitHub Actions workflow(s) for pytest and Playwright (static checks)."""
from __future__ import annotations

import re

import pytest
import yaml

import ci_helpers as h

# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def workflows():
    wfs = h.load_workflows()
    assert wfs, "no workflow files found under .github/workflows (*.yml / *.yaml)"
    return wfs


@pytest.fixture(scope="module")
def pytest_steps(workflows):
    found = [
        (wf, jid, job, i, step)
        for wf, jid, job, i, step in h.all_steps()
        if h.is_pytest_step(step)
    ]
    assert found, "no workflow step runs pytest"
    return found


@pytest.fixture(scope="module")
def playwright_steps(workflows):
    found = [
        (wf, jid, job, i, step)
        for wf, jid, job, i, step in h.all_steps()
        if h.is_playwright_run_step(wf, job, step)
    ]
    assert found, "no workflow step runs Playwright (npx playwright test / npm script)"
    return found


def _is_false_literal(v) -> bool:
    return v is False or (isinstance(v, str) and v.strip().lower() in ("false", "${{ false }}"))


# ---------------------------------------------------------------- basic shape


def test_workflow_directory_has_files():
    assert h.WORKFLOW_DIR.is_dir(), ".github/workflows does not exist"
    assert h.workflow_files(), "no .yml/.yaml files in .github/workflows"


def test_every_workflow_is_valid_yaml_mapping(workflows):
    for wf in workflows:
        assert isinstance(wf.data, dict), f"{wf.name}: top level is not a mapping"


def test_yaml_parses_with_strict_loader():
    for p in h.workflow_files():
        try:
            yaml.safe_load(h.read(p))
        except yaml.YAMLError as exc:  # pragma: no cover
            pytest.fail(f"{p.name}: invalid YAML: {exc}")


def test_every_workflow_has_name_trigger_and_jobs(workflows):
    for wf in workflows:
        assert wf.triggers, f"{wf.name}: no `on:` triggers"
        assert wf.jobs, f"{wf.name}: no jobs"


def test_every_job_has_runner_and_steps(workflows):
    for wf in workflows:
        for jid, job in wf.jobs.items():
            assert isinstance(job, dict), f"{wf.name}:{jid} is not a mapping"
            if "uses" in job:  # reusable workflow call
                continue
            assert job.get("runs-on"), f"{wf.name}:{jid} has no runs-on"
            assert job.get("steps"), f"{wf.name}:{jid} has no steps"


def test_every_step_runs_or_uses_something(workflows):
    for wf in workflows:
        for jid, job in wf.jobs.items():
            for i, step in enumerate(job.get("steps") or []):
                assert isinstance(step, dict), f"{wf.name}:{jid} step {i} not a mapping"
                assert ("run" in step) != ("uses" in step), (
                    f"{wf.name}:{jid} step {i} must have exactly one of run/uses"
                )


def test_third_party_actions_are_version_pinned(workflows):
    for wf in workflows:
        for jid, job in wf.jobs.items():
            for step in job.get("steps") or []:
                u = step.get("uses")
                if not isinstance(u, str):
                    continue
                if u.startswith("./") or u.startswith("docker://"):
                    continue
                assert "@" in u, f"{wf.name}:{jid} uses unpinned action {u!r}"


# ------------------------------------------------------------------- triggers


def test_runs_on_every_push(workflows):
    assert any(h.push_covers_every_push(wf) for wf in workflows), (
        "no workflow is triggered by an unfiltered `push` event"
    )


def test_pytest_and_playwright_workflows_run_on_push(pytest_steps, playwright_steps):
    for label, group in (("pytest", pytest_steps), ("playwright", playwright_steps)):
        assert any(h.push_covers_every_push(wf) for wf, *_ in group), (
            f"the workflow that runs {label} is not triggered on every push"
        )


def test_test_jobs_are_not_disabled_or_push_excluded(pytest_steps, playwright_steps):
    for wf, jid, job, i, step in pytest_steps + playwright_steps:
        assert not _is_false_literal(job.get("if")), f"{wf.name}:{jid} job disabled by if"
        assert not _is_false_literal(step.get("if")), f"{wf.name}:{jid} step disabled by if"
        cond = f"{job.get('if', '')} {step.get('if', '')}"
        assert not re.search(r"event_name\s*!=\s*['\"]push['\"]", cond), (
            f"{wf.name}:{jid} is excluded from push events"
        )
        assert not re.search(r"event_name\s*==\s*['\"](schedule|workflow_dispatch)['\"]", cond), (
            f"{wf.name}:{jid} only runs for non-push events"
        )


def test_failures_are_not_swallowed(pytest_steps, playwright_steps):
    for wf, jid, job, i, step in pytest_steps + playwright_steps:
        assert not step.get("continue-on-error") is True, (
            f"{wf.name}:{jid} step {i} has continue-on-error: true"
        )
        assert job.get("continue-on-error") is not True, (
            f"{wf.name}:{jid} job has continue-on-error: true"
        )
        text = h.run_text(step)
        assert "|| true" not in text and "||true" not in text, (
            f"{wf.name}:{jid} step {i} swallows failures with '|| true'"
        )
        assert "exit 0" not in text, f"{wf.name}:{jid} step {i} forces exit 0"


# --------------------------------------------------------------------- pytest


def test_pytest_job_checks_out_repo_first(pytest_steps):
    for wf, jid, job, i, step in pytest_steps:
        idx = h.first_index(job, lambda s: h.uses_of(s).startswith("actions/checkout"))
        assert idx is not None, f"{wf.name}:{jid} never checks out the repo"
        assert idx < i, f"{wf.name}:{jid} checks out after running pytest"


def test_pytest_job_sets_up_python_before_running(pytest_steps):
    wf, jid, job, i, step = pytest_steps[0]
    idx = h.first_index(job, lambda s: h.uses_of(s).startswith("actions/setup-python"))
    if idx is None and "python" in h.container_image(job):
        return
    assert idx is not None, f"{wf.name}:{jid} has no actions/setup-python step"
    assert idx < i, f"{wf.name}:{jid} sets up Python after running pytest"


def test_pytest_job_python_version_is_at_least_3_11(pytest_steps):
    for wf, jid, job, i, step in pytest_steps:
        for s in job.get("steps") or []:
            if h.uses_of(s).startswith("actions/setup-python"):
                v = (s.get("with") or {}).get("python-version")
                if v is None:
                    pv = (s.get("with") or {}).get("python-version-file")
                    assert pv, f"{wf.name}:{jid} setup-python has no version"
                    continue
                parsed = h.parse_version_floor(v)
                if parsed is None:
                    continue  # matrix expression: cannot judge statically
                assert parsed >= (3, 11), f"{wf.name}:{jid} uses Python {v}, README says 3.11+"


def test_pytest_job_installs_requirements_txt_before_pytest(pytest_steps):
    for wf, jid, job, i, step in pytest_steps:
        idx = h.first_index(job, h.is_pip_requirements_step)
        assert idx is not None, f"{wf.name}:{jid} never runs pip install -r requirements.txt"
        assert idx < i, f"{wf.name}:{jid} installs requirements after running pytest"
    assert (h.REPO / "requirements.txt").exists()


def test_pip_install_of_requirements_runs_from_repo_root(pytest_steps):
    for wf, jid, job, i, step in pytest_steps:
        for s in job.get("steps") or []:
            if h.is_pip_requirements_step(s):
                dirs = h.step_effective_dirs(wf, job, s)
                assert any(h.dir_has(d, "requirements.txt") for d in dirs), (
                    f"{wf.name}:{jid} pip install -r requirements.txt runs where the file is absent"
                )


def test_pytest_runs_where_the_repo_config_applies(pytest_steps):
    """pytest.ini lives at the repo root (testpaths = tests/unit)."""
    for wf, jid, job, i, step in pytest_steps:
        text = h.run_text(step)
        dirs = h.step_effective_dirs(wf, job, step)
        from_root = "." in dirs or any(h.dir_has(d, "pytest.ini") for d in dirs)
        explicit = re.search(r"tests[/\\]unit|--rootdir|-c\s+\S*pytest\.ini|\bcd\s+\.\.", text)
        assert from_root or explicit, (
            f"{wf.name}:{jid} pytest runs outside the repo root without pointing at the config"
        )


def test_pytest_does_not_select_live_tests(pytest_steps):
    for wf, jid, job, i, step in pytest_steps:
        text = h.run_text(step)
        assert not re.search(r"--live\b|--run-live|--meta|--real-provider", text), (
            f"{wf.name}:{jid} pytest is passed a live-API flag"
        )
        for m in re.finditer(r"-m\s+[\"']?([^\"'\n]+)", text):
            expr = m.group(1)
            assert not re.search(r"\blive\b", expr) or re.search(r"\bnot\s+live\b", expr), (
                f"{wf.name}:{jid} pytest marker expression selects live tests: {expr!r}"
            )


def test_ffmpeg_is_provided_in_the_pytest_job(pytest_steps):
    """Later tasks generate synthetic clips with ffmpeg inside pytest."""
    for wf, jid, job, i, step in pytest_steps:
        idx = h.first_index(job, h.is_ffmpeg_provision_step)
        assert idx is not None, f"{wf.name}:{jid} never installs ffmpeg for pytest"
        assert idx < i, f"{wf.name}:{jid} installs ffmpeg after running pytest"


# ----------------------------------------------------------------- Playwright


def test_playwright_job_checks_out_repo_first(playwright_steps):
    for wf, jid, job, i, step in playwright_steps:
        idx = h.first_index(job, lambda s: h.uses_of(s).startswith("actions/checkout"))
        assert idx is not None, f"{wf.name}:{jid} never checks out the repo"
        assert idx < i


def test_playwright_job_sets_up_node_before_running(playwright_steps):
    for wf, jid, job, i, step in playwright_steps:
        idx = h.first_index(job, lambda s: h.uses_of(s).startswith("actions/setup-node"))
        if idx is None and "node" in h.container_image(job):
            continue
        if idx is None and "mcr.microsoft.com/playwright" in h.container_image(job):
            continue  # Playwright image bundles Node
        assert idx is not None, f"{wf.name}:{jid} has no actions/setup-node step"
        assert idx < i, f"{wf.name}:{jid} sets up Node after running Playwright"


def test_playwright_job_node_version_is_at_least_20(playwright_steps):
    for wf, jid, job, i, step in playwright_steps:
        for s in job.get("steps") or []:
            if h.uses_of(s).startswith("actions/setup-node"):
                v = (s.get("with") or {}).get("node-version")
                if v is None:
                    assert (s.get("with") or {}).get("node-version-file"), (
                        f"{wf.name}:{jid} setup-node has no version"
                    )
                    continue
                parsed = h.parse_version_floor(v)
                if parsed is None:
                    continue
                assert parsed[0] >= 20, f"{wf.name}:{jid} uses Node {v}, README says 20+"


def test_playwright_job_uses_npm_ci_with_the_lockfile(playwright_steps):
    for wf, jid, job, i, step in playwright_steps:
        idx = h.first_index(job, h.is_npm_ci_step)
        assert idx is not None, f"{wf.name}:{jid} never runs `npm ci`"
        assert idx < i, f"{wf.name}:{jid} runs npm ci after Playwright"
        for s in job.get("steps") or []:
            if h.is_npm_ci_step(s):
                dirs = h.step_effective_dirs(wf, job, s)
                assert any(
                    h.dir_has(d, "package.json") and h.dir_has(d, "package-lock.json")
                    for d in dirs
                ), f"{wf.name}:{jid} npm ci runs where there is no package.json + lockfile"
                text = h.run_text(s)
                assert "npm install" not in text.replace("npm ci", ""), (
                    f"{wf.name}:{jid} mixes npm install with npm ci"
                )


def test_playwright_tests_run_in_the_frontend_project(playwright_steps):
    for wf, jid, job, i, step in playwright_steps:
        dirs = h.step_effective_dirs(wf, job, step)
        assert any(h.dir_has(d, "playwright.config.js") or h.dir_has(d, "playwright.config.ts")
                   or h.dir_has(d, "package.json") for d in dirs), (
            f"{wf.name}:{jid} runs Playwright outside the frontend project"
        )
        assert any(h.dir_has(d, "package.json") for d in dirs)


def test_playwright_browsers_are_installed_before_tests(playwright_steps):
    for wf, jid, job, i, step in playwright_steps:
        if "mcr.microsoft.com/playwright" in h.container_image(job):
            continue  # browsers baked into the image
        idx = h.first_index(job, h.is_playwright_install_step)
        assert idx is not None, f"{wf.name}:{jid} never runs `playwright install`"
        assert idx < i, f"{wf.name}:{jid} installs browsers after running tests"


def test_playwright_install_includes_chromium(playwright_steps):
    """frontend/playwright.config.js only defines a chromium project."""
    for wf, jid, job, i, step in playwright_steps:
        if "mcr.microsoft.com/playwright" in h.container_image(job):
            continue
        for s in job.get("steps") or []:
            if not h.is_playwright_install_step(s):
                continue
            for line in h.run_lines(s):
                if not re.search(r"\bplaywright\s+install(?!-)\b", line):
                    continue
                tail = line.split("install", 1)[1]
                args = [a for a in tail.split() if not a.startswith("-")]
                names = {"chromium", "firefox", "webkit", "chrome", "msedge", "chromium-headless-shell"}
                chosen = [a for a in args if a in names]
                assert (not chosen) or any(c.startswith("chrom") for c in chosen), (
                    f"{wf.name}:{jid} installs {chosen} but the project uses chromium"
                )


def test_playwright_install_uses_project_version_via_npx_or_npm(playwright_steps):
    """Browsers must match the locked @playwright/test, so go through the project's CLI."""
    for wf, jid, job, i, step in playwright_steps:
        for s in job.get("steps") or []:
            if h.is_playwright_install_step(s):
                text = h.run_text(s)
                assert re.search(r"\b(npx|npm exec|pnpm exec|yarn)\s+(--no-install\s+|-y\s+)?playwright\s+install", text) or (
                    "node_modules/.bin" in text
                ) or re.search(r"\bnpm\s+run\b", text), (
                    f"{wf.name}:{jid} runs a bare `playwright install` that may not match the locked version"
                )


def test_playwright_install_runs_in_frontend_dir(playwright_steps):
    for wf, jid, job, i, step in playwright_steps:
        for s in job.get("steps") or []:
            if h.is_playwright_install_step(s):
                dirs = h.step_effective_dirs(wf, job, s)
                assert any(h.dir_has(d, "package.json") for d in dirs), (
                    f"{wf.name}:{jid} playwright install runs outside the npm project"
                )


# ----------------------------------------------------- secrets and the live API

LIVE_HOST_RE = re.compile(
    r"graph\.facebook\.com|api\.llama\.com|llama\.developer\.meta\.com|"
    r"api\.meta\.ai|developers\.meta\.com|meta\.ai/api|facebook\.com/v\d",
    re.I,
)


def test_no_secret_references_other_than_github_token(workflows):
    for wf in workflows:
        for m in re.finditer(r"secrets\.([A-Za-z0-9_]+)", wf.text):
            assert m.group(1) == "GITHUB_TOKEN", (
                f"{wf.name} references secret {m.group(1)}; CI must not need any live key"
            )
        assert not re.search(r"secrets\s*\[", wf.text), f"{wf.name} indexes secrets dynamically"
        assert "secrets: inherit" not in wf.text, f"{wf.name} inherits all secrets"


def test_meta_api_key_never_populated(workflows):
    for wf in workflows:
        def check_env(env, where):
            if isinstance(env, dict):
                for k, v in env.items():
                    if str(k).upper() == "META_API_KEY":
                        assert v in (None, ""), f"{wf.name} {where} sets META_API_KEY to a value"

        check_env(wf.data.get("env"), "workflow env")
        for jid, job in wf.jobs.items():
            check_env(job.get("env"), f"job {jid} env")
            for i, step in enumerate(job.get("steps") or []):
                check_env(step.get("env"), f"job {jid} step {i} env")
                for line in h.run_lines(step):
                    if "META_API_KEY" in line:
                        assert re.search(r"unset|Remove-Item|\bis not set\b|-z\b", line), (
                            f"{wf.name}:{jid} step {i} handles META_API_KEY: {line!r}"
                        )


def test_no_hardcoded_credentials_in_workflows(workflows):
    patterns = [
        r"ghp_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}", r"sk-[A-Za-z0-9]{20,}",
        r"EAA[A-Za-z0-9]{30,}", r"LLM\|[0-9]+\|[A-Za-z0-9_-]{20,}", r"AKIA[0-9A-Z]{16}",
        r"(?i)bearer\s+[A-Za-z0-9._-]{20,}",
    ]
    for wf in workflows:
        for pat in patterns:
            assert not re.search(pat, wf.text), f"{wf.name} contains something that looks like a credential"


def test_no_live_meta_api_calls_in_workflows(workflows):
    for wf in workflows:
        assert not LIVE_HOST_RE.search(wf.text), f"{wf.name} mentions a Meta API host"
        for jid, job in wf.jobs.items():
            for i, step in enumerate(job.get("steps") or []):
                for line in h.run_lines(step):
                    if re.search(r"\b(curl|wget|Invoke-WebRequest|Invoke-RestMethod|iwr|irm)\b", line):
                        assert not re.search(r"meta|llama|facebook|muse|sam3?", line, re.I), (
                            f"{wf.name}:{jid} step {i} makes a network call that may hit Meta: {line!r}"
                        )


def test_no_step_runs_live_provider_scripts(workflows):
    for wf in workflows:
        for jid, job in wf.jobs.items():
            for i, step in enumerate(job.get("steps") or []):
                text = h.run_text(step)
                assert not re.search(r"--live\b|--real\b|--provider[ =]+(live|meta|real|sam|muse)", text, re.I), (
                    f"{wf.name}:{jid} step {i} selects a live provider"
                )


def _all_env_pairs(wf):
    def pairs(env):
        if isinstance(env, dict):
            yield from ((str(k), v) for k, v in env.items())

    yield from pairs(wf.data.get("env"))
    for job in wf.jobs.values():
        yield from pairs(job.get("env"))
        for step in job.get("steps") or []:
            yield from pairs(step.get("env"))


def test_provider_env_flags_never_select_live(workflows):
    """Any provider / live-ish env var that is set must point at the fake provider."""
    for wf in workflows:
        for k, v in _all_env_pairs(wf):
            ku = k.upper()
            sv = str(v).strip().lower()
            if "PROVIDER" in ku:
                assert "fake" in sv or sv in ("", "none", "false", "0"), (
                    f"{wf.name}: {k}={v!r} does not select the fake provider"
                )
            if re.search(r"(^|_)(LIVE|REAL)(_|$)", ku):
                assert sv in ("", "0", "false", "no", "off"), (
                    f"{wf.name}: {k}={v!r} enables a live mode"
                )


def test_playwright_and_pytest_get_fake_provider_if_a_provider_flag_exists(workflows):
    """Soft check: if CI sets a provider variable, the pytest job must see it."""
    for wf in workflows:
        provider_vars = [(k, v) for k, v in _all_env_pairs(wf) if "PROVIDER" in k.upper()]
        for k, v in provider_vars:
            assert "fake" in str(v).lower() or not str(v).strip()


def test_workflows_do_not_upload_env_or_media_artifacts(workflows):
    """Hard rules: no secrets, raw video, frames or GPX leave the machine."""
    for wf in workflows:
        for jid, job in wf.jobs.items():
            for step in job.get("steps") or []:
                if not h.uses_of(step).startswith("actions/upload-artifact"):
                    continue
                path = str((step.get("with") or {}).get("path", ""))
                assert not re.search(r"\.env\b|\.mov\b|\.mp4\b|\.gpx\b|frames", path, re.I), (
                    f"{wf.name}:{jid} uploads a sensitive path: {path!r}"
                )


def test_no_dotenv_with_key_is_created(workflows):
    for wf in workflows:
        for jid, job in wf.jobs.items():
            for step in job.get("steps") or []:
                for line in h.run_lines(step):
                    if re.search(r">>?\s*\.env\b|tee\s+\.env\b", line):
                        assert "KEY" not in line.upper(), (
                            f"{wf.name}:{jid} writes a key into .env"
                        )


def test_permissions_are_not_write_all(workflows):
    for wf in workflows:
        perms = wf.data.get("permissions")
        assert perms != "write-all", f"{wf.name} requests write-all permissions"
        for job in wf.jobs.values():
            assert job.get("permissions") != "write-all"


# ------------------------------------------------- repo consistency with CI

def test_lockfile_and_requirements_exist_for_ci():
    assert (h.REPO / "requirements.txt").exists()
    assert (h.REPO / "frontend" / "package-lock.json").exists()


def test_ci_does_not_depend_on_a_committed_env_file():
    gi = h.read(h.REPO / ".gitignore")
    assert re.search(r"^\.env\s*$", gi, re.M), ".env must stay git-ignored"
    assert not (h.WORKFLOW_DIR / ".env").exists()
