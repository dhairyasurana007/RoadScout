"""Helpers for the task 02 (CI workflow) tests.

Everything here is static: the workflow YAML files under .github/workflows are
parsed and inspected. Nothing is executed. Layout choices (one or several
workflow files, job names, step names) are not fixed by the task text, so the
helpers search across all workflows and jobs.
"""
from __future__ import annotations

import json
import os
import posixpath
import re
from pathlib import Path

import yaml


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
WORKFLOW_DIR = REPO / ".github" / "workflows"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def workflow_files() -> list[Path]:
    if not WORKFLOW_DIR.is_dir():
        return []
    return sorted(
        p for p in WORKFLOW_DIR.iterdir() if p.suffix in (".yml", ".yaml") and p.is_file()
    )


class Workflow:
    def __init__(self, path: Path):
        self.path = path
        self.text = read(path)
        self.data = yaml.safe_load(self.text)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def triggers(self):
        # PyYAML (YAML 1.1) parses the bare key `on` as boolean True.
        d = self.data
        if "on" in d:
            return d["on"]
        if True in d:
            return d[True]
        return None

    @property
    def jobs(self) -> dict:
        return self.data.get("jobs") or {}

    def defaults_dir(self, job: dict) -> str:
        for scope in (job, self.data):
            d = ((scope.get("defaults") or {}).get("run") or {}).get("working-directory")
            if d:
                return str(d)
        return "."


def load_workflows() -> list[Workflow]:
    return [Workflow(p) for p in workflow_files()]


def all_steps():
    """Yield (workflow, job_id, job, step_index, step) for every step."""
    for wf in load_workflows():
        for job_id, job in wf.jobs.items():
            for i, step in enumerate(job.get("steps") or []):
                yield wf, job_id, job, i, step


# ------------------------------------------------------------- trigger logic

MAIN_NAMES = {"main", "master"}


def trigger_names(wf: Workflow) -> set[str]:
    t = wf.triggers
    if isinstance(t, str):
        return {t}
    if isinstance(t, list):
        return {str(x) for x in t}
    if isinstance(t, dict):
        return {str(k) for k in t}
    return set()


def push_covers_every_push(wf: Workflow) -> bool:
    """True if `push` is a trigger with no filter that would skip ordinary pushes."""
    if "push" not in trigger_names(wf):
        return False
    t = wf.triggers
    cfg = t.get("push") if isinstance(t, dict) else None
    if not cfg:
        return True
    if "paths" in cfg or "paths-ignore" in cfg:
        return False
    branches = cfg.get("branches")
    if branches is not None:
        if isinstance(branches, str):
            branches = [branches]
        ok = any(b in ("**", "*") or b in MAIN_NAMES for b in branches)
        if not ok:
            return False
    ignore = cfg.get("branches-ignore")
    if ignore:
        if isinstance(ignore, str):
            ignore = [ignore]
        if any(b in MAIN_NAMES or b in ("**", "*") for b in ignore):
            return False
    if cfg.get("tags") and branches is None and "branches-ignore" not in cfg:
        return False  # tags-only
    return True


# --------------------------------------------------------------- command text

def run_lines(step: dict) -> list[str]:
    """Non-comment shell lines of a `run` step (line continuations joined)."""
    run = step.get("run")
    if not isinstance(run, str):
        return []
    joined = re.sub(r"\\\r?\n", " ", run)
    joined = re.sub(r"`\r?\n", " ", joined)  # PowerShell continuation
    out = []
    for line in joined.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def run_text(step: dict) -> str:
    return "\n".join(run_lines(step))


def uses_of(step: dict) -> str:
    u = step.get("uses")
    return u.lower() if isinstance(u, str) else ""


def step_effective_dirs(wf: Workflow, job: dict, step: dict) -> list[str]:
    """Candidate directories (repo-relative, posix) a step's commands run in.

    Combines working-directory (step, job defaults, workflow defaults) with any
    `cd X` or `--prefix X` / `-C X` in the command text. The base directory is
    always included, as is each cd target joined to it.
    """
    base = step.get("working-directory") or wf.defaults_dir(job) or "."
    base = posixpath.normpath(str(base).replace("\\", "/"))
    dirs = [base]
    text = run_text(step)
    for m in re.finditer(r"\bcd\s+(?:/d\s+)?[\"']?([\w./\\-]+)[\"']?", text):
        target = m.group(1).replace("\\", "/")
        dirs.append(posixpath.normpath(posixpath.join(base, target)))
    for m in re.finditer(r"(?:--prefix|--cwd|-C)[ =][\"']?([\w./\\-]+)[\"']?", text):
        target = m.group(1).replace("\\", "/")
        dirs.append(posixpath.normpath(posixpath.join(base, target)))
    return dirs


def dir_has(rel: str, name: str) -> bool:
    return (REPO / rel / name).exists()


def package_scripts(rel_dir: str) -> dict:
    p = REPO / rel_dir / "package.json"
    if not p.exists():
        return {}
    try:
        return json.loads(read(p)).get("scripts", {}) or {}
    except json.JSONDecodeError:
        return {}


# ------------------------------------------------------------ step classifiers

PIP_INSTALL_RE = re.compile(r"\bpip3?\s+install\b|-m\s+pip\s+install\b|\buv\s+pip\s+install\b")


def is_pytest_step(step: dict) -> bool:
    for line in run_lines(step):
        if PIP_INSTALL_RE.search(line):
            # e.g. `pip install -r requirements.txt pytest`; not a pytest run
            if not re.search(r"(&&|;|\|\|)\s*(python3?\s+-m\s+)?pytest\b", line):
                continue
        if re.search(r"(^|[\s;&|(])(python3?\s+-m\s+|py\s+-m\s+)?pytest\b", line):
            return True
    return False


def is_playwright_run_step(wf: Workflow, job: dict, step: dict) -> bool:
    for line in run_lines(step):
        if re.search(r"\bplaywright\s+test\b", line):
            return True
        m = re.search(r"\bnpm\s+(?:run|run-script)\s+([\w:.-]+)", line)
        if m:
            for d in step_effective_dirs(wf, job, step):
                script = package_scripts(d).get(m.group(1), "")
                if "playwright test" in script:
                    return True
        if re.search(r"\bnpm\s+(?:test|t)\b", line):
            for d in step_effective_dirs(wf, job, step):
                if "playwright test" in package_scripts(d).get("test", ""):
                    return True
    return False


def is_playwright_install_step(step: dict) -> bool:
    return any(re.search(r"\bplaywright\s+install(?!-)\b", l) for l in run_lines(step))


def is_npm_ci_step(step: dict) -> bool:
    return any(re.search(r"\bnpm\s+ci\b", l) for l in run_lines(step))


def is_pip_requirements_step(step: dict) -> bool:
    for line in run_lines(step):
        if PIP_INSTALL_RE.search(line) and re.search(r"(-r|--requirement)[ =]?\s*[\"']?(\./)?requirements\.txt", line):
            return True
    return False


def is_ffmpeg_provision_step(step: dict) -> bool:
    u = uses_of(step)
    if "ffmpeg" in u:
        return True
    for line in run_lines(step):
        low = line.lower()
        if "ffmpeg" in low and re.search(
            r"apt(-get)?\s+install|brew\s+install|choco\s+install|scoop\s+install|"
            r"winget\s+install|apk\s+add|dnf\s+install|yum\s+install|conda\s+install|"
            r"mamba\s+install|pip\s+install|npm\s+(i|install)|curl|wget|Invoke-WebRequest|"
            r"Install-Package",
            line,
            re.I,
        ):
            return True
    return False


def container_image(job: dict) -> str:
    c = job.get("container")
    if isinstance(c, str):
        return c.lower()
    if isinstance(c, dict):
        return str(c.get("image", "")).lower()
    return ""


# ------------------------------------------------------------- misc structure

def first_index(job: dict, pred) -> int | None:
    for i, step in enumerate(job.get("steps") or []):
        if pred(step):
            return i
    return None


def parse_version_floor(v) -> tuple[int, ...] | None:
    """Parse a literal like '3.12', 3.12, '20.x', '>=3.11' into a tuple, else None."""
    s = str(v).strip()
    if "${{" in s:
        return None
    m = re.search(r"(\d+)(?:\.(\d+|x))?", s)
    if not m:
        return None
    major = int(m.group(1))
    minor = int(m.group(2)) if m.group(2) and m.group(2).isdigit() else 0
    return (major, minor)
