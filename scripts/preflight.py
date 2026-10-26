#!/usr/bin/env python3
"""RoadScout preflight checks (task 0). Safe to re-run; exits 1 if any check fails.

Checks GitHub (gh auth, repo, Actions), the local repo and .gitignore, the
tester/coder agent files, the hidden-tests folder, the toolchain, free disk
space and whether META_API_KEY is set (never prints its value).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "dhairyasurana007/RoadScout"
ACCOUNT = "dhairyasurana007"
ROOT = Path(__file__).resolve().parent.parent
HIDDEN = ROOT.parent / "roadscout-hidden-tests"
MIN_FREE_GB = 20

failures = 0


def run(cmd):
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as exc:  # missing binary, timeout
        return 1, str(exc)


def check(name, ok, detail=""):
    global failures
    if not ok:
        failures += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def info(name, detail=""):
    print(f"[INFO] {name}" + (f": {detail}" if detail else ""))


def github():
    print("== GitHub")
    code, out = run(["gh", "auth", "status"])
    check("gh authenticated as " + ACCOUNT, code == 0 and ACCOUNT in out)
    check("gh token has repo and workflow scopes", "'repo'" in out and "'workflow'" in out)
    code, out = run(["gh", "repo", "view", REPO, "--json", "name,visibility"])
    check("repo reachable", code == 0, out if code else "")
    code, out = run(["gh", "api", f"repos/{REPO}/actions/permissions"])
    enabled = False
    if code == 0:
        try:
            enabled = bool(json.loads(out).get("enabled"))
        except ValueError:
            pass
    check("GitHub Actions enabled", enabled)


def local_repo():
    print("== Local repo")
    code, _ = run(["git", "rev-parse", "--is-inside-work-tree"])
    check("folder is a git repo", code == 0)
    if code != 0:
        return
    _, branch = run(["git", "branch", "--show-current"])
    check("on branch main", branch == "main", branch)
    _, origin = run(["git", "remote", "get-url", "origin"])
    check("origin points at " + REPO, REPO.lower() in origin.lower(), origin)
    for path in [".env", "footage/a.mov", "a.mp4", "a.gpx", "frames/a.jpg", "roadscout-hidden-tests/x.py"]:
        code, _ = run(["git", "check-ignore", "-q", path])
        check(f".gitignore ignores {path}", code == 0)
    for path in [".env.example", "tests/fixtures/a.gpx"]:
        code, _ = run(["git", "check-ignore", "-q", path])
        check(f".gitignore allows {path}", code != 0)


def agents():
    print("== Sub-agents")
    for name in ("tester", "coder"):
        f = ROOT / ".claude" / "agents" / f"{name}.md"
        text = f.read_text(encoding="utf-8") if f.exists() else ""
        check(f"{name}.md exists with matching frontmatter", text.startswith("---") and f"name: {name}" in text)
    coder = ROOT / ".claude" / "agents" / "coder.md"
    ctext = coder.read_text(encoding="utf-8") if coder.exists() else ""
    check("coder.md forbids reading the hidden tests", "roadscout-hidden-tests" in ctext and "Never read" in ctext)
    check("hidden-tests folder exists outside the repo", HIDDEN.is_dir() and ROOT not in HIDDEN.parents, str(HIDDEN))


def toolchain():
    print("== Toolchain")
    for tool in ("git", "python", "node", "npm", "ffmpeg", "ffprobe", "gh"):
        check(f"{tool} on PATH", shutil.which(tool) is not None)
    code, out = run(["ffmpeg", "-hide_banner", "-decoders"])
    check("ffmpeg can decode HEVC", code == 0 and " hevc " in out)
    code, _ = run([sys.executable, "-m", "pip", "--version"])
    check("pip available", code == 0)
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    check(f"at least {MIN_FREE_GB} GB free disk", free_gb >= MIN_FREE_GB, f"{free_gb:.1f} GB free")
    info("pytest and Playwright browsers", "checked after task 1 installs them")


def secrets():
    print("== Secrets")
    info("META_API_KEY", "set" if os.environ.get("META_API_KEY") else "not set (fine until task 26)")


if __name__ == "__main__":
    github()
    local_repo()
    agents()
    toolchain()
    secrets()
    print(f"\n{failures} check(s) failed" if failures else "\nAll checks passed")
    sys.exit(1 if failures else 0)
