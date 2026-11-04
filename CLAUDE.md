# RoadScout

Hackathon project (Meta Global AI Developer Hackathon, due Wed Nov 4, 2026, 9:00 PM ET, solo Indie track). A dashboard-mounted phone records a drive; the backend finds road hazards (SAM 3.1), verifies them (Muse Spark), geotags and dedups them, blurs faces and plates, and prefills a VDOT (Virginia) or MDOT SHA (Maryland state roads) report for a human to submit.

The full plan is `docs/architecture-plan.md`. Read it before starting any task. It holds the architecture, data model, task list (0 to 26), build order, decisions and open questions. If the user tells you something that conflicts with it, the user wins; update the plan.

## Core rules

- Deterministic code does the heavy lifting. A model is called only for detection (SAM 3.1), verification (Muse Spark) and the privacy blur (SAM 3.1).
- Do not invent Meta API details (endpoints, Muse Spark model ID, limits, request shapes). Get them from Meta's docs or ask the user.
- Build against the fake provider (task 3) until task 26; no API key or real footage is needed before then.

## Workflow for every task from 1 on

Spawn a `tester` and a `coder` sub-agent (definitions in `.claude/agents/`). You, the main agent, relay all messages between them. They never talk directly.

1. Pick the next task in the build order. Give the tester and the coder the same task text.
2. Tester writes secret unit tests from the task description alone, outside the repo at `../roadscout-hidden-tests/task-XX/`. The coder never reads or edits that directory.
3. Coder implements the task. For tasks with a UI, the coder also writes Playwright tests for that UI. Playwright is UI-only; never use it to test the core pipeline. Backend-only tasks get no Playwright tests.
4. Coder fixes until its Playwright tests (UI tasks) pass.
5. For UI tasks, coder hands its Playwright tests to the tester. Backend-only tasks: coder reports done.
6. Tester runs its hidden tests against the implementation and, for UI tasks, reviews the Playwright tests. Issues go back to the coder through you. Repeat until everything passes.
7. When the tester reports a pass, copy the tester's unit tests into `tests/unit/`, make one commit for the task (code, unit tests, UI-task Playwright tests) with the task's commit message, and push.
8. Watch GitHub Actions (`gh run watch`). On failure, send it to the tester and the coder and resume from step 6. On success, tick the task off in the plan and start the next one.

If one task loops more than about three times without passing, stop and flag it to the user.

Task 0 (preflight) and the footage spike in task 26 are done by the main agent with the user, outside this loop.

## Hard rules

- One task = one commit. Do not bundle tasks.
- CI runs against the fake provider and recorded fixtures, never the live Meta API. Live calls only in manual runs.
- Tests use short synthetic clips generated with ffmpeg, never real footage.
- Never commit secrets, real dashcam video, real GPX tracks, extracted frames, or unblurred images. `.gitignore` covers these; do not work around it. One exception, decided by the user on 2026-11-03: the `demo/` folder holds the hosted demo dataset (privacy-blurred JPEGs and a stripped database), built only by `scripts/build_demo_dataset.py` and guarded by tests; `.gitignore` has explicit exceptions for exactly those paths.
- Unblurred frames stay on the user's machine (the only exception is the Muse Spark verification call). The hosted demo (task 23) serves blurred images only.
- Never auto-submit anything to VDOT or MDOT SHA.
- Never paste or print `META_API_KEY`. It lives in a git-ignored `.env`.

## Build order

1. No API key or real footage needed: task 0, tasks 1 to 11, then 13, 14, 15, 16, 17, 18, 19, 22.
2. Needs the key and a real clip: task 26 (live provider and spike), the MVP exit check, then 12, 20, 21.
3. Last: 23, 24, and 25 only if time allows.

## Environment notes

- Windows 11, PowerShell primary. Python 3, Node and npm, ffmpeg and ffprobe are installed.
- Re-run `python scripts/preflight.py` to check GitHub, the repo, the agents and the toolchain.
- Repo: https://github.com/dhairyasurana007/RoadScout (private).
