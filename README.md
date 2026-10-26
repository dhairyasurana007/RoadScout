# RoadScout

A dashboard-mounted phone records a drive. The backend finds road hazards (SAM 3.1), verifies them (Muse Spark), geotags and dedups them, blurs faces and plates, and prefills a VDOT or MDOT SHA report for a human to submit. See `docs/architecture-plan.md` for the full plan.

Stack: Python + FastAPI, React (Vite), SQLite, pytest, Playwright, ffmpeg.

## Prerequisites

Python 3.11+, Node 20+ with npm, and ffmpeg and ffprobe on PATH.

### Install ffmpeg on Windows

Using winget (PowerShell):

```powershell
winget install Gyan.FFmpeg
```

Alternatives: `choco install ffmpeg` or `scoop install ffmpeg`. Open a new terminal afterwards and check:

```powershell
ffmpeg -version
ffprobe -version
```

## Setup

```powershell
# Backend (from the repo root)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Frontend
cd frontend
npm install
npx playwright install chromium
cd ..

# Secrets: copy the template and fill in META_API_KEY (only needed from task 26)
Copy-Item .env.example .env
```

`.env` is git-ignored. Never commit or paste the key.

## Run

Backend (http://localhost:8000, health check at `/health`, docs at `/docs`):

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

Frontend (http://localhost:5173):

```powershell
cd frontend
npm run dev
```

The SQLite file is created on startup at `data/roadscout.sqlite` (override with `ROADSCOUT_DB`). Uploaded video and GPX files go to `data/storage/<drive id>/` (override with `ROADSCOUT_STORAGE_DIR`); both folders are git-ignored.

## Uploading a drive

With the backend and the frontend running, open http://localhost:5173, pick a video (any container or codec ffprobe can read, for example iPhone `.mov`/HEVC or Android `.mp4`) and a `.gpx` file, optionally set a clock offset, and press Upload. The Vite dev server proxies `/api` to the backend on port 8000.

API:

- `POST /api/drives` multipart form with `video` (file), `gpx` (file) and optional `clock_offset_seconds` (number, default 0). Returns 201 with the drive record. Bad input returns 4xx and nothing is stored.
- `GET /api/drives/{id}` one drive; `GET /api/drives` all drives, newest first.

The video start time (`started_at`) comes from the container's `creation_time`. If the video has none, it falls back to the first GPX trackpoint time, then to the upload time; `started_at_source` and `started_at_fell_back` say which. `clock_offset_seconds` is added to video time to get GPX time, and `effective_started_at` shows the result. Uploads are streamed to disk in 1 MiB chunks, so multi-GB files are not held in memory.

## Tests

pytest (unit tests, from the repo root):

```powershell
python -m pytest
```

Playwright (UI tests only; starts the Vite dev server itself):

```powershell
cd frontend
npx playwright test
```

## CI

`.github/workflows/ci.yml` runs on every push and pull request with two jobs:

- `pytest`: installs ffmpeg and `requirements.txt`, then runs `python -m pytest`.
- `playwright`: `npm ci` in `frontend/`, installs Chromium, runs `npx playwright test`, and uploads the report and traces as an artifact if the run fails.

The workflow sets `ROADSCOUT_PROVIDER=fake` so CI uses the fake provider (and recorded fixtures once they exist). It uses no secrets and never calls the live Meta API; live calls happen only in manual runs on your machine.

## Layout

- `backend/` FastAPI app (`main.py`), settings (`config.py`), SQLite helpers (`db.py`), drive upload routes (`drives.py`), file storage (`storage.py`), ffprobe and GPX checks (`media.py`)
- `frontend/` Vite React app; Playwright tests in `frontend/e2e/`
- `tests/unit/` pytest tests
- `.github/workflows/ci.yml` CI (pytest and Playwright)
- `docs/architecture-plan.md` the plan; `scripts/preflight.py` environment checks
