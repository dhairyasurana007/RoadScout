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

The SQLite file is created on startup at `data/roadscout.sqlite` (override with `ROADSCOUT_DB`).

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

## Layout

- `backend/` FastAPI app (`main.py`), settings (`config.py`), SQLite helpers (`db.py`)
- `frontend/` Vite React app; Playwright tests in `frontend/e2e/`
- `tests/unit/` pytest tests
- `docs/architecture-plan.md` the plan; `scripts/preflight.py` environment checks
