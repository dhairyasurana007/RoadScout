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

## Preprocessing

`backend/preprocess.py` turns any uploaded video into short, uniform chunks for detection:

```python
from backend.preprocess import preprocess_video
result = preprocess_video("drive.mov", "data/work/drive-1", fps=3, chunk_seconds=10, height=720)
```

One ffmpeg pass applies rotation metadata, strips audio and all other non-video streams and container metadata, scales to at most `height` pixels tall (never upscaled, even dimensions), resamples to a constant `fps`, and writes H.264 mp4 chunks `chunk_0000.mp4`, `chunk_0001.mp4`, ... of `chunk_seconds` each (the last may be shorter). `frame_timestamps.json` in the same directory maps every sampled frame to seconds from the source video start:

```json
{"version": 1, "fps": 3.0, "width": 1280, "height": 720, "chunk_seconds": 10.0, "chunk_frames": 30,
 "frame_count": 75, "duration_seconds": 25.0, "source_name": "drive.mov",
 "chunks": [{"index": 0, "file": "chunk_0000.mp4", "start_frame": 0, "frame_count": 30,
             "start_time": 0.0, "end_time": 10.0, "duration_seconds": 10.0}],
 "frames": [{"index": 0, "chunk": 0, "frame_in_chunk": 0, "timestamp": 0.0, "chunk_timestamp": 0.0}]}
```

The result has `chunks` (index, path, start_frame, frame_count, start_time, end_time), `frame_timestamps_path`, `fps`, `width`, `height`, `duration_seconds` and `frame_count`. Defaults can be set with `ROADSCOUT_PREPROCESS_FPS` (3), `ROADSCOUT_CHUNK_SECONDS` (10) and `ROADSCOUT_PREPROCESS_MAX_HEIGHT` (720). Bad input raises `InvalidMedia`, a missing ffmpeg raises `FfmpegUnavailable`, and an ffmpeg failure raises `PreprocessError`. The full format is documented in the module docstring.

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

- `backend/` FastAPI app (`main.py`), settings (`config.py`), SQLite helpers (`db.py`), drive upload routes (`drives.py`), file storage (`storage.py`), ffprobe and GPX checks (`media.py`), ffmpeg preprocessing (`preprocess.py`)
- `frontend/` Vite React app; Playwright tests in `frontend/e2e/`
- `tests/unit/` pytest tests
- `.github/workflows/ci.yml` CI (pytest and Playwright)
- `docs/architecture-plan.md` the plan; `scripts/preflight.py` environment checks
