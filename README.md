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

## GPX position lookup

`backend/gpx_track.py` parses the drive's GPX and answers "where was the car at video second N?":

```python
from backend.gpx_track import GpxTrack, GpxRangeError, parse_gpx

track = parse_gpx("track.gpx")                # same as GpxTrack.from_file
pos = track.position_at(12.5, video_start=started_at, clock_offset_seconds=0.0)
many = track.positions_at([0, 0.33, 0.67], video_start=started_at)

track = GpxTrack.from_drive(drive_record)   # drive dict or sqlite row; remembers started_at and offset
pos = track.position_at(12.5)               # also GpxTrack.from_drive_id(drive_id) to load from the database
```

- `pos` is a `TrackPosition` with `latitude`, `longitude` (degrees), `speed_mps` (metres per second), `heading_deg` (0 up to but not including 360, 0 = north, clockwise) and `timestamp` (timezone-aware UTC datetime on the GPX clock).
- GPX time = `video_start` + `clock_offset_seconds` + `video_seconds` (naive datetimes are taken as UTC). For a track from `from_drive`, `video_start` and the offset default to the drive's values; explicit arguments override them.
- Parsing streams the file (`iterparse`), ignores XML namespaces, merges all tracks and segments, accepts fractional seconds and any UTC offset, sorts points by time and drops points that repeat an earlier timestamp. Points without a usable time are skipped; a file with none raises `GpxError` (a `ValueError`).
- Latitude and longitude are linearly interpolated by time. Speed uses `<speed>` tags (m/s) when both neighbouring points have one, otherwise haversine distance over time for the surrounding segment. Heading is the direction of travel of that segment; while stationary (under 0.5 m per segment) the last heading is kept.
- A time before the first or after the last point raises `GpxRangeError` (a `ValueError`) unless `clamp=True`, which returns the first or last point.

## SAM 3.1 detection per chunk

`backend/detection.py` sends every preprocessed chunk to the model provider (`get_provider()`, so the fake provider unless `ROADSCOUT_PROVIDER` says otherwise) with the hazard class prompts, and stores the raw boxes, masks and track IDs in the `detections` table:

```python
from backend.preprocess import preprocess_video
from backend.detection import run_detection, load_detections

result = preprocess_video("drive.mov", "data/work/drive-1")
summary = run_detection("drive-1", result)       # also accepts frame_timestamps.json or a list of Chunk
rows = load_detections("drive-1")                # dicts; box is a list, mask a decoded RLE dict or None
```

- **Prompts** live in `config/hazard_prompts.json` (override the path with `ROADSCOUT_HAZARD_PROMPTS`): five classes (pothole, debris in lane, damaged guardrail, fallen or damaged sign, faded lane markings), each with a list of phrasings. The first phrasing of each class is the one sent, so to swap one, move it to the front. The file is validated on load (`PromptConfigError`, a `ValueError`, for a bad file). All five prompts go in one `detect` call per chunk.
- **Table `detections`**: `id`, `drive_id`, `chunk`, `frame_index` (global), `frame_in_chunk`, `frame_timestamp` (global seconds from the source video start), `hazard_class`, `class_prompt`, `track_id` (the provider's id, unique only inside its chunk), `track_key` (`"<chunk>:<track_id>"`, unique across the drive), `box` (JSON `[x1, y1, x2, y2]`), `mask` (RLE dict as JSON text, or NULL) and `score`.
- **Ids and times**: the provider counts frames from 0 in each chunk; `frame_index` = the chunk's `start_frame` + that, and `frame_timestamp` comes from `frame_timestamps.json` for that global index.
- **Empty results** are success and add no rows.
- **Idempotent and all-or-nothing**: a run gathers every chunk's detections first, then replaces the drive's old rows in a single transaction. If the provider fails on any chunk (after its own retries), `DetectionError` (a `ProviderError`; `chunk_index`, `chunk_path`, original error as `__cause__`) is raised, nothing is written and the drive's earlier detections stay as they were.

## Track aggregation and geotagging

`backend/candidates.py` collapses the stored detections into one candidate per tracked object, picks each one's best frame, saves the images and geotags it:

```python
from backend.candidates import aggregate_drive, load_candidates
from backend.gpx_track import GpxTrack

track = GpxTrack.from_drive(drive_record)          # or None: the drive is looked up in the drives table
summary = aggregate_drive("drive-1", result, gpx_track=track, min_frames=3)
rows = load_candidates("drive-1")
```

- **Arguments**: `chunks` is anything `run_detection` accepts (a `PreprocessResult`, `frame_timestamps.json` or its folder, a list of `Chunk`). `gpx_track` is a `GpxTrack`, a drive record, or None. A plain `GpxTrack` needs `video_start=`; `clock_offset_seconds=` overrides the offset. `min_frames` defaults to `ROADSCOUT_MIN_TRACK_FRAMES` (3). `output_dir` defaults to `data/storage/<drive id>/candidates/`.
- **One candidate per `(hazard_class, track_key)`**. Tracks are not stitched across chunks (a `track_key` is `"<chunk>:<track_id>"`); deduplication (task 14) merges the pieces. Tracks seen in fewer than `min_frames` distinct frames are dropped; exactly `min_frames` is kept.
- **Per candidate**: first and last timestamp, frames seen, and the largest mask area (box area for a detection with no mask).
- **Best frame**: `score = 0.4 * size + 0.3 * low + 0.3 * sharp`, where `size` is the detection's mask area over the track's largest, `low` is the box's bottom edge over the frame height (lower is closer to the car), and `sharp` is `v / (v + 100)` with `v` the Laplacian variance of the box region of the real frame. The weights are constants at the top of the module. Ties go to the earlier frame. The full frame and a crop of the box are saved as JPEGs (`c<chunk>_t<track id>_<class>_best.jpg` and `..._crop.jpg`).
- **Geotag**: the GPX position at the best frame's timestamp gives `lat`, `lon` and `heading`, with `geotag_status = 'ok'`. If that time is outside the GPX track the candidate is kept with NULL location, `geotag_status = 'out_of_range'` and the reason in `geotag_note` (nothing is clamped). A drive with no usable GPX gets `'no_gps'`.
- **Table `candidates`**: `id`, `drive_id`, `hazard_id` (NULL until dedup), `hazard_class`, `track_key`, `track_id`, `chunk`, `first_timestamp`, `last_timestamp`, `frames_seen`, `max_mask_area`, `best_frame_index`, `best_timestamp`, `best_box`, `best_frame_path`, `crop_path`, `lat`, `lon`, `heading`, `score`, `geotag_status`, `geotag_note`.
- **Idempotent and all-or-nothing**: re-running replaces the drive's candidates in one transaction (and removes their old images). A missing chunk video raises `CandidateError` and leaves the old candidates alone. No detections means zero candidates, not an error. It uses OpenCV (`opencv-python-headless`) to decode frames and measure sharpness; there are no network calls.

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

- `backend/` FastAPI app (`main.py`), settings (`config.py`), SQLite helpers (`db.py`), drive upload routes (`drives.py`), file storage (`storage.py`), ffprobe and GPX checks (`media.py`), ffmpeg preprocessing (`preprocess.py`), GPX parsing and position lookup (`gpx_track.py`), hazard prompts (`hazard_prompts.py`, `config/hazard_prompts.json`), SAM 3.1 detection per chunk (`detection.py`), track aggregation and geotagging (`candidates.py`)
- `frontend/` Vite React app; Playwright tests in `frontend/e2e/`
- `tests/unit/` pytest tests
- `.github/workflows/ci.yml` CI (pytest and Playwright)
- `docs/architecture-plan.md` the plan; `scripts/preflight.py` environment checks
