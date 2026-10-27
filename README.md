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

### Skipping stationary stretches

To save cost, detection drops frames recorded while the car was not moving (red lights, traffic jams). `backend/stationary.py` computes the GPS speed (`GpxTrack.position_at`) at each sampled frame's global timestamp, using the drive's `started_at` and `clock_offset_seconds`; a frame is stationary when its speed is **below** the threshold (a speed exactly at the threshold counts as moving).

- **Settings**: `ROADSCOUT_SKIP_STATIONARY` (`0`/`false`/`off`/`no` turn it off, anything else leaves it on) and `ROADSCOUT_STATIONARY_SPEED_MPS` (default `0.5`; a garbage value falls back to the default). `flag_stationary_frames(timestamps, gpx_track, ...)` in `backend/stationary.py` returns one bool per timestamp. `run_detection(..., skip_stationary=None, stationary_speed_mps=None, gpx_track=None, video_start=None, clock_offset_seconds=None, storage=None)` and `run_pipeline(..., skip_stationary=None, stationary_speed_mps=None)` override them; `gpx_track` is a `GpxTrack` or a drive record (default: the drive's row in the database).
- **Whole chunk stationary**: the chunk is not sent to the provider at all (no call, no cost). It adds no rows and is listed in `DetectionSummary.skipped_chunks`.
- **Partly stationary chunk**: the chunk is still sent once, unchanged (chunks are never re-encoded), and detections the provider reports on its stationary frames are dropped before they are stored. Frame indices and timestamps stay global, so candidates, geotagging and verification work as before.
- **Fails open**: no GPX, an unreadable GPX, or a frame outside the GPX time range counts as moving and is never dropped; the run still succeeds and `stationary_note` says why. A drive that is stationary throughout produces zero detections and ends `done`.
- **Counts**: `DetectionSummary` has `frames_total`, `stationary_frames_skipped`, `chunks_total`, `chunks_sent`, `stationary_chunks_skipped`, `skipped_chunks`, `stationary_detections_dropped` and `stationary_note`. The pipeline also stores `frames_total`, `stationary_frames_skipped`, `stationary_chunks_skipped`, `chunks_sent` and `stationary_note` on the `drives` row (integer columns defaulting to 0, added by an idempotent migration), so `GET /api/drives/{id}` returns them next to the existing keys. They are 0 until detection finishes and reset to 0 when a run starts, so a re-run never accumulates; they are 0 too when the filter is off or the GPS is unusable.

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

## Muse Spark verification

`backend/verification.py` asks Muse Spark (through the provider interface, so the fake provider unless `ROADSCOUT_PROVIDER` says otherwise) whether each candidate is a real hazard, and stores one structured verdict per candidate in the `verdicts` table:

```python
from backend.verification import verify_drive, load_verdicts, load_accepted_candidates

summary = verify_drive("drive-1", chunks=result)         # chunks only needed for the neighbouring frames
rows = load_verdicts("drive-1")                           # every verdict, rejected ones too (accepted_only=True to filter)
hazards = load_accepted_candidates("drive-1")             # accepted candidates, each with a "verdict" dict
```

`verify_drive(drive_id, *, chunks=None, provider=None, min_confidence=None, db_path=None, prompt=None, prompts=None)`:

- **Images per candidate** (in this order): the best frame, the crop (left out if there is none), and the frames just before and after the best frame cut from the chunk video (needs `chunks`; a neighbour that does not exist is skipped). Unblurred frames are sent to Muse Spark here; this is the one place they leave your machine.
- **Prompt**: `config/verification_prompt.json` (override the path with `ROADSCOUT_VERIFY_PROMPT`) holds `{"version", "prompt", "schema"}`. The `prompt` may use `{detector_class}`, `{classes}` and `{schema}` placeholders. It is passed to the provider as `verify(images, class_prompt=<detector class>, prompt_version=<version>, prompt_text=<prompt>)`. The wording and request shape are placeholders until task 26. Change `version` whenever the prompt changes: it is stored on every verdict.
- **Verdict**: `is_hazard`, `class`, `severity` (1 to 3, 0 only for a non-hazard), `lane`, `confidence` (0 to 1), `reason`, `report_description`. A malformed answer (not a verdict, bad severity or confidence, a class not in `config/hazard_prompts.json`) raises `VerificationError` naming the candidate id.
- **Accepted** means `is_hazard and confidence >= min_confidence` (default 0.5; set `ROADSCOUT_VERIFY_MIN_CONFIDENCE` or pass `min_confidence`). Rejected verdicts are stored too, for the eval ablation.
- **Table `verdicts`**: `candidate_id` (primary key, deleted with its candidate), `drive_id`, `is_hazard`, `hazard_class`, `severity`, `lane`, `confidence`, `reason`, `report_description`, `model`, `prompt_version`, `accepted`, `created_at`.
- **Idempotent and all-or-nothing**: re-running replaces the drive's verdicts in one transaction. If the provider fails or answers badly for any candidate, `VerificationError` (`candidate_id`, original error as `__cause__`) is raised and nothing is written. A drive with no candidates is a no-op.

## Pipeline runner

`backend/pipeline.py` runs preprocessing, detection, aggregation, verification, privacy blurring and deduplication for one drive as a single job and records progress and errors on the drive:

```python
from backend.pipeline import run_pipeline
result = run_pipeline(drive_id)      # synchronous; returns a PipelineResult, status "done" or "failed"
```

API: `POST /api/drives/{id}/process` (no body) starts the job in the background and returns `202` with the drive record straight away (`status: processing`, `stage: queued`). It returns `404` for an unknown drive and `409` if the drive is already processing (the claim is one atomic `UPDATE`, so concurrent requests cannot start two runs). A `done` or `failed` drive can be processed again; its results are replaced. Poll `GET /api/drives/{id}` for progress.

- **Status** goes `uploaded` -> `processing` -> `done` or `failed`. **Stage** (`stage` column and JSON key) is `queued`, `preprocessing`, `detecting`, `aggregating`, `verifying`, `blurring`, `deduplicating`, then `done` or `failed`. Both are committed before each stage starts.
- **Errors**: on any exception the drive gets `status: failed`, `stage: failed`, `failed_stage` (the stage that raised) and `error` = `"<stage>: <message>"`, on one line, cut to 500 characters, with the Meta key and bearer tokens replaced by `***`. No stack trace goes into the API. `error` is NULL on success and cleared when a run starts. Other columns added to `drives`: `updated_at`, `finished_at` (existing databases are migrated by `init_db`).
- **What is kept after a failure**: each run first clears the drive's old detections, candidates (with verdicts and images) and work folder, so a failure at stage S leaves only the results of the stages before S from this run, and nothing stale. Each stage is all-or-nothing. A GPX that is out of range or unusable is not a failure: candidates get `geotag_status` `out_of_range` or `no_gps` and the drive ends `done`.
- **Files** live under `data/storage/<drive id>/`: the upload (never modified or deleted), `work/` (chunks and `frame_timestamps.json`, kept after the run) and `candidates/` (best-frame and crop JPEGs, unblurred and local only, plus their blurred copies `*_blur.jpg`). Only the Delete video button removes the upload (see below).
- The provider is `get_provider()` at run time (fake unless `ROADSCOUT_PROVIDER` says otherwise), one instance for detection and verification. fps, chunk length and height come from the `ROADSCOUT_PREPROCESS_*` / `ROADSCOUT_CHUNK_SECONDS` settings. If the server stops mid-run, the drive is marked `failed` ("interrupted") at the next start. Run a single server process.

## Hazard deduplication

`backend/dedup.py` merges the accepted, geotagged candidates of all drives into one row per real-world hazard (table `hazards`). It runs as the pipeline stage `deduplicating` after `verifying`, or on its own:

```python
from backend.dedup import deduplicate_drive, load_hazards, set_hazard_status
summary = deduplicate_drive("drive-1")      # DedupSummary: candidate_count, hazard_count, created/updated/removed_count, hazards
hazards = load_hazards()                    # dicts ordered by id (status= / hazard_class= filters)
set_hazard_status(1, "approved")            # new, approved, rejected, reported, possibly_fixed
```

`deduplicate_drive(drive_id, *, db_path=None, distance_m=None, heading_tolerance_deg=None)`; `rebuild_hazards(...)` is the same without a drive.

- **Takes part**: candidates with an accepted `is_hazard` verdict and `geotag_status = 'ok'` (lat and lon present). The privacy blur must not have failed (`blur_status` is not `failed`). Their class is the verdict's class. Others keep `hazard_id` NULL.
- **Same hazard**: same class, haversine distance <= `distance_m` (default 25, env `ROADSCOUT_DEDUP_DISTANCE_M`) and circular heading difference <= `heading_tolerance_deg` (default 45, env `ROADSCOUT_DEDUP_HEADING_TOLERANCE_DEG`; 350 and 10 differ by 20). A candidate without a heading never merges. Parameters beat env vars; both are read at call time. Hazards are the connected components of this relation (single linkage over all drives), so the result does not depend on row order and chaining is allowed: a row of candidates 20 m apart is one hazard.
- **Values**: `sightings` = distinct drives; `candidate_count` = member candidates; `severity` = max; `confidence` = noisy-OR over drives of each drive's best confidence, `1 - prod(1 - c)`, never below the best single one and at most 1; `direction` = circular mean of headings; `lat`/`lon` = confidence-weighted centroid; `photo_path`/`crop_path` = the highest-scoring member's BLURRED paths (`blurred_frame_path`/`blurred_crop_path`; NULL if it has none, never the unblurred ones); `first_seen`/`last_seen` = drive `started_at` + `best_timestamp` (UTC ISO).
- **Table `hazards`**: `id`, `hazard_class`, `lat`, `lon`, `direction`, `severity`, `confidence`, `sightings`, `candidate_count`, `first_seen`, `last_seen`, `photo_path`, `crop_path`, `status` (default `new`, CHECK-constrained), `created_at`, `updated_at`. `candidates.hazard_id` links each member.
- **Re-runs**: idempotent, one transaction. Surviving hazards keep their id, status and `created_at`; new sightings join an existing hazard; hazards with no member candidates left are deleted; other drives' hazards are recomputed to the same values. `clear_results` and `aggregate_drive` reconcile the table when they delete candidates, so a failed re-run leaves no hazard without candidates. If a drive is reprocessed and its only hazards had no other drive's sightings, they come back with new ids.
- **Pipeline**: a failure raises `deduplicating: ...` on the drive (`failed_stage = 'deduplicating'`); the stage is rolled back as a whole.

## Privacy blur

`backend/privacy.py` blurs faces and license plates in the saved frames. It runs as the pipeline stage `blurring` (after `verifying`, before `deduplicating`), or on its own:

```python
from backend.privacy import blur_drive, blur_image, load_privacy_prompts
summary = blur_drive("drive-1")                              # BlurSummary: ok_count, failed_count, mask_count, candidates
blur_image("frame.jpg", "frame_blur.jpg", [rle_mask, ...])   # src, dst, masks
```

- **Prompts**: `config/privacy_prompts.json` (`{"version": 1, "prompts": ["face", "license plate"]}`; override the path with `ROADSCOUT_PRIVACY_PROMPTS`).
- **What is sent**: only the saved best-frame and crop JPEGs of ACCEPTED candidates, never the video, with `provider.detect(<jpeg path>, ["face", "license plate"])`, one call per image (SAM 3.1 on a still image; there is no extra provider method, and the fake provider recognises an image path by its extension and returns a face and a plate mask, or nothing in the `no_masks` scenario). The masks of all returned detections are combined (a detection without a mask counts as its box). A valid result with no detections is success: the copy is unchanged.
- **Blur**: each masked area is pixelated into about five cells across and smoothed with a Gaussian, inside the mask only. `blur_drive` grows each mask by `ROADSCOUT_BLUR_DILATE_PX` pixels (default 4); `blur_image` defaults to 0 (exactly the mask). Masks of another size than the image are scaled to it. An empty mask list gives an unchanged copy. The original is never modified.
- **Files and columns**: `c0000_t1_pothole_best.jpg` becomes `c0000_t1_pothole_best_blur.jpg` (crop: `..._crop_blur.jpg`) in the same `candidates/` folder. The `candidates` table gets `blurred_frame_path`, `blurred_crop_path`, `blur_status` (`ok` or `failed`; NULL for rejected candidates) and `blur_note`, added by the idempotent migration in `init_db`.
- **Fails closed**: if the provider raises, returns something that is not a detection result, or a mask is invalid, that candidate gets `blur_status = 'failed'`, no blurred file and nothing is served for it. `blur_drive` records all results in one transaction and then raises `BlurError`; the pipeline ends the drive `failed` with `failed_stage = 'blurring'` and `error = 'blurring: ...'`. Re-running replaces the drive's blurred files without leaving stale ones.
- **Everything downstream uses the blurred images**: `hazards.photo_path` / `crop_path` are the blurred paths (never the unblurred ones; a candidate whose blur failed does not take part in deduplication), and the photo route serves only blurred images.
- **Unblurred frames** are kept locally in `data/storage/<drive id>/candidates/` and leave the machine only for the Muse Spark verification call (the one accepted exception). Delete video removes them.

## Delete video

`DELETE /api/drives/{id}/video` (the **Delete video** button on the drive panel) removes the drive's raw video, the `work/` folder (chunks and `frame_timestamps.json`) and the unblurred candidate JPEGs. It keeps the GPX, all database rows (detections, candidates, verdicts, hazards) and the blurred images, so the results page keeps working. Raw video is never deleted any other way and never committed.

- Returns `200` with the updated drive JSON. `video_deleted` (bool, in every drive JSON) and `video_deleted_at` say whether it happened; `video_path` keeps the stored key and `status` is not changed. New nullable `drives.video_deleted_at` column (idempotent migration).
- `404` for an unknown drive, `409` while the drive is processing (the check and the mark are one atomic `UPDATE`). Deleting again returns the same `200` (and sweeps leftovers), and files that are already gone are not an error. If a file cannot be removed it answers `500` ("Could not delete everything. Call again to retry.", with no path; the details are in the server log) and a second call retries.
- After deletion `POST /api/drives/{id}/process` returns `409` ("the raw video of this drive was deleted"), the provider is not called and nothing is changed; this holds for a drive that was never processed too.
- UI: the button opens an inline confirmation ("Delete video?" / "Yes, delete" / "Cancel"); it is disabled while the drive is processing and shows "Video deleted" afterwards, with Start processing disabled.

## Results page (local only)

The upload page doubles as the results page (`backend/results.py`, `frontend/src/App.jsx`). After an upload (or after typing a drive ID under "Open an existing drive") it shows the drive status and a **Start processing** button (`POST /api/drives/{id}/process`; disabled while the drive is processing). While the drive is `processing` the page polls `GET /api/drives/{id}` every 1.5 s (add `?pollMs=50` to the page URL to shorten it, minimum 10) and shows the stage; a `failed` drive shows the stage and error text. When the drive is `done` it lists the verified hazards, each with photo, class, severity (1 Low, 2 Medium, 3 High), confidence and location (5 decimals, or "no location"), or "No verified hazards".

**Only privacy-blurred photos are shown** (see Privacy blur); a note on the page says so and that the raw video stays on your machine. A drive panel also has the Delete video button.

API (read-only):

- `GET /api/drives/{id}/hazards` returns `200` with a JSON list of the drive's accepted hazards (verdict `accepted` and `is_hazard`), in drive-time order, or `[]`. `404` for an unknown drive. Each item has `candidate_id` (also as `id`), `drive_id`, `hazard_class` (the verdict's class), `detector_class`, `severity` (1 to 3), `confidence` (0 to 1), `lane`, `reason`, `report_description`, `lat`, `lon`, `heading` (all `null` unless `geotag_status` is `ok`; the hazard is still listed), `geotag_status`, `best_timestamp`, `first_timestamp`, `last_timestamp`, `frames_seen`, `blur_status`, `photo_url` and `crop_url` (`null` when there is no crop). No file paths are returned.
- `GET /api/drives/{id}/hazards/{candidate_id}/photo` serves the BLURRED best-frame JPEG (`image/jpeg`, `Cache-Control: no-store`); add `?variant=crop` for the blurred crop. It never serves an unblurred file: `404` for an unknown drive or candidate, a candidate of another drive, a rejected or non-hazard candidate, a candidate whose `blur_status` is not `ok` (no blurred path), a missing file, or a stored path outside `data/storage/<drive id>/candidates/`. A non-integer candidate id is `422`.

## Hazard map (local only)

The **Hazard map** tab on the page (`#map` opens it directly) shows the deduplicated hazards of all drives (`backend/hazards.py`, `frontend/src/HazardMap.jsx`). It uses Leaflet 1.9.4 (pinned) with OpenStreetMap raster tiles (`https://tile.openstreetmap.org/{z}/{x}/{y}.png`, attribution "© OpenStreetMap contributors"). Tiles are fetched by your browser from OpenStreetMap; nothing else leaves the machine and no photo is sent anywhere.

- One pin per hazard, coloured by status (new blue, approved green, rejected grey, reported purple, possibly fixed orange) and labelled with the severity. Pins are Leaflet `divIcon`s, so Leaflet's default marker images are not used. Clicking a pin opens a popup (class, severity, status, sightings, blurred photo), highlights the matching list item and scrolls it into view.
- Next to the map, a list with thumbnail, class, severity, status, sightings and location. Clicking an item highlights it, zooms to its pin (at least zoom 15) and opens the popup. Clicking an empty part of the map clears the selection.
- The view starts fitted to all pins; with no hazards it shows the Northern Virginia / Maryland commute area (centre 38.9, -77.2, zoom 9). A status filter and a **Refresh** button reload from the API. States: loading, "No hazards yet", and an error message.
- Selected state is in data attributes (`data-selected` on pins and list items, `data-selected-hazard-id` on the section); the map container carries `data-map-lat`, `data-map-lon`, `data-map-zoom`.

API (the list and photo routes only read; the review route is below):

- `GET /api/hazards` returns `200` with a bare JSON list ordered by `id`, `[]` when empty, all statuses unless filtered. Items: `id`, `hazard_class`, `lat`, `lon`, `direction`, `severity`, `confidence`, `sightings`, `candidate_count`, `first_seen`, `last_seen`, `status`, `photo_url` (`/api/hazards/{id}/photo`) and `crop_url` (`/api/hazards/{id}/photo?variant=crop`). Both URLs are `null` when the hazard has no such image or the file cannot be served, so the list never points at a 404. Hazards without a position are not listed. No file paths are returned. Filters: `?status=` (`new`, `approved`, `rejected`, `reported`, `possibly_fixed`; anything else is `422`) and `?drive_id=` (hazards with a member candidate in that drive; unknown drive gives `[]`).
- `GET /api/hazards/{id}/photo[?variant=best|crop]` serves the BLURRED photo stored in `hazards.photo_path` (or `crop_path`) as `image/jpeg` with `Cache-Control: no-store`. Fail closed, `404` unless the hazard exists, the path is set, and the file exists directly in `<storage>/<drive id>/candidates/` and is not the resolved `best_frame_path` or `crop_path` of any candidate of any drive (the unblurred originals). The check uses the candidates table, not file names. A non-integer id or an unknown `variant` is `422`; other query parameters are ignored.

### Review actions

Every list item and every popup has **Approve**, **Reject**, **Reset** (back to new) and **Edit** buttons (`aria-label`s name the hazard, for example "Approve hazard 3"). Edit opens an inline form with a class select (the five classes of `config/hazard_prompts.json`) and a severity select (1 to 3); **Save** is enabled only when something changed and sends only the changed fields, **Cancel** closes the form. The status shows as a badge, the pin is recoloured as soon as the server answers, and an "edited" marker appears next to a class or severity that was changed by hand (its tooltip shows the automatic value). Under a status filter a hazard that no longer matches leaves the list. While a request for a hazard is in flight its buttons are disabled (a double click sends one request) and other hazards stay usable. Errors stay on the hazard and the previous value is kept: 422 shows the server message, 500 the detail or the status code, a lost connection "Could not reach the server."; a 404 ("Hazard not found") also refreshes the list and shows a notice. Nothing is ever submitted to VDOT or MDOT SHA from here.

`PATCH /api/hazards/{id}` (JSON body, every field optional, at least one required):

- `status`: `new`, `approved` or `rejected`. `reported` and `possibly_fixed` are set by later tasks and are a `422` here.
- `hazard_class`: exactly one of the class names in `config/hazard_prompts.json` (for example `debris in lane`).
- `severity`: an integer 1 to 3 (`true`, `2.0`, `"2"`, `null` are `422`).
- Unknown fields (`lat`, `id`, `photo_path`, ...), `null` values, `{}`, an empty or non-object body and malformed JSON are `422`; an unknown hazard is `404` (`{"detail": "Hazard not found"}`); validation comes before the lookup, and a refused request changes nothing.
- `200` returns the updated hazard in the shape of a `GET /api/hazards` item. Those items now also carry `reviewed_at`, `class_edited`, `severity_edited` (booleans), `original_hazard_class` and `original_severity` (the automatic value while edited, else `null`).
- Any current status can be reviewed (a class or severity edit never changes the status). The change is one `BEGIN IMMEDIATE` transaction on a short-lived connection with a 30 s busy timeout, so parallel updates all succeed and leave a valid row. `updated_at` moves on every successful request; `reviewed_at` is set when the status becomes approved or rejected and cleared when it becomes new.
- DB columns added to `hazards` by `init_db` (idempotent migration, existing rows get the defaults): `reviewed_at`, `class_edited` (0/1), `severity_edited` (0/1), `original_hazard_class`, `original_severity`. Setting a class or severity back to the automatic value clears the flag and the original. The originals are kept for the eval (approve rate and edit rate per class).
- **Sticky edits**: deduplication (`deduplicate_drive`, `rebuild_hazards`, pipeline re-runs) never overwrites an edited class or severity, each field independently, and keeps the hazard id; it still rewrites sightings, confidence, position, direction, photo, first/last seen and `candidate_count`. Deduplication never clears an edit flag either (not even when the automatic value momentarily equals the edit because one drive was re-processed); only a PATCH that sets the value back to the automatic one does. Clustering ignores edits (it uses the members' own verdict classes), and a hazard is matched to its cluster through its automatic class, so a re-run never duplicates or crashes on an edited hazard.
- **Review work survives re-processing (adoption)**: when a drive is re-processed its candidates are deleted and recreated. A hazard with review work (status other than `new`, or an edited class or severity) that has lost all its members is kept ("parked") instead of deleted, and the next deduplication run lets it adopt an otherwise unclaimed new cluster of the same automatic class whose position is within the dedup distance (default 25 m) of the old hazard's position and whose direction is within the heading tolerance (default 45 degrees) of its direction; nearest pairs first, one cluster per hazard and one hazard per cluster. The adopted hazard keeps its id, status, `created_at`, `reviewed_at` and edits. A decided hazard nothing adopts (re-detected farther away, in another direction or as another class) is deleted, like every hazard without review work. A parked hazard stays in the table until the next run, so a failed re-run can leave one without candidates.
- **Known behaviour**: the review buttons stay enabled on `reported` and `possibly_fixed` hazards, so Reject or Reset on a reported hazard overwrites that status (the later reporting task will decide whether to lock them).
- **UI races**: a list reply that arrives after a review finished is dropped and the list is loaded again, so it never shows the old status; a list loaded while a review is in flight is shown and loaded again once the reviews settle. The in-flight state survives leaving and re-entering the map tab, so the hazard's buttons stay disabled.

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

- `backend/` FastAPI app (`main.py`), settings (`config.py`), SQLite helpers (`db.py`), drive upload routes (`drives.py`), file storage (`storage.py`), ffprobe and GPX checks (`media.py`), ffmpeg preprocessing (`preprocess.py`), GPX parsing and position lookup (`gpx_track.py`), hazard prompts (`hazard_prompts.py`, `config/hazard_prompts.json`), SAM 3.1 detection per chunk (`detection.py`), track aggregation and geotagging (`candidates.py`), Muse Spark verification (`verification.py`, `verification_prompt.py`, `config/verification_prompt.json`), pipeline runner (`pipeline.py`), privacy blur (`privacy.py`, `config/privacy_prompts.json`), Delete video (`video_deletion.py`), results routes (`results.py`), hazard map routes (`hazards.py`)
- `frontend/` Vite React app (upload/results page in `src/App.jsx`, hazard map in `src/HazardMap.jsx`); Playwright tests in `frontend/e2e/`
- `tests/unit/` pytest tests
- `.github/workflows/ci.yml` CI (pytest and Playwright)
- `docs/architecture-plan.md` the plan; `scripts/preflight.py` environment checks
