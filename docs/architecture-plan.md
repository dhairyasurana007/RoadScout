# RoadScout: Architecture Plan

Oct 26, 2026 · @Dhairya

## Overview

RoadScout turns a dashboard-mounted phone into an automatic road inspector: record a drive, upload it, and get back a short list of confirmed hazards with photos, locations and severity, ready to report to VDOT (or MDOT SHA for Maryland state roads) in one tap. The design rule: deterministic steps do the heavy lifting, and a model is called only where judgment is needed.

**Why it matters:** drivers see hazards every day but almost never report them. You can't stop on a highway to take a photo, and you forget the location by the time you park. VDOT accepts reports at [my.vdot.virginia.gov](https://my.vdot.virginia.gov/) and relies on the public to find problems.

**In scope for the hackathon (due Nov 4, 9:00 PM ET):**

- Record video and GPS on a phone, upload after the drive
- Detect, segment and track hazards with SAM 3.1
- Verify and rate each detection with Muse Spark
- Deduplicate, geotag, blur faces and plates
- A review screen with a map, a hazard list and approve/reject
- A prefilled agency report (VDOT or MDOT SHA) per approved hazard
- An eval on hand-labeled segments of your own commute drives

**Out of scope (mention as future work):** real-time detection while driving, automatic submission to VDOT or MDOT SHA without a human tap, multi-driver crowdsourcing, fleet dashboards, incident capture.

## Event and project context

- **Event:** Meta Global AI Developer Hackathon (Cerebral Valley), remote.
- **Deadline:** project submission due Wednesday, Nov 4, 2026 at 9:00 PM ET. Judging Nov 4 to 8, finals Nov 9.
- **Track:** Indie developer (solo). Judged on creativity, technical ambition, and whether it works. A narrow pipeline that works reliably on real footage beats a big feature list that breaks in the demo.
- **Theme:** "Push the Frontier": giving everyday drivers a new way to monitor and improve physical infrastructure.
- **Required tech:** Meta's models through the Meta Model API, specifically SAM 3.1 and Muse Spark.
- **Builder:** an AI product engineer in Centreville, VA, who commutes to Rockville, MD on I-66 and I-495, building on a local machine.
- **Repo:** [dhairyasurana007/RoadScout](https://github.com/dhairyasurana007/RoadScout), private, `main` branch, set up with the plan, `CLAUDE.md`, the tester and coder agents and the preflight script. `gh` is authenticated as `dhairyasurana007`.

## System architecture

A drive flows through eleven steps from the phone to a prefilled VDOT report. Three steps call a model: SAM 3.1 for detection, Muse Spark for verification, and SAM 3.1 again for the privacy blur.

```
Phone capture (video + GPX)
  → Ingestion (upload, store, queue job)
  → Preprocess (ffmpeg: downsample, chunk)          [deterministic]
  → SAM 3.1 (detect, segment, track)                [MODEL CALL]
  → Track aggregation (one candidate per object)    [deterministic]
  → Geotag (GPS at the best frame)                  [deterministic]
  → Muse Spark (real? class, severity)              [MODEL CALL]
  → Deduplicate (merge nearby sightings)            [deterministic]
  → Privacy blur (faces + plates, via SAM 3.1)      [MODEL CALL, saved frames only]
  → Review UI (map, approve or reject)
  → VDOT report (prefilled; a human submits)
```

Everything runs after the drive, on the backend; the phone only records and uploads.

## Component details

Three steps call a model (detection, verification and the privacy blur); everything else is plain code you can unit-test.

1. **Capture.** Video plus a GPS track with timestamps.
   - Fastest start: the phone's camera app plus a GPS logger that exports GPX, synced by clock time. Start recording on Tue Oct 27 this way.
   - Upgrade if time allows: a small Expo app that records both and uploads in one tap.
   - Rule for the demo: start recording before you drive and never touch the phone while moving.
2. **Ingestion.** Upload video and GPX to the backend, store the raw files, create a `drive` record, queue a processing job.
3. **Preprocessing (deterministic).** ffmpeg downsamples to about 720p and a few frames per second, then splits the drive into short chunks. Later (task 13): drop stretches where GPS speed is near zero (red lights, traffic jams) to save cost.
4. **Detection and tracking (SAM 3.1).** Send each chunk with text prompts for the hazard classes: pothole, debris in lane, damaged guardrail, fallen or damaged sign, faded lane markings. Keep the prompts in a config file so phrasings are easy to swap. SAM 3.1 returns boxes, masks and object identities that persist across frames, streamed one record per frame; an empty result means nothing was found, not an error. Prompt wording matters, so test a few phrasings per class early.
5. **Track aggregation (deterministic).** Collapse each tracked object into one candidate: first and last timestamp, frames seen, largest mask. Drop tracks seen in too few frames. Pick a best frame by mask size, position low in the frame (closest to the car) and sharpness (for example Laplacian variance).
6. **Geotagging (deterministic).** Interpolate the GPS track at the best frame's timestamp. Keep heading and direction of travel so the same hazard on the other side of the highway isn't merged with it.
7. **Verification (Muse Spark).** For each candidate, send the best frame, a crop and one or two neighbouring frames. Ask for structured JSON: `is_hazard`, `class`, `severity` (1 to 3), `lane`, `confidence`, `reason` (one line) and `report_description` (short). Reject below a confidence threshold. This is the only judgment call in the pipeline.
8. **Deduplication (deterministic).** Merge candidates of the same class within roughly 25 m in the same direction of travel, within a drive and across drives. Repeat sightings raise confidence; a hazard not seen on several later drives can be marked as possibly fixed.
9. **Privacy blur.** Run SAM 3.1 with `face` and `license plate` prompts on the saved frames only, blur those masks, and use only blurred images in the review UI and reports. Raw video is kept locally (git-ignored, never committed) so the eval and reruns work; a Delete video button on each drive removes it on request.
10. **Review UI.** A map with a pin per hazard, a list with photo, class, severity and sightings, and approve, edit or reject buttons.
11. **Reporting.** For each approved hazard, produce a prefilled report: agency category, location, description and blurred photo. The agency defaults from the state boundary: Virginia hazards go to VDOT ([my.vdot.virginia.gov](https://my.vdot.virginia.gov/)), and Maryland hazards go to MDOT SHA's [service request form](https://mdotsha.my.salesforce-sites.com/customercare), which covers state roads only (county and city streets are handled by the county or city). The user can change the agency on the report screen, since the app can't tell who maintains a given street. Hazards elsewhere (for example DC) get no report. Never auto-submit during the hackathon: test reports would send real crews to check them. Show the prefilled report in the demo instead.

## Data model and storage

Six tables cover the whole pipeline; SQLite is enough for one driver, with Postgres as the upgrade if you want hosted multi-user.

| Table | Key fields | Written by |
| --- | --- | --- |
| `drives` | id, started\_at, clock offset, video path, GPX path, status | Ingestion |
| `detections` | id, drive\_id, chunk, frame timestamp, class prompt, track id, box, mask | SAM 3.1 detection |
| `candidates` | id, drive\_id, hazard\_id, class, track id, first/last timestamp, frames seen, best-frame path, crop path, lat, lon, heading | Track aggregation, geotagging, deduplication |
| `verdicts` | candidate\_id, is\_hazard, class, severity, lane, confidence, reason, report\_description, model, prompt version | Verification |
| `hazards` | id, class, lat, lon, direction, severity, confidence, sightings, first/last seen, photo path, status (new, approved, rejected, reported, possibly fixed) | Deduplication, review UI |
| `reports` | hazard\_id, agency (VDOT or MDOT SHA), category, description, photo path, created\_at | Reporting |

Store the prompt version and model name on every verdict so you can compare runs. Files (raw video, frames, blurred photos) live on disk or in object storage, keyed by drive id.

## Tech stack

Recommended default: a Python backend, a small React web app, and no GPUs of your own, since SAM 3.1 and Muse Spark run on the Meta Model API. The default stack stands; Meta API details and fixture format are settled in task 26.

| Layer | Choice | Why |
| --- | --- | --- |
| Models | SAM 3.1 (`sam-3.1`) and Muse Spark via the Meta Model API | Required by the hackathon; hosted, no GPU setup |
| Backend | Python, FastAPI | Video tooling (ffmpeg, OpenCV) is best in Python |
| Jobs | A simple background worker (FastAPI background tasks or RQ) | One drive at a time is fine for a demo |
| Video and geo | ffmpeg, OpenCV, gpxpy | Downsampling, frames, sharpness, GPX parsing |
| Database | SQLite | Zero setup |
| Frontend | React (Vite) with Leaflet and OpenStreetMap tiles | Map plus list review screen |
| Unit tests | pytest | Written by the tester agent |
| UI tests | Playwright | UI behavior only |
| CI | GitHub Actions | Runs against recorded fixtures |
| Capture | Camera app plus GPX logger; Expo app as a stretch | Start collecting footage on day one |
| Hosting | Local for development; one small cloud box for the demo | A live link helps judges |
| Fallback | Open-weights SAM 3 on Modal or Baseten with partner credits | If API rate limits or cost bite |
| Coding | Muse Code alongside your usual tools | Encouraged by the organizers |

## Meta Model API: what is known

- SAM 3.1 is model `sam-3.1`, called through the Meta Model API's Responses API with Bearer token auth. Input: an image or video plus a short text prompt. Output: boxes, segmentation masks and persistent object IDs across video frames. Streaming is available for video: one record per frame, followed by a completion event.
- **Not confirmed:** exact endpoint URLs, the Muse Spark model ID, rate limits, video length caps, how video is passed (upload, URL or base64), the streaming record shape, whether Muse Spark has a JSON or structured-output mode, pricing and credits, and whether one call accepts several text prompts. Get these from Meta's official docs or from the user. Do not invent them.
- The API key comes from the environment (`META_API_KEY`, or whatever the user provides) through a git-ignored `.env`. Never commit it, and never paste it into chat.

## Evaluation plan

The headline number is precision on your own commute drives, with and without the Muse Spark check; that comparison proves the verification step earns its place.

- **Record whole drives.** Record the full 35 to 50 minute commute at 1080p and 30 fps (not 4K; the pipeline downsamples to 720p). Nothing is cut into clips.
- **Labeled set.** You label about 10 segments of 2 to 3 minutes each, as time ranges inside a few full drives. Mix 6 or 7 segments containing hazards (some from rougher local roads around Centreville, since interstates may have few hazards) with 3 or 4 clean stretches. One small label file per drive: for each hazard, a start/end timestamp, class, severity and a hazard ID that stays the same when the same hazard appears in another drive. This set measures recall and the SAM 3.1 vs SAM 3.1 plus Muse Spark comparison.
- **Precision from the review screen.** Run the full pipeline on several whole drives and approve or reject what it finds (task 17). The approve rate is the precision, on a much larger sample than the labeled segments.
- **Supplement.** Use images from a public road-damage dataset such as RDD2022 to test detection on potholes and cracks you may not catch on camera.
- **Metrics.**
  - Event-level precision and recall per class
  - False alarms per hour of driving (what a real driver would feel), measured on the reviewed full drives
  - Ablation: SAM 3.1 alone vs SAM 3.1 plus Muse Spark, on the labeled segments
  - Dedup accuracy: pick a few hazards you can see on repeated drives of the same road; each should merge into one hazard
  - Cost and processing time per minute of video
- **Prompt experiments.** Several phrasings per class for SAM 3.1, and several verification prompts for Muse Spark, versioned so runs can be compared. Run them on the labeled segments only, not whole drives, to save credits.
- **Demo artifacts.** A results table, side-by-side raw and annotated video, and the map of everything found.

## Risks and mitigations

The biggest risk is detection quality at highway speed, so test SAM 3.1 on real footage (the spike in task 26) before building anything that depends on it.

| Risk | Mitigation |
| --- | --- |
| SAM 3.1 misses hazards in fast, blurry or glare-heavy footage | Run the task 26 spike as soon as the key and a clip are ready; record in daylight; clean the windshield; lower the sampled frame rate only where speed is low |
| Sampled frame rate too low at highway speed (about 10 m between frames at 3 fps), so real hazards get dropped as short tracks | The task 26 spike compares 2, 3 and 5 fps and sets the minimum track length from real data |
| Too many false positives (shadows, patches, tar seams) | Muse Spark verification plus a minimum track length |
| Few real hazards on I-66 and I-495 | Add local roads; supplement with a public road-damage dataset |
| API limits, cost (up to $150 in Meta credits) or unknown video length caps | Short chunks, lower frame rate, skip stationary stretches, cache results; spend counter with a hard cap in the API client; the task 26 spike measures real cost per call; open-weights SAM 3 on partner credits as fallback |
| GPS places the hazard a few metres off | Tag at the frame where the object is closest; show the photo so a human confirms |
| Personal data in footage | Strip audio at preprocessing, blur faces and plates, use only blurred frames in the UI and reports, keep raw video local with a delete button, never commit it. Unblurred frames are sent to Muse Spark for verification (accepted); say so in the README |
| Accidentally filing fake VDOT or MDOT SHA reports | Prefill only; no automatic submission in the hackathon build |
| Distracted driving while filming | Start recording before the drive, mount the phone, never touch it while moving |

## Agent workflow

Every task from 1 on runs through the same loop (task 0 and the spike in task 26 are the exceptions): a tester and a coder sub-agent are spawned before implementation starts, and nothing is pushed until the tester's hidden unit tests, any UI Playwright tests, and CI all pass.

1. **Start.** The main agent picks the next task in the build order (see MVP tasks). The tester and the coder get the same task text at the same time.
2. **Tester writes secret unit tests.** It writes them from the task's description alone, outside the repo working tree (for example `../roadscout-hidden-tests/task-XX/`), so they aren't visible in the coder's workspace. The coder is told never to read that directory. Isolation is by convention only; there are no permission rules.
3. **Coder implements.** It builds the task. If the task has a UI, it also writes Playwright tests for that UI (renders, forms, buttons, navigation). Playwright is for UI only: it is not used to test the core pipeline (detection, verification, geotagging, dedup). You verify the core pipeline yourself on real footage. Backend-only tasks get no Playwright tests.
4. **Coder fixes until green.** If the Playwright tests (UI tasks) fail, the coder keeps fixing until they pass.
5. **Handoff.** For UI tasks, the coder hands its Playwright tests to the tester. For backend-only tasks, the coder just reports done.
6. **Tester reviews.** It runs its secret unit tests against the implementation and, for UI tasks, reads the Playwright tests for gaps it missed. Any issue goes back to the coder, who fixes it and hands back. This repeats until everything passes.
7. **Push.** The tester reports a pass to the main agent, which copies the tester's unit tests into the repo (for example `tests/unit/`), then makes one commit for the task with its code, unit tests and (UI tasks only) Playwright tests, using the task's commit message, and pushes to GitHub.
8. **Watch CI.** The main agent monitors GitHub Actions (for example `gh run watch`). If a run fails, it sends the failure to the tester and the coder, and the loop resumes from step 6. If it passes, the task is ticked off and the next one starts.

**Communication:** the tester and coder report to the main agent and talk to each other only through it. The main agent relays messages between them, including questions about interfaces, names or schemas, without revealing the contents of the hidden tests.

**Hard rules**

- The coder never edits or reads the tester's unit tests.
- CI runs against the fake provider and recorded Meta API responses (fixtures), never live calls. Live calls happen only in manual runs.
- Playwright tests cover UI behavior only. The core pipeline is verified by the tester's unit tests and by you on real footage.
- One task = one commit. Don't bundle tasks.
- If one task loops more than about three times without passing, the main agent stops and flags it to you instead of burning time.
- Never commit secrets, raw dashcam video, or unblurred frames.
- Tests use a short synthetic clip generated with ffmpeg at test time, never real footage.
- Unblurred frames stay on your machine; the only exception is the Muse Spark verification call. Until task 15, the local results page shows unblurred best frames. The hosted demo (task 23) serves blurred images only, so task 15 must be done before task 23.
- Never submit anything to VDOT or MDOT SHA.

## MVP tasks

The MVP is done when you can upload one real commute clip with its GPX file and see a list of verified hazards, each with a photo, on a web page. No map, dedup, blur or reporting yet. Task 0 is a one-time preflight and setup check done outside the tester/coder loop (the agents do not exist yet). Tasks 1 to 12 are each one commit and run through the agent workflow above. Task 26, which holds the footage spike, is the first task that needs the API key and real footage; the MVP is only done once it and the exit check pass. Tasks 1 to 11 need neither.

**Build order.** The goal is to build as much as possible without API access. Task numbers are IDs, not the build order. Work in this order:

1. **No API key or real footage needed** (everything runs against the fake provider and test videos): task 0, then tasks 1 to 11, then 13, 14, 15 (blur logic against synthetic masks), 16, 17, 18, 19 and 22.
2. **Needs the key and a real clip:** task 26 (live provider and spike), then the MVP exit check, then tasks 12, 20 and 21, which need live calls and your labels.
3. **Last:** tasks 23 and 24, and task 25 only if time allows.

- [x] **0. `chore: preflight and project setup`** A one-time setup and checklist, done by the main agent with you, outside the tester/coder loop because the agents don't exist yet. Keep the checks in a re-runnable script (`scripts/preflight.py`). Report pass or fail for each item, and stop and ask you if anything fails:
  - **GitHub:** `gh auth status` shows `dhairyasurana007` with the `repo` and `workflow` scopes; `gh repo view dhairyasurana007/RoadScout` works; GitHub Actions is enabled on the repo.
  - **Local repo:** this folder is a git repo on `main` with `origin` set to the GitHub repo, and a `.gitignore` covering `.env`, raw video and footage, extracted frames, `node_modules`, virtualenvs and the hidden-tests folder. First commit and push: the plan (as `docs/architecture-plan.md`), `CLAUDE.md` (workflow, hard rules, task list), the agent files and the preflight script.
  - **Sub-agents:** `.claude/agents/tester.md` and `.claude/agents/coder.md` exist, load, and encode their roles from the Agent workflow section, and a smoke dispatch confirms both respond. The hidden-tests folder (`../roadscout-hidden-tests/`) exists outside the repo working tree, and the coder's file tells it never to read it.
  - **Toolchain:** git, Python 3, Node and npm, ffmpeg and ffprobe on PATH (and able to decode an HEVC `.mov`), pytest and the Playwright browsers installable, and enough free disk space for multi-GB footage.
  - **Secrets:** `.env` is git-ignored. `META_API_KEY` is not required yet; report it as not set, which is fine until task 26.
- [x] **1. `chore: scaffold repo`** FastAPI backend, React (Vite) frontend with a hello-world page, SQLite, pytest and Playwright configured with one Playwright smoke test of that page (so CI always has a UI test to run), a `.env.example` for the Meta API key, a README with run commands (including how to install ffmpeg on Windows).
- [x] **2. `ci: add GitHub Actions for unit and Playwright tests`** Runs pytest and Playwright on every push. CI never calls the live Meta API; it uses the fake provider and, once they exist, recorded fixtures.
- [x] **3. `feat: model provider interface with a fake provider`** Define our own internal result types for detection (per-frame boxes, masks, track IDs) and verification (the verdict fields), and a provider interface for SAM 3.1 and Muse Spark. Ship a fake provider that returns canned, deterministic results (including an empty result, a short track and a false positive), a recorded-fixture provider that replays saved responses, and a placeholder live provider, chosen by a config switch. Add retries, timeouts, a cost log and a running spend counter with a hard cap (configurable, default well under the credit budget) for live calls. Everything downstream is built and tested against the fake provider, so no API key or real footage is needed until task 26.
- [x] **4. `feat: drive upload endpoint and drives table`** Upload a video and a GPX file, save them, create a drive record with a status. Accept any common video container or codec (iPhone .mov/HEVC, Android .mp4, dashcam files). Read the video's start time from its metadata, and let the user override it with a manual clock offset to line it up with the GPX. Store files through a small storage interface (local disk by default, so S3 can be swapped in later), and support multi-GB uploads without loading the file into memory. Add a minimal upload page (file pickers, offset field, submit, shows the drive status) so the pipeline can be driven from a browser from here on, with Playwright tests for it. Tests use a short synthetic clip generated with ffmpeg, not real footage.
- [x] **5. `feat: preprocessing with ffmpeg`** Normalize any input (HEVC/H.264, variable frame rate, rotation metadata) to a standard format, strip the audio track so conversations are never stored or sent anywhere, downsample to about 720p and a few frames per second, split into short chunks, save frame timestamps.
- [x] **6. `feat: GPX parsing and position lookup`** Parse the track and return interpolated latitude, longitude, speed and heading for any video timestamp, applying the clock offset from task 4.
- [x] **7. `feat: SAM 3.1 detection per chunk`** Send each chunk with the hazard class prompts from a config file; store raw boxes, masks and track IDs. Calls SAM 3.1 through the provider interface from task 3.
- [x] **8. `feat: track aggregation and best-frame selection`** One candidate per tracked object; drop short tracks; pick the best frame; geotag it.
- [x] **9. `feat: Muse Spark verification`** Structured JSON verdict per candidate (real, class, severity, lane, confidence, reason), with model and prompt version stored on each verdict. Calls Muse Spark through the provider interface from task 3.
- [ ] **10. `feat: pipeline runner`** Run tasks 5 to 9 as one background job, update drive status, record errors.
- [ ] **11. `feat: minimal results page`** Extend the upload page (task 4) with live drive status and a list of verified hazards with photo, class, severity and location. The photos are unblurred until task 15, so this page stays local.
- [ ] **12. `feat: eval script v0`** Read the hand labels (time ranges inside full drives) for a few segments, compute precision and recall per class, print a table.
- [ ] **26. `feat: live Meta provider and footage spike` (needs the API key and a real clip)** **The first task that needs a working Meta API key and one real commute clip with its GPX file.** Start with a footage spike, done by the main agent and you outside the tester/coder loop: a throwaway script (under `spike/`) samples frames at 2, 3 and 5 fps and sends them to SAM 3.1 with the hazard prompts, plus a few Muse Spark verification calls on sample candidate frames. Record how many frames each hazard appears in at highway speed, how detection holds up in glare and blur, which phrasings work, and the cost of every call, extrapolated to a full commute and to the eval set against the up-to-$150 credit budget. Findings go in `docs/spike-notes.md` and set the real defaults for frame rate, minimum track length and prompts (tasks 5, 7 and 8), and they decide whether the open-weights fallback is needed. Then implement the live provider, mapping Meta's responses into the internal types from task 3, and record fixtures from the real responses. Commit the spike script, notes and fixtures with this task, never the footage. After it, run the MVP exit check.

**MVP exit check (after task 26):** run your first real commute recording through it and look at what comes back. That tells you whether SAM 3.1 prompts need work before the evals and prompt experiments start.

## Final tasks

The final build adds everything judges see in the demo: the map, review, privacy, reports and eval numbers. Same rule: one task, one commit, through the agent workflow.

- [ ] **13. `feat: skip stationary stretches`** Drop frames where GPS speed is near zero before calling SAM 3.1.
- [ ] **14. `feat: deduplicate hazards`** Merge candidates of the same class nearby and in the same direction, within and across drives; count sightings; add the hazards table.
- [ ] **15. `feat: privacy blur`** Run SAM 3.1 face and license plate prompts on saved frames, blur the masks, use only blurred images in the review UI and reports, and add a Delete video button on each drive that removes its raw video and extracted frames on request. Raw video is otherwise kept locally and never committed. Playwright tests cover the button.
- [ ] **16. `feat: map view`** Leaflet with OpenStreetMap tiles, a pin per hazard, linked to the list.
- [ ] **17. `feat: review actions`** Approve, edit class or severity, and reject from the UI; hazard status updates.
- [ ] **18. `feat: prefilled agency report (VDOT / MDOT SHA)`** For approved hazards, generate the agency category, location, description and photo, with copy and download buttons. Default the agency from the hazard's latitude and longitude (Virginia: VDOT; Maryland: MDOT SHA state roads), let the user change it, and mark hazards in other jurisdictions "no agency". Check the real field lists of the VDOT and MDOT SHA forms when building this (the MDOT SHA page doesn't list them). No automatic submission.
- [ ] **19. `feat: possibly-fixed detection`** Mark a hazard as possibly fixed when later drives pass it without a sighting.
- [ ] **20. `feat: eval v1`** Ablation of SAM 3.1 alone vs plus Muse Spark on the labeled segments; precision and false alarms per hour from the reviewed full drives (task 17 approve/reject); dedup accuracy on hazards seen across repeated drives; cost and processing time per minute; results written to a markdown report.
- [ ] **21. `feat: prompt experiments`** Run several SAM 3.1 phrasings and verification prompts against the labeled segments only (not whole drives, to save credits); keep the best as defaults.
- [ ] **22. `feat: annotated video export`** Side-by-side raw and annotated video for the demo. Run the face and plate blur on the exported segments only (a few short clips, to save credits), so no unblurred faces or plates appear in the demo.
- [ ] **23. `chore: deploy demo`** One small cloud server, seeded with the blurred results of your real drives only (no raw video, no API key on the server), live link for judges. Requires task 15.
- [ ] **24. `docs: README and demo script`** Architecture summary, setup, eval results, and the walkthrough you'll record.
- [ ] **25. `feat: Expo capture app` (stretch)** One-tap recording of video and GPS with upload; only if everything above is done.

## Decisions and open questions

**Decided**

- Solo entry on the Indie developer track.
- The user records on an iPhone, but the app must accept footage from any device (tasks 4 and 5).
- The default stack (Python, FastAPI, React, SQLite) stands; Meta API details and fixture format are settled in task 26.
- Build as much as possible without API access: everything is built against a fake provider first. A real-footage spike in task 26 settles frame rate and prompts, and task 26 is the first task that needs the API key and footage.
- Playwright is for UI only.
- Tester and coder talk only through the main agent. Tester isolation is by convention; no permission rules.
- Sending unblurred frames to Muse Spark for verification is accepted; mention it in the README.
- The user can work on this during the day, so hours per day are not a constraint.
- The user records whole 35 to 50 minute drives at 1080p/30 (not 4K) and does the hand-labeling: about 10 segments of 2 to 3 minutes, as time ranges inside those drives. Precision comes from approve/reject clicks on the review screen.
- The user has up to $150 in Meta credits. Feasibility is unknown until the task 26 spike measures real cost per call.
- Audio is not needed; it is stripped at preprocessing.
- Raw video is not deleted automatically. It stays on the user's machine (never committed), and each drive has a Delete video button.
- Reports go to VDOT for Virginia hazards and to MDOT SHA (Maryland State Highway Administration) for Maryland state roads. Streets maintained by counties or cities are out of scope; the user can change the agency on the report screen.
- Files go through a storage interface, local disk by default. S3 (presigned uploads) is an option later, mainly if the hosted demo accepts uploads or the Meta API takes video by URL.
- The user handles scheduling; the plan has no schedule.
- The GitHub repo is `dhairyasurana007/RoadScout` (private, created).

**Still open**

- [ ] Meta Model API access: the key, endpoint URLs, Muse Spark model ID, rate limits, video length caps, credits, and whether SAM 3.1 accepts several text prompts in one call. Blocks task 26.
- [ ] Is the $150 credit pool shared by SAM 3.1 and Muse Spark, and does it expire?
- [ ] Demo as a hosted web app with a live link, or a recorded walkthrough of a local build? Suggested: record the walkthrough regardless, and host a read-only demo seeded with already-blurred results, with no public uploads and no API key on the server.
- [ ] Off-the-shelf recorder plus GPX logger (the current default), or build the Expo capture app early?

Sources: [Meta Global AI Developer Hackathon](https://cerebralvalley.ai/e/meta-ai-developer-hackathon) · [SAM 3.1 on the Meta Model API (Mind-Verse)](https://www.mind-verse.de/news/meta-sam-3-1-bild-video-analyse-api) · [VDOT report a road problem](https://my.vdot.virginia.gov/) · [MDOT SHA customer service](https://roads.maryland.gov/mdotsha/pages/Index.aspx?PageId=519)
