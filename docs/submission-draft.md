# RoadScout: submission text (draft)

Draft written before the hackathon's submission rules were seen. Rewrite the headings to match the form once the rules are known. Anything in `[brackets]` is a number or fact to fill in after the evaluation run (`docs/eval-report.md`).

## One-line pitch

A phone on your dashboard records your commute; RoadScout finds road hazards in the video, pins them on a map with a blurred photo, and prefills the state report so you only have to review and submit it.

## The problem

Reporting a road hazard to VDOT (Virginia) or MDOT SHA (Maryland) means noticing it, remembering where it was, finding the right form, and typing it all in. Most drivers never do. Agencies learn about hazards late, from complaints, instead of from the thousands of people who drive past them every day.

## What it does

1. You upload a dashcam clip and the GPS track recorded alongside it.
2. **SAM 3.1** looks for hazard candidates in the video: debris in the lane, damaged guardrails, fallen or damaged signs, faded lane markings, potholes.
3. **Muse Spark** looks at the best frame of each candidate and decides whether it really is a hazard, how severe it is, and writes a plain description.
4. Deterministic code geotags each hazard from the GPS track, merges sightings of the same hazard (25 m radius), and checks on later drives whether a hazard is still there.
5. Faces and licence plates are blurred before any image reaches the interface. The app only serves blurred images.
6. A person reviews each hazard (approve or reject) and gets a prefilled VDOT or MDOT SHA report to copy and submit. **Nothing is ever submitted automatically.**

## How it is built

- Python / FastAPI backend, SQLite, a resumable pipeline with a checkpoint after every video chunk and every verification, and a spend cap on model calls.
- A provider layer with a fake provider (used by all tests and CI), a fixture provider, and the live Meta provider. CI never calls the live API.
- React + Vite frontend with a map, a review queue and a report panel.
- Deterministic code does the heavy lifting. Models are used only for detection (SAM 3.1), verification (Muse Spark) and the privacy blur (SAM 3.1).
- Built by one person with a tester agent and a coder agent working from a written plan. The tester writes its unit tests from the task description alone, before it sees the code.

## Privacy by design

- The pipeline fails closed: if a frame cannot be blurred, it is not shown.
- The only unblurred frame that leaves the machine is the one sent to Muse Spark for verification.
- No real footage, GPS tracks or unblurred images are ever committed to the repository.

## What we measured

[Fill from docs/eval-report.md: hazards found per drive, precision from review clicks, recall on hand-labelled segments, false alarms per hour of clean road, what the Muse Spark verification step removes, cost and processing time per minute of video.]

Real run on two commutes (about 49 minutes of video, Maryland and Virginia): 512 candidates found by SAM 3.1, 5 confirmed as hazards by Muse Spark (debris on the shoulder), 0 on the second drive. Cost about `$14` for those runs.

## Honest limitations

- **No potholes were found** because none were in our footage. We do not claim pothole detection.
- The verifier sometimes accepts an object on the shoulder as "debris in lane". [State here whether the tightened verification prompt fixed this, and by how much.]
- Phone GPS from a timeline export is coarse (about one point a minute), so a hazard's position can be off by tens of metres. Hazards seen twice can fail to merge.
- The VDOT report form is a JavaScript app, so its field names are not verified; the MDOT SHA fields come from the live form.
- Speed: verification takes 10 to 15 seconds per candidate, which dominates run time.

## What's next

- A native capture app that records video and a 1 Hz GPS track together.
- Re-checking known hazards on every drive, so fixed ones close themselves.
- Measuring recall on a larger labelled set and across seasons and lighting.

## Built with

Meta SAM 3.1, Meta Muse Spark, Python, FastAPI, SQLite, ffmpeg, React, Vite, Leaflet, Playwright, pytest, GitHub Actions.
