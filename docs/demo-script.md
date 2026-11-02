# Demo script (about 3 minutes)

Record this on your own machine with the local build, on the real blurred results. Everything shown is blurred; never open a raw frame on camera. Say only what the numbers support: no potholes were found in the footage, so do not claim pothole detection.

**Before recording**
- Backend on port 8000 and frontend on port 5173 running (see `README.md`, "Run").
- Open `http://localhost:5173/#map`. All 5 hazards should be listed.
- Review the 5 hazards first: Approve the real ones, Reject any wrong ones. Keep one approved for the report.
- If labels exist, run `python scripts/eval.py` and keep the table on screen for the end.

**1. The problem (20 s)**
"Road hazards like debris and potholes are reported late because someone has to notice them and fill in a form. A phone on the dashboard already sees them."

**2. The pipeline (40 s)**
Show the Overview section of the README or the pipeline list. Say: "One drive in. SAM 3.1 finds candidate hazards in the video, Muse Spark verifies each one, then it geotags and merges repeats. Models are only used for detection, verification and the privacy blur; everything else is plain code."
Mention the real run: two drives, about 49 minutes, 512 candidates, 5 verified hazards, about $14 of API spend.

**3. The map and review list (50 s)**
Show the hazard map. Point at the clusters between Gaithersburg and Rockville. Scroll the list: class, severity, sightings, coordinates, blurred photo. Click Approve on one and Reject on a wrong one if there is one. "A person always decides."

**4. Privacy (25 s)**
Point at the banner "Photos shown here are privacy-blurred." Say: "Faces and plates are blurred by SAM 3.1 before any image reaches this screen. Audio is stripped at the start. Unblurred frames stay on this machine."

**5. The prefilled report (30 s)**
Click Prefilled report on an approved hazard. Show the agency chosen from the location, the category, the description with coordinates and the blurred photo. Say: "It fills in the form. I submit it myself on the agency's website, RoadScout never does." Do not click Mark as reported unless you really filed it.

**6. Honest results and what's next (25 s)**
- Say what was measured: the review-based precision from your approvals, and the eval table if you have labels.
- Say what was not: no potholes in this footage, recall depends on hand labels.
- Next steps: on-device detection for real-time alerts with cloud verification, denser GPS tracks for better placement.

**Tips**
- Keep the cursor slow and the window at a readable size.
- If a hosted demo link exists, show it last and say it serves blurred images only.
