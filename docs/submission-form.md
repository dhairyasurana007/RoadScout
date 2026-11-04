# Submission form: answers to paste

Fields as seen in the Cerebral Valley "Submit Project" form (starred fields are required). Draft text is ready to paste; the choice questions are the submitter's own opinions, so for those this file only lists what our run actually showed. `docs/submission-draft.md` holds the longer write-up this was cut from. Numbers come from `docs/eval-report.md` and `docs/spike-notes.md`.

## Track

Probably **Build for Your World** (a practical tool for a real local problem). The top of the form was not visible in the recording, so check the full list of options and any project-name or team fields before submitting.

## Most Viral Track: X post link (optional)

Leave empty unless you post about the project on X.

## Project Description *

RoadScout turns a phone on your dashboard into a road-hazard reporter. You upload a drive (video plus a GPS track). SAM 3.1 finds hazard candidates (debris in the lane, damaged guardrails and signs, faded lane markings, potholes). Muse Spark checks each candidate and writes a severity and a plain-language description. Deterministic code then geotags each hazard, merges repeat sightings and checks later drives to see whether a hazard is still there. Faces and license plates are blurred before any image reaches the screen. A person reviews each hazard and gets a prefilled VDOT (Virginia) or MDOT SHA (Maryland) report to submit on the agency's own site. RoadScout never submits anything itself.

On two real drives (49 minutes of video) it found 5 reports of 4 distinct hazards, all confirmed by hand (debris on the shoulder, such as a tipped construction barrel), for about $14 of API spend. It missed the one known pothole, a small one among heavy cracking, so we do not claim pothole detection, and recall is not independently measured. Live read-only demo: https://roadscout-demo.onrender.com/

## Public GitHub Repository *

https://github.com/dhairyasurana007/RoadScout (the repository must be set to public before submitting; it is private today).

## 3-Minute Demo Video *

A YouTube link (an unlisted video works). Script: `docs/demo-script.md`. Keep it under 3 minutes.

## Describe your use of Meta products to build the submitted project *

We used the Meta Model API for three jobs. (1) SAM 3.1 detects hazards: one text phrase per request (for example "pothole in the road surface" or "bent or broken guardrail") is run over 20-second, 3 fps video chunks uploaded through the Files API, and the returned masks are decoded with Meta's meta-sam-parser and tracked into candidates. (2) SAM 3.1 also finds faces and license plates in the frames we show, so they can be blurred. (3) Muse Spark (muse-spark-1.3) verifies each candidate from the best frame and a crop, answering in a JSON schema we supply (is it a hazard, class, severity, lane, confidence, reason, a one-line report description). Everything else (preprocessing, geotagging, deduplication, possibly-fixed checks, the prefilled agency report, the review UI) is ordinary deterministic code. Cost over the two real drives was about $13.80 for 49 minutes of video ($9.65 detection, $4.15 verification), roughly $0.28 per minute.

## Why are Meta's models important to your solution? *

Road hazards have no ready-made training set, and a new hazard type should only need a new phrase, not a new model. SAM 3.1 lets us describe a hazard in plain words and get per-frame masks and tracks back, and the same model handles the privacy blur, so one provider covers both detection and protecting people in the footage. Muse Spark gives the second opinion a safety tool needs: it reads the frame, applies rules (a tar seam or a shadow is not a pothole; debris must be in the lane), and returns a structured verdict plus the plain-language description the agency form needs. That second step removed 7 of 12 detections inside our labelled windows and none of them was a labelled hazard. Honest limit: on our one known pothole (small, among heavy cracking) SAM 3.1 boxed it in a single frame and Muse Spark rejected it in 10 of 10 tests, so we report that miss rather than claim pothole detection.

## Provide feedback from your experience using partner technologies (optional)

We did not use Fireworks, Modal or Baseten. The demo runs on a free Render web service.

## Choice questions (the submitter's opinions; what our run showed)

- **How likely are you to build with Meta Model API? / Overall opinion / how likely to recommend (0 to 10):** your own view. Facts: both models worked on real footage from the first live call; one request per phrase made detection slow to scan with many phrases; usage was easy to meter ($0.20 per 1,000 video frames, 50 frames minimum per request).
- **Biggest obstacle:** candidates our run supports: *Slow responses* (Muse Spark verification took 10 to 15 seconds per candidate and dominated run time) or *Code or answers from the model that were wrong or not useful* (the manhole-cover answer on the known pothole). Pick what you felt.
- **Compared with other models:** we did not compare other models for this work, so *We haven't used other models for this kind of work* fits if that is true for you.
- **What matters most when choosing a model / what makes switching hard:** your view.
- **What would stop you from using Meta's models in production:** candidates our run supports: *Model quality isn't good enough yet for our use case* (pothole miss), or *Price or unpredictable costs* (about $17 per hour of video at our settings).
- **Permission checkbox:** only the submitter can tick it.
