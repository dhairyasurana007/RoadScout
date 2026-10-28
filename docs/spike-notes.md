# Task 26 footage spike notes

Run on 2026-10-27 against the live Meta Model API with four real iPhone clips (portrait 1080p HEVC, about 30 fps): IMG_6503 (neighbourhood streets, 1:30), IMG_6504 (suburban road, 1:31), IMG_6505 (highway with a work zone, 2:49) and IMG_6508 (highway, 28:08, 50,653 frames). The footage and GPS stay local and are never committed. The clips have **no GPX track** (Google Maps Timeline was off), so geotagging was not tested. Scripts: `spike/probe_sam_image.py`, `spike/probe_sam_video.py`, `spike/show_matches.py`, `spike/probe_verify.py`.

## API facts confirmed live

- Both models answer on `POST https://api.meta.ai/v1/responses` with `Authorization: Bearer <key>`.
- SAM 3.1 image mode: one frame, about 2.5 s. SAM 3.1 video mode: upload with `POST /v1/files` (`purpose=user_data`, multipart), then `input_video` with a `file_id`. A 60-frame 20 s clip answered in about 5 s, a 507-frame 169 s clip in 14 to 27 s depending on matches. The upload is deleted afterwards with `DELETE /v1/files/{id}`.
- A non-streaming response puts the text at `output[].content[].text`. The `usage` block shows `video_frames_processed` (or `images_processed`) and zero tokens.
- Response lines are `<Nf>id<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..;c=..|><|mask;x=0;y=0;c=..;data=H,W,!payload|>`. With `include_confidence: "true"` every box carries `c`. **Frames with no match are left out of the output entirely**, so the output has fewer records than frames sent (for example 35 of 60 frames for "car").
- Confidence was identical on every frame of a track in all our runs, so it is effectively a per-track value, not per-frame.
- Object ids reached 16 to 18 in busy scenes, so expect many tracks per request (the docs cap tracking at 16 objects per frame).
- Muse Spark (`muse-spark-1.3`) with two `input_image` blocks and `text.format` = `{"type": "json_schema", "name": "verdict", "schema": ...}` returns the verdict JSON as plain text in `output[].content[].text`, in our schema. It is a reasoning model: 730 input tokens and 1,330 output tokens (1,245 of them reasoning) for one candidate, about **$0.0066 per verification** at list price.

## What the models found

- The prompt "pothole" found nothing on any clip. "traffic cone" found real work-zone cones (16 tracks in IMG_6505). The clips contain **no obvious potholes or debris** (clean pavement), so recall for the plan's hazard classes could not be measured. Hand labels and a drive over known damage are still needed (tasks 12 and 20).
- Phrasing matters a lot. "pothole in the road surface" matched small dark patches, marks and paint in the lane ahead (1 to 3 frames per track, up to 17 frames for one patch on IMG_6505). "bent or broken guardrail" matched normal guardrails. "worn lane line", "barely visible road paint" and "bent traffic sign" each returned 16 tracks on normal road furniture. "faded lane markings", "damaged guardrail", "fallen or damaged sign" and "debris in lane" alone returned nothing.
- Muse Spark correctly rejected a "pothole" candidate crop (smooth pavement plus lane paint, confidence 0.1). Verification is the filter that removes these detector false positives, so it must stay on.

## Frame rate and chunks (work-zone stretch, 100 s, "traffic cone")

| fps | frames sent and billed | frames with a match | longest track |
| --- | --- | --- | --- |
| 2 | 198 | 44 | 20 |
| 3 | 298 | 43 | 15 |
| 5 | 496 | 50 | 20 |

The number of matched frames barely moves with fps, so 2 fps costs a third less than 3 fps with no visible loss on large objects. Small hazards (potholes) were not testable, so the default stays at 3 fps until labelled hazards exist; 2 fps is the cost-saving option. At 3 fps and highway speed, frames are about 10 m apart.

## Cost (list prices)

- SAM 3.1 video: $0.20 per 1,000 frames, minimum 50 frames billed per request, one request per prompt phrase. Image: $2.50 per 1,000.
- One full drive like IMG_6508 (28 min, about 5,000 frames at 3 fps): about **$1.0 per phrase**. 15 phrases cost about $15; one phrase per class (5) costs about $5; at 2 fps about $3.4.
- The default 10 s chunk at 3 fps is 30 frames but is billed as 50, so it overpays by two thirds. Use chunks of at least 17 s at 3 fps (20 s, 60 frames, is a good default).
- Muse Spark: about $0.0066 per candidate; 100 candidates cost about $0.7.
- Spend in this spike: about $3 in total. Rate limits are not a concern (3,000 requests per minute).
- Whole-drive extrapolation for the eval set (10 segments of 2 to 3 minutes, 5 phrases, 3 fps): about $3 for SAM plus under $1 for verification per full pass, so the $150 credit pool allows many passes. The open-weights fallback is **not needed**.

## Defaults to set

- `ROADSCOUT_CHUNK_SECONDS`: 20 (was 10).
- Prompts: keep the three phrasings per class but run only the ones that work. Treat "worn lane line", "barely visible road paint", "bent traffic sign" and "bent or broken guardrail" as noisy; they depend on verification and the minimum track length. Re-tune after labelled data exists (tasks 20 and 21).
- Minimum track length stays 3 frames: single and two-frame tracks dominate the false positives.
- Frames, never the whole raw video: each 720p 3 fps chunk is about 1.6 MB per 20 s. The 15,000-frame cap per request means IMG_6508 must always be chunked.

## Open items

- No real hazards in the footage, so detection recall and the best frame rate for small hazards are unmeasured.
- No GPX, so geotagging, dedup and the report prefill were not exercised on real data.
- Muse Spark reasoning tokens dominate its cost; a lower reasoning effort setting was not tried.
- Export blur coverage: SAM omits frames with no match, so the export cannot tell a frame with nothing to blur from a missed face or plate. Plan a conservative blur (hold masks across short gaps) before the export is used for anything shared.
