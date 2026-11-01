# Hand labels

One CSV per drive: `IMG_6513.csv` (36 min, label this one first) and `IMG_6511.csv` (12 min).
Delete the EXAMPLE rows when you add real ones. Open the files in Excel or any text editor.

## Row types

- `segment`: a 2 to 3 minute stretch you looked at carefully. Aim for about 10 in total: 6 or 7 that contain hazards and 3 or 4 clean ones (nothing on the road). A clean segment has no `hazard` rows, and it is how false alarms get counted.
- `hazard`: one hazard inside a segment, with the time range it is visible.

## Columns

| Column | What to write |
|---|---|
| `row_type` | `segment` or `hazard` |
| `segment_id` | A letter or number that ties hazards to their segment, for example `A` |
| `start`, `end` | Video time as `mm:ss`, counted from the start of that video file (the timestamp in the player, not the clock time) |
| `class` | `pothole`, `debris in lane`, `damaged guardrail`, `fallen or damaged sign` or `faded lane markings` (hazard rows only) |
| `severity` | `1` low, `2` medium, `3` high (hazard rows only) |
| `hazard_id` | A short name such as `H1`. Use the same id if the same hazard shows up in another drive |
| `note` | Anything useful, for example "tire on the left shoulder" |

## Rules

- Only label what is really there. If there are no potholes, do not invent any. The eval will say that no potholes were in the footage.
- Label hazards of any class, not only potholes.
- Every `hazard` row must fall inside a `segment` row with the same `segment_id`.
- Do not label stretches where the car is stopped or the camera is blocked.
- The five hazards the pipeline already found are in the app's hazard map. Check each against the video: a real one gets a `hazard` row, and if it is wrong, just leave it out.
