# RoadScout eval v1

- drives in scope: 2 (2 with labels used, 0 without)
- labeled segments: 4, matching tolerance: 2 s
- labeled hazards: 5 -> recall 80.0% (SAM only), 80.0% (SAM + Muse Spark)
- clean segment time: 0.0 min

## Ablation: SAM only vs SAM + Muse Spark

Counts are inside the labeled segments of the drives that have a label file. A detection is a candidate; SAM + Muse Spark keeps only the candidates whose verdict was accepted.

### SAM only

| class | labeled | matched | recall | detections | correct | precision | false alarms | false alarms/hour |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pothole | 1 | 0 | 0.0% | 1 | 0 | 0.0% | 0 | n/a |
| debris in lane | 4 | 4 | 100.0% | 5 | 5 | 100.0% | 0 | n/a |
| damaged guardrail | 0 | 0 | n/a | 0 | 0 | n/a | 0 | n/a |
| fallen or damaged sign | 0 | 0 | n/a | 4 | 0 | 0.0% | 0 | n/a |
| faded lane markings | 0 | 0 | n/a | 2 | 0 | 0.0% | 0 | n/a |
| overall | 5 | 4 | 80.0% | 12 | 5 | 41.7% | 0 | n/a |

### SAM + Muse Spark

| class | labeled | matched | recall | detections | correct | precision | false alarms | false alarms/hour |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pothole | 1 | 0 | 0.0% | 0 | 0 | n/a | 0 | n/a |
| debris in lane | 4 | 4 | 100.0% | 5 | 5 | 100.0% | 0 | n/a |
| damaged guardrail | 0 | 0 | n/a | 0 | 0 | n/a | 0 | n/a |
| fallen or damaged sign | 0 | 0 | n/a | 0 | 0 | n/a | 0 | n/a |
| faded lane markings | 0 | 0 | n/a | 0 | 0 | n/a | 0 | n/a |
| overall | 5 | 4 | 80.0% | 5 | 5 | 100.0% | 0 | n/a |

Removed by the verifier: 7 detections inside labeled segments, 0 of them matched a labeled hazard.
False alarms per hour: not measured (no clean segments).

## Review precision (approved / (approved + rejected))

| class | approved | rejected | precision |
| --- | ---: | ---: | ---: |
| pothole | 0 | 0 | n/a |
| debris in lane | 0 | 0 | n/a |
| damaged guardrail | 0 | 0 | n/a |
| fallen or damaged sign | 0 | 0 | n/a |
| faded lane markings | 0 | 0 | n/a |
| overall | 0 | 0 | n/a |

Not reviewed (new, left out of the precision): 5
Rejected hazards per hour of video: 0.00
Review precision: not measured (no approved or rejected hazards).

## Dedup accuracy (hazards seen on repeated drives)

- merged correctly: 0
- split wrongly: 0
- merged wrongly: 0

Dedup accuracy: not measured (no labeled hazard was matched on two or more drives).

## Cost and processing time

- video minutes in scope: 49.1
- total cost (cost log): $13.8010
- cost per minute of video: $0.2811
  - cost of detect: $9.6500
  - cost of verify: $4.1510
- processing time per minute of video: 919.4 s (2 finished drives)
