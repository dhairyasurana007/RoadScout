"""Fake provider: canned, deterministic results. No network, and the inputs need not exist.

Select a scenario with ``FakeProvider(scenario=...)`` or ROADSCOUT_FAKE_SCENARIO
(read by ``get_provider``); an unknown scenario raises ``ValueError``:

- ``default``         one 12-frame track for the first requested prompt; verify says hazard.
- ``empty``           no detections at all (a valid result, not an error).
- ``short_track``     one 2-frame track (should be dropped by track aggregation).
- ``false_positive``  an 8-frame track that verify rejects (is_hazard False).

Frames are sampled at 3 fps starting at timestamp 0. Same inputs always give the same
outputs (no randomness, no clock). With no prompts, detect returns an empty result.
"""
from __future__ import annotations

import zlib
from typing import Sequence

from .base import Image, ModelProvider
from .errors import ProviderConfigError
from .types import Box, Detection, DetectionResult, FrameRecord, Verdict, rle_from_box

SCENARIOS = ("default", "empty", "short_track", "false_positive")

FRAME_WIDTH = 640
FRAME_HEIGHT = 360
FPS = 3.0
FAKE_VERIFY_MODEL = "fake-muse-spark"
FAKE_PROMPT_VERSION = "fake-v1"

_TRACK_FRAMES = {"default": 12, "empty": 0, "short_track": 2, "false_positive": 8}


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self, scenario: str = "default"):
        scenario = str(scenario).strip().lower()
        if scenario not in SCENARIOS:
            raise ProviderConfigError(
                f"unknown fake scenario {scenario!r}; choose one of {', '.join(SCENARIOS)}"
            )
        self.scenario = scenario

    def detect(self, video_path: str, prompts: Sequence[str]) -> DetectionResult:
        n_frames = _TRACK_FRAMES[self.scenario]
        if n_frames == 0 or not prompts:
            return DetectionResult(())
        frames = []
        for i in range(n_frames):
            # The object drifts down and grows as the car approaches it.
            box = Box(300 + 2 * i, 200 + 6 * i, 340 + 5 * i, 230 + 9 * i)
            det = Detection(
                track_id=1,
                box=box,
                score=round(min(0.95, 0.60 + 0.03 * i), 2),
                prompt=prompts[0],
                mask=rle_from_box(FRAME_WIDTH, FRAME_HEIGHT, box),
            )
            frames.append(FrameRecord(frame_index=i, timestamp=round(i / FPS, 6), detections=(det,)))
        return DetectionResult(tuple(frames))

    def verify(
        self,
        images: Sequence[Image],
        *,
        class_prompt: str | None = None,
        prompt_version: str | None = None,
    ) -> Verdict:
        hazard_class = class_prompt or "pothole"
        version = prompt_version or FAKE_PROMPT_VERSION
        if self.scenario == "false_positive":
            return Verdict(
                is_hazard=False,
                hazard_class=hazard_class,
                severity=1,
                lane=None,
                confidence=0.15,
                reason="Looks like a shadow or a patched seam, not a hazard.",
                report_description="No hazard found.",
                model=FAKE_VERIFY_MODEL,
                prompt_version=version,
            )
        # Deterministic variety from the inputs (crc32, not hash(): hash is salted per run).
        h = zlib.crc32(repr([i if isinstance(i, bytes) else str(i) for i in images]).encode("utf-8"))
        severity = 1 + h % 3
        lane = ("left", "center", "right")[(h >> 2) % 3]
        return Verdict(
            is_hazard=True,
            hazard_class=hazard_class,
            severity=severity,
            lane=lane,
            confidence=round(0.80 + (h % 20) / 100, 2),
            reason=f"Clear {hazard_class} in the {lane} lane.",
            report_description=f"A {hazard_class} in the {lane} lane.",
            model=FAKE_VERIFY_MODEL,
            prompt_version=version,
        )
