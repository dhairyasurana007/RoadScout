"""The provider interface: SAM 3.1 detection and Muse Spark verification.

Downstream code depends only on ``ModelProvider`` and the types in ``types.py``.
Implementations: ``FakeProvider``, ``FixtureProvider``, ``LiveProvider``;
pick one with ``backend.providers.get_provider()``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, Union

from .types import DetectionResult, Verdict

#: An image for verification: a file path or raw image bytes.
Image = Union[str, Path, bytes]


class ModelProvider(ABC):
    #: short provider name: "fake", "fixture" or "live"
    name: str = "base"

    @abstractmethod
    def detect(self, video_path: str, prompts: Sequence[str]) -> DetectionResult:
        """SAM 3.1: detect, segment and track objects matching the text ``prompts`` in
        one preprocessed chunk. Nothing found returns an empty ``DetectionResult``,
        never an error. Frame timestamps are relative to the start of the chunk."""

    @abstractmethod
    def verify(
        self,
        images: Sequence[Image],
        *,
        class_prompt: str | None = None,
        prompt_version: str | None = None,
    ) -> Verdict:
        """Muse Spark: judge one candidate from its images (best frame first, then the
        crop and neighbouring frames) and return a structured verdict. ``class_prompt``
        is the detector's class for the candidate."""

    def estimate_cost(self, operation: str) -> float:
        """Estimated USD cost of one call (``"detect"`` or ``"verify"``), used for the
        spend-cap check before the call. Free providers return 0."""
        return 0.0
