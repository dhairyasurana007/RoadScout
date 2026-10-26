"""Fixture provider: replays saved responses from JSON files. No network.

Format: a directory containing two files, each exactly the ``to_dict()`` of a result
serialised as JSON (see ``types.py``):

- ``detect.json``: a ``DetectionResult`` dict, ``{"frames": [{"frame_index", "timestamp",
  "detections": [{"track_id", "box": [x1, y1, x2, y2], "mask": {"size": [h, w],
  "counts": [...]} | null, "score", "prompt"}]}]}``.
- ``verify.json``: a ``Verdict`` dict with keys ``is_hazard, hazard_class, severity,
  lane, confidence, reason, report_description, model, prompt_version``.

Every ``detect`` call returns the saved detection result and every ``verify`` call the
saved verdict, ignoring the inputs (they need not exist). Files are read on each call, so
a directory may hold only the file you need. The directory is ``FixtureProvider(dir)`` or
ROADSCOUT_FIXTURES_DIR via ``get_provider``. A missing file raises ``FixtureMissing`` (a
``ProviderError`` and ``FileNotFoundError``); malformed content raises
``ProviderConfigError`` (a ``ProviderError`` and ``ValueError``).

The checked-in ``fixtures/providers/tiny/`` holds synthetic data only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .. import config
from .base import Image, ModelProvider
from .errors import FixtureMissing, ProviderConfigError
from .types import DetectionResult, Verdict


class FixtureProvider(ModelProvider):
    name = "fixture"

    def __init__(self, fixtures_dir: str | Path | None = None):
        self.fixtures_dir = Path(fixtures_dir) if fixtures_dir is not None else config.fixtures_dir()

    def _load(self, filename: str) -> dict[str, Any]:
        path = self.fixtures_dir / filename
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FixtureMissing(f"fixture file not found: {path}") from None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ProviderConfigError(f"fixture file is not valid JSON: {path}: {e}") from None
        if not isinstance(data, dict):
            raise ProviderConfigError(f"fixture file must hold a JSON object: {path}")
        return data

    def detect(self, video_path: str, prompts: Sequence[str]) -> DetectionResult:
        data = self._load("detect.json")
        try:
            return DetectionResult.from_dict(data)
        except (KeyError, TypeError, ValueError) as e:
            raise ProviderConfigError(f"bad detect.json in {self.fixtures_dir}: {e}") from None

    def verify(
        self,
        images: Sequence[Image],
        *,
        class_prompt: str | None = None,
        prompt_version: str | None = None,
    ) -> Verdict:
        data = self._load("verify.json")
        try:
            return Verdict.from_dict(data)
        except (KeyError, TypeError, ValueError) as e:
            raise ProviderConfigError(f"bad verify.json in {self.fixtures_dir}: {e}") from None
