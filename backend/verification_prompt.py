"""The versioned Muse Spark verification prompt, loaded from a JSON config file (task 9).

File: ``config/verification_prompt.json`` (override the path with ROADSCOUT_VERIFY_PROMPT)::

    {"version": "v1",
     "prompt": "... {detector_class} ... {classes} ... {schema} ...",
     "schema": {"type": "object", "properties": {"is_hazard": {...}, "class": {...}, ...}}}

- ``version``: a non-empty string (or a whole number, stored as text). It is stored on every
  verdict (``prompt_version``) so runs with different prompts can be compared. Change it whenever
  ``prompt`` or ``schema`` changes.
- ``prompt``: the instruction sent to the model. Optional placeholders, filled by ``render``:
  ``{detector_class}`` (the detector's class for the candidate), ``{classes}`` (the known class
  names, comma separated) and ``{schema}`` (the schema below as compact JSON). Other braces are
  left alone.
- ``schema`` (optional): a JSON-schema-like object describing the answer. If present it must be
  an object with a ``properties`` map that names every verdict field: ``is_hazard``, ``class``, ``severity``,
  ``lane``, ``confidence``, ``reason`` and ``report_description``. The provider maps ``class`` to
  ``Verdict.hazard_class``.

The real wording and request shape are settled against Meta's docs in task 26; until then this
file is a placeholder that the fake provider ignores. Any problem (missing file, bad JSON, wrong
shape) raises ``VerificationPromptError`` (a ``ValueError``) naming the file and the problem.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import config

REQUIRED_FIELDS = (
    "is_hazard", "class", "severity", "lane", "confidence", "reason", "report_description",
)


class VerificationPromptError(ValueError):
    """The verification prompt file is missing or invalid."""


@dataclass(frozen=True)
class VerificationPrompt:
    version: str
    text: str
    schema: dict[str, Any]
    source: str = ""

    def render(self, detector_class: str | None = None, class_names: Sequence[str] | None = None) -> str:
        """The prompt text with ``{detector_class}``, ``{classes}`` and ``{schema}`` filled in."""
        out = self.text
        out = out.replace("{schema}", json.dumps(self.schema, separators=(",", ":")))
        out = out.replace("{classes}", ", ".join(class_names) if class_names else "")
        out = out.replace("{detector_class}", detector_class or "")
        return out


def parse_verification_prompt(data: object, *, source: str = "verification prompt") -> VerificationPrompt:
    """Validate an already-loaded config object (see the module docstring for the shape)."""
    if not isinstance(data, dict):
        raise VerificationPromptError(f"{source}: verification prompt: top level must be a JSON object")
    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, (str, int)) or not str(version).strip():
        raise VerificationPromptError(f"{source}: verification prompt: 'version' must be a non-empty string")
    text = data.get("prompt")
    if not isinstance(text, str) or not text.strip():
        raise VerificationPromptError(f"{source}: verification prompt: 'prompt' must be a non-empty string")
    schema = data.get("schema", {})
    if not isinstance(schema, dict):
        raise VerificationPromptError(f"{source}: verification prompt: 'schema' must be a JSON object")
    if schema:
        props = schema.get("properties")
        if not isinstance(props, dict):
            raise VerificationPromptError(f"{source}: verification prompt: 'schema' needs a 'properties' object")
        missing = [f for f in REQUIRED_FIELDS if f not in props]
        if missing:
            raise VerificationPromptError(
                f"{source}: verification prompt: 'schema.properties' is missing {', '.join(missing)}")
    return VerificationPrompt(version=str(version).strip(), text=text, schema=schema, source=source)


def load_verification_prompt(path: str | Path | None = None) -> VerificationPrompt:
    """Read and validate the prompt file (default: ``config.verification_prompt_path()``)."""
    target = Path(path) if path is not None else config.verification_prompt_path()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise VerificationPromptError(f"verification prompt file not found: {target}") from None
    except OSError as e:
        raise VerificationPromptError(f"cannot read verification prompt file {target}: {e}") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise VerificationPromptError(f"verification prompt file {target} is not valid JSON: {e}") from None
    return parse_verification_prompt(data, source=str(target))
