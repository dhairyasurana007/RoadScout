"""Hazard class prompts for SAM 3.1, loaded from a JSON config file (task 7).

File: ``config/hazard_prompts.json`` (override the path with ROADSCOUT_HAZARD_PROMPTS)::

    {"version": 1,
     "classes": {"pothole": ["pothole", "hole in the asphalt"],
                 "debris in lane": ["debris in lane", "object lying on the road"], ...}}

``classes`` maps each hazard class name to a non-empty list of phrasings. The first phrasing of
a class is its default and the only one sent to SAM 3.1; to try another phrasing, move it to the
front of the list. Class order in the file is the order the prompts are sent.

``load_hazard_prompts(path=None)`` validates the file and returns ``HazardPrompts``; any problem
(missing file, bad JSON, wrong shape, empty or blank names or phrasings, a phrasing repeated
inside a class or shared by two classes, ignoring case) raises ``PromptConfigError`` (a
``ValueError``) with a message naming the file and the problem.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import config


class PromptConfigError(ValueError):
    """The hazard prompts file is missing or invalid."""


@dataclass(frozen=True)
class HazardPrompts:
    """Validated prompts: ``classes`` maps class name -> tuple of phrasings (first = default)."""

    classes: dict[str, tuple[str, ...]]

    @property
    def class_names(self) -> list[str]:
        return list(self.classes)

    def default_prompts(self) -> list[str]:
        """The first phrasing of every class, in file order: what is sent to SAM 3.1."""
        return [phrasings[0] for phrasings in self.classes.values()]

    def class_for_prompt(self, prompt: str) -> str | None:
        """The class a phrasing (any phrasing, not only the default) belongs to, ignoring case
        and surrounding spaces; None if it is unknown."""
        key = " ".join(str(prompt).split()).casefold()
        for name, phrasings in self.classes.items():
            if any(" ".join(p.split()).casefold() == key for p in phrasings):
                return name
        return None


def parse_hazard_prompts(data: object, *, source: str = "hazard prompts") -> HazardPrompts:
    """Validate an already-loaded config object (see the module docstring for the shape)."""
    if not isinstance(data, dict):
        raise PromptConfigError(f"{source}: top level must be a JSON object")
    classes = data.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise PromptConfigError(f"{source}: 'classes' must be a non-empty object of class name -> phrasings")
    out: dict[str, tuple[str, ...]] = {}
    owner: dict[str, str] = {}  # normalised phrasing -> class name
    seen_names: set[str] = set()
    for name, phrasings in classes.items():
        if not isinstance(name, str) or not name.strip():
            raise PromptConfigError(f"{source}: class names must be non-empty strings")
        name = name.strip()
        if name.casefold() in seen_names:
            raise PromptConfigError(f"{source}: class {name!r} is listed twice")
        seen_names.add(name.casefold())
        if not isinstance(phrasings, list) or not phrasings:
            raise PromptConfigError(f"{source}: class {name!r} needs a non-empty list of phrasings")
        cleaned: list[str] = []
        for p in phrasings:
            if not isinstance(p, str) or not p.strip():
                raise PromptConfigError(f"{source}: class {name!r} has an empty or non-string phrasing")
            p = p.strip()
            key = " ".join(p.split()).casefold()
            if key in owner:
                where = "twice in" if owner[key] == name else f"in both {owner[key]!r} and"
                raise PromptConfigError(f"{source}: phrasing {p!r} appears {where} {name!r}")
            owner[key] = name
            cleaned.append(p)
        out[name] = tuple(cleaned)
    return HazardPrompts(classes=out)


def load_hazard_prompts(path: str | Path | None = None) -> HazardPrompts:
    """Read and validate the prompts file (default: ``config.hazard_prompts_path()``)."""
    target = Path(path) if path is not None else config.hazard_prompts_path()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise PromptConfigError(f"hazard prompts file not found: {target}") from None
    except OSError as e:
        raise PromptConfigError(f"cannot read hazard prompts file {target}: {e}") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise PromptConfigError(f"hazard prompts file {target} is not valid JSON: {e}") from None
    return parse_hazard_prompts(data, source=str(target))
