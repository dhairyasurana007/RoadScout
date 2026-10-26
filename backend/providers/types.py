"""Our own internal result types for the model providers.

Nothing downstream imports Meta's response shapes; the live provider (task 26)
maps them into these types.

Detection (SAM 3.1): a ``DetectionResult`` holds ``frames``, a list of
``FrameRecord`` (``frame_index``, ``timestamp`` in seconds, ``detections``). Each
``Detection`` is one tracked object in that frame: ``track_id``, ``box``, ``mask``,
``score``, ``prompt``. Frames are strictly increasing in ``frame_index`` and
non-decreasing in ``timestamp``. An empty ``frames`` list is a valid result
(nothing found), not an error.

Box: ``Box(x1, y1, x2, y2)``, pixels in the sampled frame, origin top-left,
x2 > x1 and y2 > y1. It is a tuple, so ``x1, y1, x2, y2 = det.box`` works and it is
JSON-safe (serialised as a list of four numbers).

Mask: a plain JSON-safe dict, run-length encoded (RLE) over the frame in row-major
order (row 0 left to right, then row 1, ...)::

    {"size": [height, width], "counts": [run, run, ...]}

``counts`` alternates run lengths and always starts with a background (0) run, which
may have length 0; the runs sum to ``height * width``. Example: a 2-row, 4-column
frame whose right-hand 2x2 block is set is ``{"size": [2, 4], "counts": [2, 2, 2, 2]}``.
Helpers: ``rle_from_box``, ``rle_from_bitmap``, ``rle_to_bitmap``, ``rle_area``.

Verification (Muse Spark): a ``Verdict`` carries the fields from the plan's
``verdicts`` table. ``to_dict`` / ``from_dict`` on both result types give lossless
plain-JSON forms (used by the fixture format); ``Verdict`` dict keys are the field names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, NamedTuple, Sequence


# --- Masks (RLE dicts) -------------------------------------------------------


def validate_rle(mask: dict[str, Any]) -> dict[str, Any]:
    """Check an RLE mask dict and return it normalised (ints, list counts)."""
    try:
        height, width = (int(v) for v in mask["size"])
        counts = [int(c) for c in mask["counts"]]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"mask must be {{'size': [h, w], 'counts': [...]}}, got {mask!r}") from None
    if height <= 0 or width <= 0:
        raise ValueError("mask size must be positive")
    if any(c < 0 for c in counts):
        raise ValueError("mask run lengths must not be negative")
    if sum(counts) != height * width:
        raise ValueError(f"mask runs sum to {sum(counts)}, expected {height * width}")
    return {"size": [height, width], "counts": counts}


def rle_area(mask: dict[str, Any]) -> int:
    """Number of set pixels."""
    return sum(mask["counts"][1::2])


def rle_from_bitmap(rows: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Encode a list of rows of 0/1 values."""
    height = len(rows)
    width = len(rows[0]) if height else 0
    runs: list[int] = []
    current, length = 0, 0
    for row in rows:
        if len(row) != width:
            raise ValueError("bitmap rows must all have the same length")
        for v in row:
            bit = 1 if v else 0
            if bit == current:
                length += 1
            else:
                runs.append(length)
                current, length = bit, 1
    runs.append(length)
    return validate_rle({"size": [height, width], "counts": runs})


def rle_from_box(width: int, height: int, box: Sequence[float]) -> dict[str, Any]:
    """A filled rectangle (box rounded to whole pixels, clipped to the frame)."""
    bx1, by1, bx2, by2 = box
    x1 = max(0, min(width, round(bx1)))
    x2 = max(0, min(width, round(bx2)))
    y1 = max(0, min(height, round(by1)))
    y2 = max(0, min(height, round(by2)))
    runs: list[int] = [0]  # index parity of the last run gives its value (even = background)

    def add(value: int, n: int) -> None:
        if n == 0:
            return
        if (len(runs) - 1) % 2 == value:
            runs[-1] += n
        else:
            runs.append(n)

    for y in range(height):
        if y1 <= y < y2 and x2 > x1:
            add(0, x1)
            add(1, x2 - x1)
            add(0, width - x2)
        else:
            add(0, width)
    return validate_rle({"size": [height, width], "counts": runs})


def rle_to_bitmap(mask: dict[str, Any]) -> list[list[int]]:
    height, width = mask["size"]
    flat: list[int] = []
    value = 0
    for c in mask["counts"]:
        flat.extend([value] * c)
        value ^= 1
    return [flat[y * width:(y + 1) * width] for y in range(height)]


# --- Detection --------------------------------------------------------------


class Box(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Detection:
    """One tracked object in one frame."""

    track_id: int             # persists across frames for the same object
    box: Box
    score: float              # 0..1
    prompt: str               # the class prompt that matched, e.g. "pothole"
    mask: dict[str, Any] | None = None  # RLE dict, see module docstring

    def __post_init__(self) -> None:
        box = Box(*(float(v) for v in self.box))
        if not (box.x2 > box.x1 and box.y2 > box.y1):
            raise ValueError(f"invalid box: {tuple(self.box)}")
        object.__setattr__(self, "box", box)
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1")
        if self.mask is not None:
            object.__setattr__(self, "mask", validate_rle(self.mask))

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "box": list(self.box),
            "mask": self.mask,
            "score": self.score,
            "prompt": self.prompt,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Detection":
        return cls(
            track_id=int(d["track_id"]),
            box=Box(*d["box"]),
            score=float(d["score"]),
            prompt=str(d["prompt"]),
            mask=d.get("mask"),
        )


@dataclass(frozen=True)
class FrameRecord:
    """One streamed per-frame record: the frame and everything detected in it."""

    frame_index: int
    timestamp: float          # seconds
    detections: tuple[Detection, ...] = ()

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must not be negative")
        object.__setattr__(self, "detections", tuple(self.detections))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "detections": [d.to_dict() for d in self.detections],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FrameRecord":
        return cls(
            frame_index=int(d["frame_index"]),
            timestamp=float(d["timestamp"]),
            detections=tuple(Detection.from_dict(x) for x in d.get("detections", [])),
        )


@dataclass(frozen=True)
class DetectionResult:
    """Everything SAM 3.1 returned for one chunk. ``frames`` may be empty."""

    frames: tuple[FrameRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", tuple(self.frames))
        for prev, cur in zip(self.frames, self.frames[1:]):
            if cur.frame_index <= prev.frame_index:
                raise ValueError("frame_index must strictly increase")
            if cur.timestamp < prev.timestamp:
                raise ValueError("timestamps must not decrease")

    def all_detections(self) -> Iterator[tuple[FrameRecord, Detection]]:
        """Every (frame, detection) pair in frame order."""
        for f in self.frames:
            for d in f.detections:
                yield f, d

    @property
    def is_empty(self) -> bool:
        """True when nothing was detected (a valid result)."""
        return not any(f.detections for f in self.frames)

    def tracks(self) -> dict[int, list[tuple[FrameRecord, Detection]]]:
        """(frame, detection) pairs grouped by track id, in frame order."""
        out: dict[int, list[tuple[FrameRecord, Detection]]] = {}
        for f, d in self.all_detections():
            out.setdefault(d.track_id, []).append((f, d))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"frames": [f.to_dict() for f in self.frames]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DetectionResult":
        return cls(frames=tuple(FrameRecord.from_dict(x) for x in d.get("frames", [])))


# --- Verification -----------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """Muse Spark's judgment on one candidate (the plan's ``verdicts`` fields)."""

    is_hazard: bool
    hazard_class: str
    severity: int              # 1 (minor) to 3 (severe); 0 is allowed only when not a hazard
    lane: str | None
    confidence: float          # 0..1
    reason: str                # one line
    report_description: str    # short, for the agency report
    model: str = ""            # stored on every verdict so runs can be compared
    prompt_version: str = ""

    def __post_init__(self) -> None:
        low = 1 if self.is_hazard else 0
        if not low <= self.severity <= 3:
            raise ValueError(f"severity must be {low} to 3, got {self.severity!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_hazard": self.is_hazard,
            "hazard_class": self.hazard_class,
            "severity": self.severity,
            "lane": self.lane,
            "confidence": self.confidence,
            "reason": self.reason,
            "report_description": self.report_description,
            "model": self.model,
            "prompt_version": self.prompt_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Verdict":
        return cls(
            is_hazard=bool(d["is_hazard"]),
            hazard_class=str(d["hazard_class"]),
            severity=int(d["severity"]),
            lane=d.get("lane"),
            confidence=float(d["confidence"]),
            reason=str(d["reason"]),
            report_description=str(d["report_description"]),
            model=str(d.get("model", "")),
            prompt_version=str(d.get("prompt_version", "")),
        )
