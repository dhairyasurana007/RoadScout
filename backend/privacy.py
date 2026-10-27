"""Privacy blur (task 15): blur faces and license plates in the saved frames, serve only those.

    from backend.privacy import blur_drive, blur_image, load_privacy_prompts
    summary = blur_drive(drive_id)                       # pipeline stage 'blurring'
    blur_image("frame.jpg", "frame_blur.jpg", [mask_dict, ...])

Only SAVED FRAMES go to the model, never the video: the best-frame JPEG and the crop JPEG of every
accepted hazard candidate (task 8/9). The review UI, the photo route, ``hazards.photo_path`` and the
reports use the blurred copies only. Unblurred frames still leave the machine for the Muse Spark
verification call (the one accepted exception, see the README); nothing here reads or logs a key.

There is no extra provider method: SAM 3.1 on a still image is ``provider.detect(<jpeg path>, prompts)``
(the same call as for a chunk video; the fake provider recognises an image path by its extension).

Prompts. ``config/privacy_prompts.json`` (override the path with ROADSCOUT_PRIVACY_PROMPTS)::

    {"version": 1, "prompts": ["face", "license plate"]}

A bare JSON list of prompts is accepted too. ``load_privacy_prompts(path=None)`` validates it and returns a
tuple of strings (non-empty list of non-blank strings, no duplicates ignoring case and spacing); any
problem raises ``PrivacyConfigError`` (a ``ValueError``) naming the file.

``blur_image(src_path, dst_path, masks, *, dilate_px=0) -> BlurResult``

- ``masks``: an iterable (possibly empty) of masks, each an RLE dict (``{"size": [h, w], "counts": [...]}``,
  see ``backend/providers/types.py``), a 2-D 0/1 array, or a ``Detection``-like object (``.mask``, or,
  when that is None, ``.box`` in image pixels). A mask of another size than the image is scaled to it (a
  pixel the scaled-down mask touches at all counts as set, so scaling never uncovers anything). Masks are
  combined (union). An invalid mask raises ``InvalidMask`` (a ``PrivacyError`` and ``ValueError``), a missing
  or unreadable source raises ``MissingImage`` (a ``PrivacyError``; also ``FileNotFoundError`` when the file
  does not exist); in both cases nothing is written and no ``dst_path`` exists afterwards.
- Blur: every connected area of the union is pixelated into about 5 cells across its smaller side (cell
  averages, so the mean colour of the area is kept) and smoothed with a Gaussian, so no detail survives; it
  is a blur, not a fill. Only pixels inside the union change, plus ``dilate_px`` pixels around it when
  that is given (``blur_image`` default 0 = exactly the union; ``blur_drive`` uses
  ``config.blur_dilate_px()``, env ROADSCOUT_BLUR_DILATE_PX, default 4, as a safety margin around SAM
  masks). A JPEG output is re-encoded at quality 95, so pixels outside differ only by JPEG noise; a
  ``.png`` ``dst_path`` is written losslessly (any other extension gets JPEG data).
- No mask pixels (an empty list, empty masks): ``dst_path`` is an unchanged copy of the file.
- The source is never modified: ``dst_path`` must be another file than ``src_path`` (else
  ``PrivacyError``); the output is written through a temporary file and renamed, so it is complete or absent.
- Returns ``BlurResult(out_path, mask_count, blurred_pixels)``.

``blur_drive(drive_id, *, provider=None, db_path=None, prompts=None, dilate_px=None) -> BlurSummary``
(``blur_candidates`` is an alias)

For every candidate of the drive whose verdict is accepted and ``is_hazard``, call
``provider.detect(<saved image path as str>, ["face", "license plate"])`` (one call per image, both prompts
together) on the best frame and on the crop (if it has one), union the masks of ALL returned detections
(the box of a detection that has no mask), blur them and write the copies next to the originals, in
``<storage>/<drive id>/candidates/``::

    c0000_t1_pothole_best.jpg  ->  c0000_t1_pothole_best_blur.jpg
    c0000_t1_pothole_crop.jpg  ->  c0000_t1_pothole_crop_blur.jpg

and record on the ``candidates`` row (columns added by the idempotent migration in ``init_db``):
``blurred_frame_path``, ``blurred_crop_path`` (NULL when there is no crop), ``blur_status`` and ``blur_note``.
``blur_status`` is ``'ok'`` or ``'failed'`` (NULL for candidates that were never blurred, e.g. rejected
ones, which are never served). The provider is ``get_provider()`` by default; retries, the timeout and the
spend cap live in the provider layer. A valid result with no detections is success: the copy is unchanged.

FAIL CLOSED. A candidate fails when the provider raises, returns something that is not a
``DetectionResult`` (None, a string, a number, any other object), or has an invalid mask, or an image is
missing or unreadable. A failed candidate gets ``blur_status = 'failed'``, a one-line ``blur_note``, NULL
blurred paths and NO blurred file on disk (partial output is removed, and an older blurred copy of it is
removed too). Nothing unblurred is ever served for it (the photo route only uses ``blur_status = 'ok'``
images). All candidates are tried; then the whole result is written in ONE transaction. If any candidate
failed the call then raises ``BlurError`` (a ``ProviderError``; ``.summary``, ``.failed_candidate_ids``) AFTER
recording the failures, and the pipeline ends the drive ``failed`` with ``failed_stage = 'blurring'`` and
``error = 'blurring: ...'`` (the first failure's exception text is in it). Candidates that did blur keep
their images.

Idempotent: a re-run replaces the drive's blurred files and rows; blurred files of the drive's earlier run
that the new run does not reuse are deleted, so none is left stale. A drive without accepted candidates
makes no provider call (and clears any old blur data of its other candidates). Bad arguments (blank
``drive_id``, bad ``dilate_px``) and a bad prompts file raise ``ValueError`` before any provider call.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from . import config
from .db import get_connection, init_db
from .providers import ModelProvider, ProviderError, get_provider, validate_rle

logger = logging.getLogger(__name__)

JPEG_QUALITY = 95
BLUR_SUFFIX = "_blur"
BLUR_OK = "ok"
BLUR_FAILED = "failed"
CELLS_ACROSS = 5  # pixelation: about this many cells across the smaller side of an area
MAX_NOTE_CHARS = 300


class PrivacyConfigError(ValueError):
    """The privacy prompts file is missing or invalid."""


class PrivacyError(RuntimeError):
    """An image could not be blurred (unreadable image, invalid mask, bad output path)."""


class InvalidMask(PrivacyError, ValueError):
    """A mask given to ``blur_image`` (or returned by a provider) is not usable."""


class MissingImage(PrivacyError):
    """The source image is missing or not a readable image."""


class _NoSuchImage(MissingImage, FileNotFoundError):
    pass


class BlurError(ProviderError):
    """``blur_drive`` could not blur every accepted candidate. The failures are already recorded on
    the candidates rows (``blur_status = 'failed'``); ``summary`` is the ``BlurSummary``."""

    def __init__(self, message: str, *, summary: "BlurSummary | None" = None):
        super().__init__(message)
        self.summary = summary
        self.failed_candidate_ids = [c["candidate_id"] for c in summary.candidates
                                     if c["status"] == BLUR_FAILED] if summary else []


@dataclass(frozen=True)
class BlurResult:
    """What ``blur_image`` did: the output file, how many masks were given and how many pixels were blurred."""

    out_path: Path
    mask_count: int
    blurred_pixels: int  # pixels inside the (grown) mask; 0 means the output is an unchanged copy


@dataclass
class BlurSummary:
    """What a ``blur_drive`` call stored."""

    drive_id: str
    prompts: list[str]
    candidate_count: int  # accepted hazard candidates that were tried
    ok_count: int
    failed_count: int
    mask_count: int = 0  # masks applied over all images
    candidates: list[dict[str, Any]] = field(default_factory=list)
    # one dict per tried candidate: candidate_id, status ('ok' | 'failed'), note, blurred_frame_path,
    # blurred_crop_path, mask_count


# ----- prompts -----------------------------------------------------------------------------


def parse_privacy_prompts(data: object, *, source: str = "privacy prompts") -> tuple[str, ...]:
    """Validate an already-loaded config object (see the module docstring for the shape)."""
    items = data.get("prompts") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise PrivacyConfigError(f"{source}: 'prompts' must be a non-empty list of text prompts")
    out: list[str] = []
    seen: set[str] = set()
    for p in items:
        if not isinstance(p, str) or not p.strip():
            raise PrivacyConfigError(f"{source}: every prompt must be a non-empty string, got {p!r}")
        text = " ".join(p.split())
        if text.casefold() in seen:
            raise PrivacyConfigError(f"{source}: prompt {text!r} is listed twice")
        seen.add(text.casefold())
        out.append(text)
    return tuple(out)


def load_privacy_prompts(path: str | Path | None = None) -> tuple[str, ...]:
    """Read and validate the privacy prompts file (default ``config.privacy_prompts_path()``)."""
    target = Path(path) if path is not None else config.privacy_prompts_path()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise PrivacyConfigError(f"privacy prompts file not found: {target}") from None
    except OSError as e:
        raise PrivacyConfigError(f"cannot read privacy prompts file {target}: {e}") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise PrivacyConfigError(f"privacy prompts file {target} is not valid JSON: {e}") from None
    return parse_privacy_prompts(data, source=str(target))


# ----- masks and images --------------------------------------------------------------------


def _read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise _NoSuchImage(f"image not found: {path.name}")
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as e:
        raise MissingImage(f"cannot read image {path.name}: {e}") from None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise MissingImage(f"not a readable image: {path.name}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    if path.suffix.lower() == ".png":
        ok, buf = cv2.imencode(".png", image)
    else:
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise PrivacyError(f"could not encode {path.name}")
    _write_bytes(path, buf.tobytes())


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _rle_to_array(mask: Mapping[str, Any]) -> np.ndarray:
    """Decode an RLE mask into a (height, width) bool array (validated first)."""
    try:
        clean = validate_rle(dict(mask))
    except (ValueError, TypeError) as e:
        raise InvalidMask(f"invalid mask: {e}") from None
    height, width = clean["size"]
    counts = np.asarray(clean["counts"], dtype=np.int64)
    values = (np.arange(len(counts)) % 2).astype(np.uint8)  # runs alternate 0, 1, 0, ...
    return np.repeat(values, counts).reshape(height, width).astype(bool)


def _box_array(box: Any, width: int, height: int) -> np.ndarray:
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        raise InvalidMask(f"invalid box: {box!r}") from None
    if not all(np.isfinite([x1, y1, x2, y2])) or x2 <= x1 or y2 <= y1:
        raise InvalidMask(f"invalid box: {box!r}")
    arr = np.zeros((height, width), dtype=bool)
    arr[max(0, int(np.floor(y1))):min(height, int(np.ceil(y2))),
        max(0, int(np.floor(x1))):min(width, int(np.ceil(x2)))] = True
    return arr


def _mask_array(item: Any, width: int, height: int) -> np.ndarray:
    """One mask given to ``blur_image`` as a (h, w) bool array in its own resolution."""
    if item is None:
        raise InvalidMask("invalid mask: None")
    if hasattr(item, "box") and not isinstance(item, (Mapping, np.ndarray)):  # a Detection
        mask = getattr(item, "mask", None)
        if mask is None:
            return _box_array(item.box, width, height)  # no mask: fall back to the box
        item = mask
    if isinstance(item, Mapping):
        return _rle_to_array(item)
    if isinstance(item, (str, bytes)):
        raise InvalidMask("invalid mask: text")
    try:
        arr = np.asarray(item)
    except Exception:
        raise InvalidMask(f"invalid mask: {type(item).__name__}") from None
    if arr.ndim != 2 or arr.size == 0 or arr.dtype.kind not in "biuf":
        raise InvalidMask("invalid mask: expected an RLE dict or a non-empty 2-D array of numbers")
    if arr.dtype.kind == "f" and not np.isfinite(arr).all():
        raise InvalidMask("invalid mask: values must be finite")
    return arr != 0


def _fit_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale a mask to the image size. Shrinking keeps any pixel the mask touches (never uncovers)."""
    h, w = mask.shape
    if (w, h) == (width, height):
        return mask
    if width < w or height < h:
        scaled = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
        return scaled > 0
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def _blur_area(region: np.ndarray) -> np.ndarray:
    """Pixelate a region into about CELLS_ACROSS cells across its smaller side, then smooth it."""
    h, w = region.shape[:2]
    block = max(2, -(-min(h, w) // CELLS_ACROSS))
    small = cv2.resize(region, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    k = _odd(2 * block + 1)
    return cv2.GaussianBlur(pixelated, (k, k), 0)


def _resolve_dilate(dilate_px: Any) -> int:
    if dilate_px is None:
        return config.blur_dilate_px()
    if isinstance(dilate_px, bool) or not isinstance(dilate_px, (int, float)) or dilate_px != dilate_px \
            or dilate_px != int(dilate_px) or not 0 <= dilate_px <= 100:
        raise ValueError(f"dilate_px must be a whole number from 0 to 100, got {dilate_px!r}")
    return int(dilate_px)


def blur_image(
    src_path: str | Path,
    dst_path: str | Path,
    masks: Iterable[Any],
    *,
    dilate_px: int = 0,
) -> BlurResult:
    """Blur the masked areas of an image into ``dst_path``. See the module docstring."""
    dilate = _resolve_dilate(dilate_px)
    src, dst = Path(src_path), Path(dst_path)
    if src.resolve() == dst.resolve():
        raise PrivacyError("dst_path must be a different file than the source image")
    image = _read_image(src)
    union, mask_count = _union_of(masks, image.shape[1], image.shape[0])

    if not union.any():  # nothing to blur: an unchanged copy of the file
        if src.suffix.lower() == dst.suffix.lower():
            _write_bytes(dst, src.read_bytes())
        else:
            _write_image(dst, image)
        return BlurResult(dst, mask_count, 0)

    out, blurred_pixels = _blur_union(image, union, dilate)
    _write_image(dst, out)
    return BlurResult(dst, mask_count, blurred_pixels)


def _union_of(masks: Iterable[Any], width: int, height: int) -> tuple[np.ndarray, int]:
    """Validate every mask and return (union as a (height, width) bool array, number of masks)."""
    if masks is None or isinstance(masks, (str, bytes)):
        raise InvalidMask("masks must be a list of masks")
    if isinstance(masks, (Mapping, np.ndarray)) or hasattr(masks, "box"):
        masks = [masks]  # a single mask given without a list
    try:
        arrays = [_mask_array(m, width, height) for m in masks]  # validates every mask before anything is written
    except TypeError:
        raise InvalidMask("masks must be a list of masks") from None
    union = np.zeros((height, width), dtype=bool)
    for arr in arrays:
        union |= _fit_mask(arr, width, height)
    return union, len(arrays)


def _blur_union(image: np.ndarray, union: np.ndarray, dilate: int) -> tuple[np.ndarray, int]:
    """Blur the (non-empty) union area of an image, grown by ``dilate`` pixels. Returns (new image, pixels blurred)."""
    area = union.astype(np.uint8)
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        area = cv2.dilate(area, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(area, connectivity=8)
    out = image.copy()
    for label in range(1, count):
        x, y, w, h = (int(v) for v in stats[label, :4])
        inside = labels[y:y + h, x:x + w] == label
        patch = _blur_area(image[y:y + h, x:x + w])
        region = out[y:y + h, x:x + w]
        region[inside] = patch[inside]
    return out, int(area.sum())


def blur_frame(frame: np.ndarray, masks: Iterable[Any], *, dilate_px: int = 0) -> np.ndarray:
    """Blur the masked areas of one in-memory frame (a (height, width, 3) uint8 array, e.g. BGR) and return a NEW
    array of the same shape; ``frame`` is never modified. Same masks, validation, union, scaling, dilation and blur
    as ``blur_image`` (which shares this implementation), so an invalid mask raises ``InvalidMask`` and no
    masks / empty masks give an unchanged copy. Used by the video export (task 22)."""
    dilate = _resolve_dilate(dilate_px)
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 \
            or frame.shape[0] < 1 or frame.shape[1] < 1:
        raise PrivacyError("frame must be a non-empty (height, width, 3) uint8 array")
    union, _ = _union_of(masks, frame.shape[1], frame.shape[0])
    if not union.any():
        return frame.copy()
    return _blur_union(frame, union, dilate)[0]


# ----- candidates --------------------------------------------------------------------------


def _blur_name(path: Path) -> Path:
    return path.with_name(path.stem + BLUR_SUFFIX + ".jpg")


def _note(exc: BaseException) -> str:
    """One short line about a failure, without a key or bearer token."""
    text = str(exc).strip()
    text = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    key = config.meta_api_key()
    if key and len(key) >= 4:
        text = text.replace(key, "***")
    text = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+", "Bearer ***", text)
    text = " ".join(text.split())
    return text if len(text) <= MAX_NOTE_CHARS else text[: MAX_NOTE_CHARS - 3].rstrip() + "..."


def _detections_of(result: Any) -> list[Any]:
    """Every detection of a provider's answer; ValueError if the answer is not a DetectionResult."""
    frames = getattr(result, "frames", None)
    if frames is None or isinstance(frames, (str, bytes, Mapping)):
        raise ValueError(f"expected a DetectionResult, got {type(result).__name__}")
    out = []
    for frame in frames:
        dets = getattr(frame, "detections", None)
        if dets is None:
            raise ValueError("expected a DetectionResult with frames of detections")
        out.extend(dets)
    return out


def _remove(paths: Iterable[Path | str | None]) -> None:
    for p in paths:
        if p:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                logger.warning("could not remove %s", p)


def _blur_one(cand: Any, provider: Any, prompts: list[str], dilate: int, written: list[Path]) -> dict[str, Any]:
    """Blur the best frame and the crop of one candidate. Raises on any problem; ``written`` collects
    the files created so the caller can remove them again."""
    out: dict[str, Any] = {"blurred_frame_path": None, "blurred_crop_path": None, "mask_count": 0}
    for column, key in (("best_frame_path", "blurred_frame_path"), ("crop_path", "blurred_crop_path")):
        stored = cand[column]
        if not stored:
            if column == "best_frame_path":
                raise PrivacyError("the candidate has no saved best frame")
            continue
        src = Path(stored)
        if not src.is_file():
            raise MissingImage(f"saved image is missing: {src.name}")
        detections = _detections_of(provider.detect(str(src), list(prompts)))
        target = _blur_name(src)
        written.append(target)
        blur_image(src, target, detections, dilate_px=dilate)
        out[key] = str(target)
        out["mask_count"] += len(detections)
    return out


def blur_drive(
    drive_id: str,
    *,
    provider: ModelProvider | None = None,
    db_path: str | Path | None = None,
    prompts: Iterable[str] | None = None,
    dilate_px: int | None = None,
) -> BlurSummary:
    """Blur the saved frames of the drive's accepted hazard candidates. See the module docstring."""
    if drive_id is None or not str(drive_id).strip():
        raise ValueError("drive_id must not be blank")
    drive_id = str(drive_id).strip()
    dilate = _resolve_dilate(dilate_px)
    prompt_list = list(parse_privacy_prompts(list(prompts), source="prompts")) if prompts is not None \
        else list(load_privacy_prompts())
    target = init_db(db_path)

    conn = get_connection(target)
    try:
        rows = conn.execute(
            "SELECT c.id, c.best_frame_path, c.crop_path, c.blurred_frame_path, c.blurred_crop_path, "
            "(v.accepted = 1 AND v.is_hazard = 1) AS verified "
            "FROM candidates c LEFT JOIN verdicts v ON v.candidate_id = c.id "
            "WHERE c.drive_id = ? ORDER BY c.id", (drive_id,)
        ).fetchall()
    finally:
        conn.close()
    todo = [r for r in rows if r["verified"]]
    old_files = {Path(p) for r in rows for p in (r["blurred_frame_path"], r["blurred_crop_path"]) if p}

    summary = BlurSummary(drive_id=drive_id, prompts=prompt_list, candidate_count=len(todo),
                          ok_count=0, failed_count=0)
    if todo and provider is None:
        provider = get_provider()

    updates: list[dict[str, Any]] = []
    new_files: set[Path] = set()
    for cand in todo:
        written: list[Path] = []
        try:
            done = _blur_one(cand, provider, prompt_list, dilate, written)
        except Exception as exc:  # fail closed: this candidate gets no blurred image at all
            _remove(written)
            _remove(_expected_outputs(cand))
            note = _note(exc)
            logger.warning("privacy blur failed for candidate %s of drive %s: %s", cand["id"], drive_id, note)
            updates.append({"id": cand["id"], "frame": None, "crop": None, "status": BLUR_FAILED, "note": note})
            summary.failed_count += 1
            summary.candidates.append({
                "candidate_id": cand["id"], "status": BLUR_FAILED, "note": note,
                "blurred_frame_path": None, "blurred_crop_path": None, "mask_count": 0,
            })
            continue
        new_files.update(written)
        updates.append({"id": cand["id"], "frame": done["blurred_frame_path"], "crop": done["blurred_crop_path"],
                        "status": BLUR_OK, "note": None})
        summary.ok_count += 1
        summary.mask_count += done["mask_count"]
        summary.candidates.append({
            "candidate_id": cand["id"], "status": BLUR_OK, "note": None,
            "blurred_frame_path": done["blurred_frame_path"], "blurred_crop_path": done["blurred_crop_path"],
            "mask_count": done["mask_count"],
        })

    verified_ids = {r["id"] for r in todo}
    others = [r["id"] for r in rows if r["id"] not in verified_ids]
    conn = get_connection(target)
    try:
        with conn:  # one transaction for the whole drive
            conn.executemany(
                "UPDATE candidates SET blurred_frame_path = :frame, blurred_crop_path = :crop, "
                "blur_status = :status, blur_note = :note WHERE id = :id", updates)
            conn.executemany(
                "UPDATE candidates SET blurred_frame_path = NULL, blurred_crop_path = NULL, "
                "blur_status = NULL, blur_note = NULL WHERE id = ?", [(i,) for i in others])
    except BaseException:
        _remove(new_files - old_files)
        raise
    finally:
        conn.close()
    _remove(old_files - new_files)  # blurred copies of an earlier run that this run does not reuse

    logger.info("privacy blur for drive %s: %d candidates, %d ok, %d failed, %d masks", drive_id,
                summary.candidate_count, summary.ok_count, summary.failed_count, summary.mask_count)
    if summary.failed_count:
        first = next(c for c in summary.candidates if c["status"] == BLUR_FAILED)
        raise BlurError(
            f"could not blur {summary.failed_count} of {summary.candidate_count} candidate(s) of drive "
            f"{drive_id} (first: candidate {first['candidate_id']}: {first['note']}); "
            "no unblurred image will be shown for them",
            summary=summary,
        )
    return summary


blur_candidates = blur_drive  # alias


def _expected_outputs(cand: Any) -> list[Path]:
    """The blurred file names a candidate would get (to remove leftovers of a failed attempt)."""
    return [_blur_name(Path(p)) for p in (cand["best_frame_path"], cand["crop_path"]) if p]
