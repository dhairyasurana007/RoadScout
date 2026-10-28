"""Muse Spark verification (task 9): one structured verdict per candidate.

    from backend.verification import verify_drive, load_verdicts, load_accepted_candidates
    summary = verify_drive(drive_id, chunks=preprocess_result)     # fake provider unless configured
    rows = load_verdicts(drive_id)                                 # every verdict, rejected too
    hazards = load_accepted_candidates(drive_id)                   # accepted candidates + verdict

``verify_drive(drive_id, *, chunks=None, provider=None, min_confidence=None, db_path=None,
prompt=None, prompts=None)``

- ``drive_id``: the drive whose candidates (task 8, ``load_candidates``) are verified.
- ``chunks``: how to find the chunk videos, exactly as ``aggregate_drive`` accepts them (a
  ``PreprocessResult``, a ``frame_timestamps.json`` path or its folder, or a list of ``Chunk``).
  Optional: it is only used to cut the neighbouring frames out of the candidate's chunk video.
  Without it (or when the chunk is not among them or cannot be read) the image set is just the
  best frame and the crop.
- ``provider``: any ``ModelProvider``. Default ``get_provider()`` (the fake provider unless
  ROADSCOUT_PROVIDER says otherwise; an unknown value raises ``ValueError``). The call is
  ``provider.verify(images, class_prompt=<the candidate's class>, prompt_version=<prompt
  version>, prompt_text=<rendered prompt>)``; ``prompt_text`` is left out for a provider whose
  ``verify`` does not accept it. Retries, timeouts and the spend cap live in the provider layer.
- ``min_confidence``: minimum confidence to accept, 0 to 1 inclusive. Default
  ``config.verify_min_confidence()`` (env ``ROADSCOUT_VERIFY_MIN_CONFIDENCE``, 0.5). A value
  outside 0 to 1, NaN, inf or non-numeric (keyword or env) raises ``ValueError`` before any work.
- ``prompt``: a ``VerificationPrompt``, a path to a prompt file, or None for
  ``config/verification_prompt.json`` (env ``ROADSCOUT_VERIFY_PROMPT``; format
  ``{"version": ..., "prompt": ..., "schema": ...}``, see ``backend/verification_prompt.py``). A bad
  file raises ``VerificationPromptError`` (a ``ValueError``) before any provider call or write,
  even for a drive with no candidates.
- ``prompts``: a ``HazardPrompts`` giving the known class names (default: the hazard prompts file).
- ``db_path``: SQLite file (default ``config.db_path()``); ``init_db`` is run first.

Image set per candidate, in this order, as the ``images`` list (file paths and raw JPEG bytes):

1. the best frame (``best_frame_path``),
2. the crop (``crop_path``; left out when the candidate has none),
3. the frame just before the best frame and 4. the frame just after it, cut from the candidate's
   chunk video. A neighbour that does not exist (first or last frame of the chunk) is skipped, so
   there are 0 to 2 of them. Neighbours never cross into another chunk. They are passed as JPEG
   bytes and not written to disk.

Verdict checks (a failure is a ``VerificationError`` naming the candidate id and nothing is stored
for it): the provider must return a ``Verdict`` (or a dict that ``Verdict.from_dict`` accepts);
``is_hazard`` a bool; ``severity`` a whole number, 1 to 3 for a hazard (0 to 3 for a non-hazard);
``confidence`` a finite number from 0 to 1; ``lane`` None or text; ``reason`` and
``report_description`` text; ``hazard_class`` one of the class names in the hazard prompts file
(compared ignoring case and spacing, stored in the file's spelling), for hazards and non-hazards
alike; it may differ from the detector's class and is then stored as the model gave it.

Table ``verdicts`` (``backend/db.py``): ``candidate_id`` (primary key, foreign key to candidates,
deleted with the candidate), ``drive_id``, ``is_hazard`` (0/1), ``hazard_class``, ``severity``,
``lane``, ``confidence``, ``reason``, ``report_description``, ``model``, ``prompt_version``,
``accepted`` (0/1), ``created_at`` (UTC ISO text). ``model`` is the one the verdict reports (the
provider's name if it leaves it empty); ``prompt_version`` is always the prompt file's version.

Acceptance rule: ``accepted = is_hazard and confidence >= min_confidence``. Rejected verdicts (not a
hazard, or too unsure) are stored too, because the eval ablation (SAM 3.1 alone vs plus Muse Spark)
needs them. Only accepted candidates go on to deduplication.

Rules:

- Idempotent per drive: the drive's old verdicts are replaced in ONE transaction (other drives
  are untouched).
- All-or-nothing, with one exception: every candidate is verified in memory first. If the provider
  raises for any candidate, or returns something malformed, ``VerificationError`` (``candidate_id``,
  ``drive_id``; the original error is ``__cause__``) is raised, nothing is written and the drive's
  earlier verdicts stay as they were. The exception is a transient failure that outlived the
  provider's retries (``RetriesExhausted`` or ``ProviderTimeoutError``): that candidate is skipped (no
  verdict row, never accepted, logged by id and exception type only) and the loop goes on. The
  summary lists the skipped ids in ``failed_candidate_ids``. Spend-cap errors, missing config, permanent
  provider errors (any other ``ProviderError``), malformed verdicts and failures building the image set
  still abort. Outage guard: 3 skips in a row (a success resets the count), or every candidate skipped,
  raises ``VerificationError`` and stores nothing.
- No candidates (or an unknown drive) means success with an empty summary: no provider call, no write.
- Bad arguments (blank ``drive_id``, bad ``min_confidence``) raise ``ValueError`` before any work.
- Unblurred frames leave the machine here (the one accepted exception, see the README); no keys
  are read or logged by this module.
"""
from __future__ import annotations

import inspect
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2

from . import config
from .candidates import load_candidates
from .db import get_connection, init_db
from .detection import _normalise_chunks
from .hazard_prompts import HazardPrompts, load_hazard_prompts
from .providers import (
    ModelProvider,
    ProviderError,
    ProviderTimeoutError,
    RetriesExhausted,
    Verdict,
    get_provider,
)
from .verification_prompt import VerificationPrompt, load_verification_prompt

logger = logging.getLogger(__name__)

JPEG_QUALITY = 90

#: Transient failures in a row (a success resets the count) after which verification gives up.
MAX_CONSECUTIVE_SKIPS = 3

#: Provider errors that mean "the network or service was unavailable": that candidate is skipped.
SKIPPABLE_ERRORS = (RetriesExhausted, ProviderTimeoutError)

_INSERT = (
    "INSERT INTO verdicts (candidate_id, drive_id, is_hazard, hazard_class, severity, lane, "
    "confidence, reason, report_description, model, prompt_version, accepted, created_at) "
    "VALUES (:candidate_id, :drive_id, :is_hazard, :hazard_class, :severity, :lane, :confidence, "
    ":reason, :report_description, :model, :prompt_version, :accepted, :created_at)"
)


class VerificationError(ProviderError):
    """Verification failed for one candidate; nothing was stored. ``__cause__`` is the original error."""

    def __init__(self, message: str, *, candidate_id: int | None = None, drive_id: str | None = None):
        super().__init__(message)
        self.candidate_id = candidate_id
        self.drive_id = drive_id


@dataclass
class VerificationSummary:
    """What a ``verify_drive`` call stored."""

    drive_id: str
    min_confidence: float
    prompt_version: str  # the version of the prompt file that was sent
    candidate_count: int
    accepted_count: int
    rejected_count: int
    verdicts: list[dict[str, Any]] = field(default_factory=list)  # as load_verdicts returns
    # Candidates skipped after a transient failure (network, timeout) that outlived the retries, in
    # candidate order. The four counts and ``verdicts`` above cover the verified candidates only.
    failed_candidate_ids: list[int] = field(default_factory=list)


# ----- helpers -----------------------------------------------------------------------------


def _norm(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def _accepts_prompt_text(provider: Any) -> bool:
    """True if ``provider.verify`` takes a ``prompt_text`` keyword (older providers may not)."""
    try:
        params = inspect.signature(provider.verify).parameters
    except (TypeError, ValueError):
        return True
    return "prompt_text" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _neighbour_images(candidate: dict[str, Any], infos: dict[int, Any]) -> list[bytes]:
    """JPEG bytes of the frame before and the frame after the best frame, inside its chunk."""
    info = infos.get(candidate["chunk"])
    if info is None:
        return []
    k = int(candidate["best_frame_index"]) - info.start_frame
    wanted = [t for t in (k - 1, k + 1) if 0 <= t < info.frame_count]
    if not wanted:
        return []
    path = Path(info.path)
    cap = cv2.VideoCapture(str(path)) if path.is_file() else None
    if cap is None or not cap.isOpened():
        logger.warning("candidate %s: chunk video %s cannot be opened, no neighbouring frames",
                       candidate["id"], path)
        return []
    out: list[bytes] = []
    try:
        for pos in range(max(wanted) + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if pos in wanted:
                done, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if done:
                    out.append(buf.tobytes())
    finally:
        cap.release()
    return out


def _image_set(candidate: dict[str, Any], infos: dict[int, Any]) -> list[Any]:
    images: list[Any] = [str(candidate["best_frame_path"])]
    if candidate.get("crop_path"):
        images.append(str(candidate["crop_path"]))
    images.extend(_neighbour_images(candidate, infos))
    return images


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_verdict(raw: Any, classes: dict[str, str]) -> Verdict:
    """Validate what the provider returned. Raises ValueError with a plain reason."""
    if isinstance(raw, Mapping):
        data = dict(raw)
        if "hazard_class" not in data and "class" in data:  # the prompt schema's name for the field
            data["hazard_class"] = data.pop("class")
        try:
            raw = Verdict.from_dict(data)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"verdict dict is malformed: {type(e).__name__}: {e}") from None
    if not isinstance(raw, Verdict) and not all(
        hasattr(raw, f) for f in ("is_hazard", "hazard_class", "severity", "lane", "confidence",
                                   "reason", "report_description")
    ):
        raise ValueError(f"expected a Verdict, got {type(raw).__name__}")
    if type(raw.is_hazard) is not bool:
        raise ValueError(f"is_hazard must be true or false, got {raw.is_hazard!r}")
    if not isinstance(raw.hazard_class, str):
        raise ValueError(f"class must be text, got {raw.hazard_class!r}")
    if isinstance(raw.severity, bool) or not isinstance(raw.severity, int):
        raise ValueError(f"severity must be a whole number, got {raw.severity!r}")
    low = 1 if raw.is_hazard else 0
    if not low <= raw.severity <= 3:
        raise ValueError(f"severity must be {low} to 3, got {raw.severity!r}")
    if not _is_number(raw.confidence) or not math.isfinite(raw.confidence) or not 0.0 <= raw.confidence <= 1.0:
        raise ValueError(f"confidence must be a number from 0 to 1, got {raw.confidence!r}")
    if raw.lane is not None and not isinstance(raw.lane, str):
        raise ValueError(f"lane must be text or null, got {raw.lane!r}")
    for name in ("reason", "report_description"):
        if not isinstance(getattr(raw, name), str):
            raise ValueError(f"{name} must be text, got {getattr(raw, name)!r}")
    known = classes.get(_norm(raw.hazard_class))
    if known is None:
        raise ValueError(
            f"class {raw.hazard_class!r} is not a known hazard class ({', '.join(classes.values())})"
        )
    return Verdict(
        is_hazard=raw.is_hazard,
        hazard_class=known,
        severity=raw.severity,
        lane=raw.lane,
        confidence=float(raw.confidence),
        reason=raw.reason,
        report_description=raw.report_description,
        model=str(getattr(raw, "model", "") or ""),
        prompt_version=str(getattr(raw, "prompt_version", "") or ""),
    )


def _resolve_min_confidence(value: float | None) -> float:
    if value is None:
        return config.verify_min_confidence()
    if not _is_number(value) or not 0.0 <= float(value) <= 1.0:  # also rejects NaN
        raise ValueError(f"min_confidence must be a number from 0 to 1, got {value!r}")
    return float(value)


# ----- main --------------------------------------------------------------------------------


def verify_drive(
    drive_id: str,
    *,
    chunks: Any = None,
    provider: ModelProvider | None = None,
    min_confidence: float | None = None,
    db_path: str | Path | None = None,
    prompt: VerificationPrompt | str | Path | None = None,
    prompts: HazardPrompts | None = None,
) -> VerificationSummary:
    """Verify every candidate of a drive with Muse Spark and store the verdicts. See the module docstring."""
    if drive_id is None or not str(drive_id).strip():
        raise ValueError("drive_id must not be blank")
    drive_id = str(drive_id).strip()
    threshold = _resolve_min_confidence(min_confidence)
    provider = provider if provider is not None else get_provider()
    if prompt is None or isinstance(prompt, (str, Path)):
        prompt = load_verification_prompt(prompt)
    elif not isinstance(prompt, VerificationPrompt):
        raise TypeError("prompt must be a VerificationPrompt, a path, or None")
    target = init_db(db_path)

    candidates = load_candidates(drive_id, db_path=target)
    if not candidates:
        return VerificationSummary(drive_id, threshold, prompt.version, 0, 0, 0, [])

    infos: dict[int, Any] = {}
    if chunks is not None:
        infos = {c.index: c for c in _normalise_chunks(chunks)[0]}
    hazard_prompts = prompts if prompts is not None else load_hazard_prompts()
    class_names = hazard_prompts.class_names
    classes = {_norm(n): n for n in class_names}
    send_text = _accepts_prompt_text(provider)
    now = datetime.now(timezone.utc).isoformat()

    rows: list[dict[str, Any]] = []
    failed_ids: list[int] = []
    in_a_row = 0
    for cand in candidates:
        cid = cand["id"]
        skipped: Exception | None = None
        raw: Any = None
        try:
            images = _image_set(cand, infos)
            kwargs: dict[str, Any] = {"class_prompt": cand["hazard_class"], "prompt_version": prompt.version}
            if send_text:
                kwargs["prompt_text"] = prompt.render(cand["hazard_class"], class_names)
            try:
                raw = provider.verify(images, **kwargs)
            except SKIPPABLE_ERRORS as e:
                skipped = e
        except Exception as e:
            raise VerificationError(
                f"verification failed for candidate {cid} of drive {drive_id}; nothing was stored: "
                f"{type(e).__name__}: {e}", candidate_id=cid, drive_id=drive_id,
            ) from e
        if skipped is not None:
            failed_ids.append(cid)
            in_a_row += 1
            logger.warning("candidate %s of drive %s could not be verified (%s); skipped",
                           cid, drive_id, type(skipped).__name__)
            if in_a_row >= MAX_CONSECUTIVE_SKIPS:
                raise VerificationError(
                    f"verification gave up on drive {drive_id} after {in_a_row} candidates in a row failed "
                    f"({type(skipped).__name__}: network or service unavailable); nothing was stored",
                    candidate_id=cid, drive_id=drive_id,
                ) from skipped
            continue
        in_a_row = 0
        try:
            verdict = _check_verdict(raw, classes)
        except ValueError as e:
            raise VerificationError(
                f"malformed verdict for candidate {cid} of drive {drive_id}; nothing was stored: {e}",
                candidate_id=cid, drive_id=drive_id,
            ) from e
        rows.append({
            "candidate_id": cid,
            "drive_id": drive_id,
            "is_hazard": int(verdict.is_hazard),
            "hazard_class": verdict.hazard_class,
            "severity": verdict.severity,
            "lane": verdict.lane,
            "confidence": verdict.confidence,
            "reason": verdict.reason,
            "report_description": verdict.report_description,
            "model": verdict.model or str(getattr(provider, "name", "") or "unknown"),
            "prompt_version": prompt.version,
            "accepted": int(verdict.is_hazard and verdict.confidence >= threshold),
            "created_at": now,
        })

    if failed_ids and not rows:
        raise VerificationError(
            f"verification failed for every candidate of drive {drive_id} ({len(failed_ids)}: network or "
            f"service unavailable); nothing was stored", candidate_id=failed_ids[-1], drive_id=drive_id,
        )

    conn = get_connection(target)
    try:
        with conn:  # one transaction: commit on success, roll back on any error
            conn.execute("DELETE FROM verdicts WHERE drive_id = ?", (drive_id,))
            conn.executemany(_INSERT, rows)
    finally:
        conn.close()

    accepted = sum(r["accepted"] for r in rows)
    summary = VerificationSummary(
        drive_id=drive_id,
        min_confidence=threshold,
        prompt_version=prompt.version,
        candidate_count=len(rows),
        accepted_count=accepted,
        rejected_count=len(rows) - accepted,
        verdicts=load_verdicts(drive_id, db_path=target),
        failed_candidate_ids=failed_ids,
    )
    logger.info("verification for drive %s: %d candidates, %d accepted, %d skipped (min_confidence %s, prompt %s)",
                drive_id, summary.candidate_count, accepted, len(failed_ids), threshold, prompt.version)
    return summary


# ----- reading -----------------------------------------------------------------------------

_ORDER = "c.first_timestamp, c.chunk, c.track_id, c.hazard_class, c.id"


def _verdict_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    d = dict(row)
    d["is_hazard"] = bool(d["is_hazard"])
    d["accepted"] = bool(d["accepted"])
    return d


def load_verdicts(
    drive_id: str, *, db_path: str | Path | None = None, accepted_only: bool = False
) -> list[dict[str, Any]]:
    """The stored verdicts of a drive as dicts (column names as keys; ``is_hazard`` and ``accepted``
    are bools), in candidate order (first_timestamp, chunk, track id, class). Rejected verdicts are
    included unless ``accepted_only`` is true."""
    conn = get_connection(init_db(db_path))
    try:
        cur = conn.execute(
            "SELECT v.* FROM verdicts v JOIN candidates c ON c.id = v.candidate_id "
            "WHERE v.drive_id = ? " + ("AND v.accepted = 1 " if accepted_only else "")
            + f"ORDER BY {_ORDER}",
            (str(drive_id),),
        )
        return [_verdict_dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def load_accepted_candidates(drive_id: str, *, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """The drive's accepted candidates, each a ``load_candidates`` dict plus ``"verdict"``: its
    verdict dict (as ``load_verdicts`` returns). Candidates without a verdict, and those whose
    verdict was rejected, are left out. Same order as ``load_candidates``."""
    target = init_db(db_path)
    verdicts = {v["candidate_id"]: v for v in load_verdicts(drive_id, db_path=target, accepted_only=True)}
    out = []
    for cand in load_candidates(drive_id, db_path=target):
        verdict = verdicts.get(cand["id"])
        if verdict is not None:
            out.append({**cand, "verdict": verdict})
    return out


def get_verdict(candidate_id: int, *, db_path: str | Path | None = None) -> dict[str, Any] | None:
    """One candidate's verdict dict, or None if it has none."""
    conn = get_connection(init_db(db_path))
    try:
        row = conn.execute("SELECT * FROM verdicts WHERE candidate_id = ?", (candidate_id,)).fetchone()
        return _verdict_dict(row) if row is not None else None
    finally:
        conn.close()
