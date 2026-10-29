"""Live provider: the real Meta Model API calls (task 26).

It needs META_API_KEY (read from the environment, never logged, echoed, put in an error
message or shown in ``repr``). Requests go to ``base_url`` (argument, else
ROADSCOUT_META_BASE_URL, else https://api.meta.ai/v1) with ``Authorization: Bearer <key>``.
Models: ROADSCOUT_SAM_MODEL (default ``sam-3.1``) and ROADSCOUT_VERIFY_MODEL (default
``muse-spark-1.3``). An ``httpx.Client`` can be injected with ``client=`` (tests use a
``MockTransport``); the default client is built here.

Detection (``_do_detect``): the chunk is uploaded once (``POST /files``, multipart, field
``file`` plus ``purpose=user_data``), then each prompt phrase gets one non-streaming
``POST /responses`` with ``input_video`` and ``metadata {"mask_encoding": "one_bit",
"include_confidence": "true"}``, and the upload is always deleted afterwards (a failed delete
is ignored). The reply text has one line per frame that has matches::

    <3f>0<|box;x1=72;y1=382;x2=95;y2=396;w=406;h=720;c=0.64|><|mask;x=0;y=0;c=0.64;data=15,24,!PAYLOAD|>,2<|box;...|>...

``<Nf>`` is the zero-based frame index in the uploaded chunk, then comma-separated records
``<object id><|box;...|><|mask;...|>``. Box corners are inclusive pixels, so ``Detection.box``
is ``(x1, y1, x2 + 1, y2 + 1)``. The mask raster covers exactly the box region and is decoded
with Meta's ``meta_sam_parser``, pasted at (x1, y1) into a full-frame RLE mask. A mask that
cannot be decoded or does not fit its box becomes the filled box (the privacy blur fails
closed); a record that cannot be parsed at all is skipped. Mapping: timestamp = N / fps (fps
of the uploaded chunk, from ffprobe), score = ``c`` (0.5 when missing), track id =
prompt index * 1000 + object id (ids from different phrases never collide), and the records
of all phrases for one frame form that frame's single ``FrameRecord``. Empty text means
nothing found: an empty ``DetectionResult``.

Images (task 26d): ``detect`` on a file that is an image (JPEG or PNG, decided by its magic bytes, the
extension only as a fallback) is not uploaded and not sent as video. The privacy blur uses this. Each
prompt gets one ``POST /responses`` whose media part is ``{"type": "input_image", "image_url":
"data:<mime>;base64,..."}``, with the same model, metadata and parsing as video; the result has at most
one frame (``frame_index`` 0, ``timestamp`` 0.0, no ffprobe). No ``/files`` call is made. An image over
50,000,000 bytes, an empty one or an unreadable one raises ``ProviderError`` before any request. The
estimate is $0.0025 per prompt (SAM image pricing) unless ROADSCOUT_COST_PER_DETECT_USD is set, which
still wins and stays per prompt. Errors, retries and key redaction are the same as for video.

Verification (``_do_verify``): one ``POST /responses`` to Muse Spark with the prompt and one
``input_image`` data URL per image (at most 50), asking for a JSON reply in the verdict
schema. The reply (Markdown fences tolerated) becomes a ``Verdict``; anything that does not
fit the schema raises ``ProviderError``.

Errors: HTTP 429, 500, 502 and 503 and transport errors (including timeouts) raise
``TransientProviderError`` (the guard layer retries them, honouring Retry-After up to 30 s);
every other failure status (400, 401, 402, 403, 404, 413, 504, ...) raises a plain
``ProviderError`` that is not retried. Messages come from the body's ``error.message`` and
never hold the key, the request body or a file path.

Cost: the spend cap is checked before the upload and before each request. Estimates are
$0.012 per detect prompt and $0.007 per verify (ROADSCOUT_COST_PER_DETECT_USD /
ROADSCOUT_COST_PER_VERIFY_USD override them) and the estimate is what is charged.

The guard layer (``_call``): spend-cap check before the call, per-call timeout (scaled by
the number of prompts for detect, which makes one request per prompt), retries with
exponential backoff on transient errors and timeouts, and a charge plus cost-log line after
success. A retry repeats the whole call (upload included). Defaults: five attempts with waits of
2, 4, 8 and 16 s between them (plus Retry-After), so a brief network outage is ridden out.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

import httpx
import numpy as np

from .. import config
from ..media import InvalidMedia, ProbeUnavailable, probe_video
from .base import Image, ModelProvider
from .cost import SpendTracker
from .errors import (
    ProviderConfigError,
    ProviderError,
    ProviderTimeoutError,
    TransientProviderError,
)
from .retry import call_with_retries, call_with_timeout
from .types import (
    Box,
    Detection,
    DetectionResult,
    FrameRecord,
    Verdict,
    rle_from_box,
    validate_rle,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_SAM_MODEL = "sam-3.1"
DEFAULT_VERIFY_MODEL = "muse-spark-1.3"

# detect is per prompt; detect_image is SAM image pricing ($2.50 per 1,000 images) per prompt
DEFAULT_COST_USD = {"detect": 0.012, "verify": 0.007, "detect_image": 0.0025}

MAX_IMAGE_BYTES = 50_000_000  # largest image file sent to SAM

MAX_VERIFY_IMAGES = 50
DEFAULT_SCORE = 0.5
TRACK_ID_STRIDE = 1000  # track id = prompt index * 1000 + object id

UPLOAD_TIMEOUT_S = 300.0
RESPONSE_TIMEOUT_S = 600.0
DELETE_TIMEOUT_S = 60.0
MAX_ERROR_CHARS = 300

TRANSIENT_STATUSES = frozenset({429, 500, 502, 503})

#: The verdict schema (same as ``config/verification_prompt.json``), sent as the JSON-schema format.
VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_hazard": {"type": "boolean"},
        "class": {"type": "string"},
        "severity": {"type": "integer", "minimum": 0, "maximum": 3},
        "lane": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "report_description": {"type": "string"},
    },
    "required": ["is_hazard", "class", "severity", "lane", "confidence", "reason", "report_description"],
}

# One frame marker, or one record: object id, box, optional mask. Base85 payloads never hold "|",
# so a record is always consumed whole and a marker-looking string inside a payload is not seen.
_TOKEN = re.compile(
    r"<(?P<frame>\d+)f>"
    r"|(?P<oid>\d+)<\|box;(?P<box>[^|]*)\|>(?:<\|mask;(?P<mask>[^|]*)\|>)?",
    re.S,
)
_MASK_DATA = re.compile(r"data=(\d+),(\d+),(.*)\Z", re.S)  # H,W,payload; the payload may hold ";" or ","
_FENCE = re.compile(r"^\s*```[A-Za-z0-9_-]*[ \t]*\r?\n?(.*?)\r?\n?[ \t]*```\s*$", re.S)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# --- helpers ------------------------------------------------------------------


def _output_text(payload: Any) -> str:
    """Every text part of a Responses payload (``output[].content[].text``), concatenated."""
    parts: list[str] = []
    if isinstance(payload, dict):
        for item in payload.get("output") or []:
            if not isinstance(item, dict):
                continue
            for part in item.get("content") or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
    return "".join(parts)


def _fields(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for piece in text.split(";"):
        key, sep, value = piece.partition("=")
        if sep:
            out[key.strip()] = value.strip()
    return out


def _number(raw: str | None) -> float | None:
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _full_frame_mask(
    width: int, height: int, x1: int, y1: int, x2: int, y2: int, mask_field: str | None
) -> dict[str, Any]:
    """Full-frame RLE mask from Meta's box-sized raster, or the filled box if it is unusable."""
    from meta_sam_parser import SegmentationMask, decode_mask_to_raster  # official decoder

    filled = rle_from_box(width, height, (x1, y1, x2 + 1, y2 + 1))
    if not mask_field:
        return filled
    try:
        match = _MASK_DATA.search(mask_field)
        if match is None:
            return filled
        raster_h, raster_w, payload = int(match.group(1)), int(match.group(2)), match.group(3)
        box_w, box_h = x2 - x1 + 1, y2 - y1 + 1
        if raster_w != box_w or raster_h != box_h:
            return filled
        encoding = "lossless" if payload.startswith("~") else "one_bit"
        raster = decode_mask_to_raster(
            SegmentationMask(encoding=encoding, payload=payload, width=raster_w, height=raster_h)
        )
        if len(raster) != raster_w * raster_h:
            return filled
        patch = np.frombuffer(raster, dtype=np.uint8).reshape(raster_h, raster_w) != 0
        canvas = np.zeros((height, width), dtype=np.uint8)
        left, top = max(x1, 0), max(y1, 0)
        right, bottom = min(x1 + raster_w, width), min(y1 + raster_h, height)
        if right > left and bottom > top:
            canvas[top:bottom, left:right] = patch[top - y1:bottom - y1, left - x1:right - x1]
        flat = canvas.ravel()
        change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        runs = np.diff(np.concatenate(([0], change, [flat.size]))).tolist()
        if flat[0]:
            runs.insert(0, 0)  # counts always start with a background run
        return validate_rle({"size": [height, width], "counts": runs})
    except Exception:  # noqa: BLE001 - any decode problem must fail closed to the box
        return filled


def _parse_record(oid: str, box_text: str, mask_text: str | None, prompt: str, prompt_index: int) -> Detection | None:
    """One ``<id><|box|><|mask|>`` record as a Detection, or None if the box cannot be read."""
    try:
        box = _fields(box_text)
        x1, y1, x2, y2 = (_number(box.get(k)) for k in ("x1", "y1", "x2", "y2"))
        if None in (x1, y1, x2, y2):
            return None
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # type: ignore[arg-type]
        score = _number(box.get("c"))
        score = DEFAULT_SCORE if score is None else min(1.0, max(0.0, score))
        width, height = _number(box.get("w")), _number(box.get("h"))
        mask = None
        if width and height and width >= 1 and height >= 1:
            mask = _full_frame_mask(int(width), int(height), x1, y1, x2, y2, mask_text)
        return Detection(
            track_id=prompt_index * TRACK_ID_STRIDE + int(oid),
            box=Box(x1, y1, x2 + 1, y2 + 1),
            score=score,
            prompt=prompt,
            mask=mask,
        )
    except (ValueError, TypeError, OverflowError):
        return None


def _parse_detections(text: str, prompt: str, prompt_index: int, into: dict[int, list[Detection]],
                      default_frame: int | None = None) -> None:
    """Add every parsable record of one response text to ``into`` (frame index -> detections).
    Records before any frame marker go to ``default_frame`` (None: they are skipped)."""
    frame: int | None = default_frame
    for m in _TOKEN.finditer(text):
        if m.group("frame") is not None:
            frame = int(m.group("frame"))
            continue
        if frame is None:
            continue
        det = _parse_record(m.group("oid"), m.group("box"), m.group("mask"), prompt, prompt_index)
        if det is not None:
            into.setdefault(frame, []).append(det)


def _image_mime(data: bytes) -> str:
    return "image/png" if data.startswith(_PNG_MAGIC) else "image/jpeg"


_JPEG_MAGIC = b"\xff\xd8\xff"
_EXTENSION_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def _sniff_image_mime(path: Path) -> str | None:
    """"image/jpeg" or "image/png" when the file is an image: decided by its magic bytes, with the
    extension only as a fallback when the content matches neither. None for anything else (a video)."""
    try:
        with path.open("rb") as handle:
            head = handle.read(len(_PNG_MAGIC))
    except OSError:
        return None
    if head.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    if head.startswith(_PNG_MAGIC):
        return "image/png"
    return _EXTENSION_MIME.get(path.suffix.lower())


def _json_object(text: str) -> Any:
    """Parse the model's reply as one JSON object, tolerating Markdown code fences or stray text."""
    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(body[start:end + 1])
            except json.JSONDecodeError:
                pass
    raise ProviderError("Muse Spark reply was not valid JSON")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _verdict_from_reply(text: str, model: str, prompt_version: str) -> Verdict:
    data = _json_object(text)
    if not isinstance(data, dict):
        raise ProviderError("Muse Spark reply was not a JSON object")
    missing = [k for k in VERDICT_SCHEMA["required"] if k not in data]
    if missing:
        raise ProviderError(f"Muse Spark reply is missing {', '.join(missing)}")
    bad: str | None = None
    severity = data["severity"]
    if isinstance(severity, float) and severity.is_integer():
        severity = int(severity)
    if type(data["is_hazard"]) is not bool:
        bad = "is_hazard"
    elif not isinstance(data["class"], str):
        bad = "class"
    elif isinstance(severity, bool) or not isinstance(severity, int) or not 0 <= severity <= 3:
        bad = "severity"
    elif data["lane"] is not None and not isinstance(data["lane"], str):
        bad = "lane"
    elif not _is_number(data["confidence"]) or not 0 <= data["confidence"] <= 1:
        bad = "confidence"
    elif not isinstance(data["reason"], str):
        bad = "reason"
    elif not isinstance(data["report_description"], str):
        bad = "report_description"
    if bad:
        raise ProviderError(f"Muse Spark reply does not match the verdict schema: bad {bad}")
    try:
        return Verdict(
            is_hazard=data["is_hazard"],
            hazard_class=data["class"],
            severity=severity,
            lane=data["lane"],
            confidence=float(data["confidence"]),
            reason=data["reason"],
            report_description=data["report_description"],
            model=model,
            prompt_version=prompt_version,
        )
    except ValueError as e:  # e.g. a hazard with severity 0
        raise ProviderError(f"Muse Spark reply does not match the verdict schema: {e}") from None


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        value = float(resp.headers.get("Retry-After", ""))
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


class LiveProvider(ModelProvider):
    name = "live"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        tracker: SpendTracker | None = None,
        max_attempts: int = 5,
        base_delay: float = 2.0,
        backoff: float = 2.0,
        timeout_seconds: float | None = 120.0,
        sleep: Callable[[float], None] = time.sleep,
        client: httpx.Client | None = None,
        base_url: str | None = None,
    ):
        key = api_key or config.meta_api_key()
        if not key:
            raise ProviderConfigError(
                "ROADSCOUT_PROVIDER=live needs META_API_KEY (set it in the environment or a "
                "git-ignored .env). Use ROADSCOUT_PROVIDER=fake to run without a key."
            )
        self._api_key = key  # never print or log this
        self.tracker = tracker if tracker is not None else SpendTracker(log_path=config.cost_log_path())
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.backoff = backoff
        self.timeout_seconds = timeout_seconds
        self._sleep = sleep
        self._base_url = base_url
        self._client = client if client is not None else httpx.Client(timeout=RESPONSE_TIMEOUT_S)

    def __repr__(self) -> str:
        return "LiveProvider(api_key=<hidden>)"

    # --- settings -------------------------------------------------------------

    @property
    def base_url(self) -> str:
        raw = self._base_url or os.environ.get("ROADSCOUT_META_BASE_URL") or DEFAULT_BASE_URL
        return raw.strip().rstrip("/")

    @property
    def sam_model(self) -> str:
        return (os.environ.get("ROADSCOUT_SAM_MODEL") or "").strip() or DEFAULT_SAM_MODEL

    @property
    def verify_model(self) -> str:
        return (os.environ.get("ROADSCOUT_VERIFY_MODEL") or "").strip() or DEFAULT_VERIFY_MODEL

    def estimate_cost(self, operation: str) -> float:
        """USD per verify call, or per detect prompt ("detect_image": per prompt on one image, cheaper than
        a video chunk). ROADSCOUT_COST_PER_DETECT_USD / ROADSCOUT_COST_PER_VERIFY_USD win when set."""
        env_operation = "detect" if operation == "detect_image" else operation
        raw = os.environ.get(f"ROADSCOUT_COST_PER_{env_operation.upper()}_USD")
        if raw is not None and raw.strip():
            return config.cost_per_call_usd(env_operation)
        return DEFAULT_COST_USD.get(operation, 0.0)

    # --- guard layer ---------------------------------------------------------

    def _call(
        self,
        operation: str,
        fn: Callable[[], T],
        *,
        estimate: float | None = None,
        timeout: float | None = ...,  # type: ignore[assignment]
    ) -> T:
        if estimate is None:
            estimate = self.estimate_cost(operation)
        if timeout is ...:
            timeout = self.timeout_seconds
        self.tracker.check(estimate)  # raises SpendCapExceeded before any call is made
        started = time.monotonic()
        attempts = 0

        def attempt() -> T:
            nonlocal attempts
            attempts += 1
            return call_with_timeout(fn, timeout)

        result = call_with_retries(
            attempt,
            max_attempts=self.max_attempts,
            base_delay=self.base_delay,
            backoff=self.backoff,
            sleep=self._sleep,
            retry_on=(TransientProviderError, ProviderTimeoutError),
        )
        self.tracker.charge(
            estimate,
            model=f"live-{operation}",
            operation=operation,
            attempts=attempts,
            duration_s=round(time.monotonic() - started, 3),
        )
        return result

    # --- HTTP ------------------------------------------------------------------

    def _http_error(self, resp: httpx.Response) -> ProviderError:
        status = resp.status_code
        detail = ""
        try:
            body = resp.json()
        except ValueError:
            body = None
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            bits = [str(err[k]) for k in ("message", "type", "code") if err.get(k)]
            detail = " / ".join(bits[:1]) + (f" ({', '.join(bits[1:])})" if len(bits) > 1 else "")
        elif isinstance(err, str):
            detail = err
        detail = " ".join(detail.replace(self._api_key, "<hidden>").split())[:MAX_ERROR_CHARS]
        message = f"Meta API returned HTTP {status}" + (f": {detail}" if detail else "")
        if status in TRANSIENT_STATUSES:
            return TransientProviderError(message, retry_after=_retry_after(resp))
        return ProviderError(message)

    def _request(self, method: str, path: str, *, timeout: float, **kwargs: Any) -> httpx.Response:
        try:
            resp = self._client.request(
                method,
                f"{self.base_url}{path}",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=timeout,
                **kwargs,
            )
        except httpx.TransportError as e:  # connection problems and timeouts
            raise TransientProviderError(f"network error talking to Meta ({type(e).__name__})") from e
        if not 200 <= resp.status_code < 300:
            raise self._http_error(resp)
        return resp

    def _respond(self, body: dict[str, Any]) -> str:
        """POST /responses and return the reply text (``output[].content[].text``)."""
        resp = self._request("POST", "/responses", json=body, timeout=RESPONSE_TIMEOUT_S)
        try:
            payload = resp.json()
        except ValueError:
            raise ProviderError("Meta API returned a reply that is not JSON") from None
        err = payload.get("error") if isinstance(payload, dict) else None
        if err:
            message = err.get("message") if isinstance(err, dict) else err
            detail = " ".join(str(message or "unknown error").replace(self._api_key, "<hidden>").split())
            raise ProviderError(f"Meta API reported an error: {detail[:MAX_ERROR_CHARS]}")
        return _output_text(payload)

    def _upload(self, path: Path) -> str:
        try:
            handle = path.open("rb")
        except OSError:
            raise ProviderError("the video chunk could not be read") from None
        with handle:
            resp = self._request(
                "POST",
                "/files",
                data={"purpose": "user_data"},
                files={"file": (f"chunk{path.suffix or '.mp4'}", handle, "application/octet-stream")},
                timeout=UPLOAD_TIMEOUT_S,
            )
        try:
            file_id = resp.json()["id"]
        except (ValueError, KeyError, TypeError):
            file_id = None
        if not isinstance(file_id, str) or not file_id:
            raise ProviderError("Meta API upload reply had no file id")
        return file_id

    def _delete_upload(self, file_id: str) -> None:
        """Best effort: a failed delete never hides the real result or error."""
        try:
            self._request("DELETE", f"/files/{file_id}", timeout=DELETE_TIMEOUT_S)
        except Exception:  # noqa: BLE001
            logger.warning("could not delete the uploaded chunk from Meta")

    # --- the real calls (task 26) -------------------------------------------

    @staticmethod
    def _chunk_fps(video_path: str) -> float:
        try:
            fps = probe_video(Path(video_path)).fps
        except (InvalidMedia, ProbeUnavailable):
            raise ProviderError("the video chunk could not be probed with ffprobe") from None
        if not fps:
            raise ProviderError("the video chunk has no readable frame rate")
        return fps

    def _do_detect(self, video_path: str, prompts: Sequence[str]) -> DetectionResult:
        phrases = [str(p) for p in prompts]
        path = Path(video_path)
        if not path.is_file():
            raise ProviderError("the video chunk does not exist")
        mime = _sniff_image_mime(path)
        if mime is not None:
            return self._detect_image(path, mime, phrases)
        fps = self._chunk_fps(video_path)
        per_prompt = self.estimate_cost("detect")
        self.tracker.check(per_prompt * len(phrases))  # before the upload
        found: dict[int, list[Detection]] = {}
        file_id = self._upload(path)
        try:
            for index, phrase in enumerate(phrases):
                self.tracker.check(per_prompt * (index + 1))  # before each request
                text = self._respond({
                    "model": self.sam_model,
                    "input": [{"type": "message", "role": "user", "content": [
                        {"type": "input_text", "text": phrase},
                        {"type": "input_video", "file_id": file_id},
                    ]}],
                    "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
                })
                _parse_detections(text, phrase, index, found)
        finally:
            self._delete_upload(file_id)
        frames = [
            FrameRecord(frame_index=n, timestamp=n / fps, detections=tuple(found[n]))
            for n in sorted(found)
        ]
        return DetectionResult(frames=frames)

    def _detect_image(self, path: Path, mime: str, phrases: list[str]) -> DetectionResult:
        """SAM 3.1 on one image file (for example the privacy blur): the image goes inline as a data URL, one
        request per phrase, no upload. Frame index 0, timestamp 0.0."""
        try:
            size = path.stat().st_size
            if size > MAX_IMAGE_BYTES:
                raise ProviderError(
                    f"the image is {size} bytes; SAM accepts at most {MAX_IMAGE_BYTES} bytes per image")
            data = path.read_bytes()
        except OSError:
            raise ProviderError("the image file could not be read") from None
        if not data:
            raise ProviderError("the image file is empty")
        image_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        per_prompt = self.estimate_cost("detect_image")
        found: dict[int, list[Detection]] = {}
        for index, phrase in enumerate(phrases):
            self.tracker.check(per_prompt * (index + 1))  # before each request
            text = self._respond({
                "model": self.sam_model,
                "input": [{"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": phrase},
                    {"type": "input_image", "image_url": image_url},
                ]}],
                "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
            })
            _parse_detections(text, phrase, index, found, default_frame=0)
        merged = [d for n in sorted(found) for d in found[n]]  # an image is one frame
        frames = [FrameRecord(frame_index=0, timestamp=0.0, detections=tuple(merged))] if merged else []
        return DetectionResult(frames=frames)

    @staticmethod
    def _image_url(image: Image) -> str:
        if isinstance(image, (bytes, bytearray)):
            data = bytes(image)
        else:
            try:
                data = Path(image).read_bytes()
            except OSError:
                raise ProviderError("an image for verification could not be read") from None
        return f"data:{_image_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"

    def _default_prompt(self, class_prompt: str | None) -> str:
        """Prompt for the 3-argument form (no ``prompt_text``): the configured verification prompt."""
        try:
            from ..hazard_prompts import load_hazard_prompts
            from ..verification_prompt import load_verification_prompt

            return load_verification_prompt().render(class_prompt, load_hazard_prompts().class_names)
        except ValueError:
            return (
                f"Is this a real road hazard (detector class: {class_prompt or 'unknown'})? Answer with one "
                f"JSON object matching this schema: {json.dumps(VERDICT_SCHEMA)}"
            )

    def _do_verify(
        self,
        images: Sequence[Image],
        class_prompt: str | None,
        prompt_version: str | None,
        prompt_text: str | None = None,
    ) -> Verdict:
        images = list(images)
        if len(images) > MAX_VERIFY_IMAGES:
            raise ProviderError(
                f"verify got {len(images)} images; Muse Spark accepts at most {MAX_VERIFY_IMAGES} per request"
            )
        text = prompt_text if prompt_text is not None else self._default_prompt(class_prompt)
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
        content.extend({"type": "input_image", "image_url": self._image_url(img)} for img in images)
        model = self.verify_model
        reply = self._respond({
            "model": model,
            "input": [{"type": "message", "role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": "verdict", "schema": VERDICT_SCHEMA}},
        })
        return _verdict_from_reply(reply, model, prompt_version or "")

    def detect(self, video_path: str, prompts: Sequence[str]) -> DetectionResult:
        phrases = list(prompts)
        if not phrases:
            return DetectionResult()
        # one request per phrase, so the whole-call timeout and the estimate scale with their number
        timeout = None if self.timeout_seconds is None else self.timeout_seconds * len(phrases)
        is_image = _sniff_image_mime(Path(video_path)) is not None and Path(video_path).is_file()
        return self._call(
            "detect",
            lambda: self._do_detect(video_path, phrases),
            estimate=self.estimate_cost("detect_image" if is_image else "detect") * len(phrases),
            timeout=timeout,
        )

    def verify(
        self,
        images: Sequence[Image],
        *,
        class_prompt: str | None = None,
        prompt_version: str | None = None,
        prompt_text: str | None = None,
    ) -> Verdict:
        if prompt_text is None:  # keep the 3-argument form for subclasses that override _do_verify
            return self._call("verify", lambda: self._do_verify(images, class_prompt, prompt_version))
        return self._call(
            "verify",
            lambda: self._do_verify(images, class_prompt, prompt_version, prompt_text=prompt_text),
        )
