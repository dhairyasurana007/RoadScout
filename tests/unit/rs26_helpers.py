"""Helpers for the task 26 hidden tests: a recording fake Meta API on httpx.MockTransport,
response-text builders for SAM 3.1 output, and expected-value helpers."""
import base64
import json
from pathlib import Path

import httpx

REPO = None  # set by conftest

KEY = "tk-DUMMY-7f3a91c2"
BASE = "https://api.meta.ai/v1"
FILE_ID = "file_test_0001"


# ------------------------------------------------------------------ fake API

def out_json(text, extra_items=()):
    return {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "output": [*extra_items, {"type": "message", "role": "assistant",
                                  "content": [{"type": "output_text", "text": text}]}],
        "usage": {"video_frames_processed": 60},
    }


def err_json(message, type_="invalid_request_error", code="bad"):
    return {"error": {"message": message, "type": type_, "code": code}}


class Req:
    def __init__(self, request):
        self.method = request.method
        self.url = str(request.url)
        self.path = request.url.path
        self.headers = request.headers
        self.body = request.read()
        self.json = None
        ctype = request.headers.get("content-type", "")
        if "json" in ctype and self.body:
            try:
                self.json = json.loads(self.body)
            except ValueError:
                pass


class FakeMeta:
    """Records every request. /responses answers come from, in order: ``queue`` items
    (httpx.Response, Exception, dict -> JSON, str -> output text), then ``by_prompt``
    (phrase -> output text), then ``default_text``."""

    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.requests = []
        self.queue = []
        self.by_prompt = {}
        self.default_text = ""
        self.upload_n = 0
        self.upload_fn = None   # callable(Req) -> response-like or None
        self.delete_fn = None   # callable(Req) -> response-like or None
        self.file_ids = []

    # ---- helpers
    @property
    def uploads(self):
        return [r for r in self.requests if r.method == "POST" and r.path.endswith("/files")]

    @property
    def responses(self):
        return [r for r in self.requests if r.method == "POST" and r.path.endswith("/responses")]

    @property
    def deletes(self):
        return [r for r in self.requests if r.method == "DELETE"]

    def kinds(self):
        out = []
        for r in self.requests:
            if r.method == "DELETE":
                out.append("delete")
            elif r.path.endswith("/files"):
                out.append("upload")
            else:
                out.append("responses")
        return out

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    def _coerce(self, item, request):
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, httpx.Response):
            return item
        if isinstance(item, dict):
            return httpx.Response(200, json=item)
        if isinstance(item, str):
            return httpx.Response(200, json=out_json(item))
        raise TypeError(item)

    def _handle(self, request):
        req = Req(request)
        self.requests.append(req)
        self.events.append(("req", request.method, req.path))
        p = request.url.path
        if request.method == "POST" and p.endswith("/files"):
            if self.upload_fn is not None:
                r = self.upload_fn(req)
                if r is not None:
                    return self._coerce(r, request)
            self.upload_n += 1
            fid = FILE_ID if self.upload_n == 1 else f"file_test_{self.upload_n:04d}"
            self.file_ids.append(fid)
            return httpx.Response(200, json={"id": fid, "object": "file", "purpose": "user_data"})
        if request.method == "DELETE" and "/files/" in p:
            if self.delete_fn is not None:
                r = self.delete_fn(req)
                if r is not None:
                    return self._coerce(r, request)
            return httpx.Response(200, json={"id": p.rsplit("/", 1)[1], "deleted": True})
        if request.method == "POST" and p.endswith("/responses"):
            if self.queue:
                return self._coerce(self.queue.pop(0), request)
            payload = req.json or {}
            try:
                phrase = payload["input"][0]["content"][0]["text"]
            except Exception:  # noqa: BLE001
                phrase = None
            return httpx.Response(200, json=out_json(self.by_prompt.get(phrase, self.default_text)))
        return httpx.Response(404, json=err_json("no such route"))


def make_provider(api, tracker=None, **kw):
    from backend.providers.cost import SpendTracker
    from backend.providers.live import LiveProvider

    import inspect

    kw.setdefault("api_key", KEY)
    if "sleep" in inspect.signature(LiveProvider.__init__).parameters:
        kw.setdefault("sleep", lambda seconds=0.0, *a, **k: api.events.append(("sleep", float(seconds))))
    if tracker is None:
        tracker = SpendTracker(cap_usd=100.0, log_path=None)
    return LiveProvider(client=api.client(), tracker=tracker, **kw)


# ------------------------------------------------------------ SAM text builders

def make_raster(w, h, kind="pattern"):
    """Row-major list of 0/1 of length h*w."""
    if kind == "ones":
        return [1] * (w * h)
    if kind == "zeros":
        return [0] * (w * h)
    return [1 if (x + 2 * y) % 3 == 0 else 0 for y in range(h) for x in range(w)]


def payload_for(raster, w, h):
    from meta_sam_parser._mask_codec import _encode_raster

    return _encode_raster(list(raster), w, h)


def box_tok(x1, y1, x2, y2, w, h, c=None):
    s = f"<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h}"
    if c is not None:
        s += f";c={c}"
    return s + "|>"


def mask_tok(hh, ww, payload, c=None, x=0, y=0):
    s = f"<|mask;x={x};y={y}"
    if c is not None:
        s += f";c={c}"
    return s + f";data={hh},{ww},{payload}|>"


def rec_str(obj, x1, y1, x2, y2, w, h, c=None, raster="pattern", payload=None, mask=True):
    """One record: ``<id><|box..|><|mask..|>`` (raster covers the box region)."""
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    s = f"{obj}" + box_tok(x1, y1, x2, y2, w, h, c)
    if mask:
        if payload is None:
            r = make_raster(bw, bh, raster) if isinstance(raster, str) else list(raster)
            payload = payload_for(r, bw, bh)
        s += mask_tok(bh, bw, payload, c)
    return s


def frame_line(n, *records):
    return f"<{n}f>" + ",".join(records)


def expected_bitmap(W, H_, x1, y1, raster, bw, bh):
    """Full-frame H_ x W bitmap with the bw x bh raster pasted at (x1, y1), clipped."""
    rows = [[0] * W for _ in range(H_)]
    for j in range(bh):
        for i in range(bw):
            X, Y = x1 + i, y1 + j
            if 0 <= X < W and 0 <= Y < H_ and raster[j * bw + i]:
                rows[Y][X] = 1
    return rows


def bitmap_of(det):
    from backend.providers.types import rle_to_bitmap

    return rle_to_bitmap(det.mask)


def dets_of(result):
    return [d for f in result.frames for d in f.detections]


def frame_of(result, n):
    for f in result.frames:
        if f.frame_index == n:
            return f
    raise AssertionError(f"no frame {n} in {[f.frame_index for f in result.frames]}")


def by_track(frame):
    return {d.track_id: d for d in frame.detections}


# --------------------------------------------------------------- misc helpers

def verdict_obj(**over):
    d = {
        "is_hazard": True, "class": "pothole", "severity": 2, "lane": "left", "confidence": 0.83,
        "reason": "Dark crater with broken edges.", "report_description": "A pothole in the left lane.",
    }
    d.update(over)
    return d


def jpeg_bytes(color=(200, 30, 30), size=(8, 8)):
    import io

    from PIL import Image

    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "JPEG")
    return b.getvalue()


def png_bytes(color=(30, 30, 200), size=(8, 8)):
    import io

    from PIL import Image

    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "PNG")
    return b.getvalue()


def data_url(mime, raw):
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def verification_schema():
    path = Path(REPO) / "config" / "verification_prompt.json"
    return json.loads(path.read_text(encoding="utf-8"))["schema"]
