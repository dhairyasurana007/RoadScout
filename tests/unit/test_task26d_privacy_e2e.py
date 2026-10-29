"""Task 26d: the privacy blur works end to end with the live provider on a mock transport (images as input_image)."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

import rs26d_setup  # noqa: F401
import rs26_helpers as H
from rs26_conftest import _safe_env, rec, api, clips, clip, provider_factory, provider  # noqa: F401
from rs26_helpers import FakeMeta, KEY, frame_line, out_json, rec_str
from rs15_helpers import (FAILED, make_stub, rs15_env, status_of, DONE)  # noqa: F401


def checker_jpeg(path, w=80, h=60, cell=2, fmt="JPEG"):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if ((x // cell) + (y // cell)) % 2 == 0:
                arr[y, x] = (230, 40, 40)
            else:
                arr[y, x] = (20, 20, 220)
    Image.fromarray(arr).save(path, fmt, **({"quality": 95} if fmt == "JPEG" else {}))
    return path


def gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float64)


class DynMeta(FakeMeta):
    """Answers every image request with one detection: a rectangle covering the middle of the image."""

    def __init__(self, events=None, status=None):
        super().__init__(events)
        self.status = status
        self.region = {}

    def _handle(self, request):
        if request.method == "POST" and request.url.path.endswith("/responses"):
            import json

            body = json.loads(request.read())
            content = body["input"][0]["content"]
            img = next((p for p in content if p.get("type") == "input_image"), None)
            if img is not None:
                if self.status is not None:
                    super()._handle(request)
                    return httpx.Response(self.status, json=H.err_json(
                        "unsupported media type" if self.status == 400 else "failure"))
                raw = base64.b64decode(img["image_url"].split(",", 1)[1])
                with Image.open(io.BytesIO(raw)) as im:
                    w, h = im.size
                x1, y1, x2, y2 = w // 4, h // 4, 3 * w // 4 - 1, 3 * h // 4 - 1
                self.region[(w, h)] = (x1, y1, x2, y2)
                super()._handle(request)     # record it
                text = frame_line(0, rec_str(0, x1, y1, x2, y2, w, h, c=0.9, raster="ones"))
                return httpx.Response(200, json=out_json(text))
        return super()._handle(request)


# ------------------------------------------------------------------ blur_image with the live provider

@pytest.mark.parametrize("fmt,ext", [("JPEG", ".jpg"), ("PNG", ".png")])
def test_blur_image_changes_only_the_mask_area(api, tmp_path, fmt, ext):
    from backend.privacy import blur_image

    meta = DynMeta(api.events)
    p = H.make_provider(meta)
    src = checker_jpeg(tmp_path / f"in{ext}", fmt=fmt)
    res = p.detect(str(src), ["face", "license plate"])
    dets = H.dets_of(res)
    assert dets and meta.uploads == [] and meta.deletes == []
    dst = tmp_path / f"out{ext}"
    blur_image(str(src), str(dst), dets)
    before, after = gray(src), gray(dst)
    assert before.shape == after.shape
    x1, y1, x2, y2 = meta.region[(80, 60)]
    inside = np.zeros(before.shape, bool)
    inside[y1:y2 + 1, x1:x2 + 1] = True
    diff = np.abs(before - after)
    core = np.zeros_like(inside)
    core[y1 + 2:y2 - 1, x1 + 2:x2 - 1] = True
    assert diff[core].mean() > 10, "the masked area must be visibly blurred"
    far = ~inside
    far[max(0, y1 - 4):y2 + 5, max(0, x1 - 4):x2 + 5] = False
    assert diff[far].mean() < 3, "pixels away from the mask must be unchanged (JPEG noise only)"


def test_blur_image_with_a_malformed_mask_still_blurs_the_whole_box(api, tmp_path):
    from backend.privacy import blur_image

    class BadMask(FakeMeta):
        def _handle(self, request):
            r = super()._handle(request)
            return r

    meta = BadMask(api.events)
    meta.default_text = ("<0f>0<|box;x1=20;y1=15;x2=59;y2=44;w=80;h=60;c=0.9|>"
                         "<|mask;x=0;y=0;c=0.9;data=30,40,!zzzz|>")
    p = H.make_provider(meta)
    src = checker_jpeg(tmp_path / "in.jpg")
    dets = H.dets_of(p.detect(str(src), ["face"]))
    dst = tmp_path / "out.jpg"
    blur_image(str(src), str(dst), dets)
    diff = np.abs(gray(src) - gray(dst))
    assert diff[17:42, 22:57].mean() > 10, "fail closed: the whole box is blurred"
    assert diff[:10, :].mean() < 3


def test_blur_image_with_an_empty_answer_is_an_unchanged_copy(api, tmp_path):
    from backend.privacy import blur_image

    meta = FakeMeta(api.events)
    meta.default_text = ""
    p = H.make_provider(meta)
    src = checker_jpeg(tmp_path / "in.jpg")
    dets = H.dets_of(p.detect(str(src), ["face", "license plate"]))
    assert dets == []
    dst = tmp_path / "out.jpg"
    blur_image(str(src), str(dst), dets)
    assert np.abs(gray(src) - gray(dst)).mean() < 3


# ------------------------------------------------------------------ the pipeline's blurring stage

def live_stub(api, **meta_kw):
    meta = DynMeta(api.events, **meta_kw)
    live = H.make_provider(meta)
    stub = make_stub(image_result=lambda s, path, prompts: live.detect(path, prompts))
    return stub, meta


def test_pipeline_blurs_saved_frames_through_the_live_provider(rs15_env, api):
    e = rs15_env
    stub, meta = live_stub(api)
    did = e.process_ok(stub)
    assert stub.image_calls, "the blurring stage called detect() on the saved JPEGs"
    assert meta.uploads == [] and meta.deletes == [], "images are never uploaded"
    assert meta.responses
    for r in meta.responses:
        parts = r.json["input"][0]["content"]
        assert [p["type"] for p in parts] == ["input_text", "input_image"]
        assert parts[1]["image_url"].startswith("data:image/jpeg;base64,")
        assert r.json["model"] == "sam-3.1"
    assert len(meta.responses) == 2 * len(stub.image_calls), "two privacy prompts per image"
    blurred = e.blurred_paths(did)
    originals = e.unblurred_paths(did)
    assert blurred and len(blurred) == len(originals)
    for orig, blur in zip(originals, blurred):
        assert Path(blur).is_file() and Path(blur) != Path(orig)
        a, b = gray(orig), gray(blur)
        assert a.shape == b.shape
        h, w = a.shape
        x1, y1, x2, y2 = meta.region[(w, h)]
        diff = np.abs(a - b)
        assert diff[y1:y2 + 1, x1:x2 + 1].max() > 15, "the masked area was blurred in %s" % Path(blur).name
        far = np.ones(a.shape, bool)
        far[max(0, y1 - 8):y2 + 9, max(0, x1 - 8):x2 + 9] = False
        if far.any():
            assert diff[far].mean() < 3, "pixels away from the mask are unchanged"
    c = e.cands(did)
    assert all(str(x.get("blur_status")).lower() in ("ok",) for x in c if x.get("blurred_frame_path"))


def test_a_400_from_meta_fails_closed_with_no_unblurred_file_served(rs15_env, api):
    e = rs15_env
    stub, meta = live_stub(api, status=400)
    did = e.upload()
    res, exc = e.run(did, stub)
    d = e.get(did)
    assert status_of(d) in FAILED, d
    assert d.get("failed_stage") == "blurring"
    assert "blurring" in (d.get("error") or "")
    assert KEY not in (d.get("error") or "")
    assert meta.uploads == []
    cands = e.cands(did)
    assert cands
    for c in cands:
        assert str(c.get("blur_status")).lower() == "failed"
        assert not c.get("blurred_frame_path") and not c.get("blurred_crop_path")
    assert e.blurred_paths(did) == []
    assert list(Path(e.tmp).rglob("*_blur.jpg")) == [] if hasattr(e, "tmp") else True
    for c in cands:
        r = e.photo(did, c["id"])
        assert r.status_code != 200, "an unblurred frame must never be served"
        r = e.photo(did, c["id"], crop=True)
        assert r.status_code != 200


def test_a_server_error_from_meta_also_fails_closed(rs15_env, api, rec):
    e = rs15_env
    stub, meta = live_stub(api, status=500)
    did = e.upload()
    e.run(did, stub)
    d = e.get(did)
    assert status_of(d) in FAILED and d.get("failed_stage") == "blurring"
    assert e.blurred_paths(did) == []
    for c in e.cands(did):
        assert e.photo(did, c["id"]).status_code != 200
