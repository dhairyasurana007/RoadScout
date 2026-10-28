"""Task 26: mask decoding, pasting into the full frame, clipping and the fail-closed fallback."""
import time

import pytest

import rs26_helpers as H
from rs26_conftest import _safe_env, rec, api, clips, clip, provider_factory, provider  # noqa: F401
from rs26_helpers import box_tok, frame_line, make_raster, mask_tok, payload_for, rec_str

W, HH = 100, 80


def one(api, provider, clip, record, w=W, h=HH):
    api.default_text = frame_line(1, record)
    res = provider.detect(clip, ["x"])
    dets = H.dets_of(res)
    assert len(dets) == 1, "the detection must never be dropped"
    return dets[0]


def box_fill_bitmap(d, w=W, h=HH):
    from backend.providers.types import rle_from_box, rle_to_bitmap

    return rle_to_bitmap(rle_from_box(w, h, tuple(d.box)))


def test_mask_is_pasted_at_the_box_corner(api, provider, clip):
    x1, y1, x2, y2 = 20, 30, 33, 41
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    raster = make_raster(bw, bh)
    d = one(api, provider, clip, rec_str(0, x1, y1, x2, y2, W, HH, c=0.5, raster=raster))
    assert d.mask["size"] == [HH, W]
    assert H.bitmap_of(d) == H.expected_bitmap(W, HH, x1, y1, raster, bw, bh)


def test_mask_x_y_fields_do_not_move_the_paste(api, provider, clip):
    """The mask token carries x=0;y=0 (as in real output); the raster is still placed at the box corner."""
    x1, y1, x2, y2 = 40, 10, 49, 19
    raster = make_raster(10, 10)
    rec = "0" + box_tok(x1, y1, x2, y2, W, HH, 0.5) + mask_tok(10, 10, payload_for(raster, 10, 10), 0.5, x=0, y=0)
    d = one(api, provider, clip, rec)
    assert H.bitmap_of(d) == H.expected_bitmap(W, HH, x1, y1, raster, 10, 10)


def test_asymmetric_raster_is_not_transposed(api, provider, clip):
    """A 3 wide x 5 tall raster with a single marked cell in its top-right corner."""
    raster = [0] * 15
    raster[0 * 3 + 2] = 1
    raster[4 * 3 + 0] = 1
    d = one(api, provider, clip, rec_str(0, 10, 20, 12, 24, W, HH, c=0.5, raster=raster))
    bm = H.bitmap_of(d)
    assert bm[20][12] == 1 and bm[24][10] == 1
    assert sum(map(sum, bm)) == 2


def test_mask_size_comes_from_the_box_frame_dims_not_the_clip(api, provider, clip):
    d = one(api, provider, clip, rec_str(0, 5, 5, 14, 14, 640, 360, c=0.5), w=640, h=360)
    assert d.mask["size"] == [360, 640]


def test_clipped_at_right_and_bottom_edges(api, provider, clip):
    x1, y1, x2, y2 = 90, 70, 109, 89   # 20 x 20, frame is 100 x 80
    raster = make_raster(20, 20, "ones")
    d = one(api, provider, clip, rec_str(0, x1, y1, x2, y2, W, HH, c=0.5, raster=raster))
    bm = H.bitmap_of(d)
    assert len(bm) == HH and len(bm[0]) == W
    assert bm == H.expected_bitmap(W, HH, x1, y1, raster, 20, 20)
    assert sum(map(sum, bm)) == 10 * 10


def test_clipped_at_left_and_top_edges(api, provider, clip):
    x1, y1, x2, y2 = -5, -3, 14, 16
    bw, bh = 20, 20
    raster = make_raster(bw, bh)
    d = one(api, provider, clip, rec_str(0, x1, y1, x2, y2, W, HH, c=0.5, raster=raster))
    assert H.bitmap_of(d) == H.expected_bitmap(W, HH, x1, y1, raster, bw, bh)


def test_mask_touching_the_last_row_and_column(api, provider, clip):
    x1, y1, x2, y2 = 95, 75, 99, 79
    raster = make_raster(5, 5, "ones")
    d = one(api, provider, clip, rec_str(0, x1, y1, x2, y2, W, HH, c=0.5, raster=raster))
    bm = H.bitmap_of(d)
    assert bm[79][99] == 1 and bm[75][95] == 1 and sum(map(sum, bm)) == 25


def test_inclusive_x2_y2_raster_size(api, provider, clip):
    """W = x2-x1+1 and H = y2-y1+1: the last column/row of the box is covered."""
    raster = make_raster(4, 3, "ones")
    d = one(api, provider, clip, rec_str(0, 10, 10, 13, 12, W, HH, c=0.5, raster=raster))
    bm = H.bitmap_of(d)
    assert sum(map(sum, bm)) == 12
    assert bm[12][13] == 1 and bm[12][14] == 0 and bm[13][13] == 0


def test_all_zero_raster_is_kept_as_decoded(api, provider, clip):
    d = one(api, provider, clip, rec_str(0, 10, 10, 19, 19, W, HH, c=0.5, raster="zeros"))
    assert d.mask is not None
    assert sum(map(sum, H.bitmap_of(d))) == 0


def test_mask_rle_is_valid(api, provider, clip):
    from backend.providers.types import validate_rle

    d = one(api, provider, clip, rec_str(0, 10, 10, 29, 29, W, HH, c=0.5))
    validate_rle(d.mask)
    assert d.mask["counts"][0] >= 0


# ------------------------------------------------------------------ fail closed

def _fallback_cases():
    box = (20, 30, 33, 41)  # 14 x 12
    bw, bh = 14, 12
    good = payload_for(make_raster(bw, bh), bw, bh)
    wrong_size_payload = payload_for(make_raster(5, 5), 5, 5)
    return {
        "garbage_payload": dict(payload="!zzzzzzzz"),
        "empty_payload": dict(payload=""),
        "no_marker_payload": dict(payload=good[1:]),
        "truncated_payload": dict(payload=good[: len(good) // 2]),
        "payload_for_other_size": dict(payload=wrong_size_payload),
        "declared_size_mismatch": dict(declared=(bh + 3, bw + 2), payload=good),
        "declared_zero": dict(declared=(0, 0), payload=good),
        "valid_payload_but_larger_than_box": dict(declared=(20, 20), payload=payload_for(make_raster(20, 20), 20, 20)),
        "declared_not_numbers": dict(raw_data="data=a,b,%s" % good),
        "missing_data_field": dict(raw_data=""),
    }


@pytest.mark.parametrize("name", sorted(_fallback_cases()))
def test_bad_mask_falls_back_to_the_filled_box(api, provider, clip, name):
    case = _fallback_cases()[name]
    x1, y1, x2, y2 = 20, 30, 33, 41
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    if "raw_data" in case:
        data = case["raw_data"]
        mask = "<|mask;x=0;y=0;c=0.5" + (";" + data if data else "") + "|>"
    else:
        hh, ww = case.get("declared", (bh, bw))
        mask = mask_tok(hh, ww, case["payload"], 0.5)
    rec = "3" + box_tok(x1, y1, x2, y2, W, HH, 0.5) + mask
    d = one(api, provider, clip, rec)
    assert tuple(d.box) == (x1, y1, x2 + 1, y2 + 1)
    assert d.mask is not None
    assert d.mask["size"] == [HH, W]
    assert H.bitmap_of(d) == box_fill_bitmap(d)
    assert sum(map(sum, H.bitmap_of(d))) == bw * bh


def test_missing_mask_token_falls_back_to_the_filled_box(api, provider, clip):
    d = one(api, provider, clip, rec_str(0, 20, 30, 33, 41, W, HH, c=0.5, mask=False))
    assert d.mask is not None
    assert H.bitmap_of(d) == box_fill_bitmap(d)


def test_fallback_clips_the_filled_box_to_the_frame(api, provider, clip):
    rec = "0" + box_tok(90, 70, 109, 89, W, HH, 0.5) + mask_tok(20, 20, "!zzzz", 0.5)
    d = one(api, provider, clip, rec)
    bm = H.bitmap_of(d)
    assert len(bm) == HH and len(bm[0]) == W
    assert sum(map(sum, bm)) == 100


def test_bad_mask_does_not_affect_other_records(api, provider, clip):
    good_raster = make_raster(10, 10)
    bad = "1" + box_tok(50, 50, 59, 59, W, HH, 0.5) + mask_tok(10, 10, "!zzzz", 0.5)
    good = rec_str(0, 5, 5, 14, 14, W, HH, c=0.5, raster=good_raster)
    api.default_text = frame_line(1, good, bad)
    res = provider.detect(clip, ["x"])
    t = H.by_track(res.frames[0])
    assert set(t) == {0, 1}
    assert H.bitmap_of(t[0]) == H.expected_bitmap(W, HH, 5, 5, good_raster, 10, 10)
    assert sum(map(sum, H.bitmap_of(t[1]))) == 100


# ------------------------------------------------------------------ speed

def test_many_masks_in_a_720p_frame_are_fast(api, provider, clip):
    """30 small masks in a 1280x720 frame. The official decoder is slow for huge rasters, so the
    masks are small; building the full-frame RLE must not cost much per detection."""
    fw, fh = 1280, 720
    bw, bh = 60, 40
    payload = payload_for(make_raster(bw, bh), bw, bh)
    recs = []
    for i in range(30):
        x1, y1 = 20 + 38 * i, 30 + 20 * i
        recs.append(rec_str(i, x1, y1, x1 + bw - 1, y1 + bh - 1, fw, fh, c=0.5, payload=payload))
    api.default_text = "\n".join(frame_line(n, *recs) for n in (1, 2))
    t0 = time.perf_counter()
    res = provider.detect(clip, ["x"])
    elapsed = time.perf_counter() - t0
    assert sum(len(f.detections) for f in res.frames) == 60
    assert all(d.mask["size"] == [fh, fw] for d in H.dets_of(res))
    assert elapsed < 12.0, f"60 masks in 720p took {elapsed:.1f}s"


def test_many_fallback_masks_in_a_720p_frame_are_fast(api, provider, clip):
    fw, fh = 1280, 720
    recs = []
    for i in range(30):
        x1, y1 = 20 + 38 * i, 30 + 20 * i
        recs.append("%d" % i + box_tok(x1, y1, x1 + 99, y1 + 79, fw, fh, 0.5) + mask_tok(80, 100, "!zzzz", 0.5))
    api.default_text = "\n".join(frame_line(n, *recs) for n in (1, 2, 3))
    t0 = time.perf_counter()
    res = provider.detect(clip, ["x"])
    elapsed = time.perf_counter() - t0
    assert sum(len(f.detections) for f in res.frames) == 90
    assert elapsed < 12.0, f"90 fallback masks in 720p took {elapsed:.1f}s"


def test_one_large_mask(api, provider, clip):
    fw, fh = 406, 720
    bw, bh = 200, 400
    raster = [1 if (x // 5 + y // 5) % 2 == 0 else 0 for y in range(bh) for x in range(bw)]
    d = one(api, provider, clip, rec_str(0, 100, 200, 100 + bw - 1, 200 + bh - 1, fw, fh, c=0.5, raster=raster),
            w=fw, h=fh)
    assert d.mask["size"] == [fh, fw]
    from backend.providers.types import rle_area

    assert rle_area(d.mask) == sum(raster)
