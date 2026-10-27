"""Task 15: the pure blur function (image + RLE masks -> blurred JPEG copy)."""
from __future__ import annotations

from pathlib import Path

import pytest

from rs15_helpers import (blur_image_fn, checker_array, lap_var, load_gray, outside_boxes, rle_box, save_gray_jpeg,
                          sha)

W, H = 320, 240
A = (96, 72, 224, 168)      # block aligned (multiples of 8)
B1 = (32, 32, 112, 112)
B2 = (208, 128, 288, 208)


@pytest.fixture()
def src(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    p = tmp_path / "in" / "frame.jpg"
    save_gray_jpeg(p, checker_array(W, H))
    return p


def blur(src, tmp_path, masks, name="blurred.jpg"):
    dst = tmp_path / "out" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    f = blur_image_fn()
    f(str(src), str(dst), list(masks))
    assert dst.is_file(), "blur function did not write the destination file"
    return dst


def assert_blurred(orig, out, box, ratio=0.15, erode=12):
    vo = lap_var(orig, box, erode)
    vb = lap_var(out, box, erode)
    assert vo > 1000, "test image should be high contrast (var %s)" % vo
    assert vb < ratio * vo, "mask area not blurred strongly enough: laplacian var %.1f -> %.1f" % (vo, vb)


def assert_untouched(orig, out, boxes, margin=16):
    m = outside_boxes(orig.shape, boxes, margin)
    d = abs(out - orig)[m]
    assert d.size > 1000
    assert d.mean() < 2.5, "pixels well outside the mask changed (mean abs diff %.2f)" % d.mean()
    assert (d > 40).mean() < 0.002, "pixels well outside the mask changed noticeably"


def test_blur_inside_mask_only(src, tmp_path):
    orig = load_gray(src)
    dst = blur(src, tmp_path, [rle_box(W, H, A)])
    out = load_gray(dst)
    assert out.shape == orig.shape
    assert_blurred(orig, out, A)
    assert_untouched(orig, out, [A])


def test_blur_is_a_blur_not_a_black_box(src, tmp_path):
    orig = load_gray(src)
    out = load_gray(blur(src, tmp_path, [rle_box(W, H, A)]))
    x1, y1, x2, y2 = A
    mo = orig[y1 + 12:y2 - 12, x1 + 12:x2 - 12].mean()
    mb = out[y1 + 12:y2 - 12, x1 + 12:x2 - 12].mean()
    assert abs(mo - mb) < 40, "masked area was filled instead of blurred (mean %.1f -> %.1f)" % (mo, mb)


def test_original_file_never_modified(src, tmp_path):
    before = sha(src)
    mtime = src.stat().st_mtime_ns
    dst = blur(src, tmp_path, [rle_box(W, H, A)])
    assert dst.resolve() != src.resolve()
    assert sha(src) == before
    assert src.stat().st_mtime_ns == mtime


def test_output_is_a_jpeg_of_the_same_size(src, tmp_path):
    from PIL import Image

    dst = blur(src, tmp_path, [rle_box(W, H, A)])
    with Image.open(dst) as im:
        assert im.format == "JPEG"
        assert im.size == (W, H)
        assert im.mode == "RGB"


@pytest.mark.parametrize("kind", ["no_masks", "all_zero_mask", "zero_area_box"])
def test_empty_mask_gives_unchanged_copy(src, tmp_path, kind):
    masks = {"no_masks": [],
             "all_zero_mask": [{"size": [H, W], "counts": [H * W]}],
             "zero_area_box": [rle_box(W, H, (0, 0, 0, 0))]}[kind]
    orig = load_gray(src)
    dst = blur(src, tmp_path, masks)
    out = load_gray(dst)
    assert out.shape == orig.shape
    assert abs(out - orig).mean() < 2.5, "an empty mask must leave the image unchanged"
    assert (abs(out - orig) > 40).mean() < 0.002
    assert sha(src) == sha(src)  # original still readable


def test_multiple_masks_are_combined(src, tmp_path):
    orig = load_gray(src)
    out = load_gray(blur(src, tmp_path, [rle_box(W, H, B1), rle_box(W, H, B2)]))
    assert_blurred(orig, out, B1)
    assert_blurred(orig, out, B2)
    assert_untouched(orig, out, [B1, B2])
    # the area between the two boxes
    between = (130, 120, 190, 150)
    assert lap_var(out, between, 4) > 0.8 * lap_var(orig, between, 4)


def test_overlapping_masks_are_combined(src, tmp_path):
    a = (64, 64, 160, 160)
    b = (120, 100, 232, 200)
    orig = load_gray(src)
    out = load_gray(blur(src, tmp_path, [rle_box(W, H, a), rle_box(W, H, b)]))
    assert_blurred(orig, out, a)
    assert_blurred(orig, out, b)
    assert_untouched(orig, out, [a, b])


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_mask_of_another_size_is_scaled_to_the_image(src, tmp_path, scale):
    mw, mh = int(W * scale), int(H * scale)
    box = tuple(v * scale for v in A)
    orig = load_gray(src)
    out = load_gray(blur(src, tmp_path, [rle_box(mw, mh, box)]))
    assert_blurred(orig, out, A)
    assert_untouched(orig, out, [A], margin=20)


def test_full_image_mask_blurs_everything(src, tmp_path):
    orig = load_gray(src)
    out = load_gray(blur(src, tmp_path, [{"size": [H, W], "counts": [0, H * W]}]))
    assert_blurred(orig, out, (0, 0, W, H), erode=16)


def test_colour_image_stays_colour(tmp_path):
    import numpy as np
    from PIL import Image

    pytest.importorskip("numpy")
    g = checker_array(W, H)
    rgb = np.stack([g, np.zeros_like(g), 255 - g], axis=-1)
    p = tmp_path / "in" / "c.jpg"
    p.parent.mkdir()
    Image.fromarray(rgb, "RGB").save(str(p), "JPEG", quality=95, subsampling=0)
    dst = blur(p, tmp_path, [rle_box(W, H, A)])
    with Image.open(dst) as im:
        assert im.mode == "RGB" and im.size == (W, H)
        arr = np.array(im).astype("float64")
    x1, y1, x2, y2 = A
    inner = arr[y1 + 12:y2 - 12, x1 + 12:x2 - 12]
    assert inner[..., 0].std() < 0.3 * rgb[y1 + 12:y2 - 12, x1 + 12:x2 - 12, 0].std()


def test_destination_can_be_overwritten(src, tmp_path):
    dst = blur(src, tmp_path, [rle_box(W, H, A)])
    first = sha(dst)
    dst2 = blur(src, tmp_path, [rle_box(W, H, B1)])
    assert dst2 == dst
    assert sha(dst) != first


@pytest.mark.parametrize("bad", [
    {"size": [H, W], "counts": [5, 5]},          # runs do not sum to h*w
    {"size": [H, W]},                            # no counts
    "not a mask",
    {"size": [0, 0], "counts": []},
])
def test_invalid_mask_fails_closed(src, tmp_path, bad):
    dst = tmp_path / "out" / "bad.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    f = blur_image_fn()
    with pytest.raises(Exception):  # noqa: B017
        f(str(src), str(dst), [bad])
    assert not dst.exists(), "an invalid mask must not leave an (unblurred) output behind"


def test_missing_source_raises_and_writes_nothing(tmp_path):
    f = blur_image_fn()
    dst = tmp_path / "out.jpg"
    with pytest.raises(Exception):  # noqa: B017
        f(str(tmp_path / "nope.jpg"), str(dst), [])
    assert not dst.exists()


def test_accepts_path_objects(src, tmp_path):
    f = blur_image_fn()
    dst = tmp_path / "out" / "p.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    f(Path(src), Path(dst), [rle_box(W, H, A)])
    assert dst.is_file()
