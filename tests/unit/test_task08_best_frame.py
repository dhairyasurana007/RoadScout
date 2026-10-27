"""Task 8: best-frame selection (mask size, low position, sharpness), determinism, saved images."""
from __future__ import annotations

import pytest

from rs8_helpers import (  # noqa: F401
    FPC, W, H, identify_frame, image_size, laplacian_var, make_spec, mean_level, rs8_cache, rs8_env, ts_of,
)

TRACK_FRAMES = [0, 1, 2, 3, 4]
ALL_FRAMES = [(0, f) for f in range(FPC)]
BASE = (60, 10, 100, 40)  # 40 x 30 box, high in the frame


def best_of(e):
    (c,) = e.cands()
    assert e.resolve(c["best_path"]).is_file(), "best frame file missing: %r" % (c["best_path"],)
    return identify_frame(e, c["best_path"], ALL_FRAMES), c


@pytest.mark.parametrize("winner", [0, 2, 4])
def test_larger_mask_beats_smaller(rs8_env, winner):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    for f in TRACK_FRAMES:
        # widening keeps the bottom edge and the vertical centre identical
        box = (40, 10, 130, 40) if f == winner else BASE
        e.add_det(0, f, 1, box)
    e.run()
    (chunk, frame), c = best_of(e)
    assert (chunk, frame) == (0, winner)


@pytest.mark.parametrize("winner", [0, 2, 4])
def test_lower_position_beats_higher(rs8_env, winner):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    for f in TRACK_FRAMES:
        box = (60, 50, 100, 80) if f == winner else BASE  # same size, 40 px lower
        e.add_det(0, f, 1, box)
    e.run()
    (chunk, frame), c = best_of(e)
    assert (chunk, frame) == (0, winner)


@pytest.mark.parametrize("winner", [0, 2, 4])
def test_sharper_frame_beats_blurred(rs8_env, winner):
    e = rs8_env
    blur = {(0, f): 2.5 for f in TRACK_FRAMES if f != winner}
    e.use_spec(make_spec(n_chunks=1, blur=blur))
    for f in TRACK_FRAMES:
        e.add_det(0, f, 1, BASE)
    e.run()
    (chunk, frame), c = best_of(e)
    assert (chunk, frame) == (0, winner)


def test_frames_without_detections_are_never_the_best_frame(rs8_env):
    """Frames 5 to 7 are perfectly sharp but the object was not detected there."""
    e = rs8_env
    blur = {(0, f): 2.5 for f in TRACK_FRAMES}
    e.use_spec(make_spec(n_chunks=1, blur=blur))
    for f in TRACK_FRAMES:
        e.add_det(0, f, 1, BASE)
    e.run()
    (chunk, frame), c = best_of(e)
    assert frame in TRACK_FRAMES


def test_frame_better_on_every_factor_wins_over_worse_on_every_factor(rs8_env):
    e = rs8_env
    blur = {(0, f): 2.5 for f in TRACK_FRAMES if f != 3}
    e.use_spec(make_spec(n_chunks=1, blur=blur))
    for f in TRACK_FRAMES:
        box = (30, 45, 130, 85) if f == 3 else (70, 5, 90, 20)  # bigger, lower, and sharp vs small, high, blurred
        e.add_det(0, f, 1, box)
    e.run()
    (chunk, frame), c = best_of(e)
    assert (chunk, frame) == (0, 3)


def test_best_frame_is_chosen_within_the_track_only(rs8_env):
    """Two tracks in one chunk: each candidate gets the best frame of its own track."""
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    for f in (0, 1, 2):  # track 1: biggest at frame 1
        e.add_det(0, f, 1, (60, 10, 140, 40) if f == 1 else (60, 10, 80, 40))
    for f in (3, 4, 5):  # track 2: biggest at frame 5
        e.add_det(0, f, 2, (10, 50, 70, 80) if f == 5 else (10, 50, 30, 80))
    e.run(min_frames=1)
    cs = {c["track_key"]: c for c in e.cands()}
    assert identify_frame(e, cs["0:1"]["best_path"], [(0, f) for f in (0, 1, 2)]) == (0, 1)
    assert identify_frame(e, cs["0:2"]["best_path"], [(0, f) for f in (3, 4, 5)]) == (0, 5)


def test_best_frame_comes_from_the_right_chunk(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=3))
    for f in range(4):
        e.add_det(2, f, 7, (60, 10, 140, 40) if f == 2 else (60, 10, 80, 40))
    e.run(min_frames=1)
    (c,) = e.cands()
    cand = [(2, f) for f in range(4)]
    assert identify_frame(e, c["best_path"], cand) == (2, 2)


# ----------------------------------------------------------------------------- determinism


def test_selection_is_deterministic_across_runs_and_insertion_order(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    widths = {0: 30, 1: 45, 2: 80, 3: 60, 4: 50}

    def insert(order):
        e.delete_dets()
        for f in order:
            e.add_det(0, f, 1, (30, 10, 30 + widths[f], 40))

    picks = []
    for order in ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 0, 4, 1, 3]):
        insert(order)
        e.run()
        e.run()  # twice on the same data
        picks.append(best_of(e)[0])
    assert picks == [(0, 2)] * 3


def test_ties_resolve_the_same_way_every_time(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    for f in TRACK_FRAMES:
        e.add_det(0, f, 1, BASE)
    picks = []
    for _ in range(3):
        e.run()
        picks.append(best_of(e)[0])
    assert len(set(picks)) == 1
    assert picks[0][0] == 0 and picks[0][1] in TRACK_FRAMES


# ----------------------------------------------------------------------------- saved files


def test_best_frame_and_crop_files_exist_and_have_sane_sizes(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    box = (50, 20, 110, 60)
    for f in TRACK_FRAMES:
        e.add_det(0, f, 1, box)
    e.run()
    (c,) = e.cands()
    best = e.resolve(c["best_path"])
    crop = e.resolve(c["crop_path"])
    assert best.is_file() and best.stat().st_size > 0
    assert crop.is_file() and crop.stat().st_size > 0
    assert best != crop
    assert image_size(best) == (W, H)
    cw, ch = image_size(crop)
    assert 0 < cw <= W and 0 < ch <= H
    # the crop covers the box (a margin is allowed) and is clearly smaller than the whole frame
    assert cw >= (box[2] - box[0]) - 2 and ch >= (box[3] - box[1]) - 2
    assert cw * ch < W * H
    # the crop shows the same frame as the best-frame image
    cand = ALL_FRAMES
    assert identify_frame(e, c["crop_path"], cand) == identify_frame(e, c["best_path"], cand)


@pytest.mark.parametrize("box", [(-10, -5, 60, 50), (120, 60, 200, 130), (-20, -20, 400, 300)])
def test_boxes_reaching_outside_the_frame_still_yield_a_crop_inside_it(rs8_env, box):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    for f in TRACK_FRAMES:
        e.add_det(0, f, 1, box)
    e.run()
    (c,) = e.cands()
    crop = e.resolve(c["crop_path"])
    assert crop.is_file()
    cw, ch = image_size(crop)
    assert 0 < cw <= W and 0 < ch <= H


def test_images_are_written_under_output_dir(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    for f in TRACK_FRAMES:
        e.add_det(0, f, 1, BASE)
    e.run(output_dir=e.out)
    (c,) = e.cands()
    for key in ("best_path", "crop_path"):
        p = e.resolve(c[key]).resolve()
        assert e.out.resolve() in p.parents, "%s is not under output_dir: %s" % (key, p)


def test_default_output_location_is_under_the_storage_dir(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    for f in TRACK_FRAMES:
        e.add_det(0, f, 1, BASE)
    e.run(output_dir=None)
    (c,) = e.cands()
    for key in ("best_path", "crop_path"):
        p = e.resolve(c[key]).resolve()
        assert p.is_file(), "%s does not exist: %s" % (key, p)
        assert (e.storage / "drive-a").resolve() in p.parents, "default images must live under storage/<drive_id>, got %s" % p


def test_rerun_keeps_files_valid(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    for f in TRACK_FRAMES:
        e.add_det(0, f, 1, BASE)
    e.run()
    e.run()
    (c,) = e.cands()
    assert e.resolve(c["best_path"]).is_file() and e.resolve(c["crop_path"]).is_file()


def test_sharp_and_blurred_synthetic_frames_really_differ(rs8_env):
    """Guards the test fixtures themselves: the sharp frame has a far higher Laplacian variance."""
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1, blur={(0, 1): 2.5}))
    for f in (0, 1, 2):
        e.add_det(0, f, 1, BASE)
    e.run(min_frames=1)
    (c,) = e.cands()
    best = e.resolve(c["best_path"])
    assert identify_frame(e, c["best_path"], [(0, 0), (0, 1), (0, 2)]) != (0, 1)
    assert laplacian_var(best) > 50
