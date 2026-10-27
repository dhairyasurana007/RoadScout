"""Task 22: the annotated (right) half: boxes / mask tint and the class label from the detections table.

The thresholds count strongly different pixels between the right (annotated) and the left (raw) half; they leave room
for codec noise but a missing or a spurious overlay is far outside it.
"""
from __future__ import annotations

import pytest

import rs22_helpers as H
from rs22_helpers import (FAR_BOX, OTHER_BOX, OVERLAY_BOX, SRC_FPS, SRC_H, SRC_W, StubProvider, decode_frames,
                          diff_map, region_count, rs22_env, rs22_std, total_count)  # noqa: F401

pytestmark = pytest.mark.filterwarnings("ignore")

PRESENT_MIN = 150     # pixels with a strong difference inside the (expanded) box of a drawn detection
ABSENT_MAX = 60       # the same count where nothing is drawn (codec noise only)
MARGIN = 40           # the label must be drawn within this distance of its box


def _idx(s, t):
    return int(round((t - s.start) * SRC_FPS))


def _frame_at(fr, s, t):
    i = _idx(s, t)
    assert 0 <= i < fr.shape[0], (t, i, fr.shape[0])
    return fr[i]


def test_annotation_appears_in_the_right_half_at_the_detection_times(rs22_std):
    s = rs22_std
    fr = s.frames
    for t in (7.0, 7.5, 8.0, 8.5):
        d = diff_map(_frame_at(fr, s, t))
        n = region_count(d, OVERLAY_BOX, margin=MARGIN)
        assert n >= PRESENT_MIN, "no visible annotation at t=%.1f (changed pixels near the box: %d)" % (t, n)
        inside = region_count(d, OVERLAY_BOX, margin=4)
        assert inside >= 60, "the box / mask tint itself is not visible at t=%.1f (%d)" % (t, inside)


def test_annotation_is_absent_where_there_is_no_detection(rs22_std):
    s = rs22_std
    fr = s.frames
    for t in (5.3, 5.8, 6.2, 9.8, 10.5, 11.5, 12.5):
        d = diff_map(_frame_at(fr, s, t))
        assert region_count(d, OVERLAY_BOX, margin=MARGIN) <= ABSENT_MAX, "spurious overlay at t=%.1f" % t
        assert total_count(d) <= 400, "t=%.1f: the halves differ although nothing is detected (%d px)" % (t, total_count(d))
        assert float(d.mean()) < 8.0


def test_annotation_stays_near_its_box_and_the_left_half_is_clean(rs22_std):
    import numpy as np

    s = rs22_std
    fr = s.frames
    for t in (7.0, 8.0):
        d = diff_map(_frame_at(fr, s, t))
        near = region_count(d, OVERLAY_BOX, margin=MARGIN)
        assert total_count(d) - near <= 300, "changes far from the detection box at t=%.1f" % t
    # the left half must equal the clean source: no overlay pixels in the box area of the raw half
    src = H.src_frames()
    i = _idx(s, 8.0)
    left = fr[i, :, :SRC_W].astype("float64")
    j0 = H.est_index(fr[i, :, :SRC_W])
    best = min(float(np.abs(left - src[j].astype("float64"))[50:100, 110:190].mean())
               for j in range(max(0, j0 - 4), min(src.shape[0], j0 + 5)))
    assert best < 25.0, "the raw half contains the overlay (box area MAD %.1f)" % best


def test_no_detections_means_no_overlay_in_any_frame(rs22_env):
    env = rs22_env
    env.use_provider(StubProvider("empty"))
    did = env.add_drive("d22a")
    hid = env.seed(did, best_t=9.0, others=())
    env.export_ok(hid)
    fr = decode_frames(env.export_videos(did)[0])
    for i in range(0, fr.shape[0], 5):
        d = diff_map(fr[i])
        assert total_count(d) <= 400, "frame %d: halves differ with an empty detections table (%d px)" % (i, total_count(d))
        assert float(d.mean()) < 8.0


def test_class_label_is_drawn_next_to_a_tiny_box(rs22_env):
    env = rs22_env
    env.use_provider(StubProvider("empty"))
    did = env.add_drive("d22a")
    hid = env.seed(did, best_t=9.0, others=())
    cls = max(H.known_classes(), key=len)
    tiny = (150, 100, 160, 110)
    for t in (7.0, 7.5, 8.0, 8.5):
        env.add_det(did, t, tiny, cls=cls)
    rec = env.export_ok(hid)
    fr = decode_frames(env.export_videos(did)[0])
    start = H.rec_start(rec)
    for t in (7.0, 8.0):
        d = diff_map(fr[int(round((t - start) * SRC_FPS))])
        ring = region_count(d, tiny, margin=MARGIN, exclude=tiny, exclude_margin=5)
        assert ring >= 40, "no class label drawn near the box at t=%.1f (%d changed pixels around it)" % (t, ring)


def test_every_detection_in_the_window_is_drawn_not_only_one_class(rs22_env):
    env = rs22_env
    env.use_provider(StubProvider("empty"))
    did = env.add_drive("d22a")
    hid = env.seed(did, best_t=9.0, others=())
    classes = H.known_classes()
    other_cls = [c for c in classes if c != "pothole"][0]
    for t in (7.0, 7.5, 8.0, 8.5):
        env.add_det(did, t, OVERLAY_BOX, cls="pothole", track=1)
        env.add_det(did, t, FAR_BOX, cls=other_cls, track=2)
    rec = env.export_ok(hid)
    fr = decode_frames(env.export_videos(did)[0])
    start = H.rec_start(rec)
    d = diff_map(fr[int(round((8.0 - start) * SRC_FPS))])
    assert region_count(d, OVERLAY_BOX, margin=MARGIN) >= PRESENT_MIN
    assert region_count(d, FAR_BOX, margin=10) >= PRESENT_MIN, "the second detection is not drawn"


def test_detections_of_other_drives_and_outside_the_window_are_not_drawn(rs22_env):
    env = rs22_env
    env.use_provider(StubProvider("empty"))
    did = env.add_drive("d22a")
    env.add_drive("d22x")
    hid = env.seed(did, best_t=9.0, others=())
    for t in (7.0, 7.5, 8.0, 8.5):
        env.add_det(did, t, OVERLAY_BOX)
        env.add_det("d22x", t, OTHER_BOX)               # another drive, same time
    for t in (1.0, 1.5, 2.0, 14.0, 14.5, 15.0):
        env.add_det(did, t, FAR_BOX)                     # same drive, outside the exported window
    rec = env.export_ok(hid)
    fr = decode_frames(env.export_videos(did)[0])
    start = H.rec_start(rec)
    for t in (7.0, 7.5, 8.0, 8.5):
        d = diff_map(fr[int(round((t - start) * SRC_FPS))])
        assert region_count(d, OVERLAY_BOX, margin=MARGIN) >= PRESENT_MIN
        assert region_count(d, OTHER_BOX, margin=6) <= ABSENT_MAX, "a detection of another drive was drawn"
        assert region_count(d, FAR_BOX, margin=6) <= ABSENT_MAX, "a detection outside the window was drawn"
    for i in range(0, fr.shape[0], 7):
        d = diff_map(fr[i])
        assert region_count(d, FAR_BOX, margin=6) <= ABSENT_MAX
        assert region_count(d, OTHER_BOX, margin=6) <= ABSENT_MAX


def test_mask_detections_are_drawn_too(rs22_env):
    from backend.providers.types import rle_from_box

    env = rs22_env
    env.use_provider(StubProvider("empty"))
    did = env.add_drive("d22a")
    hid = env.seed(did, best_t=9.0, others=())
    for t in (7.0, 7.5, 8.0, 8.5):
        env.add_det(did, t, OVERLAY_BOX, mask=rle_from_box(SRC_W, SRC_H, OVERLAY_BOX))
    rec = env.export_ok(hid)
    fr = decode_frames(env.export_videos(did)[0])
    d = diff_map(fr[int(round((8.0 - H.rec_start(rec)) * SRC_FPS))])
    assert region_count(d, OVERLAY_BOX, margin=MARGIN) >= PRESENT_MIN
    assert total_count(d) - region_count(d, OVERLAY_BOX, margin=MARGIN) <= 300
