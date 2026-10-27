"""Task 22: the exported file (format, side by side layout, window, cap, clipping, raw half fidelity)."""
from __future__ import annotations

import pytest

import rs22_helpers as H
from rs22_helpers import (OVERLAY_BOX, SRC_DUR, SRC_FPS, SRC_H, SRC_W, StubProvider, decode_frames, est_index,
                          export_env, probe, rs22_env, rs22_std, src_frames, vstreams, astreams)  # noqa: F401

pytestmark = pytest.mark.filterwarnings("ignore")


# ------------------------------------------------------------------ format

def test_export_is_one_h264_mp4_without_audio_side_by_side(rs22_std):
    s = rs22_std
    f = s.video
    assert f.suffix.lower() == ".mp4"
    info = probe(f)
    assert len(info["streams"]) == 1, [x.get("codec_type") for x in info["streams"]]
    v = vstreams(info)
    assert len(v) == 1 and v[0]["codec_name"] == "h264", v
    assert not astreams(info), "the export must have no audio track"
    assert v[0].get("pix_fmt") == "yuv420p", v[0].get("pix_fmt")
    w, h = int(v[0]["width"]), int(v[0]["height"])
    assert w % 2 == 0 and h % 2 == 0
    assert w == 2 * SRC_W and h == SRC_H, "expected [raw | annotated] at the source size, got %dx%d" % (w, h)
    fps = H.fps_of(v[0])
    assert abs(fps - SRC_FPS) < 0.6, "frame rate must match the source (%s), got %s" % (SRC_FPS, fps)
    dur = H.duration_of(info)
    assert 7.4 <= dur <= 8.6, "default window is about 4 s before and 4 s after: got %.2f s" % dur
    n = s.frames.shape[0]
    assert abs(n - 80) <= 4, "expected about 80 frames, got %d" % n
    assert "mp4" in info["format"]["format_name"]


def test_export_is_decodable_end_to_end(rs22_std):
    fr = rs22_std.frames
    assert fr.shape[0] > 40 and fr.shape[2] == 2 * SRC_W


def test_record_window_matches_the_default_around_the_best_candidate(rs22_std):
    s = rs22_std
    assert abs(H.rec_start(s.rec) - 5.0) <= 0.3, s.rec
    assert abs(H.rec_end(s.rec) - 13.0) <= 0.3, s.rec


def test_segment_content_is_taken_around_the_best_scoring_candidate(rs22_std):
    """Frame time is read back from the red channel ramp of the raw (left) half."""
    s = rs22_std
    fr = s.frames
    first = est_index(fr[0, :, :SRC_W]) / float(SRC_FPS)
    last = est_index(fr[-1, :, :SRC_W]) / float(SRC_FPS)
    assert abs(first - 5.0) <= 0.7, "segment starts at %.2f s, expected about 5 (best candidate at 9 s, not 3 s)" % first
    assert abs(last - 13.0) <= 0.8, "segment ends at %.2f s, expected about 13" % last
    mid = est_index(fr[fr.shape[0] // 2, :, :SRC_W]) / float(SRC_FPS)
    assert abs(mid - 9.0) <= 0.8


# ------------------------------------------------------------------ window configuration

def test_window_before_and_after_are_configurable(rs22_env):
    env = rs22_env
    env.mp.setenv(export_env("before"), "2")
    env.mp.setenv(export_env("after"), "1")
    env.use_provider(StubProvider("empty"))
    did = env.add_drive("d22a")
    hid = env.seed(did, best_t=9.0, others=())
    rec = env.export_ok(hid)
    assert abs(H.rec_start(rec) - 7.0) <= 0.3 and abs(H.rec_end(rec) - 10.0) <= 0.3, rec
    f = env.export_videos(did)[0]
    dur = H.duration_of(probe(f))
    assert 2.5 <= dur <= 3.5, dur


def test_hard_cap_limits_the_length_and_the_provider_segment(rs22_env):
    env = rs22_env
    env.mp.setenv(export_env("before"), "15")
    env.mp.setenv(export_env("after"), "15")
    env.mp.setenv(export_env("max"), "6")
    stub = env.use_provider(StubProvider("empty"))
    did = env.add_drive("d22a")
    hid = env.seed(did, best_t=8.0, others=())
    rec = env.export_ok(hid)
    dur = H.duration_of(probe(env.export_videos(did)[0]))
    assert 5.4 <= dur <= 6.5, "cap 6 s: got %.2f s" % dur
    assert H.rec_end(rec) - H.rec_start(rec) <= 6.2
    assert len(stub.calls) == 1 and stub.calls[0]["dur"] <= 6.6, stub.calls


def test_default_cap_is_twenty_seconds(rs22_env):
    env = rs22_env
    env.mp.setenv(export_env("before"), "30")
    env.mp.setenv(export_env("after"), "30")
    stub = env.use_provider(StubProvider("empty"))
    did = env.add_drive("d22b", clip=H.clip_b_bytes(), duration=40.0)
    hid = env.seed(did, best_t=20.0, others=())
    rec = env.export_ok(hid)
    dur = H.duration_of(probe(env.export_videos(did)[0]))
    assert 18.5 <= dur <= 20.6, "default hard cap is 20 s: got %.2f s" % dur
    assert stub.calls[0]["dur"] <= 20.6


# ------------------------------------------------------------------ clipping to the video bounds

def test_window_is_clipped_at_the_start_of_the_video(rs22_env):
    env = rs22_env
    env.use_provider(StubProvider("empty"))
    did = env.add_drive("d22a")
    hid = env.seed(did, best_t=1.0, others=())
    rec = env.export_ok(hid)
    assert H.rec_start(rec) >= -0.001 and H.rec_start(rec) <= 0.3, rec
    assert abs(H.rec_end(rec) - 5.0) <= 0.4, rec
    dur = H.duration_of(probe(env.export_videos(did)[0]))
    assert 4.4 <= dur <= 5.6, dur


def test_window_is_clipped_at_the_end_of_the_video(rs22_env):
    env = rs22_env
    env.use_provider(StubProvider("empty"))
    did = env.add_drive("d22a")
    hid = env.seed(did, best_t=15.0, others=())
    rec = env.export_ok(hid)
    assert H.rec_end(rec) <= SRC_DUR + 0.05 and H.rec_end(rec) >= SRC_DUR - 0.4, rec
    assert abs(H.rec_start(rec) - 11.0) <= 0.4, rec
    dur = H.duration_of(probe(env.export_videos(did)[0]))
    assert 4.3 <= dur <= 5.7, dur


# ------------------------------------------------------------------ best candidate in another drive

def test_export_uses_the_drive_of_the_best_scoring_candidate(rs22_env):
    env = rs22_env
    stub = env.use_provider(StubProvider("empty"))
    env.add_drive("d22a")
    env.add_drive("d22b")
    hid = env.seed_other_best("d22b", "d22a", best_t=9.0, other_t=3.0)
    rec = env.export_ok(hid)
    assert str(rec.get("drive_id")) == "d22b", rec
    assert env.export_videos("d22b") and not env.export_videos("d22a")
    assert abs(H.rec_start(rec) - 5.0) <= 0.3


# ------------------------------------------------------------------ raw half fidelity

def _free_mask(shape_h, shape_w, boxes, margin):
    import numpy as np

    m = np.ones((shape_h, shape_w), dtype=bool)
    for (x1, y1, x2, y2) in boxes:
        m[max(0, int(y1 - margin)):int(y2 + margin), max(0, int(x1 - margin)):int(x2 + margin)] = False
    return m


def test_raw_half_matches_the_source_outside_the_blur_masks(rs22_std):
    import numpy as np

    s = rs22_std
    fr = s.frames
    src = src_frames()
    boxes = list(H.mask_boxes(SRC_W, SRC_H).values()) + [OVERLAY_BOX]
    free = _free_mask(SRC_H, SRC_W, boxes, 14)
    start_idx = int(round(H.rec_start(s.rec) * SRC_FPS))
    for i in (3, 15, 30, 44, 60, fr.shape[0] - 4):
        left = fr[i, :, :SRC_W].astype("float64")
        j0 = est_index(fr[i, :, :SRC_W])
        best_j, best = None, 1e9
        for j in range(max(0, j0 - 4), min(src.shape[0], j0 + 5)):
            mad = float(np.abs(left - src[j].astype("float64"))[free].mean())
            if mad < best:
                best_j, best = j, mad
        assert best < 25.0, "frame %d of the raw half does not look like any source frame (best MAD %.1f)" % (i, best)
        assert abs(best_j - (start_idx + i)) <= 3, "frame %d shows source frame %d, expected about %d" % (
            i, best_j, start_idx + i)


def test_frame_rate_and_frame_count_follow_the_source_even_for_a_short_window(rs22_env):
    env = rs22_env
    env.mp.setenv(export_env("before"), "1")
    env.mp.setenv(export_env("after"), "1")
    env.use_provider(StubProvider("empty"))
    did = env.add_drive("d22a")
    hid = env.seed(did, best_t=8.0, others=())
    env.export_ok(hid)
    f = env.export_videos(did)[0]
    info = probe(f)
    assert abs(H.fps_of(vstreams(info)[0]) - SRC_FPS) < 0.6
    n = decode_frames(f).shape[0]
    assert 18 <= n <= 22, n
