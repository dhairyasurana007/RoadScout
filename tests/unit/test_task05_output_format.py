"""Task 5: output format (audio stripped, ~720p, few fps, standard H.264 mp4, upright, chunked)."""
import inspect

import pytest

import rs5_helpers as h
from rs5_helpers import clip, clips  # noqa: F401  (fixtures)

pytestmark = h.pytestmark_ffmpeg


def _all_chunk_streams(paths):
    for p in paths:
        for s in h.streams(p):
            yield p, s


# ------------------------------------------------------------------ interface


def test_module_and_function_exist():
    assert callable(h.find_func())


def test_function_takes_input_and_output_dir_and_is_configurable():
    sig = inspect.signature(h.find_func())
    params = list(sig.parameters)
    assert len(params) >= 2
    assert h._pick(sig, h.FPS_NAMES), "no configurable fps parameter (expected one of %s)" % h.FPS_NAMES
    assert h._pick(sig, h.CHUNK_NAMES), "no configurable chunk duration (expected one of %s)" % h.CHUNK_NAMES


def test_default_fps_is_a_few_fps(tmp_path, clip):
    _, out, paths, frames = h.run_pre(tmp_path, clip("h264_audio"))
    ts = [t for t, _, _ in frames]
    assert 2 <= (len(ts) - 1) / (ts[-1] - ts[0]) <= 5.3


def test_default_chunk_length_is_short(tmp_path, clip):
    _, out, paths, _ = h.run_pre(tmp_path, clip("h264_audio"))
    for p in paths:
        assert h.container_duration(p) <= 61


def test_default_height_is_720(tmp_path, clip):
    _, out, paths, _ = h.run_pre(tmp_path, clip("h264_audio"))
    assert 700 <= h.vstream(paths[0])["height"] <= 740


# ------------------------------------------------------------------ audio


def test_audio_stripped_from_h264_input_with_audio(tmp_path, clip):
    src = clip("h264_audio")
    assert any(s["codec_type"] == "audio" for s in h.streams(src)), "fixture should have audio"
    _, out, paths, _ = h.run_pre(tmp_path, src, fps=3, chunk_seconds=3)
    assert paths
    for p, s in _all_chunk_streams(paths):
        assert s["codec_type"] != "audio", "audio stream left in %s" % p


def test_audio_stripped_from_hevc_mov_with_audio(tmp_path, clip):
    src = clip("hevc")
    assert any(s["codec_type"] == "audio" for s in h.streams(src))
    _, out, paths, _ = h.run_pre(tmp_path, src, fps=3, chunk_seconds=2)
    for p, s in _all_chunk_streams(paths):
        assert s["codec_type"] == "video", "unexpected %s stream in %s" % (s["codec_type"], p)


def test_no_audio_file_is_written_anywhere_in_output_dir(tmp_path, clip):
    _, out, _, _ = h.run_pre(tmp_path, clip("h264_audio"), fps=3, chunk_seconds=3)
    bad = {".wav", ".aac", ".m4a", ".mp3", ".ogg", ".opus", ".flac", ".mka"}
    found = [p for p in out.rglob("*") if p.suffix.lower() in bad]
    assert not found, "audio artifacts written: %s" % found


# ------------------------------------------------------------------ standard format


def test_chunks_are_h264_mp4_yuv420p(tmp_path, clip):
    _, out, paths, _ = h.run_pre(tmp_path, clip("h264_audio"), fps=3, chunk_seconds=3)
    assert paths
    for p in paths:
        assert str(p).lower().endswith(".mp4")
        fmt = h.ffprobe_json(p, "-show_format")["format"]["format_name"]
        assert "mp4" in fmt
        v = h.vstream(p)
        assert v["codec_name"] == "h264"
        assert v["pix_fmt"] == "yuv420p"


def test_hevc_mov_input_is_normalized_to_h264_mp4(tmp_path, clip):
    src = clip("hevc")
    assert h.vstream(src)["codec_name"] == "hevc"
    _, out, paths, _ = h.run_pre(tmp_path, src, fps=3, chunk_seconds=2)
    assert paths
    for p in paths:
        assert h.vstream(p)["codec_name"] == "h264"
        assert "mp4" in h.ffprobe_json(p, "-show_format")["format"]["format_name"]


def test_chunks_decode_cleanly(tmp_path, clip):
    _, out, paths, _ = h.run_pre(tmp_path, clip("h264_audio"), fps=3, chunk_seconds=3)
    for p in paths:
        rc, err = h.decode_errors(p)
        assert rc == 0 and err == "", "%s: %s" % (p, err)


def test_rotation_metadata_not_left_on_output(tmp_path, clip):
    src = clip("rotated")
    assert h.has_rotation(src)
    _, out, paths, _ = h.run_pre(tmp_path, src, fps=3, chunk_seconds=2)
    for p in paths:
        assert not h.has_rotation(p), "rotation metadata still present on %s (would double-rotate)" % p


# ------------------------------------------------------------------ resolution


def test_1080p_input_is_downsampled_to_about_720p(tmp_path, clip):
    src = clip("h264_audio")
    assert (h.vstream(src)["width"], h.vstream(src)["height"]) == (1920, 1080)
    _, out, paths, _ = h.run_pre(tmp_path, src, fps=3, chunk_seconds=3)
    for p in paths:
        v = h.vstream(p)
        assert 700 <= v["height"] <= 740, v["height"]
        assert v["height"] % 2 == 0 and v["width"] % 2 == 0
        assert abs(v["width"] / v["height"] - 16 / 9) < 0.02, "aspect ratio changed: %sx%s" % (v["width"], v["height"])


def test_hevc_1080p_input_is_downsampled_to_about_720p(tmp_path, clip):
    _, out, paths, _ = h.run_pre(tmp_path, clip("hevc"), fps=3, chunk_seconds=2)
    for p in paths:
        assert 700 <= h.vstream(p)["height"] <= 740


def test_all_chunks_share_the_same_dimensions_and_fps(tmp_path, clip):
    _, out, paths, _ = h.run_pre(tmp_path, clip("h264_audio"), fps=3, chunk_seconds=2)
    assert len(paths) >= 3
    dims = {(h.vstream(p)["width"], h.vstream(p)["height"]) for p in paths}
    rates = {h.vstream(p)["avg_frame_rate"] for p in paths}
    assert len(dims) == 1, dims
    assert len(rates) == 1, rates


def test_small_input_is_not_upscaled(tmp_path, clip):
    src = clip("small")
    assert h.vstream(src)["height"] == 360
    _, out, paths, _ = h.run_pre(tmp_path, src, fps=3, chunk_seconds=2)
    for p in paths:
        v = h.vstream(p)
        assert v["height"] <= 364, "small input was upscaled to %sx%s" % (v["width"], v["height"])


def test_height_is_configurable(tmp_path, clip):
    sig_has = h._pick(__import__("inspect").signature(h.find_func()), h.HEIGHT_NAMES)
    if not sig_has:
        pytest.skip("height is not configurable")
    _, out, paths, _ = h.run_pre(tmp_path, clip("h264_audio"), fps=3, chunk_seconds=3, height=480)
    for p in paths:
        assert 470 <= h.vstream(p)["height"] <= 490


def test_rotated_input_ends_up_portrait_and_sensible(tmp_path, clip):
    src = clip("rotated")
    raw = h.vstream(src)
    assert raw["width"] > raw["height"], "fixture stores landscape pixels"
    _, out, paths, _ = h.run_pre(tmp_path, src, fps=3, chunk_seconds=2)
    assert paths
    for p in paths:
        v = h.vstream(p)
        w, ht = v["width"], v["height"]
        assert ht > w, "portrait phone video should come out portrait, got %dx%d" % (w, ht)
        assert min(w, ht) <= 724
        assert max(w, ht) <= 1284
        assert max(w, ht) >= 600
        assert abs(w / ht - 9 / 16) < 0.02, "aspect ratio changed: %dx%d" % (w, ht)
        assert w % 2 == 0 and ht % 2 == 0


def test_rotated_input_comes_out_upright(tmp_path, clip):
    """Compare quadrant colours to ffmpeg's own autorotated decode of the input."""
    src = clip("rotated")
    ref = h.quadrant_rgb(src, autorotate=True)
    unrot = h.quadrant_rgb(src, autorotate=False)
    if all(abs(a - b) < 40 for a, b in zip(ref, unrot)):
        pytest.skip("this ffmpeg did not apply the rotation; cannot build a reference")
    _, out, paths, _ = h.run_pre(tmp_path, src, fps=3, chunk_seconds=2)
    got = h.quadrant_rgb(paths[0], autorotate=True)
    assert all(abs(a - b) < 45 for a, b in zip(ref, got)), "orientation differs: ref=%s got=%s" % (ref, got)


# ------------------------------------------------------------------ frame rate


@pytest.mark.parametrize("fps", [2, 4])
def test_output_frame_rate_matches_requested_and_is_constant(tmp_path, clip, fps):
    _, out, paths, _ = h.run_pre(tmp_path, clip("h264_audio"), fps=fps, chunk_seconds=3, name="o%d" % fps)
    for p in paths:
        v = h.vstream(p)
        assert abs(h.frac(v["avg_frame_rate"]) - fps) < 0.05, v["avg_frame_rate"]
        assert abs(h.frac(v["r_frame_rate"]) - fps) < 0.05, v["r_frame_rate"]
        pts = h.packet_pts(p)
        if len(pts) > 1:
            deltas = [b - a for a, b in zip(pts, pts[1:])]
            assert max(deltas) - min(deltas) < 0.05 * (1.0 / fps) + 1e-3, deltas


def test_default_fps_run_yields_a_few_fps(tmp_path, clip):
    result, out, paths, frames = h.run_pre(tmp_path, clip("h264_audio"))
    ts = [t for t, _, _ in frames]
    assert len(ts) >= 2
    rate = (len(ts) - 1) / (ts[-1] - ts[0])
    assert 1.8 <= rate <= 5.3, rate
    for p in paths:
        assert 1.8 <= h.frac(h.vstream(p)["avg_frame_rate"]) <= 5.3


def test_variable_frame_rate_input_comes_out_constant_frame_rate(tmp_path, clip):
    src = clip("vfr")
    v = h.vstream(src)
    assert v["avg_frame_rate"] != v["r_frame_rate"] or h.frac(v["r_frame_rate"]) > 25, "fixture should be VFR"
    _, out, paths, _ = h.run_pre(tmp_path, src, fps=3, chunk_seconds=3)
    for p in paths:
        pv = h.vstream(p)
        assert abs(h.frac(pv["avg_frame_rate"]) - 3) < 0.05
        assert abs(h.frac(pv["r_frame_rate"]) - 3) < 0.05
        pts = h.packet_pts(p)
        deltas = [b - a for a, b in zip(pts, pts[1:])]
        assert not deltas or max(deltas) - min(deltas) < 0.02


# ------------------------------------------------------------------ chunking


def test_chunk_count_and_durations(tmp_path, clip):
    src = clip("h264_audio")  # 7 s
    dur = h.container_duration(src)
    _, out, paths, _ = h.run_pre(tmp_path, src, fps=3, chunk_seconds=3)
    assert len(paths) == h.ceil_div(dur, 3) == 3
    ds = [h.container_duration(p) for p in paths]
    for d in ds[:-1]:
        assert abs(d - 3) <= 1 / 3 + 0.25, ds
    assert 0.3 <= ds[-1] <= 3.6, ds


def test_chunk_length_is_configurable(tmp_path, clip):
    src = clip("h264_audio")
    _, _, p2, _ = h.run_pre(tmp_path, src, fps=3, chunk_seconds=2, name="a")
    _, _, p5, _ = h.run_pre(tmp_path, src, fps=3, chunk_seconds=5, name="b")
    assert len(p2) == 4
    assert len(p5) == 2


def test_chunks_exist_inside_output_dir_and_are_ordered(tmp_path, clip):
    _, out, paths, _ = h.run_pre(tmp_path, clip("h264_audio"), fps=3, chunk_seconds=2)
    out_r = out.resolve()
    for p in paths:
        assert p.is_file() and p.stat().st_size > 0
        assert out_r in p.resolve().parents
    assert [str(p) for p in paths] == sorted(str(p) for p in paths), "chunk file names should sort in time order"
    assert len({str(p) for p in paths}) == len(paths)


def test_result_chunk_metadata_when_present(tmp_path, clip):
    result, out, paths, _ = h.run_pre(tmp_path, clip("h264_audio"), fps=3, chunk_seconds=3)
    chunks = h.get(result, "chunks")
    assert chunks is not None and len(chunks) == len(paths)
    starts = [h.get(c, "start_time", "start", "start_s") for c in chunks]
    if any(s is None for s in starts):
        pytest.skip("chunk items carry no start time")
    starts = [float(s) for s in starts]
    assert starts == sorted(starts) and abs(starts[0]) < 0.4
    for s, expected in zip(starts, (0, 3, 6)):
        assert abs(s - expected) <= 0.5, starts


def test_very_short_clip_yields_exactly_one_chunk(tmp_path, clip):
    _, out, paths, frames = h.run_pre(tmp_path, clip("short"), fps=3, chunk_seconds=10)
    assert len(paths) == 1
    assert len(frames) >= 2
    assert h.count_frames(paths[0]) == len(frames)


def test_clip_shorter_than_default_chunk_yields_one_chunk(tmp_path, clip):
    _, out, paths, _ = h.run_pre(tmp_path, clip("short"))
    assert len(paths) == 1
