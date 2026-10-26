"""Task 5: saved frame timestamps (global, monotonic, gap-free, matching decoded frames)."""
import json

import pytest

import rs5_helpers as h
from rs5_helpers import clip, clips  # noqa: F401  (fixtures)

pytestmark = h.pytestmark_ffmpeg


def test_timestamps_file_exists_is_json_and_in_output_dir(tmp_path, clip):
    result, out, _, _ = h.run_pre(tmp_path, clip("h264_audio"), fps=3, chunk_seconds=3)
    p = h.timestamps_path(result, out)
    assert p.is_file()
    assert out.resolve() in p.resolve().parents
    json.loads(p.read_text(encoding="utf-8"))  # valid JSON


def test_default_timestamps_file_name(tmp_path, clip):
    _, out, _, _ = h.run_pre(tmp_path, clip("h264_audio"), fps=3, chunk_seconds=3)
    assert (out / "frame_timestamps.json").is_file()


@pytest.mark.parametrize("name", ["h264_audio", "hevc", "vfr", "rotated"])
def test_frame_count_matches_decoded_frames(tmp_path, clip, name):
    _, out, paths, frames = h.run_pre(tmp_path, clip(name), fps=3, chunk_seconds=2)
    decoded = sum(h.count_frames(p) for p in paths)
    assert len(frames) == decoded, "record has %d frames, chunks decode to %d" % (len(frames), decoded)
    assert decoded > 0


def test_timestamps_strictly_increasing_and_in_seconds(tmp_path, clip):
    src = clip("h264_audio")
    dur = h.container_duration(src)
    _, out, paths, frames = h.run_pre(tmp_path, src, fps=3, chunk_seconds=2)
    ts = [t for t, _, _ in frames]
    assert all(b > a for a, b in zip(ts, ts[1:])), "timestamps must strictly increase"
    assert ts[0] >= 0
    assert ts[-1] <= dur + 0.1, "last timestamp %.3f beyond clip duration %.3f (seconds expected)" % (ts[-1], dur)


def test_timestamps_are_global_not_chunk_relative(tmp_path, clip):
    """Across chunk boundaries the clock keeps going: no reset to zero."""
    src = clip("h264_audio")
    _, out, paths, frames = h.run_pre(tmp_path, src, fps=3, chunk_seconds=2)
    assert len(paths) >= 3
    ts = [t for t, _, _ in frames]
    assert ts[-1] > 5.0, "timestamps look chunk-relative (last = %.3f)" % ts[-1]
    # one monotonic series over the whole drive
    assert ts == sorted(ts) and len(set(ts)) == len(ts)


def test_no_gaps_and_covers_whole_input(tmp_path, clip):
    src = clip("h264_audio")
    dur = h.container_duration(src)
    fps = 3
    _, out, paths, frames = h.run_pre(tmp_path, src, fps=fps, chunk_seconds=2)
    ts = [t for t, _, _ in frames]
    assert ts[0] <= 1.0 / fps + 1e-3, "first sampled frame should be at the start, got %.3f" % ts[0]
    assert ts[-1] >= dur - 1.5 / fps, "record stops at %.3f of %.3f s" % (ts[-1], dur)
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    assert max(gaps) <= 1.5 / fps, "gap of %.3f s (including across chunk boundaries)" % max(gaps)
    assert min(gaps) >= 0.5 / fps, "frames sampled too densely/duplicated: %.3f" % min(gaps)


def test_sampled_count_matches_duration_times_fps(tmp_path, clip):
    src = clip("h264_audio")
    dur = h.container_duration(src)
    fps = 3
    _, out, paths, frames = h.run_pre(tmp_path, src, fps=fps, chunk_seconds=3)
    assert abs(len(frames) - dur * fps) <= len(paths) + 2


def test_timestamp_spacing_is_about_one_over_fps(tmp_path, clip):
    _, out, _, frames = h.run_pre(tmp_path, clip("h264_audio"), fps=4, chunk_seconds=3)
    ts = [t for t, _, _ in frames]
    mean_gap = (ts[-1] - ts[0]) / (len(ts) - 1)
    assert abs(mean_gap - 0.25) < 0.03, mean_gap


def test_chunk_index_on_frames_is_consistent_with_chunk_files(tmp_path, clip):
    _, out, paths, frames = h.run_pre(tmp_path, clip("h264_audio"), fps=3, chunk_seconds=3)
    chunk_ids = [c for _, c, _ in frames]
    if any(c is None for c in chunk_ids):
        pytest.skip("frame entries do not carry a chunk index")
    assert chunk_ids == sorted(chunk_ids), "frames must be listed in chunk order"
    base = min(chunk_ids)
    for i, p in enumerate(paths):
        n = sum(1 for c in chunk_ids if c - base == i)
        assert n == h.count_frames(p), "chunk %d: %d recorded vs %d decoded" % (i, n, h.count_frames(p))
    assert sorted(set(chunk_ids)) == list(range(base, base + len(paths)))


def test_chunk_relative_timestamps_when_present(tmp_path, clip):
    _, out, paths, frames = h.run_pre(tmp_path, clip("h264_audio"), fps=3, chunk_seconds=3)
    rel = []
    for t, c, e in frames:
        if not isinstance(e, dict):
            pytest.skip("entries are bare numbers")
        k = next((k for k in ("chunk_timestamp", "chunk_time", "local_timestamp", "offset") if k in e), None)
        if k is None:
            pytest.skip("no chunk-relative field recorded (optional)")
        rel.append((t, c, float(e[k])))
    for t, c, r in rel:
        assert 0 <= r <= 3.5
    # the relative time restarts in every chunk, and the global time does not
    firsts = {}
    for t, c, r in rel:
        firsts.setdefault(c, r)
    assert all(v <= 0.4 for v in firsts.values())


def test_timestamps_map_to_source_time_cfr(tmp_path, clip):
    """The ramp clip's brightness encodes source time; recorded timestamps must agree with the pixels."""
    src = clip("ramp")  # 10 s, luma = 25.5 * t
    _, out, paths, frames = h.run_pre(tmp_path, src, fps=3, chunk_seconds=3)
    luma = []
    for p in paths:
        luma += h.frame_luma(p)
    assert len(luma) == len(frames)
    est = h.ramp_time_from_luma(clip("ramp"))
    checked = 0
    bad = []
    for (t, _, _), y in zip(frames, luma):
        t_pix = est(y)
        if t_pix is None:
            continue
        checked += 1
        if abs(t_pix - t) > 0.45:
            bad.append((round(t, 2), round(t_pix, 2)))
    assert checked >= 15
    assert not bad, "timestamps disagree with frame content (t_recorded, t_from_pixels): %s" % bad[:6]


def test_timestamps_map_to_source_time_variable_frame_rate(tmp_path, clip):
    """Same check on a VFR source (10 fps first half, 30 fps second half): no drift."""
    src = clip("vfr")
    _, out, paths, frames = h.run_pre(tmp_path, src, fps=3, chunk_seconds=3)
    luma = []
    for p in paths:
        luma += h.frame_luma(p)
    assert len(luma) == len(frames)
    est = h.ramp_time_from_luma(clip("ramp"))
    checked = 0
    bad = []
    for (t, _, _), y in zip(frames, luma):
        t_pix = est(y)
        if t_pix is None:
            continue
        checked += 1
        if abs(t_pix - t) > 0.45:
            bad.append((round(t, 2), round(t_pix, 2)))
    assert checked >= 15
    assert not bad, "VFR timestamps drift from content (t_recorded, t_from_pixels): %s" % bad[:6]
    ts = [t for t, _, _ in frames]
    assert max(b - a for a, b in zip(ts, ts[1:])) <= 0.5


def test_rotated_input_timestamps_are_complete(tmp_path, clip):
    src = clip("rotated")
    dur = h.container_duration(src)
    _, out, paths, frames = h.run_pre(tmp_path, src, fps=3, chunk_seconds=2)
    ts = [t for t, _, _ in frames]
    assert ts[-1] >= dur - 0.5 and all(b > a for a, b in zip(ts, ts[1:]))
