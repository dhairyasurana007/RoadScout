"""Task 5: errors, input immutability, path handling, reruns."""
import os
from pathlib import Path

import pytest

import rs5_helpers as h
from rs5_helpers import clip, clips  # noqa: F401  (fixtures)

pytestmark = h.pytestmark_ffmpeg

BAD_EXC = (TypeError, AttributeError, NameError, ImportError, KeyError, IndexError, UnboundLocalError)


def _expect_clear_error(tmp_path, bad_input, name="o"):
    out = Path(tmp_path) / name
    with pytest.raises(Exception) as ei:
        h.call(bad_input, out, fps=3, chunk_seconds=3)
    e = ei.value
    assert not isinstance(e, BAD_EXC), "unclear error type %s: %s" % (type(e).__name__, e)
    assert str(e).strip(), "error has no message"
    return e


# ------------------------------------------------------------------ invalid input


def test_missing_file_raises_clear_error(tmp_path):
    _expect_clear_error(tmp_path, tmp_path / "nope.mp4")


def test_text_file_with_video_extension_raises(tmp_path):
    f = tmp_path / "fake.mp4"
    f.write_text("this is not a video at all\n" * 50)
    _expect_clear_error(tmp_path, f)


def test_empty_file_raises(tmp_path):
    f = tmp_path / "empty.mov"
    f.write_bytes(b"")
    _expect_clear_error(tmp_path, f)


def test_truncated_garbage_bytes_raise(tmp_path):
    f = tmp_path / "garbage.mp4"
    f.write_bytes(os.urandom(4096))
    _expect_clear_error(tmp_path, f)


def test_directory_as_input_raises(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    _expect_clear_error(tmp_path, d)


def test_audio_only_input_raises(tmp_path):
    f = tmp_path / "audio_only.m4a"
    ok, err = h.ffmpeg_ok(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-t", "2",
                           "-c:a", "aac", str(f)])
    assert ok, err
    _expect_clear_error(tmp_path, f)


def test_failed_run_leaves_no_chunk_files_that_look_valid(tmp_path):
    f = tmp_path / "fake.mp4"
    f.write_text("nope")
    out = tmp_path / "o"
    with pytest.raises(Exception):
        h.call(f, out, fps=3, chunk_seconds=3)
    assert not list(out.rglob("*.mp4")) if out.exists() else True


# ------------------------------------------------------------------ input immutability / paths


def test_original_input_is_not_modified_and_nothing_written_next_to_it(tmp_path, clip):
    src = Path(tmp_path) / "inbox" / "input.mp4"
    src.parent.mkdir()
    src.write_bytes(clip("h264_audio").read_bytes())
    before_hash = h.sha256(src)
    before_stat = (src.stat().st_size, src.stat().st_mtime_ns)
    before_dir = sorted(os.listdir(src.parent))
    h.run_pre(tmp_path, src, fps=3, chunk_seconds=3)
    assert src.exists()
    assert h.sha256(src) == before_hash
    assert (src.stat().st_size, src.stat().st_mtime_ns) == before_stat
    assert sorted(os.listdir(src.parent)) == before_dir, "files were written next to the input"
    assert any(s["codec_type"] == "audio" for s in h.streams(src)), "audio was stripped from the original"


def test_accepts_str_and_pathlib_paths(tmp_path, clip):
    src = clip("short")
    r1 = h.call(str(src), str(tmp_path / "as_str"), fps=3, chunk_seconds=5)
    r2 = h.call(src, tmp_path / "as_path", fps=3, chunk_seconds=5)
    assert h.result_chunk_paths(r1, tmp_path / "as_str")
    assert h.result_chunk_paths(r2, tmp_path / "as_path")


def test_creates_missing_output_dir_including_parents(tmp_path, clip):
    out = tmp_path / "a" / "b" / "c"
    assert not out.exists()
    r = h.call(clip("short"), out, fps=3, chunk_seconds=5)
    assert out.is_dir()
    assert h.result_chunk_paths(r, out)


def test_works_with_existing_empty_output_dir(tmp_path, clip):
    out = tmp_path / "exists"
    out.mkdir()
    r = h.call(clip("short"), out, fps=3, chunk_seconds=5)
    assert h.result_chunk_paths(r, out)


def test_input_with_spaces_in_path(tmp_path, clip):
    src = tmp_path / "my drive 1.mp4"
    src.write_bytes(clip("short").read_bytes())
    r = h.call(src, tmp_path / "out dir", fps=3, chunk_seconds=5)
    assert h.result_chunk_paths(r, tmp_path / "out dir")


def test_two_inputs_into_separate_output_dirs_are_independent(tmp_path, clip):
    _, _, p1, f1 = h.run_pre(tmp_path, clip("short"), fps=3, chunk_seconds=5, name="x")
    _, _, p2, f2 = h.run_pre(tmp_path, clip("h264_audio"), fps=3, chunk_seconds=3, name="y")
    assert len(p1) == 1 and len(p2) == 3
    assert len(f1) < len(f2)


def test_rerun_into_same_dir_leaves_no_stale_chunks(tmp_path, clip):
    src = clip("h264_audio")
    out = tmp_path / "again"
    h.call(src, out, fps=3, chunk_seconds=2)  # 4 chunks
    r = h.call(src, out, fps=3, chunk_seconds=5)  # 2 chunks
    paths = h.result_chunk_paths(r, out)
    on_disk = sorted(out.rglob("*.mp4"))
    assert len(paths) == 2
    assert len(on_disk) == 2, "stale chunks from the previous run left behind: %s" % on_disk
    frames = h.load_frames(h.timestamps_path(r, out))
    assert len(frames) == sum(h.count_frames(p) for p in paths)


# ------------------------------------------------------------------ other containers / input shapes


def test_audio_free_input_still_works(tmp_path, clip):
    _, out, paths, frames = h.run_pre(tmp_path, clip("h264_silent"), fps=3, chunk_seconds=3)
    assert len(paths) == 3 and frames
    for p in paths:
        assert all(s["codec_type"] == "video" for s in h.streams(p))


def test_mkv_container_input(tmp_path, clip):
    mkv = tmp_path / "in.mkv"
    ok, err = h.ffmpeg_ok(["-i", str(clip("small")), "-c", "copy", str(mkv)])
    assert ok, err
    _, out, paths, frames = h.run_pre(tmp_path, mkv, fps=3, chunk_seconds=2)
    assert paths and frames
    assert h.vstream(paths[0])["codec_name"] == "h264"


def test_portrait_native_1080x1920_input_is_downsampled(tmp_path):
    src = tmp_path / "portrait.mp4"
    h.make_h264(src, duration=2, size="1080x1920", fps=30)
    _, out, paths, _ = h.run_pre(tmp_path, src, fps=3, chunk_seconds=5)
    v = h.vstream(paths[0])
    assert v["height"] > v["width"]
    assert min(v["width"], v["height"]) <= 724
    assert max(v["width"], v["height"]) <= 1284
    assert abs(v["width"] / v["height"] - 9 / 16) < 0.02
