"""Shared helpers and fixtures for the task 5 (preprocessing) hidden tests.

Interface assumed (see the report to the main agent):

    backend/preprocess.py
        preprocess_video(input_path, output_dir, *, fps=<2..5>, chunk_seconds=<short>, height=720)
    returns an object (dataclass or dict) with:
        chunks                  list of items with .path (and optionally .index, .start_time, .end_time)
        frame_timestamps_path   path of the JSON timestamps record
    timestamps JSON: {"fps": f, "frames": [{"index": n, "chunk": c, "timestamp": seconds, ...}, ...]}
    default timestamps file: <output_dir>/frame_timestamps.json
"""
import hashlib
import importlib
import inspect
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark_ffmpeg = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not on PATH"
)

FUNC_NAMES = ["preprocess_video", "preprocess", "run_preprocessing", "preprocess_drive"]
FPS_NAMES = ["fps", "target_fps", "sample_fps", "frame_rate"]
CHUNK_NAMES = ["chunk_seconds", "chunk_duration", "chunk_duration_s", "chunk_length", "chunk_len", "chunk_s"]
HEIGHT_NAMES = ["height", "target_height", "max_height", "short_side"]

TS_KEYS = ["timestamp", "time", "t", "pts_time", "timestamp_s", "source_time", "global_timestamp"]
CHUNK_KEYS = ["chunk", "chunk_index", "chunk_id"]


# ---------------------------------------------------------------- subprocess helpers


def run(cmd, timeout=180):
    p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    return p


def ffmpeg_ok(args, timeout=180):
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y"] + [str(a) for a in args]
    p = run(cmd, timeout)
    return p.returncode == 0, p.stderr.decode("utf-8", "replace")


def ffprobe_json(path, *args):
    cmd = [FFPROBE, "-v", "error", "-of", "json"] + list(args) + [str(path)]
    p = run(cmd)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    return json.loads(p.stdout.decode("utf-8"))


def streams(path):
    return ffprobe_json(path, "-show_streams", "-show_format")["streams"]


def vstream(path):
    vs = [s for s in streams(path) if s.get("codec_type") == "video"]
    assert len(vs) == 1, "expected exactly one video stream in %s, got %d" % (path, len(vs))
    return vs[0]


def container_duration(path):
    d = ffprobe_json(path, "-show_format")["format"].get("duration")
    return float(d)


def frac(s):
    n, _, d = str(s).partition("/")
    return float(n) / float(d) if d else float(n)


def count_frames(path):
    j = ffprobe_json(
        path, "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames"
    )
    return int(j["streams"][0]["nb_read_frames"])


def packet_pts(path):
    j = ffprobe_json(path, "-select_streams", "v:0", "-show_entries", "packet=pts_time")
    return sorted(float(p["pts_time"]) for p in j["packets"] if "pts_time" in p)


def decode_errors(path):
    p = run([FFMPEG, "-v", "error", "-i", str(path), "-f", "null", "-"])
    return p.returncode, p.stderr.decode("utf-8", "replace").strip()


def frame_luma(path):
    """Mean luma (0..255) of every decoded frame, in order (1x1 area-scaled gray)."""
    cmd = [FFMPEG, "-v", "error", "-i", str(path), "-vf", "scale=1:1:flags=area,format=gray",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-"]
    p = run(cmd)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    return list(p.stdout)


def quadrant_rgb(path, autorotate=True):
    """12 bytes: mean RGB of the 4 quadrants (TL, TR, BL, BR) of the first frame."""
    cmd = [FFMPEG, "-v", "error"]
    if not autorotate:
        cmd += ["-noautorotate"]
    cmd += ["-i", str(path), "-frames:v", "1", "-vf", "scale=2:2:flags=area,format=rgb24",
            "-f", "rawvideo", "-"]
    p = run(cmd)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    assert len(p.stdout) == 12, "unexpected raw frame size %d" % len(p.stdout)
    return list(p.stdout)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- clip generation


def _gen(out, vf_src, duration, extra_in=None, enc=None, audio=False, vf=None, post=None):
    args = ["-f", "lavfi", "-i", vf_src]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100"]
    args += ["-t", str(duration)]
    if vf:
        args += ["-vf", vf]
    args += enc or ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-b:a", "64k", "-shortest"]
    if post:
        args += post
    args += [str(out)]
    ok, err = ffmpeg_ok(args)
    assert ok, err
    return out


def make_h264(out, duration=7, size="1920x1080", fps=30, audio=False):
    return _gen(out, "testsrc2=size=%s:rate=%d" % (size, fps), duration, audio=audio)


def make_hevc_mov(out, duration=4, size="1920x1080", fps=30, audio=True):
    enc = ["-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
           "-x265-params", "log-level=error"]
    return _gen(out, "testsrc2=size=%s:rate=%d" % (size, fps), duration, enc=enc, audio=audio)


RAMP_SRC = "color=c=black:s=64x36:r=30,geq=lum='clip(16+219*T/%s,16,235)':cb=128:cr=128"


def make_ramp(out, duration=10, vfr=False):
    """Brightness encodes source time: luma ~= 255 * t / duration. Optionally variable frame rate."""
    src = RAMP_SRC % duration
    vf = "scale=1920:1080:flags=neighbor,format=yuv420p"
    if vfr:
        vf = "select='lt(n,%d)*not(mod(n,3))+gte(n,%d)'," % (int(duration * 30 * 0.5), int(duration * 30 * 0.5)) + vf
    enc = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if vfr:
        enc = enc + ["-fps_mode", "vfr"]
    return _gen(out, src, duration, enc=enc, vf=vf)


QUAD_SRC = (
    "color=c=red:s=640x360:r=30[a];color=c=lime:s=640x360:r=30[b];"
    "color=c=blue:s=640x360:r=30[c];color=c=yellow:s=640x360:r=30[d];"
    "[a][b][c][d]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0"
)


def make_rotated(out, duration=3, rotation=90):
    """Landscape 1280x720 stored pixels (coloured quadrants) plus rotation metadata (phone portrait).

    Returns None if this ffmpeg cannot write rotation metadata.
    """
    base = Path(out).with_name("rot_base.mp4")
    ok, err = ffmpeg_ok(["-f", "lavfi", "-i", QUAD_SRC, "-t", str(duration),
                         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(base)])
    assert ok, err
    attempts = [
        ["-display_rotation", str(rotation), "-i", str(base), "-c", "copy", str(out)],
        ["-i", str(base), "-c", "copy", "-metadata:s:v:0", "rotate=%d" % rotation, str(out)],
    ]
    for a in attempts:
        if Path(out).exists():
            Path(out).unlink()
        ok, _ = ffmpeg_ok(a)
        if ok and has_rotation(out):
            return Path(out)
    return None


def has_rotation(path):
    v = [s for s in streams(path) if s.get("codec_type") == "video"][0]
    if str(v.get("tags", {}).get("rotate", "0")) not in ("0", ""):
        return True
    for sd in v.get("side_data_list", []) or []:
        if sd.get("rotation") not in (None, 0, "0"):
            return True
    return False


# ---------------------------------------------------------------- interface discovery


def load_module():
    return importlib.import_module("backend.preprocess")


def find_func():
    mod = load_module()
    for n in FUNC_NAMES:
        f = getattr(mod, n, None)
        if callable(f):
            return f
    pytest.fail("backend.preprocess has none of the expected functions %s" % FUNC_NAMES)


def _pick(sig, names):
    for n in names:
        if n in sig.parameters:
            return n
    return None


def call(input_path, output_dir, fps=None, chunk_seconds=None, height=None, **extra):
    """Call preprocess_video with logical kwargs mapped to whatever the signature calls them."""
    f = find_func()
    sig = inspect.signature(f)
    kw = dict(extra)
    for val, names, label in ((fps, FPS_NAMES, "fps"),
                              (chunk_seconds, CHUNK_NAMES, "chunk_seconds"),
                              (height, HEIGHT_NAMES, "height")):
        if val is None:
            continue
        n = _pick(sig, names)
        assert n, "preprocess function has no configurable '%s' parameter (expected one of %s)" % (label, names)
        kw[n] = val
    return f(input_path, output_dir, **kw)


def signature_default(names):
    sig = inspect.signature(find_func())
    n = _pick(sig, names)
    if n is None:
        return None
    d = sig.parameters[n].default
    return None if d is inspect.Parameter.empty else d


def get(obj, *names, default=None):
    for n in names:
        if isinstance(obj, dict):
            if n in obj:
                return obj[n]
        elif hasattr(obj, n):
            return getattr(obj, n)
    return default


def result_chunk_paths(result, output_dir):
    chunks = get(result, "chunks", "chunk_files", "chunk_paths")
    paths = []
    if chunks:
        for c in chunks:
            p = c if isinstance(c, (str, Path)) else get(c, "path", "file", "filename")
            assert p is not None, "chunk item has no 'path'"
            p = Path(p)
            if not p.is_absolute() and not p.exists():
                p = Path(output_dir) / p
            paths.append(p)
        return paths
    return sorted(Path(output_dir).rglob("*.mp4"))


def timestamps_path(result, output_dir):
    p = get(result, "frame_timestamps_path", "timestamps_path", "frame_timestamps_file", "timestamps_file")
    if p is not None and not isinstance(p, (list, dict)):
        p = Path(p)
        return p if p.is_absolute() or p.exists() else Path(output_dir) / p
    return Path(output_dir) / "frame_timestamps.json"


def load_frames(path):
    """Return list of (timestamp, chunk_or_None, raw_entry) from the JSON record."""
    assert Path(path).is_file(), "timestamps record not found at %s" % path
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        frames = None
        for k in ("frames", "timestamps", "frame_timestamps", "items"):
            if isinstance(data.get(k), list):
                frames = data[k]
                break
        assert frames is not None, "timestamps JSON object has no 'frames' list: keys=%s" % list(data)
    else:
        frames = data
    out = []
    for e in frames:
        if isinstance(e, (int, float)):
            out.append((float(e), None, e))
            continue
        t = None
        for k in TS_KEYS:
            if k in e:
                t = float(e[k])
                break
        assert t is not None, "frame entry has no timestamp key among %s: %s" % (TS_KEYS, e)
        c = None
        for k in CHUNK_KEYS:
            if k in e:
                c = int(e[k])
                break
        out.append((t, c, e))
    return out


def run_pre(tmp_path, clip, name="out", **kw):
    out = Path(tmp_path) / name
    result = call(clip, out, **kw)
    paths = result_chunk_paths(result, out)
    frames = load_frames(timestamps_path(result, out))
    return result, out, paths, frames


# ---------------------------------------------------------------- session fixtures (imported by tests)


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    if not (FFMPEG and FFPROBE):
        pytest.skip("ffmpeg/ffprobe not available")
    d = tmp_path_factory.mktemp("rs5clips")
    c = {}
    c["h264_audio"] = make_h264(d / "h264_audio.mp4", duration=7, audio=True)
    c["h264_silent"] = make_h264(d / "h264_silent.mp4", duration=7, audio=False)
    c["short"] = make_h264(d / "short.mp4", duration=1, size="1280x720", audio=True)
    c["small"] = make_h264(d / "small.mp4", duration=3, size="640x360", fps=24)
    try:
        c["hevc"] = make_hevc_mov(d / "hevc.mov", duration=4)
    except AssertionError:
        c["hevc"] = None
    c["ramp"] = make_ramp(d / "ramp.mp4", duration=10, vfr=False)
    c["vfr"] = make_ramp(d / "vfr.mp4", duration=10, vfr=True)
    c["rotated"] = make_rotated(d / "rotated.mp4", duration=3, rotation=90)
    c["dir"] = d
    return c


@pytest.fixture()
def clip(clips):
    def _get(name):
        v = clips[name]
        if v is None:
            pytest.skip("could not generate clip %r with this ffmpeg" % name)
        return Path(v)
    return _get


def ceil_div(a, b):
    return int(math.ceil(a / b - 1e-9))


def ramp_time_from_luma(ramp_clip):
    """Calibrate luma->source time from the constant-frame-rate ramp clip itself (same readout path)."""
    curve = frame_luma(ramp_clip)
    dur = len(curve) / 30.0

    def est(y):
        if y <= 3 or y >= 252:
            return None  # clipped region, no information
        best = None
        for i in range(len(curve) - 1):
            a, b = curve[i], curve[i + 1]
            if min(a, b) <= y <= max(a, b) and a != b:
                best = (i + (y - a) / (b - a)) / 30.0
                break
        return best
    return est
