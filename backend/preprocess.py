"""Preprocessing (task 5): normalize, strip audio, downsample, sample, chunk, timestamp.

    result = preprocess_video("drive.mov", "data/work/drive-1")

One ffmpeg pass turns any input (HEVC or H.264, variable frame rate, rotation
metadata, audio, extra streams) into short H.264 mp4 chunks with:

- exactly one video stream, no audio, subtitle or data streams, and no container
  metadata (so no embedded GPS tags or creation time either)
- the rotation metadata applied, so the picture is upright
- a constant frame rate of `fps` frames per second (default 3)
- a height of at most `height` pixels (default 720); never upscaled, aspect
  ratio kept, width and height both even
- `chunk_frames` frames per chunk (the last chunk may be shorter), each chunk a
  standalone file whose own timestamps start at 0

Output layout (everything stays inside `output_dir`; the input is never modified):

    chunk_0000.mp4, chunk_0001.mp4, ...   in time order
    frame_timestamps.json                 maps every sampled frame to source time

frame_timestamps.json (version 1):

    {
      "version": 1,
      "source_name": "drive.mov",          # file name only, no directories
      "fps": 3.0,                          # sampled frames per second
      "width": 1280, "height": 720,        # of the chunk files
      "chunk_seconds": 10.0,               # nominal chunk length (chunk_frames / fps)
      "chunk_frames": 30,                  # frames per chunk, except possibly the last
      "frame_count": 905,                  # total frames over all chunks
      "duration_seconds": 301.666667,      # frame_count / fps
      "chunks": [
        {"index": 0, "file": "chunk_0000.mp4",
         "start_frame": 0,                 # global index of its first frame
         "frame_count": 30,                # decoded frames in the file (checked with ffprobe)
         "start_time": 0.0,                # timestamp of its first frame
         "end_time": 10.0,                 # exclusive: last frame timestamp + 1/fps
         "duration_seconds": 10.0},
        ...
      ],
      "frames": [
        {"index": 0,                       # global frame index, 0-based, no gaps
         "chunk": 0,                       # chunk index
         "frame_in_chunk": 0,              # position inside that chunk file
         "timestamp": 0.0,                 # seconds from the source video start
         "chunk_timestamp": 0.0},          # seconds from the start of its chunk file
        ...
      ]
    }

Timestamps are seconds from the first video frame of the source (the video start
that GPS lookup uses). The frame at global index i has timestamp i / fps, rounded
to 6 decimals; it is the source frame nearest that instant. A frame's time inside
its chunk file is chunk_timestamp (= timestamp - chunk start_time).

Python result: PreprocessResult with `chunks` (list of Chunk: index, path, start_frame,
frame_count, start_time, end_time), `frame_timestamps_path`, fps, width, height,
duration_seconds, frame_count, chunk_seconds, chunk_frames, output_dir, plus the
convenience properties `chunk_paths`, `timestamps_path` and `chunk_count`.

Errors: InvalidMedia (a ValueError, from backend.media) for a missing, unreadable
or video-less input; FfmpegUnavailable (a ProbeUnavailable, itself a RuntimeError)
when ffmpeg or ffprobe is not installed; PreprocessError (a RuntimeError) when
ffmpeg fails. Invalid fps, height or chunk_seconds raise ValueError.

Video is streamed through ffmpeg and never loaded into Python memory. Commands are
argv lists, never shell strings.
"""
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from . import config
from .media import InvalidMedia, ProbeUnavailable, probe_video

TIMESTAMPS_FILENAME = "frame_timestamps.json"
CHUNK_NAME_PATTERN = "chunk_%04d.mp4"  # ffmpeg segment pattern
_CHUNK_NAME_RE = re.compile(r"^chunk_\d{4,}\.mp4$")
FORMAT_VERSION = 1

_CRF = "23"
_X264_PRESET = "veryfast"


class FfmpegUnavailable(ProbeUnavailable):
    """ffmpeg or ffprobe is not installed or could not run."""


class PreprocessError(RuntimeError):
    """ffmpeg failed or produced output that does not add up."""


@dataclass
class Chunk:
    index: int
    path: Path
    start_frame: int  # global index of the chunk's first frame
    frame_count: int  # decoded frames in the file
    start_time: float  # seconds from source start, timestamp of the first frame
    end_time: float  # exclusive: last frame timestamp + 1/fps


@dataclass
class PreprocessResult:
    chunks: list[Chunk]  # in time order, all inside output_dir
    frame_timestamps_path: Path  # output_dir / frame_timestamps.json
    fps: float
    width: int
    height: int
    duration_seconds: float  # frame_count / fps
    frame_count: int
    chunk_seconds: float  # nominal chunk length, chunk_frames / fps
    chunk_frames: int
    output_dir: Path = field(default=Path("."))

    @property
    def chunk_paths(self) -> list[Path]:
        return [c.path for c in self.chunks]

    @property
    def timestamps_path(self) -> Path:
        return self.frame_timestamps_path

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(argv, capture_output=True, text=True, errors="replace")
    except FileNotFoundError as exc:
        raise FfmpegUnavailable(f"{argv[0]} is not installed or not on PATH") from exc


def _check_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise FfmpegUnavailable(f"{tool} is not installed or not on PATH")


def _count_chunk_frames(path: Path) -> tuple[int, int, int]:
    """Decode-count one chunk with ffprobe. Returns (frames, width, height)."""
    proc = _run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,width,height", "-of", "json", str(path),
    ])
    try:
        stream = json.loads(proc.stdout)["streams"][0]
        return int(stream["nb_read_frames"]), int(stream["width"]), int(stream["height"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise PreprocessError(f"Could not read back chunk {path.name}") from exc


def _clean_previous_output(output_dir: Path, input_path: Path) -> None:
    """Remove chunk files and the timestamps file from an earlier run (never the input)."""
    candidates = [p for p in output_dir.iterdir() if p.is_file() and _CHUNK_NAME_RE.match(p.name)]
    candidates.append(output_dir / TIMESTAMPS_FILENAME)
    for path in candidates:
        if path.resolve() == input_path:
            raise InvalidMedia(f"Input file {path.name} would be overwritten by preprocessing output; "
                               "use a different output directory")
        path.unlink(missing_ok=True)


def _ffmpeg_args(src: Path, stream_index: int, output_dir: Path, fps: Fraction, max_height: int,
                 chunk_frames: int) -> list[str]:
    max_h = max_height - (max_height % 2)
    # Rotation is applied by ffmpeg's automatic rotation before these filters run, so
    # ih is the upright height. -2 keeps the aspect ratio with an even width.
    vf = (
        "setpts=PTS-STARTPTS,"
        f"fps={fps.numerator}/{fps.denominator},"
        f"scale=-2:'trunc(min({max_h},ih)/2)*2',"
        "setsar=1,format=yuv420p"
    )
    segment_seconds = chunk_frames / fps
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-fflags", "+genpts",
        "-i", str(src),
        "-map", f"0:{stream_index}",
        "-an", "-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1",
        "-vf", vf,
        "-c:v", "libx264", "-preset", _X264_PRESET, "-crf", _CRF,
        "-bf", "0", "-g", str(chunk_frames), "-keyint_min", str(chunk_frames), "-sc_threshold", "0",
        "-f", "segment",
        "-segment_time", f"{float(segment_seconds):.9f}",
        # Cut at the keyframe that lands exactly on a chunk boundary, not an earlier one.
        "-segment_time_delta", f"{float(Fraction(1, 4) / fps):.9f}",
        "-segment_format", "mp4",
        "-segment_format_options", "movflags=+faststart",
        "-reset_timestamps", "1",
        str(output_dir / CHUNK_NAME_PATTERN),
    ]


def preprocess_video(
    input_path,
    output_dir,
    *,
    fps: float | None = None,
    chunk_seconds: float | None = None,
    height: int | None = None,
    max_height: int | None = None,
) -> PreprocessResult:
    """Normalize `input_path` into chunks under `output_dir`. See the module docstring.

    `height` is the maximum output height (`max_height` is an alias). Defaults (None) come
    from config: ROADSCOUT_PREPROCESS_FPS (3), ROADSCOUT_CHUNK_SECONDS (20) and
    ROADSCOUT_PREPROCESS_MAX_HEIGHT (720).
    """
    if height is not None and max_height is not None and height != max_height:
        raise ValueError("pass only one of height and max_height")
    max_height = height if height is not None else max_height
    fps = config.preprocess_fps() if fps is None else fps
    max_height = config.preprocess_max_height() if max_height is None else max_height
    chunk_seconds = config.chunk_seconds() if chunk_seconds is None else chunk_seconds
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not 0 < fps <= 120:
        raise ValueError(f"fps must be a number between 0 and 120, got {fps!r}")
    if isinstance(max_height, bool) or not isinstance(max_height, int) or max_height < 2:
        raise ValueError(f"max_height must be an integer of at least 2, got {max_height!r}")
    if isinstance(chunk_seconds, bool) or not isinstance(chunk_seconds, (int, float)) or not chunk_seconds > 0 \
            or chunk_seconds == float("inf"):
        raise ValueError(f"chunk_seconds must be a positive number, got {chunk_seconds!r}")

    _check_tools()

    src = Path(input_path)
    if not src.is_file():
        raise InvalidMedia(f"Input video not found: {src.name}")
    src = src.resolve()
    info = probe_video(src)  # InvalidMedia if there is no decodable video stream
    if info.stream_index is None:
        raise InvalidMedia("File has no video stream")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out = out.resolve()
    _clean_previous_output(out, src)

    fps_exact = Fraction(fps).limit_denominator(1000)
    chunk_frames = max(1, round(float(chunk_seconds) * float(fps_exact)))
    args = _ffmpeg_args(src, info.stream_index, out, fps_exact, max_height, chunk_frames)
    try:
        proc = _run(args)
        if proc.returncode != 0:
            tail = " | ".join((proc.stderr or "").strip().splitlines()[-3:])
            raise PreprocessError(f"ffmpeg failed (exit {proc.returncode}): {tail or 'no error output'}")
        chunk_paths = sorted(p for p in out.iterdir() if p.is_file() and _CHUNK_NAME_RE.match(p.name))
        if not chunk_paths:
            raise PreprocessError("ffmpeg produced no frames (is the video empty?)")

        chunks, frames = [], []
        width = height = 0
        start_frame = 0
        for index, path in enumerate(chunk_paths):
            count, width, height = _count_chunk_frames(path)
            if count < 1:
                raise PreprocessError(f"Chunk {path.name} has no frames")
            start_time = start_frame / fps_exact
            end_time = (start_frame + count) / fps_exact
            chunks.append(Chunk(index, path, start_frame, count,
                                round(float(start_time), 6), round(float(end_time), 6)))
            for k in range(count):
                frames.append({
                    "index": start_frame + k,
                    "chunk": index,
                    "frame_in_chunk": k,
                    "timestamp": round(float((start_frame + k) / fps_exact), 6),
                    "chunk_timestamp": round(float(Fraction(k) / fps_exact), 6),
                })
            start_frame += count
    except BaseException:
        for path in out.iterdir():
            if path.is_file() and _CHUNK_NAME_RE.match(path.name):
                path.unlink(missing_ok=True)
        raise

    fps_float = float(fps_exact)
    duration = round(float(Fraction(start_frame) / fps_exact), 6)
    nominal = round(float(chunk_frames / fps_exact), 6)
    document = {
        "version": FORMAT_VERSION,
        "source_name": src.name,
        "fps": fps_float,
        "width": width,
        "height": height,
        "chunk_seconds": nominal,
        "chunk_frames": chunk_frames,
        "frame_count": start_frame,
        "duration_seconds": duration,
        "chunks": [
            {
                "index": c.index,
                "file": c.path.name,
                "start_frame": c.start_frame,
                "frame_count": c.frame_count,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "duration_seconds": round(c.end_time - c.start_time, 6),
            }
            for c in chunks
        ],
        "frames": frames,
    }
    timestamps_path = out / TIMESTAMPS_FILENAME
    tmp = timestamps_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(document, indent=1), encoding="utf-8")
    tmp.replace(timestamps_path)

    return PreprocessResult(
        chunks=chunks,
        frame_timestamps_path=timestamps_path,
        fps=fps_float,
        width=width,
        height=height,
        duration_seconds=duration,
        frame_count=start_frame,
        chunk_seconds=nominal,
        chunk_frames=chunk_frames,
        output_dir=out,
    )


def load_preprocess_result(output_dir) -> PreprocessResult | None:
    """Rebuild the ``PreprocessResult`` of an earlier ``preprocess_video`` run from its output folder
    (task 26c, used to resume a drive), or None when the folder is not complete: no readable
    ``frame_timestamps.json``, or a chunk file it lists is missing. Runs no ffmpeg."""
    out = Path(output_dir)
    ts_path = out / TIMESTAMPS_FILENAME
    try:
        doc = json.loads(ts_path.read_text(encoding="utf-8"))
        chunks = []
        for c in doc["chunks"]:
            path = out / str(c["file"])
            if not path.is_file():
                return None
            chunks.append(Chunk(int(c["index"]), path.resolve(), int(c["start_frame"]), int(c["frame_count"]),
                                float(c["start_time"]), float(c["end_time"])))
        if not chunks:
            return None
        return PreprocessResult(
            chunks=chunks,
            frame_timestamps_path=ts_path.resolve(),
            fps=float(doc["fps"]),
            width=int(doc["width"]),
            height=int(doc["height"]),
            duration_seconds=float(doc["duration_seconds"]),
            frame_count=int(doc["frame_count"]),
            chunk_seconds=float(doc["chunk_seconds"]),
            chunk_frames=int(doc["chunk_frames"]),
            output_dir=out.resolve(),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
