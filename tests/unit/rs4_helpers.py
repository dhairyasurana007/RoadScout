"""Shared helpers for the task 04 (drive upload + drives table) tests.

The task text fixes very few names, so everything here discovers the layout
from the running app (OpenAPI schema, PRAGMA table_info, module scanning)
instead of hard-coding it. The preferred names are listed in the tester's
report to the main agent.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ------------------------------------------------------------------ repo root


def find_repo_root() -> Path:
    override = os.environ.get("ROADSCOUT_REPO")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "CLAUDE.md").exists() and (parent / "docs").exists():
            return parent
    for parent in here.parents:
        sibling = parent / "RoadScout"
        if (sibling / "CLAUDE.md").exists():
            return sibling
    raise RuntimeError("cannot locate the RoadScout repo root; set ROADSCOUT_REPO")


REPO = find_repo_root()

# ---------------------------------------------------------------------- env

STORAGE_ENV_VARS = (
    "ROADSCOUT_STORAGE_DIR",
    "ROADSCOUT_STORAGE_PATH",
    "ROADSCOUT_STORAGE_ROOT",
    "ROADSCOUT_UPLOAD_DIR",
    "ROADSCOUT_UPLOADS_DIR",
    "ROADSCOUT_FILES_DIR",
    "ROADSCOUT_DATA_DIR",
    "STORAGE_DIR",
)


@dataclass
class Env:
    db: Path
    storage: Path
    extra: dict = field(default_factory=dict)


def apply_env(monkeypatch, tmp_path: Path) -> Env:
    db = tmp_path / "db" / "roadscout.sqlite"
    storage = tmp_path / "storage"
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("ROADSCOUT_DB", str(db))
    for var in STORAGE_ENV_VARS:
        monkeypatch.setenv(var, str(storage))
    monkeypatch.setenv("ROADSCOUT_PROVIDER", "fake")
    return Env(db=db, storage=storage)


def load_app():
    sys.path.insert(0, str(REPO))
    try:
        mod = importlib.import_module("backend.main")
    finally:
        pass
    from fastapi import FastAPI

    for attr in ("app", "application", "api"):
        obj = getattr(mod, attr, None)
        if isinstance(obj, FastAPI):
            return obj
    for obj in vars(mod).values():
        if isinstance(obj, FastAPI):
            return obj
    raise AssertionError("backend.main exposes no FastAPI app")


# ----------------------------------------------------------------- ffmpeg


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


_ENCODERS: str | None = None


def encoders() -> str:
    global _ENCODERS
    if _ENCODERS is None:
        try:
            _ENCODERS = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=60,
            ).stdout
        except Exception:  # noqa: BLE001
            _ENCODERS = ""
    return _ENCODERS


def has_encoder(name: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", encoders()) is not None


# kind -> (extension, video args, audio args, extra muxer args, encoder to require)
CLIP_KINDS = {
    "mp4-h264": ("mp4", ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"], [], [], "libx264"),
    "mp4-h264-aac": ("mp4", ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"], ["-c:a", "aac"], [], "libx264"),
    "mov-hevc": ("mov", ["-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                          "-tag:v", "hvc1", "-x265-params", "log-level=error"], [], [], "libx265"),
    "mkv-h264": ("mkv", ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"], [], [], "libx264"),
    "avi-mpeg4": ("avi", ["-c:v", "mpeg4"], [], [], "mpeg4"),
    "ts-h264": ("ts", ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"], [], ["-f", "mpegts"], "libx264"),
    "webm-vp8": ("webm", ["-c:v", "libvpx", "-b:v", "200k"], [], [], "libvpx"),
}

COLORS = ("red", "green", "blue", "yellow", "purple", "orange")

DEFAULT_CREATION_TIME = "2026-03-14T12:34:56Z"


def make_clip(dest_dir: Path, kind: str, color: str = "red",
              creation_time: str | None = DEFAULT_CREATION_TIME) -> bytes:
    """Generate a ~1 second synthetic clip and return its bytes (skips if unavailable)."""
    if not (have("ffmpeg") and have("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not installed")
    ext, vargs, aargs, margs, need = CLIP_KINDS[kind]
    if not has_encoder(need):
        pytest.skip(f"local ffmpeg has no {need} encoder")
    dest_dir.mkdir(parents=True, exist_ok=True)
    tag = creation_time or "none"
    out = dest_dir / f"{kind}-{color}-{re.sub(r'[^0-9A-Za-z]', '', tag)}.{ext}"
    if not out.exists():
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", f"color=c={color}:s=320x240:r=10:d=1"]
        if aargs:
            cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1"]
        cmd += vargs + aargs
        if creation_time:
            cmd += ["-metadata", f"creation_time={creation_time}"]
        cmd += margs + [str(out)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if res.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            out.unlink(missing_ok=True)
            pytest.skip(f"ffmpeg could not build a {kind} clip: {res.stderr[-300:]}")
    return out.read_bytes()


def make_gpx(version: str = "1.1", start: str = "2026-03-14T12:34:56Z", points: int = 6) -> bytes:
    """A tiny synthetic GPX track (made-up coordinates, not a real drive)."""
    base = datetime.fromisoformat(start.replace("Z", "+00:00"))
    ns = {
        "1.1": "http://www.topografix.com/GPX/1/1",
        "1.0": "http://www.topografix.com/GPX/1/0",
    }[version]
    pts = []
    for i in range(points):
        t = datetime.fromtimestamp(base.timestamp() + i, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        pts.append(
            f'<trkpt lat="{38.8000 + i * 0.0001:.6f}" lon="{-77.4000 - i * 0.0001:.6f}">'
            f"<ele>100.0</ele><time>{t}</time></trkpt>"
        )
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<gpx version="{version}" creator="roadscout-tests" xmlns="{ns}">'
        "<trk><name>synthetic</name><trkseg>" + "".join(pts) + "</trkseg></trk></gpx>\n"
    )
    return body.encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pad_mp4_with_free_box(src: Path, dst: Path, extra_bytes: int) -> int:
    """Copy a valid mp4 and append a `free` box of extra_bytes (still a valid mp4)."""
    import struct

    shutil.copyfile(src, dst)
    total = 16 + extra_bytes
    with open(dst, "ab") as fh:
        # 64-bit size form: size==1, type, largesize(8 bytes)
        fh.write(struct.pack(">I4sQ", 1, b"free", total))
        remaining = extra_bytes
        chunk = b"\0" * (1 << 20)
        while remaining > 0:
            n = min(remaining, len(chunk))
            fh.write(chunk[:n])
            remaining -= n
    return dst.stat().st_size


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------- time / JSON utils


def parse_ts(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1e11:
            v /= 1000.0
        return datetime.fromtimestamp(v, tz=timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def secs(a: datetime, b: datetime) -> float:
    return (a - b).total_seconds()


def find_key(d: dict, *patterns: str):
    for p in patterns:
        for k in d:
            if re.search(p, k, re.I):
                return k
    return None


def unwrap_drive(body) -> dict:
    assert isinstance(body, dict), f"expected a JSON object, got {type(body)}: {body!r}"
    inner = body.get("drive")
    return inner if isinstance(inner, dict) else body


def drive_id(d: dict):
    key = "id" if "id" in d else find_key(d, r"^drive_id$")
    assert key is not None and d[key] not in (None, ""), f"no id in {d}"
    return d[key]


def started_key(d: dict):
    return find_key(d, r"^started_at$", r"^(video_)?start(ed)?(_at|_time)?$")


def offset_key(d: dict):
    return find_key(d, r"^clock_offset", r"offset")


def effective_key(d: dict):
    return find_key(d, r"effective|aligned|adjusted|corrected|synced|gpx_start")


# ------------------------------------------------------------- upload API


@dataclass
class UploadApi:
    path: str
    video_field: str
    gpx_field: str
    offset_field: str | None
    offset_in_query: bool
    base: str = ""  # collection path for GET list / GET one


def _resolve_schema(spec: dict, schema: dict) -> dict:
    seen = 0
    while isinstance(schema, dict) and "$ref" in schema and seen < 10:
        name = schema["$ref"].split("/")[-1]
        schema = spec.get("components", {}).get("schemas", {}).get(name, {})
        seen += 1
    return schema


def discover_upload(app) -> UploadApi:
    spec = app.openapi()
    found = []
    for path, item in spec.get("paths", {}).items():
        op = item.get("post")
        if not op:
            continue
        content = (op.get("requestBody") or {}).get("content", {})
        mp = content.get("multipart/form-data")
        if mp is None:
            continue
        schema = _resolve_schema(spec, mp.get("schema", {}))
        props = list((schema.get("properties") or {}).keys())
        video = next((p for p in props if re.search(r"video|clip|movie", p, re.I)), None)
        gpx = next((p for p in props if re.search(r"gpx|track", p, re.I)), None)
        if not (video and gpx):
            continue
        offset = next((p for p in props if re.search(r"offset", p, re.I)), None)
        in_query = False
        if offset is None:
            for prm in op.get("parameters", []) or []:
                if prm.get("in") == "query" and re.search(r"offset", prm.get("name", ""), re.I):
                    offset, in_query = prm["name"], True
        found.append((path, video, gpx, offset, in_query))
    assert found, (
        "no POST route with a multipart/form-data body having a video field and a GPX field "
        f"in the OpenAPI schema; POST paths: "
        f"{[p for p, i in spec.get('paths', {}).items() if 'post' in i]}"
    )
    found.sort(key=lambda t: ("drive" not in t[0], len(t[0])))
    path, video, gpx, offset, in_query = found[0]
    return UploadApi(path=path, video_field=video, gpx_field=gpx, offset_field=offset,
                     offset_in_query=in_query)


def discover_base(client, api: UploadApi) -> str:
    candidates = [api.path.rstrip("/")]
    stripped = re.sub(r"/(upload|uploads|create|new)$", "", api.path.rstrip("/"))
    candidates += [stripped, "/drives", "/api/drives"]
    seen = []
    for c in candidates:
        if c in seen or not c:
            continue
        seen.append(c)
        r = client.get(c)
        if r.status_code == 200:
            body = r.json()
            if isinstance(body, list) or (isinstance(body, dict) and any(
                    isinstance(v, list) for v in body.values())):
                return c
    raise AssertionError(f"no GET list route for drives among {seen}")


def upload(client, api: UploadApi, video=None, gpx=None, offset=None, extra_data=None):
    """video/gpx are (filename, bytes, content_type) tuples or None to omit the part."""
    files = {}
    if video is not None:
        files[api.video_field] = video
    if gpx is not None:
        files[api.gpx_field] = gpx
    data = dict(extra_data or {})
    params = {}
    if offset is not None:
        if api.offset_field is None:
            raise AssertionError("the upload route has no clock offset field")
        (params if api.offset_in_query else data)[api.offset_field] = str(offset)
    return client.post(api.path, files=files or None, data=data or None, params=params or None)


def list_drives(client, api: UploadApi) -> list:
    r = client.get(api.base)
    assert r.status_code == 200, r.text
    body = r.json()
    if isinstance(body, list):
        return body
    for key in ("drives", "items", "results", "data"):
        if isinstance(body.get(key), list):
            return body[key]
    for v in body.values():
        if isinstance(v, list):
            return v
    raise AssertionError(f"cannot find the drive list in {body!r}")


def ok_status(code: int) -> bool:
    return code in (200, 201, 202)


# ----------------------------------------------------------------- DB / disk


def db_rows(env: Env) -> list[dict]:
    if not env.db.exists():
        return []
    con = sqlite3.connect(env.db)
    con.row_factory = sqlite3.Row
    try:
        try:
            return [dict(r) for r in con.execute("SELECT * FROM drives")]
        except sqlite3.OperationalError:
            return []
    finally:
        con.close()


def table_columns(path: Path, table: str = "drives") -> list[sqlite3.Row]:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        return list(con.execute(f"PRAGMA table_info({table})"))
    finally:
        con.close()


def storage_files(env: Env) -> list[Path]:
    if not env.storage.exists():
        return []
    return [p for p in env.storage.rglob("*") if p.is_file()]


def storage_hashes(env: Env) -> dict[str, Path]:
    return {sha_file(p): p for p in storage_files(env)}


def col(cols, pattern: str):
    for c in cols:
        if re.search(pattern, c, re.I):
            return c
    return None


def resolve_stored(env: Env, value: str) -> Path | None:
    """Map a stored path/key (as written in the DB) to a real file under the storage dir."""
    if not value:
        return None
    value = re.sub(r"^file://", "", str(value))
    p = Path(value)
    if p.is_absolute() and p.is_file():
        return p
    cand = env.storage / value
    if cand.is_file():
        return cand
    norm = value.replace("\\", "/").lstrip("./")
    for f in storage_files(env):
        rel = f.relative_to(env.storage).as_posix()
        if rel.endswith(norm) or norm.endswith(rel):
            return f
    cwd_rel = (REPO / value)
    if cwd_rel.is_file():
        return cwd_rel
    return None


# ---------------------------------------------------------- call utilities


def call(fn, *args, **kwargs):
    res = fn(*args, **kwargs)
    if inspect.isawaitable(res):
        res = asyncio.run(res)
    return res


def read_all(obj) -> bytes:
    if obj is None:
        raise AssertionError("storage open() returned None")
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if hasattr(obj, "read"):
        try:
            data = obj.read()
            if inspect.isawaitable(data):
                data = asyncio.run(data)
            return data
        finally:
            close = getattr(obj, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
    if isinstance(obj, (str, os.PathLike)) and Path(obj).is_file():
        return Path(obj).read_bytes()
    if hasattr(obj, "__enter__"):
        with obj as fh:
            return read_all(fh)
    if hasattr(obj, "__iter__"):
        return b"".join(obj)
    raise AssertionError(f"cannot read content from {type(obj)}")
