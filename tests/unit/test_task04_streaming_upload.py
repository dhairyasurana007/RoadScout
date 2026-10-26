"""A multi-hundred-MB upload must be streamed to disk, not buffered in memory.

TestClient builds the whole request body in memory, so it cannot measure this. These
tests serve the real app with uvicorn on a loopback port inside this process, upload a
large (padded, still valid) synthetic mp4 with a streaming httpx request, and sample
this process's resident memory while the upload runs.
"""
import ctypes
import gc
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from rs4_fixtures import *  # noqa: F401,F403
from rs4_helpers import (
    CLIP_KINDS,
    ok_status,
    make_gpx,
    pad_mp4_with_free_box,
    sha_file,
    storage_files,
    unwrap_drive,
    drive_id,
)

BIG_EXTRA_BYTES = 320 * 1024 * 1024
MAX_GROWTH_BYTES = 160 * 1024 * 1024  # well under the file size; a read() of the whole file blows it


def _rss_now() -> int | None:
    if sys.platform == "win32":
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return None
        return int(pmc.WorkingSetSize)
    statm = Path("/proc/self/statm")
    if statm.exists():
        return int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return None


class RssSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.peak = 0

    def run(self):
        while not self.stop.is_set():
            v = _rss_now() or 0
            self.peak = max(self.peak, v)
            time.sleep(0.005)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def live_server(rs4_app, rs4_env):
    uvicorn = pytest.importorskip("uvicorn")
    port = _free_port()
    cfg = uvicorn.Config(rs4_app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(timeout=30)


def test_large_upload_is_streamed_not_buffered(live_server, rs4_api, rs4_clip, rs4_env, tmp_path):
    httpx = pytest.importorskip("httpx")
    if _rss_now() is None:
        pytest.skip("cannot measure process memory on this platform")

    small = tmp_path / "small.mp4"
    small.write_bytes(rs4_clip("mp4-h264"))
    big = tmp_path / "big.mp4"
    size = pad_mp4_with_free_box(small, big, BIG_EXTRA_BYTES)
    assert size > 300 * 1024 * 1024
    gpx = make_gpx()
    expected = sha_file(big)

    gc.collect()
    baseline = _rss_now()
    sampler = RssSampler()
    sampler.start()
    try:
        with open(big, "rb") as fh:
            r = httpx.post(
                live_server + rs4_api.path,
                files={rs4_api.video_field: ("big.mp4", fh, "video/mp4"),
                       rs4_api.gpx_field: ("t.gpx", gpx, "application/gpx+xml")},
                timeout=httpx.Timeout(600.0),
            )
    finally:
        sampler.stop.set()
        sampler.join(timeout=5)
    assert ok_status(r.status_code), r.text[:500]
    assert drive_id(unwrap_drive(r.json()))

    growth = sampler.peak - baseline
    assert growth < MAX_GROWTH_BYTES, (
        f"process memory grew by {growth / 2**20:.0f} MiB while uploading a "
        f"{size / 2**20:.0f} MiB file; the upload is being buffered in memory")

    stored = [p for p in storage_files(rs4_env) if p.stat().st_size == size]
    assert stored, "the full-size video was not stored intact"
    assert sha_file(stored[0]) == expected


def test_live_server_small_upload_end_to_end(live_server, rs4_api, rs4_clip):
    """Same route over real HTTP (a real socket, real multipart streaming)."""
    httpx = pytest.importorskip("httpx")
    r = httpx.post(
        live_server + rs4_api.path,
        files={rs4_api.video_field: ("clip.mp4", rs4_clip("mp4-h264"), "video/mp4"),
               rs4_api.gpx_field: ("t.gpx", make_gpx(), "application/gpx+xml")},
        data={rs4_api.offset_field: "1.5"} if (rs4_api.offset_field and not rs4_api.offset_in_query) else None,
        params={rs4_api.offset_field: "1.5"} if (rs4_api.offset_field and rs4_api.offset_in_query) else None,
        timeout=60,
    )
    assert ok_status(r.status_code), r.text
    d = unwrap_drive(r.json())
    got = httpx.get(live_server + f"{rs4_api.base}/{drive_id(d)}", timeout=30)
    assert got.status_code == 200


def test_rejected_large_garbage_upload_leaves_no_files(live_server, rs4_api, rs4_env, tmp_path):
    """A big non-video body is refused (4xx) and nothing is left in storage."""
    httpx = pytest.importorskip("httpx")
    junk = tmp_path / "junk.mp4"
    with open(junk, "wb") as fh:
        for _ in range(32):
            fh.write(b"not a video at all " * 55_000)  # ~1 MB per write
    with open(junk, "rb") as fh:
        r = httpx.post(
            live_server + rs4_api.path,
            files={rs4_api.video_field: ("junk.mp4", fh, "video/mp4"),
                   rs4_api.gpx_field: ("t.gpx", make_gpx(), "application/gpx+xml")},
            timeout=httpx.Timeout(300.0),
        )
    assert 400 <= r.status_code < 500, r.status_code
    assert storage_files(rs4_env) == []
    assert CLIP_KINDS  # keep the import used
