"""Task 10: against a real uvicorn server (real concurrency): the endpoint returns promptly while the job runs."""
from __future__ import annotations

import socket
import threading
import time

import pytest

from rs10_helpers import (CHUNKS, DONE, FAILED, INITIAL, clip_bytes, err_of, gpx_bytes, rs10_env,  # noqa: F401
                          spy, stage_pair, status_of)


class Server:
    def __init__(self, app):
        import uvicorn

        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.time() + 30
        while not self.server.started and time.time() < deadline:
            time.sleep(0.05)
        assert self.server.started, "uvicorn did not start"

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=30)


def test_endpoint_returns_promptly_and_job_runs_in_background(rs10_env):
    pytest.importorskip("uvicorn")
    import httpx

    from backend.main import app

    e = rs10_env
    gate = threading.Event()
    started = threading.Event()
    seen = {}

    def on_detect(p, n, video_path):
        if n == 0:
            seen["detect_row"] = e.row(did)
            started.set()
            assert gate.wait(60), "test never released the gate"

    def on_verify(p, n):
        if n == 0:
            seen["verify_row"] = e.row(did)

    p = spy(on_detect=on_detect, on_verify=on_verify)
    e.use_provider(p)
    srv = Server(app)
    try:
        with httpx.Client(base_url="http://127.0.0.1:%d" % srv.port, timeout=30) as c:
            r = c.post("/api/drives", files={"video": ("clip.mp4", clip_bytes(), "video/mp4"),
                                             "gpx": ("track.gpx", gpx_bytes(), "application/gpx+xml")})
            assert r.status_code == 201, r.text
            did = r.json()["id"]
            path = e.process_route().replace("{drive_id}", did).replace("{id}", did)

            t = time.time()
            r1 = c.post(path)
            elapsed = time.time() - t
            assert r1.status_code in (200, 201, 202), r1.text
            assert elapsed < 8, "the endpoint must return while the job is still running (took %.1fs)" % elapsed

            assert started.wait(90), "the job never reached detection"
            d = c.get("/api/drives/%s" % did).json()
            st = status_of(d)
            assert st not in INITIAL and st not in DONE and st not in FAILED, d
            detect_pair = stage_pair(d)

            # starting it again while it runs: rejected (409) or accepted as a no-op; never an error
            r2 = c.post(path)
            assert r2.status_code in (200, 201, 202, 409), r2.text
            d = c.get("/api/drives/%s" % did).json()
            assert status_of(d) not in FAILED

            gate.set()
            deadline = time.time() + 120
            while time.time() < deadline:
                d = c.get("/api/drives/%s" % did).json()
                if status_of(d) in DONE | FAILED:
                    break
                time.sleep(0.1)
            assert status_of(d) in DONE, d
            assert err_of(d) == ""
            assert stage_pair(d) != detect_pair

            # exactly one run happened: 3 chunks detected once, 3 candidates verified once
            assert len(p.detect_calls) == CHUNKS, "a second POST while running must not start a second run"
            assert len(p.verify_calls) == CHUNKS
            assert e.counts(did)["verdicts"] == CHUNKS
            assert stage_pair(seen["detect_row"]) != stage_pair(seen["verify_row"])
    finally:
        gate.set()
        srv.stop()


def test_failed_job_over_real_server_leaves_server_responsive(rs10_env):
    pytest.importorskip("uvicorn")
    import httpx

    from backend.main import app

    e = rs10_env
    e.use_provider(spy(fail_detect_on=0))
    srv = Server(app)
    try:
        with httpx.Client(base_url="http://127.0.0.1:%d" % srv.port, timeout=30) as c:
            r = c.post("/api/drives", files={"video": ("clip.mp4", clip_bytes(), "video/mp4"),
                                             "gpx": ("track.gpx", gpx_bytes(), "application/gpx+xml")})
            did = r.json()["id"]
            path = e.process_route().replace("{drive_id}", did).replace("{id}", did)
            assert c.post(path).status_code in (200, 201, 202)
            deadline = time.time() + 120
            d = {}
            while time.time() < deadline:
                d = c.get("/api/drives/%s" % did).json()
                if status_of(d) in DONE | FAILED:
                    break
                time.sleep(0.1)
            assert status_of(d) in FAILED, d
            assert "boom-detect-xyz" in err_of(d)
            assert c.get("/health").status_code == 200
            assert c.get("/api/drives").status_code == 200
    finally:
        srv.stop()
