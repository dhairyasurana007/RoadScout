"""Task 26 hidden tests: shared fixtures (imported by each task 26 test module; not a conftest, so it
cannot affect other tests).

Safety: no test here may reach the real Meta API.  Every provider gets an
httpx.Client on a MockTransport, a dummy key is used, any real META_API_KEY is
removed from the environment, and outbound sockets are blocked in-process.
"""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

_here = Path(__file__).resolve()
_REPO = None
for cand in (_here.parents[2], Path(os.environ.get("ROADSCOUT_REPO", r"C:\Users\dhair\RoadScout"))):
    if (cand / "backend").is_dir():
        _REPO = cand
        break
assert _REPO is not None, "cannot find the RoadScout repo (set ROADSCOUT_REPO)"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_UNIT = Path(__file__).resolve().parent
if str(_UNIT) not in sys.path:
    sys.path.insert(0, str(_UNIT))

import rs26_helpers as H  # noqa: E402

H.REPO = _REPO


class Recorder:
    """Shared ordered log of HTTP requests and sleeps."""

    def __init__(self):
        self.events = []

    @property
    def sleeps(self):
        return [e[1] for e in self.events if e[0] == "sleep"]


@pytest.fixture(autouse=True)
def _safe_env(monkeypatch):
    for name in (
        "META_API_KEY", "ROADSCOUT_META_BASE_URL", "ROADSCOUT_SAM_MODEL", "ROADSCOUT_VERIFY_MODEL",
        "ROADSCOUT_COST_PER_DETECT_USD", "ROADSCOUT_COST_PER_VERIFY_USD", "ROADSCOUT_PROVIDER",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROADSCOUT_COST_LOG", "off")

    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def _is_loopback(addr):
        try:
            host = addr[0]
        except Exception:  # noqa: BLE001
            return False
        return host in ("127.0.0.1", "::1", "localhost") or str(host).startswith("127.")

    def no_connect(self, addr, *a, **k):
        if _is_loopback(addr):
            return real_connect(self, addr, *a, **k)
        raise AssertionError("a test tried to open a real network connection")

    def no_connect_ex(self, addr, *a, **k):
        if _is_loopback(addr):
            return real_connect_ex(self, addr, *a, **k)
        raise AssertionError("a test tried to open a real network connection")

    monkeypatch.setattr(socket.socket, "connect", no_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", no_connect_ex)


@pytest.fixture()
def rec(monkeypatch):
    """Never really sleep. ``rec.sleeps`` / ("sleep", s) events are only the provider's own retry waits."""
    r = Recorder()

    def fake_sleep(seconds=0.0, *a, **k):
        # Every time.sleep call in the process lands here (e.g. subprocess/ffprobe polling with tiny, doubling
        # waits on a slow machine). Tagged "global_sleep" so it is never mistaken for a retry wait: retry waits
        # are only the ("sleep", s) events written by the provider's own sleep= argument (see rs26_helpers).
        r.events.append(("global_sleep", float(seconds)))

    monkeypatch.setattr(time, "sleep", fake_sleep)
    try:
        from backend.providers import retry

        kd = getattr(retry.call_with_retries, "__kwdefaults__", None) or {}
        if "sleep" in kd:
            monkeypatch.setitem(kd, "sleep", fake_sleep)
        for mod_name in ("backend.providers.live", "backend.providers.retry"):
            mod = sys.modules.get(mod_name)
            if mod is not None and hasattr(mod, "sleep"):
                monkeypatch.setattr(mod, "sleep", fake_sleep)
    except Exception:  # noqa: BLE001
        pass
    return r


@pytest.fixture()
def api(rec):
    return H.FakeMeta(rec.events)


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("task26_clips")
    out = {}

    def make(name, rate, secs, size="64x48"):
        p = d / name
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}",
             "-t", str(secs), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)],
            check=True,
        )
        return str(p)

    out["fps3"] = make("uniquechunkname_7731.mp4", "3", 3)
    out["ntsc"] = make("uniquentscchunk_8842.mp4", "30000/1001", 2)
    out["fps5"] = make("uniquefive_9953.mp4", "5", 2)
    return out


@pytest.fixture()
def clip(clips):
    return clips["fps3"]


@pytest.fixture()
def provider_factory(api):
    def make(**kw):
        return H.make_provider(api, **kw)

    return make


@pytest.fixture()
def provider(provider_factory):
    return provider_factory()
