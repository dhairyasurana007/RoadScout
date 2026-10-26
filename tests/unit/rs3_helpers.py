"""Shared helpers for the task 3 hidden tests (provider interface).

Names are discovered tolerantly: the package backend.providers and all of its
submodules are searched, so the coder may split modules however they like.
"""
import dataclasses
import importlib
import inspect
import json
import pkgutil
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

MISSING = object()
PKG = "backend.providers"

# Aliases tried in order for each required public name.
ALIASES = {
    "ModelProvider": ["ModelProvider", "Provider", "BaseProvider"],
    "FakeProvider": ["FakeProvider"],
    "FixtureProvider": ["FixtureProvider", "RecordedFixtureProvider", "RecordedProvider", "ReplayProvider"],
    "LiveProvider": ["LiveProvider", "MetaProvider", "LiveMetaProvider"],
    "get_provider": ["get_provider", "make_provider", "create_provider", "build_provider"],
    "DetectionResult": ["DetectionResult", "DetectionResults"],
    "Verdict": ["Verdict", "VerificationResult", "VerificationVerdict"],
    "ProviderError": ["ProviderError"],
    "TransientProviderError": ["TransientProviderError", "TransientError", "RetryableError"],
    "ProviderTimeoutError": ["ProviderTimeoutError", "ProviderTimeout", "CallTimeoutError"],
    "SpendCapExceeded": ["SpendCapExceeded", "SpendCapExceededError", "BudgetExceeded", "SpendLimitExceeded"],
    "SpendTracker": ["SpendTracker", "SpendCounter", "CostTracker"],
    "call_with_retries": ["call_with_retries", "with_retries", "retry_call"],
    "call_with_timeout": ["call_with_timeout", "with_timeout", "run_with_timeout"],
}


def _modules():
    pkg = importlib.import_module(PKG)
    mods = [pkg]
    for info in pkgutil.walk_packages(pkg.__path__, PKG + "."):
        mods.append(importlib.import_module(info.name))
    return mods


def find(key):
    """Return the public object registered under any alias of `key`."""
    mods = _modules()
    for name in ALIASES.get(key, [key]):
        for m in mods:  # package namespace first, then submodules
            if hasattr(m, name):
                return getattr(m, name)
    raise AssertionError(
        f"{PKG} must expose {ALIASES.get(key, [key])[0]!r} (searched package and submodules)"
    )


def get(obj, *names, default=MISSING):
    for n in names:
        if isinstance(obj, dict):
            if n in obj:
                return obj[n]
        elif hasattr(obj, n):
            return getattr(obj, n)
    if default is MISSING:
        raise AssertionError(f"{type(obj).__name__} has none of the fields {names}")
    return default


def normalize(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return normalize(dataclasses.asdict(obj))
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return normalize(obj.to_dict())
    if hasattr(obj, "model_dump"):
        return normalize(obj.model_dump())
    if isinstance(obj, dict):
        return {str(k): normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]
    if hasattr(obj, "tolist"):
        return normalize(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def canon(obj):
    return json.dumps(normalize(obj), sort_keys=True, default=str)


def frames_of(result):
    return list(get(result, "frames", "records"))


def dets_of(frame):
    return list(get(frame, "detections", "objects", "instances"))


def all_dets(result):
    return [d for f in frames_of(result) for d in dets_of(f)]


def track_id(det):
    return get(det, "track_id", "object_id", "track")


def frames_per_track(result):
    seen = {}
    for f in frames_of(result):
        for d in dets_of(f):
            seen.setdefault(track_id(d), set()).add(id(f))
    return {k: len(v) for k, v in seen.items()}


def verdict_field(v, name):
    if name == "hazard_class":
        return get(v, "hazard_class", "class_", "class", "cls", "klass", "category")
    return get(v, name)


# ---------------------------------------------------------------- fixtures

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any socket use during these tests is a failure: no live Meta API calls."""
    calls = []

    def blocker(*a, **k):
        calls.append(a)
        raise RuntimeError("network access blocked by hidden tests")

    monkeypatch.setattr(socket.socket, "connect", blocker)
    monkeypatch.setattr(socket.socket, "connect_ex", blocker)
    monkeypatch.setattr(socket, "create_connection", blocker)
    monkeypatch.setattr(socket, "getaddrinfo", blocker)
    yield calls
    assert not calls, "a network call was attempted"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("META_API_KEY", "ROADSCOUT_PROVIDER", "ROADSCOUT_SPEND_CAP_USD",
              "ROADSCOUT_FIXTURES_DIR"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    """A tiny synthetic clip (ffmpeg testsrc). Never real footage."""
    d = tmp_path_factory.mktemp("rs3clip")
    p = d / "chunk.mp4"
    if shutil.which("ffmpeg"):
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
             "testsrc=size=160x120:rate=5:duration=1", "-pix_fmt", "yuv420p", str(p)],
            capture_output=True)
        if r.returncode == 0 and p.exists():
            return p
    p.write_bytes(b"\x00" * 64)
    return p


@pytest.fixture(scope="session")
def frame_image(tmp_path_factory):
    d = tmp_path_factory.mktemp("rs3img")
    p = d / "frame.png"
    if shutil.which("ffmpeg"):
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
             "testsrc=size=160x120:rate=1:duration=1", "-frames:v", "1", str(p)],
            capture_output=True)
        if r.returncode == 0 and p.exists():
            return p
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    return p


PROMPTS = ["pothole", "debris in lane"]
