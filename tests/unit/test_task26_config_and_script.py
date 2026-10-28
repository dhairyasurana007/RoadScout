"""Task 26: chunk default of 20 s, docs, requirements, and scripts/record_fixtures.py."""
import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import rs26_helpers as H
from rs26_conftest import _safe_env, rec, api, clips, clip, provider_factory, provider  # noqa: F401
from rs26_helpers import KEY, frame_line, rec_str, verdict_obj


def repo():
    return Path(H.REPO)


# ------------------------------------------------------------------ chunk default

def test_default_chunk_seconds_constant_is_20():
    from backend import config

    assert config.DEFAULT_CHUNK_SECONDS == 20


def test_chunk_seconds_default_is_20(monkeypatch):
    from backend import config

    monkeypatch.delenv("ROADSCOUT_CHUNK_SECONDS", raising=False)
    assert config.chunk_seconds() == 20


def test_chunk_seconds_env_still_overrides(monkeypatch):
    from backend import config

    monkeypatch.setenv("ROADSCOUT_CHUNK_SECONDS", "4")
    assert config.chunk_seconds() == 4
    monkeypatch.setenv("ROADSCOUT_CHUNK_SECONDS", "12.5")
    assert config.chunk_seconds() == 12.5


def test_chunk_seconds_docstring_not_stale():
    from backend import config

    assert not re.search(r"default\s+10\b", config.chunk_seconds.__doc__ or "")


def test_env_example_documents_20_not_10():
    text = (repo() / ".env.example").read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if re.search(r"chunk", l, re.I)]
    assert lines, "the chunk setting must stay documented in .env.example"
    assert any(re.search(r"ROADSCOUT_CHUNK_SECONDS\s*=\s*20\b", l) for l in lines)
    for l in lines:
        assert not re.search(r"\b10\b\s*(s\b|sec|second)", l, re.I), l
        assert not re.search(r"default\D{0,5}\b10\b", l, re.I), l
    # the real key placeholder stays a placeholder
    m = re.search(r"^META_API_KEY=(.*)$", text, re.M)
    assert m and ("your" in m.group(1).lower() or m.group(1).strip() == "")


def test_readme_states_the_default_of_20():
    text = (repo() / "README.md").read_text(encoding="utf-8")
    assert not re.search(r"ROADSCOUT_CHUNK_SECONDS`?\s*\(\s*10\s*\)", text)
    assert re.search(r"ROADSCOUT_CHUNK_SECONDS`?\s*\(\s*20\s*\)", text)


def test_requirements_list_the_sam_parser():
    text = (repo() / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^meta-sam-parser\s*>=\s*0\.1\.0\s*$", text, re.M | re.I)


def test_default_preprocess_chunks_are_20_seconds(tmp_path, monkeypatch):
    from backend.preprocess import preprocess_video

    monkeypatch.delenv("ROADSCOUT_CHUNK_SECONDS", raising=False)
    src = tmp_path / "long.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=6", "-t", "45",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    res = preprocess_video(str(src), str(tmp_path / "out"))
    assert len(res.chunks) == 3, "45 s at the default 20 s chunk length is 3 chunks (20 + 20 + 5)"
    first = res.chunks[0]
    frames = first["frame_count"] if isinstance(first, dict) else first.frame_count
    assert frames == pytest.approx(60, abs=2), "20 s at the default 3 fps is 60 frames"


# ------------------------------------------------------------------ record script

SCRIPT_REL = Path("scripts") / "record_fixtures.py"


def run_script(args, env_over, timeout=90):
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(repo()), "ROADSCOUT_COST_LOG": "off",
                "ROADSCOUT_META_BASE_URL": "http://127.0.0.1:9/v1"})
    for k, v in env_over.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return subprocess.run([sys.executable, str(repo() / SCRIPT_REL), *map(str, args)], cwd=str(repo()), env=env,
                          capture_output=True, text=True, timeout=timeout)


def out_files(d):
    return sorted(p.name for p in Path(d).glob("*")) if Path(d).exists() else []


def test_script_exists():
    assert (repo() / SCRIPT_REL).is_file()


def test_script_without_arguments_fails_with_usage():
    cp = run_script([], {"ROADSCOUT_PROVIDER": "live", "META_API_KEY": KEY})
    assert cp.returncode != 0
    assert "usage" in (cp.stdout + cp.stderr).lower()
    assert KEY not in cp.stdout + cp.stderr


def test_script_with_one_argument_fails():
    cp = run_script(["only_one.mp4"], {"ROADSCOUT_PROVIDER": "live", "META_API_KEY": KEY})
    assert cp.returncode != 0
    assert KEY not in cp.stdout + cp.stderr


@pytest.mark.parametrize("provider", [None, "fake", "fixture", "FAKE", ""])
def test_script_refuses_unless_provider_is_live(tmp_path, clip, provider):
    out = tmp_path / "out"
    cp = run_script([clip, out], {"ROADSCOUT_PROVIDER": provider, "META_API_KEY": KEY})
    assert cp.returncode != 0
    text = cp.stdout + cp.stderr
    assert "ROADSCOUT_PROVIDER" in text or "live" in text.lower()
    assert KEY not in text
    assert "detect.json" not in out_files(out) and "verify.json" not in out_files(out)


@pytest.mark.parametrize("key", [None, ""])
def test_script_refuses_without_a_key(tmp_path, clip, key):
    out = tmp_path / "out"
    cp = run_script([clip, out], {"ROADSCOUT_PROVIDER": "live", "META_API_KEY": key if key is not None else ""})
    assert cp.returncode != 0
    assert "META_API_KEY" in cp.stdout + cp.stderr
    assert "detect.json" not in out_files(out) and "verify.json" not in out_files(out)


def test_script_failure_never_prints_the_key(tmp_path, clip):
    """Provider live and a key, but the base URL is a closed local port: the call fails, the key stays hidden."""
    out = tmp_path / "out"
    cp = run_script([clip, out], {"ROADSCOUT_PROVIDER": "live", "META_API_KEY": KEY})
    assert cp.returncode != 0
    assert KEY not in cp.stdout + cp.stderr
    for p in Path(out).glob("*") if out.exists() else []:
        assert KEY not in p.read_text(encoding="utf-8", errors="ignore")


def test_script_missing_chunk_fails_cleanly(tmp_path):
    cp = run_script([tmp_path / "nope.mp4", tmp_path / "out"], {"ROADSCOUT_PROVIDER": "live", "META_API_KEY": KEY})
    assert cp.returncode != 0
    assert KEY not in cp.stdout + cp.stderr


# ----- end to end against a local stub server (never the real API)

class Stub:
    def __init__(self):
        self.requests = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(n) if n else b""

            def do_POST(self):
                body = self._body()
                stub.requests.append(("POST", self.path, dict(self.headers), body))
                if self.path.endswith("/files"):
                    return self._send(200, {"id": "file_stub_1", "object": "file"})
                payload = json.loads(body)
                parts = payload["input"][0]["content"]
                if any(p.get("type") == "input_video" for p in parts):
                    text = "\n".join(
                        frame_line(n, rec_str(0, 4, 4, 33, 33, 64, 48, c=0.8)) for n in (0, 1, 2))
                else:
                    text = json.dumps(verdict_obj())
                return self._send(200, H.out_json(text))

            def do_DELETE(self):
                stub.requests.append(("DELETE", self.path, dict(self.headers), b""))
                return self._send(200, {"deleted": True})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()


def test_script_end_to_end_writes_fixture_files(tmp_path, clip):
    from backend.providers.fixture import FixtureProvider
    from backend.providers.types import DetectionResult, Verdict

    out = tmp_path / "fixtures_out"
    with Stub() as stub:
        cp = run_script([clip, out], {
            "ROADSCOUT_PROVIDER": "live", "META_API_KEY": KEY, "ROADSCOUT_SPEND_CAP_USD": "5",
            "ROADSCOUT_META_BASE_URL": f"http://127.0.0.1:{stub.port}/v1"})
    text = cp.stdout + cp.stderr
    assert cp.returncode == 0, text
    assert KEY not in text
    assert (out / "detect.json").is_file() and (out / "verify.json").is_file()
    det = DetectionResult.from_dict(json.loads((out / "detect.json").read_text(encoding="utf-8")))
    assert not det.is_empty
    ver = Verdict.from_dict(json.loads((out / "verify.json").read_text(encoding="utf-8")))
    assert ver.hazard_class == "pothole" and ver.model == "muse-spark-1.3"
    for name in ("detect.json", "verify.json"):
        assert KEY not in (out / name).read_text(encoding="utf-8")
    # the recorded files replay through the existing fixture provider
    fp = FixtureProvider(out)
    assert fp.detect("whatever.mp4", ["x"]).to_dict() == det.to_dict()
    assert fp.verify([b"x"]).to_dict() == ver.to_dict()
    # what the stub saw: auth on everything, SAM then Muse Spark, upload deleted, one upload
    assert all(r[2].get("Authorization") == f"Bearer {KEY}" for r in stub.requests)
    assert sum(1 for r in stub.requests if r[0] == "POST" and r[1].endswith("/files")) >= 1
    assert sum(1 for r in stub.requests if r[0] == "DELETE") == sum(
        1 for r in stub.requests if r[0] == "POST" and r[1].endswith("/files"))
    bodies = [json.loads(r[3]) for r in stub.requests if r[1].endswith("/responses")]
    sam = [b for b in bodies if b["model"] == "sam-3.1"]
    mus = [b for b in bodies if b["model"] == "muse-spark-1.3"]
    assert sam and len(mus) == 1
    prompts_cfg = json.loads((repo() / "config" / "hazard_prompts.json").read_text(encoding="utf-8"))["classes"]
    allowed = {p for phrases in prompts_cfg.values() for p in phrases}
    for b in sam:
        assert b["input"][0]["content"][0]["text"] in allowed
    assert any(p.get("type") == "input_image" for p in mus[0]["input"][0]["content"])
