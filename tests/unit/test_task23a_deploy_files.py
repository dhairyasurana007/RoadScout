"""Task 23a: Dockerfile, .dockerignore and render.yaml - parsed as text, no docker needed."""
from __future__ import annotations

import fnmatch
import glob
import re

import pytest

from rs23a_helpers import ROOT

SECRET_WORDS = re.compile(r"(?i)(api[_-]?key|secret|token|passw(or)?d|credential|private[_-]?key)")


def read(name):
    p = ROOT / name
    assert p.is_file(), f"{name} must exist at the repo root"
    return p.read_text(encoding="utf-8")


def logical_lines(text):
    """Dockerfile lines with continuations joined and comments dropped."""
    out, cur = [], ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if not cur and line.lstrip().startswith("#"):
            continue
        if line.endswith("\\"):
            cur += line[:-1] + " "
            continue
        cur += line
        if cur.strip():
            out.append(cur.strip())
        cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out


# ------------------------------------------------------------------ Dockerfile

@pytest.fixture(scope="module")
def dockerfile():
    return logical_lines(read("Dockerfile"))


def instructions(lines, word):
    return [l for l in lines if l.split(None, 1)[0].upper() == word]


def test_dockerfile_is_multi_stage_node_then_python_slim(dockerfile):
    froms = instructions(dockerfile, "FROM")
    assert len(froms) >= 2
    assert re.match(r"(?i)FROM\s+(--platform=\S+\s+)?node:", froms[0]), froms[0]
    assert any(re.match(r"(?i)FROM\s+(--platform=\S+\s+)?python:\S*slim", f) for f in froms[1:]), froms
    assert re.match(r"(?i)FROM\s+(--platform=\S+\s+)?python:\S*slim", froms[-1]), "the final stage is the Python image"


def test_node_stage_builds_the_frontend(dockerfile):
    text = "\n".join(dockerfile)
    assert re.search(r"npm\s+(ci|install)", text)
    assert re.search(r"npm\s+run\s+build", text)
    assert "frontend" in text


def test_final_image_copies_the_built_dist_from_the_node_stage(dockerfile):
    copies = [l for l in instructions(dockerfile, "COPY") if "--from" in l]
    assert any("dist" in l for l in copies), copies


def test_final_image_installs_the_python_requirements(dockerfile):
    text = "\n".join(dockerfile)
    assert "requirements.txt" in text and re.search(r"pip\s+install", text)


def env_pairs(lines):
    pairs = {}
    for l in instructions(lines, "ENV"):
        body = l.split(None, 1)[1]
        if "=" in body.split(None, 1)[0]:
            for m in re.finditer(r'(\w+)=("[^"]*"|\'[^\']*\'|\S*)', body):
                pairs[m.group(1)] = m.group(2).strip("\"'")
        else:
            k, _, v = body.partition(" ")
            pairs[k] = v.strip().strip("\"'")
    return pairs


def test_environment_variables_for_the_demo(dockerfile):
    env = env_pairs(dockerfile)
    assert env.get("ROADSCOUT_DEMO") == "1"
    assert env.get("ROADSCOUT_DB") == "/app/demo/demo.sqlite"
    assert env.get("ROADSCOUT_STORAGE_DIR") == "/app/demo/storage"
    assert env.get("ROADSCOUT_STATIC_DIR"), "the static directory must be set"
    assert env["ROADSCOUT_STATIC_DIR"].startswith("/"), "an absolute path inside the image"
    assert "dist" in env["ROADSCOUT_STATIC_DIR"]


def test_static_dir_matches_where_dist_is_copied(dockerfile):
    env = env_pairs(dockerfile)
    target = env["ROADSCOUT_STATIC_DIR"].rstrip("/")
    copies = [l for l in instructions(dockerfile, "COPY") if "--from" in l and "dist" in l]
    assert copies
    # A relative COPY destination is resolved against the final stage's WORKDIR (/app).
    dests = [(d if d.startswith("/") else "/app/" + d) for d in (l.split()[-1].rstrip("/") for l in copies)]
    assert any(target == d or target.startswith(d + "/") or d.startswith(target) or d == target for d in dests), (
        target, dests)


def test_dockerfile_starts_uvicorn_on_port_from_env(dockerfile):
    cmd = [l for l in dockerfile if l.split(None, 1)[0].upper() in ("CMD", "ENTRYPOINT")]
    assert cmd, "a CMD is required"
    flat = re.sub(r'[\[\]",]', " ", " ".join(cmd))
    flat = re.sub(r"\s+", " ", flat)
    assert "python -m uvicorn backend.main:app" in flat, flat
    assert "--host 0.0.0.0" in flat
    assert re.search(r"--port\s+\$\{?PORT\}?", flat), "the port comes from $PORT"


def test_dockerfile_has_no_secrets_or_private_data(dockerfile):
    text = "\n".join(dockerfile)
    assert "META_API_KEY" not in text
    assert not SECRET_WORDS.search(text), SECRET_WORDS.search(text).group(0)
    for l in instructions(dockerfile, "COPY") + instructions(dockerfile, "ADD"):
        assert ".env" not in l.replace(".env.example", ""), l
        for bad in ("data/", "spike", "labels", "footage", ".git"):
            assert not re.search(r"(^|\s)%s" % re.escape(bad), l.split(None, 1)[1]), l
    assert not [l for l in instructions(dockerfile, "ARG") if SECRET_WORDS.search(l)]


def test_dockerfile_does_not_run_a_server_for_real_footage(dockerfile):
    text = "\n".join(dockerfile).lower()
    assert "roadscout_provider=live" not in text and "roadscout_provider live" not in text


def test_copy_sources_exist_in_the_repo(dockerfile):
    for l in instructions(dockerfile, "COPY"):
        if "--from" in l:
            continue
        parts = [p for p in l.split()[1:] if not p.startswith("--")]
        for src in parts[:-1]:
            if "demo" in src:           # the prebuilt dataset is added by a later task
                continue
            hits = glob.glob(str(ROOT / src))
            assert hits, f"COPY source {src!r} does not exist in the repo"


def test_copies_what_the_backend_needs(dockerfile):
    text = "\n".join(l for l in instructions(dockerfile, "COPY") if "--from" not in l)
    for needed in ("backend", "config", "requirements.txt"):
        assert needed in text or re.search(r"COPY\s+\.\s", text), needed


# ------------------------------------------------------------------ .dockerignore

@pytest.fixture(scope="module")
def ignore_rules():
    rules = []
    for raw in read(".dockerignore").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            rules.append(line)
    return rules


def _regex(pat):
    """Docker .dockerignore pattern -> regex: '*' stays inside one path part, '**' crosses parts."""
    out, i = "", 0
    while i < len(pat):
        if pat.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pat.startswith("**", i):
            out += ".*"
            i += 2
        elif pat[i] == "*":
            out += "[^/]*"
            i += 1
        elif pat[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pat[i])
            i += 1
    return re.compile(out)


def ignored(rules, path):
    """Docker's rule semantics: root-relative matching, a matched directory excludes its contents,
    a later '!' rule re-includes."""
    parts = path.split("/")
    prefixes = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
    state = False
    for rule in rules:
        neg = rule.startswith("!")
        pat = (rule[1:] if neg else rule).lstrip("/").rstrip("/")
        rx = _regex(pat)
        if any(rx.fullmatch(pre) for pre in prefixes):
            state = not neg
    return state


@pytest.mark.parametrize("path", [".env", "data/roadscout.sqlite", "data/storage/x/video.mov", "spike/probe.py",
                                  "spike/footage/IMG_6503.MOV", "labels/IMG_6513.csv", ".git/config",
                                  "node_modules/react/index.js", "frontend/node_modules/react/index.js",
                                  "tests/unit/test_x.py", "clip.mov", "track.gpx", "movie.mp4"])
def test_dockerignore_excludes(ignore_rules, path):
    assert ignored(ignore_rules, path), f"{path} must be excluded from the image build context"


@pytest.mark.parametrize("path", ["backend/main.py", "config/hazard_prompts.json", "requirements.txt",
                                  "frontend/package.json", "frontend/package-lock.json", "frontend/src/main.jsx",
                                  "frontend/index.html", "README.md", "demo/demo.sqlite", "demo/storage/x/y.jpg",
                                  "scripts/eval.py", "fixtures/providers/tiny/detect.json"])
def test_dockerignore_does_not_exclude_what_the_image_needs(ignore_rules, path):
    if path.startswith("demo/"):
        pytest.skip("covered once the prebuilt dataset exists")
    if path in ("frontend/package-lock.json", "frontend/src/main.jsx") and not (ROOT / path).exists():
        pytest.skip("file not present")
    assert not ignored(ignore_rules, path), f"{path} is needed in the image"


@pytest.mark.xfail(strict=False, reason="advisory: nested media files need '**/*.mov' style rules")
@pytest.mark.parametrize("path", ["sub/dir/drive.mov", "sub/dir/track.gpx", "sub/dir/movie.mp4", "footage/clip.mov"])
def test_dockerignore_excludes_nested_media_files_too(ignore_rules, path):
    assert ignored(ignore_rules, path)


@pytest.mark.xfail(strict=False, reason="advisory: docker's ignore rules are case-sensitive (uppercase iPhone names)")
@pytest.mark.parametrize("path", ["IMG_6513.MOV", "track.GPX", "clip.MP4"])
def test_dockerignore_excludes_uppercase_extensions_too(ignore_rules, path):
    assert ignored(ignore_rules, path)


def test_dockerignore_has_no_secrets(ignore_rules):
    assert not any(SECRET_WORDS.search(r) and not r.startswith(".env") for r in ignore_rules)


# ------------------------------------------------------------------ render.yaml

def test_render_yaml_text_has_no_secrets():
    text = read("render.yaml")
    assert "META_API_KEY" not in text
    assert not SECRET_WORDS.search(text), SECRET_WORDS.search(text).group(0)
    assert "sync: false" not in text.replace(" ", " ")
    assert "generateValue" not in text and "fromDatabase" not in text and "fromGroup" not in text


def test_render_yaml_health_check_and_runtime_in_text():
    text = read("render.yaml")
    assert re.search(r"(?m)^\s*healthCheckPath:\s*/health\s*$", text)
    assert re.search(r"(?m)^\s*runtime:\s*docker\s*$", text)


def test_render_blueprint_structure():
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(read("render.yaml"))
    assert isinstance(doc, dict)
    services = doc["services"]
    assert len(services) == 1, "one web service"
    svc = services[0]
    assert svc["type"] == "web"
    assert svc["runtime"] == "docker"
    assert svc["healthCheckPath"] == "/health"
    assert svc.get("name")
    env = svc.get("envVars", [])
    assert [e["key"] for e in env] == ["ROADSCOUT_DEMO"], "the only environment variable is ROADSCOUT_DEMO"
    assert str(env[0]["value"]) == "1"
    for e in env:
        assert "sync" not in e and "generateValue" not in e and "fromService" not in e and "fromDatabase" not in e
    assert set(doc) <= {"services", "version"}, "no databases, key-value stores or env groups"
    assert "disk" not in svc and "disks" not in svc


def test_render_blueprint_has_no_secret_looking_keys_anywhere():
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(read("render.yaml"))

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                assert not SECRET_WORDS.search(str(k)), k
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        else:
            assert not SECRET_WORDS.search(str(o)), o

    walk(doc)


def test_render_dockerfile_path_if_given_exists():
    yaml = pytest.importorskip("yaml")
    svc = yaml.safe_load(read("render.yaml"))["services"][0]
    path = svc.get("dockerfilePath", "./Dockerfile")
    assert (ROOT / path).is_file()
    ctx = svc.get("dockerContext", ".")
    assert (ROOT / ctx).is_dir()
