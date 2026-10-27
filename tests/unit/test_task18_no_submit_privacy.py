"""Task 18: nothing is ever submitted to an agency; the report never exposes paths or unblurred images."""
import ast
import builtins
import re
import socket
import sys
from pathlib import Path

import pytest

import rs18_helpers
from rs18_helpers import MD_PT, ROOT, dump, rep_text, rs18_env, walk  # noqa: F401

CLIENT_LIBS = ("requests", "httpx", "urllib.request", "urllib3", "http.client", "aiohttp", "smtplib", "ftplib",
               "selenium", "playwright", "mechanize", "webbrowser", "pycurl", "websockets")


def is_client(mod):
    return any(mod == c or mod.startswith(c + ".") for c in CLIENT_LIBS)


def client_imports(tree):
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names if is_client(a.name)}
        elif isinstance(node, ast.ImportFrom) and node.module:
            if is_client(node.module) or any(is_client(node.module + "." + a.name) for a in node.names):
                found.add(node.module)
    return found
NETWORK_NAMES = re.compile(r"agency|report|vdot|mdot|submit", re.I)


# ------------------------------------------------------------------ runtime: no outbound call of any kind

class NetSpy:
    def __init__(self, mp):
        self.calls = []
        mp.setattr(socket, "getaddrinfo", self._wrap(socket.getaddrinfo, "getaddrinfo"))
        mp.setattr(socket, "create_connection", self._wrap(socket.create_connection, "create_connection"))
        try:
            import httpx

            mp.setattr(httpx.HTTPTransport, "handle_request", self._boom("httpx.HTTPTransport"))
            mp.setattr(httpx.AsyncHTTPTransport, "handle_async_request", self._aboom("httpx.AsyncHTTPTransport"))
        except Exception:
            pass
        try:
            import requests

            mp.setattr(requests.adapters.HTTPAdapter, "send", self._boom("requests"))
        except Exception:
            pass
        import urllib.request
        import http.client

        mp.setattr(urllib.request, "urlopen", self._boom("urllib.request.urlopen"))
        mp.setattr(http.client.HTTPConnection, "connect", self._boom("http.client.HTTPConnection.connect"))
        mp.setattr(http.client.HTTPSConnection, "connect", self._boom("http.client.HTTPSConnection.connect"))

    def _wrap(self, real, label):
        def f(host, *a, **k):
            h = host[0] if isinstance(host, tuple) else host
            if str(h) not in rs18_helpers.LOOPBACK:
                self.calls.append((label, str(h)))
                raise AssertionError("outbound network call %s(%r)" % (label, h))
            return real(host, *a, **k)

        return f

    def _boom(self, label):
        def f(*a, **k):
            self.calls.append((label, ""))
            raise AssertionError("outbound HTTP call via " + label)

        return f

    def _aboom(self, label):
        async def f(*a, **k):
            self.calls.append((label, ""))
            raise AssertionError("outbound HTTP call via " + label)

        return f


def exercise(env, hid):
    """Touch every report related route for hazard hid."""
    c = env.client
    rp = env.report_path()
    base = env.sub(rp, hid)
    out = [c.get(base)]
    for suffix in (".txt", "/download", "/text"):
        out.append(c.get(base.rstrip("/") + suffix))
    for q in ("format=txt", "download=1", "format=text"):
        out.append(c.get(base + "?" + q))
    pp = env.photo_path()
    if pp:
        out.append(c.get(env.sub(pp, hid)))
        out.append(c.get(env.sub(pp, hid) + "?download=1"))
    out.append(env.set_agency(hid, "MDOT SHA"))
    out.append(c.get(base))
    out.append(env.set_agency(hid, "VDOT"))
    out.append(env.set_agency(hid, "none"))
    out.append(env.set_agency(hid, None))
    out.append(env.set_status(hid, "reported"))
    out.append(c.get(base))
    out.append(c.get("/api/hazards"))
    return out


def test_no_outbound_calls_from_any_report_route(rs18_env, monkeypatch):
    env = rs18_env
    hid, _ = env.seed()
    spy = NetSpy(monkeypatch)
    out = exercise(env, hid)
    assert spy.calls == [], "outbound calls: %s" % spy.calls
    assert all(r.status_code < 500 for r in out), [r.status_code for r in out]


def test_no_outbound_calls_for_a_maryland_hazard(rs18_env, monkeypatch):
    env = rs18_env
    hid, _ = env.seed(lat=MD_PT[0], lon=MD_PT[1])
    spy = NetSpy(monkeypatch)
    exercise(env, hid)
    assert spy.calls == []


def test_marking_reported_is_only_a_local_status_change(rs18_env, monkeypatch):
    env = rs18_env
    hid, _ = env.seed()
    spy = NetSpy(monkeypatch)
    r = env.set_status(hid, "reported")
    assert r.status_code == 200
    assert env.row(hid)["status"] == "reported"
    assert spy.calls == []


# ------------------------------------------------------------------ static: no submitting machinery

def test_no_submit_routes(rs18_env):
    for p, ops in rs18_env.paths().items():
        for m in ops:
            if m in ("post", "put", "patch", "delete"):
                assert not re.search(r"submit|send|file-report|file_report|transmit|forward", p, re.I), \
                    "%s %s looks like an automatic submission" % (m.upper(), p)


def agency_modules():
    out = []
    for p in (ROOT / "backend").glob("*.py"):
        if re.search(r"agenc|report|vdot|mdot|jurisdic", p.stem, re.I):
            out.append(p)
    return out


def test_there_is_an_agency_or_report_module():
    assert agency_modules(), "expected a backend module for agencies / reports"


@pytest.mark.parametrize("path", agency_modules(), ids=lambda p: p.name)
def test_agency_and_report_modules_import_no_network_client(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = sorted(client_imports(tree))
    assert not bad, "%s imports a network-capable library: %s" % (path.name, bad)


@pytest.mark.parametrize("path", agency_modules(), ids=lambda p: p.name)
def test_agency_and_report_modules_call_no_network_or_dynamic_import(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            assert name not in ("urlopen", "urlretrieve", "create_connection", "__import__", "system", "Popen",
                                "check_output", "startfile", "open_new_tab"), \
                "%s calls %s()" % (path.name, name)


def test_backend_never_posts_to_the_agency_sites():
    hits = []
    for p in (ROOT / "backend").rglob("*.py"):
        src = p.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"vdot\.virginia\.gov|salesforce-sites\.com", src):
            continue
        imports = client_imports(ast.parse(src))
        if imports:
            hits.append((p.name, sorted(imports)))
    assert not hits, "modules that mention the agency sites and import a network client: %s" % hits


# ------------------------------------------------------------------ privacy

def test_report_and_text_expose_no_paths(rs18_env):
    env = rs18_env
    classes = rs18_helpers.known_classes()
    for i, cls in enumerate(classes):
        hid, info = env.seed(name="p%d" % i, hazard_class=cls)
        for agency in (None, "MDOT SHA", "VDOT"):
            env.ok_agency(hid, agency)
            body = env.report(hid)
            blob = dump(body) + "\n" + rep_text(body)
            for forbidden in (str(env.tmp), str(env.tmp).replace("\\", "/"), str(env.storage),
                              str(info["blur"]), str(info["orig"]), "zqblur", "zqorig", "drv18"):
                assert forbidden not in blob, "report leaks %r" % forbidden
            real = "\n".join(rs18_helpers.all_strings(body)) + "\n" + rep_text(body)
            assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", real.replace("https://", "").replace("http://", ""))
            assert "photo_path" not in blob and "crop_path" not in blob
            for path, _v in walk(body):
                if path:
                    assert "path" not in path[-1].lower(), "key %r looks like a filesystem path" % path[-1]


def test_report_has_no_image_bytes_or_data_urls(rs18_env):
    env = rs18_env
    hid, info = env.seed()
    blob = dump(env.report(hid))
    assert "data:image" not in blob and "base64" not in blob.lower()
    assert "\\xff\\xd8" not in blob and "ÿØ" not in blob


def test_photo_url_never_resolves_to_the_unblurred_frame(rs18_env):
    env = rs18_env
    hid, info = env.seed()
    url = rs18_helpers.rep_photo_url(env.report(hid))
    assert url
    r = env.client.get(url)
    assert r.content == info["blur"].read_bytes() != info["orig"].read_bytes()


def test_reports_table_if_any_holds_no_original_paths(rs18_env):
    env = rs18_env
    hid, info = env.seed()
    env.report(hid)
    tables = {r["name"] for r in env.q("SELECT name FROM sqlite_master WHERE type='table'")}
    if "reports" not in tables:
        return
    for row in env.q("SELECT * FROM reports"):
        for v in row.values():
            assert "zqorig" not in str(v), "reports table references the unblurred frame"
