"""Task 18: the copy / download side: text download and blurred photo download."""
import re

import pytest

from rs18_helpers import MD_PT, rep_text, rs18_env  # noqa: F401

TEXT_PARAMS = {"format": ["txt", "text", "plain"], "fmt": ["txt", "text"], "download": ["1", "true"],
               "as": ["txt", "text"], "type": ["txt", "text"], "attachment": ["1", "true"]}
PHOTO_PARAMS = {"download": ["1", "true"], "attachment": ["1", "true"], "dl": ["1", "true"], "disposition": ["attachment"]}


def text_candidates(env, hid):
    """Yield (label, url, params) for every plausible text-download variant."""
    rp = env.report_path()
    base = env.sub(rp, hid)
    for p, ops in env.paths().items():
        if "get" in ops and p != rp and p.rstrip("/").startswith(rp.rstrip("/")) and "photo" not in p:
            yield "path " + p, env.sub(p, hid), None
    for p, ops in env.paths().items():
        if "get" in ops and p != rp and re.search(r"report", p) and re.search(r"download|\.txt|/text|/txt", p) and "photo" not in p:
            yield "path " + p, env.sub(p, hid), None
    names = {x["name"] for x in env.paths()[rp]["get"].get("parameters", []) if x.get("in") == "query"}
    for n, vals in TEXT_PARAMS.items():
        for v in vals:
            yield "?%s=%s" % (n, v), base, {n: v}
    for n in names - set(TEXT_PARAMS):
        for v in ("txt", "text", "1", "true"):
            yield "?%s=%s" % (n, v), base, {n: v}
    yield "path .txt", base.rstrip("/") + ".txt", None
    yield "path /download", base.rstrip("/") + "/download", None
    yield "path /text", base.rstrip("/") + "/text", None


def text_download(env, hid):
    tried = []
    for label, url, params in text_candidates(env, hid):
        r = env.client.get(url, params=params)
        cd = r.headers.get("content-disposition", "")
        ct = r.headers.get("content-type", "").split(";")[0].strip().lower()
        tried.append("%s -> %s %s %s" % (label, r.status_code, ct, cd[:40]))
        if r.status_code == 200 and "attachment" in cd.lower() and ct == "text/plain":
            return r, label
    pytest.fail("no text download variant found (Content-Disposition: attachment + text/plain). Tried: " + "; ".join(tried[:12]))


def photo_candidates(env, hid):
    pp = env.photo_path()
    base = env.sub(pp, hid)
    names = {x["name"] for x in env.paths()[pp]["get"].get("parameters", []) if x.get("in") == "query"}
    for n, vals in PHOTO_PARAMS.items():
        for v in vals:
            yield "?%s=%s" % (n, v), base, {n: v}
    for n in names - set(PHOTO_PARAMS) - {"variant"}:
        for v in ("1", "true"):
            yield "?%s=%s" % (n, v), base, {n: v}
    for p, ops in env.paths().items():
        if "get" in ops and "photo" in p and p != pp and ("download" in p or p.startswith(pp.rstrip("/"))):
            yield "path " + p, env.sub(p, hid), None
    rp = env.report_path()
    if rp:
        yield "path report/photo", env.sub(rp, hid).rstrip("/") + "/photo", None
        yield "path report/photo/download", env.sub(rp, hid).rstrip("/") + "/photo/download", None


def photo_download(env, hid):
    tried = []
    for label, url, params in photo_candidates(env, hid):
        r = env.client.get(url, params=params)
        cd = r.headers.get("content-disposition", "")
        ct = r.headers.get("content-type", "").split(";")[0].strip().lower()
        tried.append("%s -> %s %s %s" % (label, r.status_code, ct, cd[:40]))
        if r.status_code == 200 and "attachment" in cd.lower() and ct.startswith("image/"):
            return r, label
    pytest.fail("no photo download variant found (Content-Disposition: attachment + image/*). Tried: " + "; ".join(tried[:12]))


# ------------------------------------------------------------------ text

def test_text_download_matches_the_json_text(rs18_env):
    env = rs18_env
    hid, _ = env.seed(lat=MD_PT[0], lon=MD_PT[1])
    r, _label = text_download(env, hid)
    assert r.text.strip() == rep_text(env.report(hid)).strip()


def test_text_download_has_a_safe_filename(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    r, _ = text_download(env, hid)
    cd = r.headers["content-disposition"]
    m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", cd, re.I)
    assert m, "attachment without a filename: " + cd
    name = m.group(1)
    assert name.lower().endswith(".txt")
    assert not re.search(r"[\\/:*?<>|]|\.\.", name)
    assert str(hid) in name or "hazard" in name.lower() or "report" in name.lower()


def test_text_download_is_utf8_and_has_no_paths(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    r, _ = text_download(env, hid)
    r.content.decode("utf-8")
    assert str(env.tmp) not in r.text and "zqblur" not in r.text and "zqorig" not in r.text


def test_text_download_of_a_non_approved_hazard_is_refused(rs18_env):
    env = rs18_env
    ok, _ = env.seed(name="ok")
    _r, label = text_download(env, ok)
    bad, _ = env.seed(name="bad", status="new")
    rp = env.report_path()
    for lab, url, params in text_candidates(env, bad):
        if lab == label:
            r = env.client.get(url, params=params)
            assert r.status_code in (404, 409, 422), "%s on a 'new' hazard -> %s" % (lab, r.status_code)
            assert "attachment" not in r.headers.get("content-disposition", "").lower()
            return
    pytest.skip("variant not reproducible")


def test_text_download_unknown_hazard(rs18_env):
    env = rs18_env
    ok, _ = env.seed(name="ok")
    _r, label = text_download(env, ok)
    for lab, url, params in text_candidates(env, 777777):
        if lab == label:
            assert env.client.get(url, params=params).status_code == 404
            return


def test_json_report_is_not_an_attachment(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    r = env.get_report(hid)
    assert r.headers["content-type"].split(";")[0].strip() == "application/json"
    assert "attachment" not in r.headers.get("content-disposition", "").lower()


# ------------------------------------------------------------------ photo

def test_photo_download_is_the_blurred_file_as_attachment(rs18_env):
    env = rs18_env
    hid, info = env.seed()
    r, _ = photo_download(env, hid)
    assert r.content == info["blur"].read_bytes()
    assert r.content != info["orig"].read_bytes()
    assert r.headers["content-type"].split(";")[0].strip().lower() == "image/jpeg"
    cd = r.headers["content-disposition"]
    m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", cd, re.I)
    assert m and m.group(1).lower().endswith((".jpg", ".jpeg")), cd
    assert not re.search(r"[\\/:]|zqblur|zqorig", m.group(1)), "filename must not leak storage names: " + m.group(1)


def test_plain_photo_route_is_still_inline(rs18_env):
    env = rs18_env
    hid, info = env.seed()
    r = env.client.get(env.sub(env.photo_path(), hid))
    assert r.status_code == 200 and r.content == info["blur"].read_bytes()
    assert "attachment" not in r.headers.get("content-disposition", "").lower()


def test_photo_download_without_photo_is_404(rs18_env):
    env = rs18_env
    ok, _ = env.seed(name="ok")
    _r, label = photo_download(env, ok)
    nop, _ = env.seed(name="nop", with_photo=False)
    for lab, url, params in photo_candidates(env, nop):
        if lab == label:
            r = env.client.get(url, params=params)
            assert r.status_code == 404
            return


def test_photo_download_never_serves_the_original(rs18_env):
    env = rs18_env
    hid, info = env.seed()
    orig = info["orig"].read_bytes()
    for lab, url, params in photo_candidates(env, hid):
        r = env.client.get(url, params=params)
        assert r.content != orig, lab
    # the blurred photo is gone: nothing else may be served instead
    info["blur"].unlink()
    for lab, url, params in photo_candidates(env, hid):
        r = env.client.get(url, params=params)
        assert r.content != orig, lab
        assert r.status_code < 500, lab
