"""Task 11 backend: live drive status as the results page polls it."""
from __future__ import annotations

import threading
import time

import pytest

from rs11_helpers import (DONE, FAILED, make_hook_provider, patch_everywhere, rs11_done, rs11_env,  # noqa: F401
                          status_of)


def _use(env, provider):
    patch_everywhere(env.mp, "backend.providers", "get_provider", lambda *a, **k: provider)


def test_status_fields_after_upload(rs11_env):
    env = rs11_env
    did = env.upload()
    d = env.get(did)
    assert d["id"] == did
    assert status_of(d) == "uploaded"
    assert not d.get("error")
    for k in ("stage", "error", "failed_stage"):
        assert k in d, "drive JSON must carry %r (null allowed): %s" % (k, sorted(d))


def test_status_fields_after_done(rs11_done):
    env, did = rs11_done
    d = env.get(did)
    assert status_of(d) in DONE
    assert not d.get("error")
    assert not d.get("failed_stage")
    assert "stage" in d


def test_live_status_visible_while_processing(rs11_env):
    env = rs11_env

    def slow(n):
        time.sleep(0.6)

    _use(env, make_hook_provider(on_detect=slow))
    did = env.upload()
    # TestClient runs background tasks inside the POST, so start the job from a thread and poll meanwhile.
    t = threading.Thread(target=env.start, args=(did,), daemon=True)
    t.start()
    seen, stages, list_codes = [], [], []
    deadline = time.time() + 120
    while time.time() < deadline:
        d = env.get(did)
        seen.append(status_of(d))
        if d.get("stage"):
            stages.append(d["stage"])
        if status_of(d) not in DONE | FAILED:
            r = env.hazards_response(did)  # results page polls this too; must never 5xx mid-run
            list_codes.append(r.status_code)
            assert r.status_code == 200, r.text
        else:
            break
        time.sleep(0.05)
    t.join(timeout=60)
    assert seen[-1] in DONE
    assert any(s not in DONE | FAILED | {"uploaded"} for s in seen), "never saw an in-progress status: %s" % set(seen)
    assert stages, "stage never reported while processing"
    assert list_codes


def test_drives_list_carries_live_status(rs11_env):
    env = rs11_env
    did = env.upload_and_process()
    r = env.client.get("/api/drives")
    assert r.status_code == 200
    body = r.json()
    items = body if isinstance(body, list) else next(v for v in body.values() if isinstance(v, list))
    mine = [i for i in items if i["id"] == did]
    assert len(mine) == 1
    assert status_of(mine[0]) in DONE
    assert "stage" in mine[0] or "status" in mine[0]


def test_failed_drive_reports_error_and_stage(rs11_env):
    env = rs11_env
    _use(env, make_hook_provider(fail_detect_on=1))
    did = env.upload()
    env.start(did)
    d = env.wait(did)
    assert status_of(d) in FAILED
    assert d.get("error"), "failed drive must expose an error message"
    assert d.get("failed_stage"), "failed drive must expose failed_stage"
    # the results list stays usable after a failure
    r = env.hazards_response(did)
    assert r.status_code == 200
    assert isinstance(env.hazards(did), list)


def test_hazards_available_right_after_done(rs11_env):
    """When status flips to done the verified hazards must already be listed (no second wait for the page)."""
    env = rs11_env
    did = env.upload()
    env.start(did)
    d = env.wait(did)
    assert status_of(d) in DONE
    assert env.hazards(did), "done drive should list its verified hazards immediately"
