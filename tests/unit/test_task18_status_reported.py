"""Task 18: the user can mark an approved hazard as 'reported'; possibly_fixed stays refused."""
import warnings

import pytest

from rs18_helpers import rs18_env  # noqa: F401


def test_approved_to_reported(rs18_env):
    env = rs18_env
    hid, _ = env.seed(status="approved")
    r = env.set_status(hid, "reported")
    assert r.status_code == 200, "%s %s" % (r.status_code, r.text[:200])
    body = r.json()
    body = body.get("hazard", body) if isinstance(body, dict) and "id" not in body else body
    assert body["status"] == "reported"
    assert env.row(hid)["status"] == "reported"
    assert {int(h["id"]): h for h in env.hazards()}[hid]["status"] == "reported"


def test_reported_keeps_review_marker_and_other_fields(rs18_env):
    env = rs18_env
    hid, _ = env.seed(status="new")
    assert env.set_status(hid, "approved").status_code == 200
    before = env.row(hid)
    assert env.set_status(hid, "reported").status_code == 200
    after = env.row(hid)
    review_cols = [c for c in after if "review" in c.lower()]
    for c in review_cols:
        assert after[c] is not None, "%s lost after marking reported" % c
    for k in ("hazard_class", "severity", "lat", "lon", "sightings", "photo_path"):
        assert after[k] == before[k]


def test_reported_twice_is_fine(rs18_env):
    env = rs18_env
    hid, _ = env.seed(status="approved")
    assert env.set_status(hid, "reported").status_code == 200
    assert env.set_status(hid, "reported").status_code == 200
    assert env.row(hid)["status"] == "reported"


@pytest.mark.parametrize("start", ["new", "rejected"])
def test_other_statuses_to_reported_is_consistent(rs18_env, start):
    env = rs18_env
    hid, _ = env.seed(status=start)
    r = env.set_status(hid, "reported")
    assert r.status_code in (200, 409, 422), r.status_code
    if r.status_code == 200:
        warnings.warn("a %r hazard may be marked reported directly" % start)
        assert env.row(hid)["status"] == "reported"
    else:
        warnings.warn("a %r hazard cannot be marked reported (%s)" % (start, r.status_code))
        assert env.row(hid)["status"] == start


def test_reported_can_go_back_or_is_clearly_refused(rs18_env):
    env = rs18_env
    hid, _ = env.seed(status="reported")
    r = env.set_status(hid, "approved")
    assert r.status_code in (200, 409, 422)
    if r.status_code == 200:
        assert env.row(hid)["status"] == "approved"
    else:
        assert env.row(hid)["status"] == "reported"


@pytest.mark.parametrize("start", ["new", "approved", "rejected", "reported"])
def test_possibly_fixed_stays_refused(rs18_env, start):
    env = rs18_env
    hid, _ = env.seed(status=start)
    r = env.set_status(hid, "possibly_fixed")
    assert r.status_code in (400, 409, 422), r.status_code
    assert env.row(hid)["status"] == start


@pytest.mark.parametrize("bad", ["banana", "", 3])
def test_invalid_statuses_still_422(rs18_env, bad):
    env = rs18_env
    hid, _ = env.seed(status="approved")
    r = env.set_status(hid, bad)
    assert r.status_code in (400, 422), (bad, r.status_code)
    assert env.row(hid)["status"] == "approved"


def test_class_and_severity_edits_on_a_reported_hazard_keep_status(rs18_env):
    env = rs18_env
    hid, _ = env.seed(status="reported")
    g = env.generic_route()
    r = getattr(env.client, g[0])(env.sub(g[1], hid), json={"severity": 1})
    assert r.status_code in (200, 409, 422)
    assert env.row(hid)["status"] == "reported"


def test_report_remains_reachable_after_reported_or_cleanly_refused(rs18_env):
    env = rs18_env
    hid, _ = env.seed(status="approved")
    assert env.set_status(hid, "reported").status_code == 200
    r = env.get_report(hid)
    assert r.status_code in (200, 409, 422)


def test_reported_survives_dedup_rerun(rs18_env):
    from rs18_helpers import T0, VA_PT, offset

    env = rs18_env
    env.add_drive("A", T0)
    env.add_cand("A", *offset(*VA_PT), 0.0)
    env.dedup("A")
    hid = env.rows()[0]["id"]
    assert env.set_status(hid, "approved").status_code == 200
    assert env.set_status(hid, "reported").status_code == 200
    env.dedup("A")
    rows = env.rows()
    assert len(rows) == 1 and rows[0]["id"] == hid and rows[0]["status"] == "reported"


def test_openapi_status_filter_accepts_reported(rs18_env):
    env = rs18_env
    hid, _ = env.seed(status="approved")
    env.set_status(hid, "reported")
    ops = env.paths()["/api/hazards"]["get"]
    if "status" not in {p["name"] for p in ops.get("parameters", [])}:
        pytest.skip("no status filter")
    r = env.client.get("/api/hazards", params={"status": "reported"})
    assert r.status_code == 200
    assert [int(h["id"]) for h in r.json()] == [hid]
