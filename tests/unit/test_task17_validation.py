"""Task 17 backend: errors, validation, atomicity, concurrency, non-reviewable statuses."""
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest

from rs17_helpers import rs17_env, unwrap, known_classes  # noqa: F401

BAD = (400, 422)


def snap(env):
    return env.rows()


# ------------------------------------------------------------------ unknown hazards

def test_unknown_hazard_is_404(rs17_env):
    env = rs17_env
    env.add_hazard()
    r = env.update(99999, status="approved")
    assert r.status_code == 404, r.text[:200]
    assert len(env.rows()) == 1 and env.rows()[0]["status"] == "new"


@pytest.mark.parametrize("fields", [{"severity": 3}, {"hazard_class": "pothole"}])
def test_unknown_hazard_edits_are_404(rs17_env, fields):
    env = rs17_env
    assert env.update(424242, **fields).status_code == 404


def test_non_integer_id_is_rejected(rs17_env):
    env = rs17_env
    env.add_hazard()
    r = env.update("abc", status="approved")
    assert r.status_code in (404, 422), r.status_code
    assert env.rows()[0]["status"] == "new"


def test_negative_or_zero_id_is_404(rs17_env):
    env = rs17_env
    for bad in (0, -1):
        assert env.update(bad, status="approved").status_code in (404, 422)


def test_action_routes_unknown_hazard_404(rs17_env):
    env = rs17_env
    for name in ("approve", "reject"):
        route = env.action_route(name)
        if not route:
            continue
        r = getattr(env.client, route[0])(env.url(31337, route[1]))
        assert r.status_code == 404


# ------------------------------------------------------------------ invalid values change nothing

@pytest.mark.parametrize("body", [
    {"status": "banana"},
    {"status": ""},
    {"status": 7},
    {"status": ["approved"]},
    {"status": {"a": 1}},
    {"status": "APPROVE"},
    {"hazard_class": "banana"},
    {"hazard_class": ""},
    {"hazard_class": 5},
    {"hazard_class": ["pothole"]},
    {"hazard_class": "Potholes"},
    {"severity": 0},
    {"severity": 4},
    {"severity": -1},
    {"severity": 100},
    {"severity": 2.5},
    {"severity": "high"},
    {"severity": [2]},
    {"severity": {"v": 2}},
])
def test_invalid_value_is_rejected_and_nothing_changes(rs17_env, body):
    env = rs17_env
    hid = env.add_hazard(hazard_class="pothole", severity=2, status="new")
    other = env.add_hazard(lat=38.9)
    before = snap(env)
    r = env.send(hid, json=body)
    assert r.status_code in BAD, "%s -> %s %s" % (body, r.status_code, r.text[:200])
    assert snap(env) == before


@pytest.mark.parametrize("body", [
    {"lat": 1.0},
    {"lon": 1.0},
    {"id": 99},
    {"confidence": 0.1},
    {"sightings": 50},
    {"photo_path": "/etc/passwd"},
    {"crop_path": "x"},
    {"created_at": "2000-01-01"},
    {"updated_at": "2000-01-01"},
    {"direction": 12.0},
    {"nonsense": True},
    {"status": "approved", "lat": 1.0},
    {"status": "approved", "bogus": 1},
    {"severity": 3, "sightings": 99},
])
def test_unknown_or_forbidden_fields_are_rejected(rs17_env, body):
    env = rs17_env
    hid = env.add_hazard()
    before = snap(env)
    r = env.send(hid, json=body)
    assert r.status_code in BAD, "%s -> %s" % (body, r.status_code)
    assert snap(env) == before


def test_empty_body_is_rejected(rs17_env):
    env = rs17_env
    hid = env.add_hazard()
    before = snap(env)
    r = env.send(hid, json={})
    assert r.status_code in BAD, r.status_code
    assert snap(env) == before


@pytest.mark.parametrize("content", [b"", b"not json", b"[1, 2]", b'"approved"', b"123", b"null", b"{"])
def test_malformed_or_non_object_body_is_rejected(rs17_env, content):
    env = rs17_env
    hid = env.add_hazard()
    before = snap(env)
    r = env.send(hid, content=content)
    assert r.status_code in BAD, "%r -> %s" % (content, r.status_code)
    assert snap(env) == before


def test_no_body_at_all_is_rejected(rs17_env):
    env = rs17_env
    hid = env.add_hazard()
    before = snap(env)
    g = env.generic_route()
    if g is None:
        pytest.skip("no generic update route")
    r = getattr(env.client, g[0])(env.url(hid, g[1]))
    assert r.status_code in BAD
    assert snap(env) == before


def test_null_values_do_not_clobber(rs17_env):
    env = rs17_env
    hid = env.add_hazard(hazard_class="pothole", severity=2)
    before = snap(env)
    r = env.send(hid, json={"hazard_class": None, "severity": None, "status": None})
    assert r.status_code == 200 or r.status_code in BAD
    after = env.row(hid)
    assert (after["hazard_class"], after["severity"], after["status"]) == ("pothole", 2, "new")


def test_partially_invalid_update_is_atomic(rs17_env):
    env = rs17_env
    hid = env.add_hazard(hazard_class="pothole", severity=2, status="new")
    before = snap(env)
    for body in ({"status": "approved", "severity": 9},
                 {"status": "approved", "hazard_class": "banana"},
                 {"severity": 3, "hazard_class": "banana"},
                 {"status": "approved", "severity": 3, "hazard_class": 42}):
        r = env.send(hid, json=body)
        assert r.status_code in BAD, body
        assert snap(env) == before, "partial change after %s" % (body,)


def test_failed_updates_do_not_set_review_marker_or_updated_at(rs17_env):
    env = rs17_env
    hid = env.add_hazard()
    before = env.row(hid)
    env.send(hid, json={"status": "nope"})
    env.send(hid, json={})
    assert env.row(hid) == before


# ------------------------------------------------------------------ non-review statuses (either behaviour)

@pytest.mark.parametrize("status", ["reported", "possibly_fixed"])
def test_reported_and_possibly_fixed_via_review_endpoint_is_consistent(rs17_env, status):
    env = rs17_env
    hid = env.add_hazard()
    before = snap(env)
    r = env.send(hid, json={"status": status})
    if r.status_code == 200:
        warnings.warn("review endpoint ACCEPTS status %r" % status)
        assert unwrap(r.json())["status"] == status
        assert env.row(hid)["status"] == status
        # and the CHECK constraint still lets it back to a review status
        assert env.ok(hid, status="new")["status"] == "new"
    else:
        warnings.warn("review endpoint REJECTS status %r with %s" % (status, r.status_code))
        assert r.status_code in (400, 409, 422)
        assert snap(env) == before


@pytest.mark.parametrize("status", ["reported", "possibly_fixed"])
def test_class_or_severity_edit_on_reported_hazard_keeps_status(rs17_env, status):
    env = rs17_env
    hid = env.add_hazard(status=status)
    r = env.update(hid, severity=3)
    if r.status_code == 200:
        assert env.row(hid)["status"] == status
        assert env.row(hid)["severity"] == 3
    else:
        assert r.status_code in (400, 409, 422)
        assert env.row(hid)["status"] == status


# ------------------------------------------------------------------ concurrency

def test_quick_back_to_back_updates_leave_a_valid_row(rs17_env):
    env = rs17_env
    hid = env.add_hazard()
    for i in range(10):
        env.ok(hid, status=("approved", "rejected")[i % 2], severity=1 + i % 3)
    r = env.row(hid)
    assert (r["status"], r["severity"]) == ("rejected", 1)


def test_parallel_updates_leave_a_valid_consistent_row(rs17_env):
    from fastapi.testclient import TestClient

    from backend.main import app

    env = rs17_env
    hid = env.add_hazard()
    classes = known_classes()
    pairs = [("approved", 1, classes[0]), ("rejected", 3, classes[1]), ("approved", 2, classes[2]),
             ("new", 3, classes[3]), ("rejected", 1, classes[4]), ("approved", 3, classes[0]),
             ("rejected", 2, classes[1]), ("approved", 1, classes[2])]

    def work(p):
        c = TestClient(app)
        r = env.update(hid, client=c, status=p[0], severity=p[1], hazard_class=p[2])
        return r.status_code

    with ThreadPoolExecutor(max_workers=8) as ex:
        codes = list(ex.map(work, pairs))
    assert all(c == 200 for c in codes), codes
    r = env.row(hid)
    assert (r["status"], r["severity"], r["hazard_class"]) in set(pairs)
    assert len(env.rows()) == 1
    assert env.by_id()[hid]["status"] == r["status"]


def test_parallel_updates_on_different_hazards(rs17_env):
    from fastapi.testclient import TestClient

    from backend.main import app

    env = rs17_env
    ids = [env.add_hazard(lat=38.8 + i * 0.01) for i in range(8)]

    def work(i):
        r = env.update(ids[i], client=TestClient(app), status="approved" if i % 2 else "rejected", severity=1 + i % 3)
        return r.status_code

    with ThreadPoolExecutor(max_workers=8) as ex:
        codes = list(ex.map(work, range(8)))
    assert all(c == 200 for c in codes), codes
    for i, h in enumerate(ids):
        r = env.row(h)
        assert r["status"] == ("approved" if i % 2 else "rejected")
        assert r["severity"] == 1 + i % 3
