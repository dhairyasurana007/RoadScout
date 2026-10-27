"""Task 17 backend: the review/update endpoint (approve, reject, edit class or severity)."""
import pytest

from rs17_helpers import (LIST_KEYS, UNTOUCHED, SEED_TS, known_classes, parse_time, rs17_env,  # noqa: F401
                          unwrap)
from datetime import datetime, timedelta, timezone

CLASSES = known_classes()


def test_an_update_route_exists(rs17_env):
    env = rs17_env
    assert env.generic_route() is not None or (env.action_route("approve") and env.action_route("reject")), \
        "no update route found (expected PATCH /api/hazards/{id})"


def test_known_classes_are_the_five_config_classes():
    assert len(CLASSES) == 5 and "pothole" in CLASSES


@pytest.mark.parametrize("status", ["approved", "rejected"])
def test_set_status_returns_updated_hazard(rs17_env, status):
    env = rs17_env
    hid = env.add_hazard()
    body = env.ok(hid, status=status)
    assert int(body["id"]) == hid
    assert body["status"] == status
    for k in LIST_KEYS:
        assert k in body, "response lacks %r (same shape as a list item expected)" % k


def test_response_matches_the_list_item(rs17_env):
    env = rs17_env
    hid = env.add_hazard()
    other = env.add_hazard(lat=38.90)
    body = env.ok(hid, status="approved", severity=3, hazard_class="faded lane markings")
    item = env.by_id()[hid]
    for k in LIST_KEYS:
        assert body[k] == item[k], "response/list mismatch on %r: %r vs %r" % (k, body[k], item[k])
    assert set(item) == set(body), "response keys differ from list item keys: %s vs %s" % (sorted(body), sorted(item))
    assert env.by_id()[other]["status"] == "new"


def test_status_persists_in_sqlite(rs17_env):
    env = rs17_env
    hid = env.add_hazard()
    env.ok(hid, status="approved")
    assert env.row(hid)["status"] == "approved"
    env.ok(hid, status="rejected")
    assert env.row(hid)["status"] == "rejected"


def test_back_to_new_and_round_trips(rs17_env):
    env = rs17_env
    hid = env.add_hazard()
    for st in ("approved", "rejected", "approved", "new", "rejected", "new"):
        body = env.ok(hid, status=st)
        assert body["status"] == st
        assert env.row(hid)["status"] == st
        assert env.by_id()[hid]["status"] == st


def test_same_status_twice_is_fine(rs17_env):
    env = rs17_env
    hid = env.add_hazard()
    env.ok(hid, status="approved")
    assert env.ok(hid, status="approved")["status"] == "approved"


def test_list_reflects_status_and_status_filter(rs17_env):
    env = rs17_env
    a, b, c = env.add_hazard(lat=38.80), env.add_hazard(lat=38.81), env.add_hazard(lat=38.82)
    env.ok(a, status="approved")
    env.ok(b, status="rejected")
    items = env.by_id()
    assert (items[a]["status"], items[b]["status"], items[c]["status"]) == ("approved", "rejected", "new")
    if "status" not in env.list_params():
        pytest.skip("GET /api/hazards has no status filter")
    for st, want in (("approved", {a}), ("rejected", {b}), ("new", {c})):
        got = {int(i["id"]) for i in env.hazards(status=st)}
        assert got == want, "status=%s -> %s" % (st, got)


@pytest.mark.parametrize("cls", CLASSES)
def test_every_known_class_is_accepted(rs17_env, cls):
    env = rs17_env
    hid = env.add_hazard(hazard_class="pothole" if cls != "pothole" else "faded lane markings")
    body = env.ok(hid, hazard_class=cls)
    assert body["hazard_class"] == cls
    assert env.row(hid)["hazard_class"] == cls
    assert env.by_id()[hid]["hazard_class"] == cls


@pytest.mark.parametrize("sev", [1, 2, 3])
def test_severity_values_accepted(rs17_env, sev):
    env = rs17_env
    hid = env.add_hazard(severity=1 if sev != 1 else 3)
    body = env.ok(hid, severity=sev)
    assert body["severity"] == sev
    assert env.row(hid)["severity"] == sev
    assert env.by_id()[hid]["severity"] == sev


def test_edit_does_not_change_status(rs17_env):
    env = rs17_env
    hid = env.add_hazard(status="approved")
    env.ok(hid, severity=3)
    env.ok(hid, hazard_class="damaged guardrail")
    assert env.row(hid)["status"] == "approved"


def test_status_does_not_change_edit_fields(rs17_env):
    env = rs17_env
    hid = env.add_hazard(hazard_class="pothole", severity=2)
    env.ok(hid, status="rejected")
    r = env.row(hid)
    assert (r["hazard_class"], r["severity"]) == ("pothole", 2)


def test_combined_update(rs17_env):
    env = rs17_env
    hid = env.add_hazard(hazard_class="pothole", severity=1)
    body = env.ok(hid, status="approved", hazard_class="fallen or damaged sign", severity=3)
    r = env.row(hid)
    assert (r["status"], r["hazard_class"], r["severity"]) == ("approved", "fallen or damaged sign", 3)
    assert (body["status"], body["hazard_class"], body["severity"]) == ("approved", "fallen or damaged sign", 3)


def test_unchanged_fields_stay_untouched(rs17_env):
    env = rs17_env
    hid = env.add_hazard(direction=None, photo_path="x/p.jpg", crop_path="x/c.jpg", confidence=0.77,
                         sightings=4, candidate_count=5)
    before = env.row(hid)
    env.ok(hid, status="approved")
    env.ok(hid, severity=3)
    env.ok(hid, hazard_class="faded lane markings")
    after = env.row(hid)
    for k in UNTOUCHED:
        assert after[k] == before[k], "%s changed: %r -> %r" % (k, before[k], after[k])
    assert after["id"] == hid


def test_updated_at_changes_and_created_at_does_not(rs17_env):
    env = rs17_env
    hid = env.add_hazard()
    for fields in ({"status": "approved"}, {"severity": 3}, {"hazard_class": "damaged guardrail"}):
        prev = env.row(hid)["updated_at"]
        env.ok(hid, **fields)
        r = env.row(hid)
        assert r["updated_at"] != prev, "updated_at not refreshed by %s" % (fields,)
        assert parse_time(r["updated_at"]) >= parse_time(prev)
        assert r["created_at"] == SEED_TS
    now = datetime.now(timezone.utc)
    assert abs(parse_time(env.row(hid)["updated_at"]) - now) < timedelta(hours=24)


def test_review_decision_is_recorded(rs17_env):
    env = rs17_env
    cols = env.review_cols()
    assert cols, "hazards table has no review-time column (name containing 'review', e.g. reviewed_at); columns=%s" \
        % env.columns()
    hid = env.add_hazard()
    other = env.add_hazard(lat=38.9)
    for c in cols:
        assert env.row(hid)[c] in (None, 0, ""), "%s is set before any review" % c
    for st in ("approved", "rejected"):
        for h in (hid,):
            env.x("UPDATE hazards SET %s = NULL WHERE id = ?" % cols[0], (h,))
        env.ok(hid, status=st)
        val = env.row(hid)[cols[0]]
        assert val not in (None, "", 0), "%s not recorded after %s" % (cols[0], st)
        if isinstance(val, str):
            assert abs(parse_time(val) - datetime.now(timezone.utc)) < timedelta(hours=24)
    assert env.row(other)[cols[0]] in (None, 0, ""), "review time leaked onto another hazard"


def test_list_still_works_after_updates_and_keeps_all_hazards(rs17_env):
    env = rs17_env
    ids = [env.add_hazard(lat=38.8 + i * 0.01) for i in range(4)]
    env.ok(ids[0], status="approved")
    env.ok(ids[1], status="rejected")
    got = env.by_id()
    assert set(got) == set(ids)
    assert got[ids[2]]["status"] == got[ids[3]]["status"] == "new"


def test_other_hazards_and_photos_untouched(rs17_env):
    env = rs17_env
    trio = env.trio()
    before_rows = {h: env.row(h) for h, _ in trio}
    before_bytes = {p: p.read_bytes() for _, p in trio}
    target = trio[0][0]
    env.ok(target, status="approved", severity=3, hazard_class="faded lane markings")
    env.ok(target, status="rejected")
    for h, _ in trio[1:]:
        assert env.row(h) == before_rows[h], "hazard %s was modified" % h
    for _, p in trio:
        assert p.read_bytes() == before_bytes[p]
    assert env.row(target)["photo_path"] == before_rows[target]["photo_path"]
    assert len(env.rows()) == 3


def test_update_via_http_never_changes_photo_endpoint(rs17_env):
    env = rs17_env
    (hid, p), *_ = env.trio()
    items = env.by_id()
    url_before = items[hid].get("photo_url")
    env.ok(hid, status="approved")
    assert env.by_id()[hid].get("photo_url") == url_before
