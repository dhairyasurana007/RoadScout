"""Task 11 backend: list of verified hazards of a drive."""
from __future__ import annotations

import pytest

from rs11_helpers import (CLASS_KEYS, LAT_KEYS, LON_KEYS, PHOTO_KEYS, REASON_KEYS, cid_of, first_key,  # noqa: F401
                          rs11_done, rs11_env, rs11_env_empty, rs11_env_fp, status_of, val)


def _ids(items):
    return sorted(str(cid_of(i)) for i in items)


def test_list_route_exists_and_returns_json_list(rs11_done):
    env, did = rs11_done
    r = env.hazards_response(did)
    assert r.status_code == 200
    assert "json" in r.headers["content-type"]
    items = env.hazards(did)
    assert isinstance(items, list)


def test_default_scenario_lists_exactly_the_accepted_hazards(rs11_done):
    env, did = rs11_done
    expected = env.accepted_candidates(did)
    assert expected, "default fake scenario should produce accepted hazards"
    items = env.hazards(did)
    assert _ids(items) == sorted(str(r["id"]) for r in expected)
    assert len({str(cid_of(i)) for i in items}) == len(items), "duplicate hazards in the list"


def test_item_fields_present_and_match_db(rs11_done):
    env, did = rs11_done
    by_id = {str(r["id"]): r for r in env.accepted_candidates(did)}
    for it in env.hazards(did):
        row = by_id[str(cid_of(it))]
        assert first_key(it, CLASS_KEYS), "no hazard class key in %s" % sorted(it)
        cls = val(it, CLASS_KEYS)
        assert isinstance(cls, str) and cls
        assert cls in (row["v_class"], row.get("hazard_class"))
        assert it["severity"] == row["v_severity"]
        assert isinstance(it["severity"], int) and 1 <= it["severity"] <= 3
        assert it["confidence"] == pytest.approx(row["v_confidence"])
        assert 0.0 <= it["confidence"] <= 1.0
        assert first_key(it, LAT_KEYS) and first_key(it, LON_KEYS), "lat/lon keys must be present"
        if row["lat"] is not None:
            assert val(it, LAT_KEYS) == pytest.approx(row["lat"])
            assert val(it, LON_KEYS) == pytest.approx(row["lon"])
        text = val(it, REASON_KEYS)
        assert isinstance(text, str) and text.strip()
        assert env.photo_url(it), "no photo URL in %s (expected one of %s)" % (sorted(it), PHOTO_KEYS)


def test_response_does_not_leak_server_paths(rs11_done):
    env, did = rs11_done
    r = env.hazards_response(did)
    body = r.text
    assert str(env.tmp) not in body
    assert str(env.tmp).replace("\\", "/") not in body
    assert "\\\\" not in body.replace("\\\\/", ""), "windows path separators in JSON"
    for it in env.hazards(did):
        for v in it.values():
            if isinstance(v, str) and v.startswith(("/api", "http")):
                continue
            assert not (isinstance(v, str) and (v.endswith(".jpg") or v.endswith(".jpeg"))), \
                "filesystem path leaked: %r" % (v,)


def test_rejected_verdict_not_listed(rs11_done):
    env, did = rs11_done
    accepted = env.accepted_candidates(did)
    assert len(accepted) >= 2
    victim = accepted[0]["id"]
    env.x("UPDATE verdicts SET accepted = 0 WHERE candidate_id = ?", (victim,))
    ids = _ids(env.hazards(did))
    assert str(victim) not in ids
    assert ids == sorted(str(r["id"]) for r in accepted[1:])


def test_non_hazard_verdict_not_listed(rs11_done):
    env, did = rs11_done
    accepted = env.accepted_candidates(did)
    assert len(accepted) >= 2
    victim = accepted[-1]["id"]
    env.x("UPDATE verdicts SET is_hazard = 0, accepted = 0 WHERE candidate_id = ?", (victim,))
    ids = _ids(env.hazards(did))
    assert str(victim) not in ids
    assert len(ids) == len(accepted) - 1


def test_all_rejected_gives_empty_list(rs11_done):
    env, did = rs11_done
    env.x("UPDATE verdicts SET accepted = 0")
    r = env.hazards_response(did)
    assert r.status_code == 200
    assert env.hazards(did) == []


def test_candidate_without_verdict_not_listed(rs11_done):
    env, did = rs11_done
    accepted = env.accepted_candidates(did)
    victim = accepted[0]["id"]
    env.x("DELETE FROM verdicts WHERE candidate_id = ?", (victim,))
    assert str(victim) not in _ids(env.hazards(did))


def test_false_positive_scenario_matches_db(rs11_env_fp):
    env = rs11_env_fp
    did = env.upload_and_process()
    expected = sorted(str(r["id"]) for r in env.accepted_candidates(did))
    assert _ids(env.hazards(did)) == expected


def test_empty_scenario_gives_empty_list(rs11_env_empty):
    env = rs11_env_empty
    did = env.upload_and_process()
    r = env.hazards_response(did)
    assert r.status_code == 200
    assert env.hazards(did) == []


def test_unprocessed_drive_gives_empty_list(rs11_env):
    env = rs11_env
    did = env.upload()
    assert status_of(env.get(did)) == "uploaded"
    r = env.hazards_response(did)
    assert r.status_code == 200
    assert env.hazards(did) == []


def test_unknown_drive_is_404(rs11_env):
    env = rs11_env
    for bad in ("does-not-exist", "00000000-0000-0000-0000-000000000000", "999999"):
        assert env.hazards_response(bad).status_code == 404, bad


def test_null_lat_lon_allowed_and_still_listed(rs11_done):
    env, did = rs11_done
    accepted = env.accepted_candidates(did)
    victim = accepted[0]["id"]
    try:
        env.x("UPDATE candidates SET lat = NULL, lon = NULL WHERE id = ?", (victim,))
    except Exception as e:  # noqa: BLE001
        pytest.skip("lat/lon not nullable: %s" % e)
    items = {str(cid_of(i)): i for i in env.hazards(did)}
    assert str(victim) in items
    it = items[str(victim)]
    assert val(it, LAT_KEYS) is None and val(it, LON_KEYS) is None


def test_drives_are_isolated(rs11_env):
    env = rs11_env
    a = env.upload_and_process()
    b = env.upload_and_process()
    ia, ib = _ids(env.hazards(a)), _ids(env.hazards(b))
    assert ia and ib
    assert not set(ia) & set(ib)
    assert ia == sorted(str(r["id"]) for r in env.accepted_candidates(a))
    assert ib == sorted(str(r["id"]) for r in env.accepted_candidates(b))


def test_list_is_read_only_and_repeatable(rs11_done):
    env, did = rs11_done
    first = env.hazards(did)
    counts = env.q("SELECT (SELECT COUNT(*) FROM candidates) c, (SELECT COUNT(*) FROM verdicts) v")
    second = env.hazards(did)
    assert first == second
    assert env.q("SELECT (SELECT COUNT(*) FROM candidates) c, (SELECT COUNT(*) FROM verdicts) v") == counts


def test_write_methods_not_allowed_on_list(rs11_done):
    env, did = rs11_done
    url = env.list_url(did)
    for m in ("post", "put", "delete", "patch"):
        r = getattr(env.client, m)(url)
        assert r.status_code in (404, 405), (m, r.status_code)
