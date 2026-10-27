"""Task 19: hazard JSON (miss count, flag time) and the user dismissing a possibly_fixed flag with PATCH."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rs19_helpers import HAZARD_AT, OLD_KEYS, day, rs19_env  # noqa: F401

HL, HO = HAZARD_AT


def soon(k):
    """Start of a drive k days from now (after any dismissal done in the test)."""
    return datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=k)


def flag(env, status="approved"):
    hid = env.make_hazard(status=status)
    env.miss_drives(hid, 3)
    env.check()
    assert env.status(hid) == "possibly_fixed"
    return hid


# ------------------------------------------------------------------ JSON

def test_list_items_keep_every_old_key_and_add_two(rs19_env):
    env = rs19_env
    hid = env.make_hazard(status="approved")
    item = env.item(hid)
    for k in OLD_KEYS:
        assert k in item, "lost key %r" % k
    assert env.misses(hid) == 0
    assert env.flagged_at(hid) is None
    assert isinstance(item[next(k for k in item if "miss" in k.lower())], int)


def test_miss_count_is_exposed_before_the_hazard_is_flagged(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    env.miss_drives(hid, 2)
    env.check()
    item = env.item(hid)
    assert env.misses(hid) == 2
    assert not isinstance(env.misses(hid), bool)
    assert item["status"] == "new"
    assert env.flagged_at(hid) is None


def test_flagged_hazard_json(rs19_env):
    env = rs19_env
    hid = flag(env)
    item = env.item(hid)
    assert item["status"] == "possibly_fixed"
    assert env.misses(hid) >= 3
    at = env.flagged_at(hid)
    assert isinstance(at, str) and at
    datetime.fromisoformat(at.replace("Z", "+00:00"))
    for k in OLD_KEYS:
        assert k in item


def test_status_filter_finds_the_flagged_hazard(rs19_env):
    env = rs19_env
    hid = flag(env)
    other = env.make_hazard(*[HL + 0.02, HO], heading=0.0, status="approved")
    got = {i["id"] for i in env.hazards(status="possibly_fixed")}
    assert got == {hid}
    assert hid not in {i["id"] for i in env.hazards(status="approved")}
    assert other in {i["id"] for i in env.hazards(status="approved")}


def test_patch_response_and_list_item_have_identical_keys(rs19_env):
    env = rs19_env
    plain = env.make_hazard(*[HL + 0.02, HO], heading=0.0)
    hid = flag(env)
    for target in (plain, hid):
        body = env.patch_ok(target, severity=3)
        item = env.item(target)
        assert set(body) == set(item), (sorted(body), sorted(item))
        assert body["status"] == item["status"]


def test_every_item_has_the_same_key_set(rs19_env):
    env = rs19_env
    flag(env)
    env.make_hazard(*[HL + 0.02, HO], heading=0.0)
    env.make_hazard(*[HL + 0.04, HO], heading=0.0, status="rejected")
    keys = [set(i) for i in env.hazards()]
    assert len(keys) == 3 and keys[0] == keys[1] == keys[2]


# ------------------------------------------------------------------ the user dismisses the flag

@pytest.mark.parametrize("target", ["approved", "new"])
def test_manual_status_change_dismisses_the_flag(rs19_env, target):
    env = rs19_env
    hid = flag(env)
    body = env.patch_ok(hid, status=target)
    assert body["status"] == target
    env.check()
    assert env.status(hid) == target, "the same old passes flagged the hazard again straight away"
    assert env.misses(hid) == 0
    assert env.flagged_at(hid) is None
    env.check()
    assert env.status(hid) == target


@pytest.mark.parametrize("target", ["approved", "new"])
def test_new_misses_after_a_dismissal_can_flag_it_again(rs19_env, target):
    env = rs19_env
    hid = flag(env)
    env.patch_ok(hid, status=target)
    env.check()
    for k in (1, 2):
        env.add_pass_drive("after%d" % k, soon(k), env.row(hid)["lat"], env.row(hid)["lon"], 0.0)
    env.check()
    assert env.status(hid) == target
    assert env.misses(hid) == 2
    env.add_pass_drive("after3", soon(3), env.row(hid)["lat"], env.row(hid)["lon"], 0.0)
    env.check()
    assert env.status(hid) == "possibly_fixed"
    assert env.misses(hid) == 3
    assert env.flagged_at(hid) is not None


def test_dismissed_hazard_remembers_its_new_status_when_flagged_again(rs19_env):
    env = rs19_env
    hid = flag(env, status="approved")
    env.patch_ok(hid, status="new")
    env.check()
    for k in (1, 2, 3):
        env.add_pass_drive("again%d" % k, soon(k), env.row(hid)["lat"], env.row(hid)["lon"], 0.0)
    env.check()
    assert env.status(hid) == "possibly_fixed"
    row = env.row(hid)
    cols = env.prev_status_cols()
    assert any(row[c] == "new" for c in cols), {c: row[c] for c in cols}
    assert not any(row[c] == "possibly_fixed" for c in cols)


def test_rejecting_a_flagged_hazard_ends_the_story(rs19_env):
    env = rs19_env
    hid = flag(env)
    env.patch_ok(hid, status="rejected")
    for k in range(1, 6):
        env.add_pass_drive("rej%d" % k, soon(k), env.row(hid)["lat"], env.row(hid)["lon"], 0.0)
    env.check()
    assert env.status(hid) == "rejected"
    assert env.flagged_at(hid) is None


def test_editing_something_else_does_not_dismiss(rs19_env):
    env = rs19_env
    hid = flag(env)
    at = env.flagged_at(hid)
    env.patch_ok(hid, severity=3)
    env.check()
    assert env.status(hid) == "possibly_fixed"
    assert env.flagged_at(hid) == at
