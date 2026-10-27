"""Task 19: edge cases specific to the coder's documented rules."""
from __future__ import annotations

import pytest

from rs19_helpers import HAZARD_AT, day, find_check, rs19_env  # noqa: F401

HL, HO = HAZARD_AT


def test_unknown_direction_does_not_restrict_the_heading(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    env.x("UPDATE hazards SET direction = NULL WHERE id = ?", (hid,))
    env.miss_drives(hid, 1, heading=180.0)
    env.miss_drives(hid, 1, first=2, heading=40.0)
    env.check()
    assert env.misses(hid) == 2


def test_a_hand_set_possibly_fixed_status_is_left_alone(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    env.x("UPDATE hazards SET status = 'possibly_fixed' WHERE id = ?", (hid,))
    env.miss_drives(hid, 1)
    env.sight(hid, "back", day(2))       # even a sighting must not restore a status the system did not set
    env.check()
    assert env.status(hid) == "possibly_fixed"


def test_include_drive_ids_counts_a_processing_drive(rs19_env):
    env = rs19_env
    f, name = find_check()
    if f is None or "include_drive_ids" not in f.__code__.co_varnames:
        pytest.skip("no include_drive_ids parameter")
    hid = env.make_hazard()
    env.configure(misses=1)
    env.add_pass_drive("busy", day(1), HL, HO, 0.0, status="processing")
    env.check()
    assert env.misses(hid) == 0
    f(db_path=str(env.db), min_misses=1, include_drive_ids=["busy"])
    assert env.status(hid) == "possibly_fixed"
    assert env.misses(hid) == 1


def test_restore_when_previous_status_is_unknown_goes_to_new(rs19_env):
    env = rs19_env
    hid = env.make_hazard(status="approved")
    env.miss_drives(hid, 3)
    env.check()
    assert env.status(hid) == "possibly_fixed"
    env.x("UPDATE hazards SET previous_status = NULL WHERE id = ?", (hid,))
    env.sight(hid, "back", day(4))
    env.check()
    assert env.status(hid) == "new"


def test_dismissal_columns_after_manual_change(rs19_env):
    env = rs19_env
    hid = env.make_hazard(status="approved")
    env.miss_drives(hid, 3)
    env.check()
    env.patch_ok(hid, status="rejected")
    row = env.row(hid)
    assert row["status"] == "rejected"
    assert row["previous_status"] is None and row["possibly_fixed_at"] is None and row["fix_misses"] == 0
    assert row["fix_check_dismissed_at"]
    item = env.item(hid)
    assert item["misses"] == 0 and item["possibly_fixed_at"] is None and item["previous_status"] is None
    assert "fix_check_dismissed_at" not in item


def test_json_exposes_previous_status_while_flagged(rs19_env):
    env = rs19_env
    hid = env.make_hazard(status="reported")
    env.miss_drives(hid, 3)
    env.check()
    item = env.item(hid)
    assert item["previous_status"] == "reported" and item["misses"] == 3
    assert item["possibly_fixed_at"]


def test_dedup_rerun_keeps_the_fix_bookkeeping(rs19_env):
    env = rs19_env
    hid = env.make_hazard(status="approved")
    env.miss_drives(hid, 3)
    env.check()
    before = {k: env.row(hid)[k] for k in ("status", "previous_status", "possibly_fixed_at", "fix_misses")}
    env.dedup("s00")
    env.dedup("s00")
    after = {k: env.row(hid)[k] for k in before}
    assert after == before
