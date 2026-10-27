"""Task 19 safety: possibly_fixed is never set by a user, rejected is never touched, no paths in the JSON."""
from __future__ import annotations

import re

import pytest

from rs19_helpers import HAZARD_AT, OLD_KEYS, day, rs19_env  # noqa: F401

HL, HO = HAZARD_AT
RAW_PATH_KEYS = {"best_frame_path", "crop_path", "blurred_frame_path", "blurred_crop_path", "photo_path",
                 "video_path", "gpx_path", "path", "file", "filename"}


@pytest.mark.parametrize("start", ["new", "approved", "rejected", "reported"])
def test_patch_cannot_set_possibly_fixed(rs19_env, start):
    env = rs19_env
    hid = env.make_hazard(status=start)
    before = env.row(hid)
    r = env.patch(hid, status="possibly_fixed")
    assert r.status_code in (400, 409, 422), (r.status_code, r.text[:200])
    assert env.row(hid) == before


def test_patch_cannot_set_the_fix_bookkeeping(rs19_env):
    env = rs19_env
    hid = env.make_hazard(status="approved")
    before = env.row(hid)
    for body in ({"misses": 5}, {"possibly_fixed_at": "2026-01-01T00:00:00Z"}, {"previous_status": "new"},
                 {"fix_misses": 9}, {"status": "approved", "misses": 3}):
        r = env.patch(hid, **body)
        assert r.status_code == 422, (body, r.status_code, r.text[:200])
    assert env.row(hid) == before


def test_flagged_hazard_cannot_be_put_back_to_possibly_fixed_by_patch(rs19_env):
    env = rs19_env
    hid = env.make_hazard(status="approved")
    env.miss_drives(hid, 3)
    env.check()
    assert env.status(hid) == "possibly_fixed"
    r = env.patch(hid, status="possibly_fixed")
    assert r.status_code in (400, 409, 422)
    assert env.status(hid) == "possibly_fixed"


def test_rejected_row_is_byte_identical_after_many_checks(rs19_env):
    env = rs19_env
    hid = env.make_hazard(status="rejected")
    before = env.row(hid)
    env.miss_drives(hid, 6)
    for _ in range(3):
        env.check()
    assert env.row(hid) == before          # every column, including updated_at and the fix bookkeeping
    assert env.item(hid)["status"] == "rejected"


def test_rejected_hazard_that_is_seen_again_stays_rejected(rs19_env):
    env = rs19_env
    hid = env.make_hazard(status="rejected")
    env.sight(hid, "back", day(2))
    env.check()
    assert env.status(hid) == "rejected"


def test_no_filesystem_paths_in_any_hazard_json(rs19_env):
    env = rs19_env
    flagged = env.make_hazard(status="approved")
    env.make_hazard(*[HL + 0.02, HO], heading=0.0)
    env.miss_drives(flagged, 3)
    env.check()
    bodies = env.hazards() + [env.patch_ok(flagged, severity=3)]
    for it in bodies:
        assert not ({k.lower() for k in it} & RAW_PATH_KEYS), sorted(it)
        for k, v in it.items():
            if isinstance(v, str):
                assert not re.search(r"[A-Za-z]:[\\/]|^/(?:home|tmp|var|Users)|\\\\", v), (k, v)
                assert str(env.tmp) not in v, (k, v)
        for k in ("photo_url", "crop_url"):
            assert it[k] is None or it[k].startswith("/api/")


def test_check_makes_no_network_calls_and_needs_no_key(rs19_env):
    env = rs19_env
    hid = env.make_hazard()
    env.miss_drives(hid, 3)
    env.check()            # the fixture's socket guard fails on any non-loopback connection
    assert env.status(hid) == "possibly_fixed"
