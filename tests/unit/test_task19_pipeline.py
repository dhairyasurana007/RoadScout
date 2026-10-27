"""Task 19: the pipeline runs the possibly-fixed check (fake provider, 24 s synthetic clip)."""
from __future__ import annotations

import pytest

from rs19_helpers import (BASE, M_PER_DEG, OLD_KEYS, config_env, offset, rs19_env, run_pipeline_for,  # noqa: F401
                          upload)

ORIGINAL_STAGES = ["preprocessing", "detecting", "aggregating", "verifying", "blurring", "deduplicating"]


def seed_hazard(env, status="approved"):
    """A hazard 20 m east of the clip's northbound track, 120 m from its start (video second 10), seen only by an
    old drive; the pipeline's own candidates are kept >5 m away from it by a tiny dedup distance."""
    radius_env = config_env("radius")
    if radius_env:       # preferred: wide pass radius, hazard 40 m away, default dedup distance (25 m)
        env.mp.setenv(radius_env, "60")
        east = 40.0
    else:                # fallback: 20 m away and a 5 m dedup distance
        env.mp.setenv("ROADSCOUT_DEDUP_DISTANCE_M", "5")
        east = 20.0
    lat, lon = offset(38.88 + 120.0 / M_PER_DEG, -77.30, east_m=east)
    return env.make_hazard(lat, lon, heading=0.0, status=status)


def test_stages_keep_the_old_order_and_add_at_most_the_check():
    import backend.pipeline as p

    stages = list(p.STAGES)
    assert stages[:len(ORIGINAL_STAGES)] == ORIGINAL_STAGES, stages
    extra = stages[len(ORIGINAL_STAGES):]
    assert len(extra) <= 1, "more than one new stage: %r" % (extra,)
    if extra:
        assert "fix" in extra[0].lower(), "new stage should be named after the fix check: %r" % (extra,)


def test_pipeline_flags_a_hazard_the_new_drive_passed_without_seeing(rs19_env):
    env = rs19_env
    name = config_env("misses")
    if name is None:
        pytest.fail("no ROADSCOUT_* env variable for the miss threshold found in backend/*.py")
    env.mp.setenv(name, "1")
    hid = seed_hazard(env)
    did = upload(env.client)
    res = run_pipeline_for(did, env.db)
    assert res.status == "done", getattr(res, "error", res)
    drive = env.client.get("/api/drives/%s" % did).json()
    assert drive["status"] == "done" and drive["stage"] == "done"
    assert env.status(hid) == "possibly_fixed", (
        "the drive that was just processed (still 'processing' while its stages run) must count as a later drive")
    assert env.misses(hid) == 1
    assert env.flagged_at(hid) is not None


def test_processing_the_same_drive_twice_counts_it_once(rs19_env):
    env = rs19_env
    name = config_env("misses")
    if name is None:
        pytest.fail("no ROADSCOUT_* env variable for the miss threshold found in backend/*.py")
    env.mp.setenv(name, "2")
    hid = seed_hazard(env)
    did = upload(env.client)
    assert run_pipeline_for(did, env.db).status == "done"
    assert env.misses(hid) == 1
    assert env.status(hid) == "approved"
    assert run_pipeline_for(did, env.db).status == "done"
    assert env.misses(hid) == 1, "re-processing a drive counted it a second time"
    assert env.status(hid) == "approved"


def test_default_threshold_needs_three_drives_not_one(rs19_env):
    env = rs19_env
    hid = seed_hazard(env, status="new")
    did = upload(env.client)
    assert run_pipeline_for(did, env.db).status == "done"
    assert env.status(hid) == "new"
    assert env.misses(hid) == 1


def test_hazard_json_after_a_run_keeps_all_keys(rs19_env):
    env = rs19_env
    seed_hazard(env)
    did = upload(env.client)
    assert run_pipeline_for(did, env.db).status == "done"
    items = env.hazards()
    assert items
    for it in items:
        for k in OLD_KEYS:
            assert k in it, k
        assert any("miss" in k.lower() for k in it)
        assert any("fixed" in k.lower() and k != "status" for k in it)
    assert len({frozenset(i) for i in items}) == 1


def test_a_failed_drive_does_not_count(rs19_env, monkeypatch):
    env = rs19_env
    name = config_env("misses")
    if name is None:
        pytest.fail("no ROADSCOUT_* env variable for the miss threshold found in backend/*.py")
    env.mp.setenv(name, "1")
    hid = seed_hazard(env)
    did = upload(env.client)
    import backend.pipeline as p

    def boom(*a, **k):
        raise RuntimeError("detection exploded")

    monkeypatch.setattr(p, "run_detection", boom)
    res = run_pipeline_for(did, env.db)
    assert res.status == "failed"
    assert env.status(hid) == "approved"
    assert env.misses(hid) == 0
