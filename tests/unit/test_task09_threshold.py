"""Task 9: confidence threshold, acceptance flag, rejected verdicts still stored."""
from __future__ import annotations

import pytest

from rs9_helpers import THRESH_ENV, class_names, make_verdict, rs9_env, stub  # noqa: F401


def conf_script(values, **kw):
    """Hazard verdicts with the given confidences, by candidate index."""
    return {i: make_verdict(True, class_names()[0], 2, "center", v, **kw) for i, v in enumerate(values)}


def accepted_list(e, cands):
    acc = e.accepted()
    return [acc[c["id"]] for c in cands]


def test_default_threshold_accepts_confident_and_rejects_unsure(rs9_env):
    e = rs9_env
    cands = e.add_candidates(3)
    e.run(provider=stub(e, script=conf_script([0.95, 0.05, 0.99])))
    assert len(e.rows()) == 3, "rejected verdicts are still stored for the eval ablation"
    assert accepted_list(e, cands) == [True, False, True]


def test_threshold_boundary_is_inclusive(rs9_env):
    e = rs9_env
    e.set_threshold("0.6")
    cands = e.add_candidates(3)
    e.run(provider=stub(e, script=conf_script([0.6, 0.59, 0.61])))
    assert accepted_list(e, cands) == [True, False, True]
    assert len(e.rows()) == 3


def test_threshold_from_env_changes_the_outcome(rs9_env):
    e = rs9_env
    cands = e.add_candidates(2)
    script = conf_script([0.75, 0.45])
    e.set_threshold("0.9")
    e.run(provider=stub(e, script=script))
    assert accepted_list(e, cands) == [False, False]
    e.set_threshold("0.4")
    e.run(provider=stub(e, script=script))
    assert accepted_list(e, cands) == [True, True]
    assert len(e.rows()) == 2


def test_kwarg_overrides_env_threshold(rs9_env):
    e = rs9_env
    cands = e.add_candidates(2)
    script = conf_script([0.5, 0.95])
    e.set_threshold("0.9")
    e.run(provider=stub(e, script=script), min_confidence=0.2)
    assert accepted_list(e, cands) == [True, True]
    e.set_threshold("0.1")
    e.run(provider=stub(e, script=script), min_confidence=0.9)
    assert accepted_list(e, cands) == [False, True]


def test_threshold_zero_accepts_every_hazard_and_one_only_certain_ones(rs9_env):
    e = rs9_env
    cands = e.add_candidates(3)
    script = conf_script([0.0, 0.99, 1.0])
    e.set_threshold("0")
    e.run(provider=stub(e, script=script))
    assert accepted_list(e, cands) == [True, True, True]
    e.set_threshold("1")
    e.run(provider=stub(e, script=script))
    assert accepted_list(e, cands) == [False, False, True]


def test_non_hazard_verdict_is_never_accepted_but_is_stored(rs9_env):
    e = rs9_env
    cands = e.add_candidates(3)
    names = class_names()
    script = {
        0: make_verdict(False, names[0], 0, None, 0.99, "Shadow.", "No hazard."),
        1: make_verdict(False, names[1], 1, None, 1.0, "Patch.", "No hazard."),
        2: make_verdict(True, names[2], 2, "left", 0.99, "Real.", "Real hazard."),
    }
    e.set_threshold("0")
    e.run(provider=stub(e, script=script))
    got = e.by_candidate()
    assert len(got) == 3
    assert accepted_list(e, cands) == [False, False, True]
    assert not bool(got[cands[0]["id"]]["is_hazard"])
    assert got[cands[0]["id"]]["confidence"] == 0.99


@pytest.mark.parametrize("bad", ["abc", "1.5", "-0.1", "nan", "inf", "2"])
def test_invalid_threshold_env_is_a_value_error_before_any_call(rs9_env, bad):
    e = rs9_env
    e.add_candidates(2)
    e.set_threshold(bad)
    p = stub(e)
    with pytest.raises(ValueError):
        e.run(provider=p)
    assert p.calls == [] and e.rows() == []


@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan")])
def test_invalid_threshold_kwarg_is_a_value_error(rs9_env, bad):
    e = rs9_env
    e.add_candidates(2)
    p = stub(e)
    with pytest.raises(ValueError):
        e.run(provider=p, min_confidence=bad)
    assert p.calls == [] and e.rows() == []


def test_fake_default_scenario_is_accepted_under_the_default_threshold(rs9_env):
    """The fake default verdicts have confidence 0.80 to 0.99: the default threshold must be <= 0.8."""
    e = rs9_env
    cands = e.add_candidates(4)
    e.run()
    assert accepted_list(e, cands) == [True] * 4


def test_fake_false_positive_scenario_is_rejected_but_stored(rs9_env, monkeypatch):
    e = rs9_env
    cands = e.add_candidates(3)
    monkeypatch.setenv("ROADSCOUT_FAKE_SCENARIO", "false_positive")
    e.run()
    assert len(e.rows()) == 3
    assert accepted_list(e, cands) == [False] * 3
    assert all(not bool(r["is_hazard"]) for r in e.rows())


def test_acceptance_is_recomputed_on_rerun_with_a_new_threshold(rs9_env):
    e = rs9_env
    cands = e.add_candidates(2)
    script = conf_script([0.7, 0.3])
    e.set_threshold("0.5")
    e.run(provider=stub(e, script=script))
    assert accepted_list(e, cands) == [True, False]
    e.set_threshold("0.2")
    e.run(provider=stub(e, script=script))
    assert accepted_list(e, cands) == [True, True]
    assert len(e.rows()) == 2
