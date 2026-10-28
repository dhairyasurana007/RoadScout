"""Task 9: malformed provider output and provider errors are clear, and leave no partial row."""
from __future__ import annotations

import pytest

from rs9_helpers import class_names, cls_of, corrupt, make_verdict, rs9_env, stub  # noqa: F401

BAD_ID = 9102  # the middle candidate (added with base_id 9101)


def bad_verdicts():
    names = class_names()

    def v(**kw):
        return lambda c, cp, pv: corrupt(make_verdict(True, names[0], 2, "left", 0.9), **kw)

    return {
        "none": lambda c, cp, pv: None,
        "plain_string": lambda c, cp, pv: "not json at all",
        "dict_missing_fields": lambda c, cp, pv: {"is_hazard": True},
        "hazard_severity_zero": v(severity=0),
        "severity_four": v(severity=4),
        "severity_negative": v(severity=-1),
        "severity_text": v(severity="high"),
        "confidence_above_one": v(confidence=1.5),
        "confidence_negative": v(confidence=-0.1),
        "confidence_nan": v(confidence=float("nan")),
        "confidence_text": v(confidence="very sure"),
        "unknown_class": v(hazard_class="unicorn sighting"),
        "blank_class": v(hazard_class=""),
        "non_hazard_severity_nine": lambda c, cp, pv: corrupt(
            make_verdict(False, names[0], 0, None, 0.3), severity=9),
    }


@pytest.mark.parametrize("name", sorted(bad_verdicts()))
def test_malformed_output_names_the_candidate_and_stores_no_row_for_it(rs9_env, name):
    e = rs9_env
    cands = e.add_candidates(3)
    p = stub(e, script={1: bad_verdicts()[name]})
    with pytest.raises(Exception) as ei:
        e.run(provider=p)
    assert str(BAD_ID) in str(ei.value), "error must name the candidate id: %s" % ei.value
    ids = {r["candidate_id"] for r in e.rows()}
    assert BAD_ID not in ids
    for r in e.rows():  # nothing garbage was stored for the others either
        assert r["severity"] in (0, 1, 2, 3)
        assert 0.0 <= r["confidence"] <= 1.0
        assert cls_of(r) in class_names()


def test_bad_candidate_is_not_silently_stored_as_rejected(rs9_env):
    """A malformed verdict is an error, not a quiet rejected row."""
    e = rs9_env
    e.add_candidates(3)
    p = stub(e, script={1: bad_verdicts()["severity_four"]})
    with pytest.raises(Exception):
        e.run(provider=p)
    assert BAD_ID not in {r["candidate_id"] for r in e.rows()}


# Task 26b: RetriesExhausted and ProviderTimeoutError (a transient failure that outlived the retries) no longer
# abort the stage: that candidate is skipped (see test_task26b_verification.py). Everything else still aborts.
ERRORS = ["ProviderError", "SpendCapExceeded", "TransientProviderError", "RuntimeError", "ValueError"]


def make_error(kind):
    import backend.providers as bp

    msg = "boom-%s-unique" % kind
    if kind == "RetriesExhausted":
        return bp.RetriesExhausted(msg, 3), msg
    if kind in ("RuntimeError", "ValueError"):
        return {"RuntimeError": RuntimeError, "ValueError": ValueError}[kind](msg), msg
    return getattr(bp, kind)(msg), msg


@pytest.mark.parametrize("kind", ERRORS)
def test_provider_error_surfaces_with_candidate_id_and_no_row(rs9_env, kind):
    e = rs9_env
    e.add_candidates(3)
    err, msg = make_error(kind)
    p = stub(e, script={1: err})
    with pytest.raises(Exception) as ei:
        e.run(provider=p)
    exc = ei.value
    assert str(BAD_ID) in str(exc), "error must name the candidate id: %s" % exc
    chain = []
    cur = exc
    while cur is not None and cur not in chain:
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__
    assert any(c is err for c in chain) or msg in str(exc), "the original provider error must stay visible"
    assert BAD_ID not in {r["candidate_id"] for r in e.rows()}


def test_error_on_the_first_candidate_leaves_nothing(rs9_env):
    e = rs9_env
    e.add_candidates(3)
    p = stub(e, script={0: RuntimeError("first one fails")})
    with pytest.raises(Exception) as ei:
        e.run(provider=p)
    assert "9101" in str(ei.value)
    assert 9101 not in {r["candidate_id"] for r in e.rows()}


def test_failed_rerun_never_leaves_duplicates_or_garbage(rs9_env):
    e = rs9_env
    e.add_candidates(3)
    e.run(provider=stub(e))
    before = {r["candidate_id"]: r for r in e.rows()}
    assert len(before) == 3
    p = stub(e, script={1: bad_verdicts()["confidence_above_one"]})
    with pytest.raises(Exception) as ei:
        e.run(provider=p)
    assert str(BAD_ID) in str(ei.value)
    after = e.rows()
    ids = [r["candidate_id"] for r in after]
    assert len(ids) == len(set(ids)), "no duplicate verdict rows after a failed re-run"
    for r in after:
        assert 0.0 <= r["confidence"] <= 1.0
        # whatever survives is a complete earlier verdict (the old one), never a half-written one
        assert r == before[r["candidate_id"]]


def test_a_later_good_run_recovers_after_an_error(rs9_env):
    e = rs9_env
    e.add_candidates(3)
    with pytest.raises(Exception):
        e.run(provider=stub(e, script={1: RuntimeError("temporary")}))
    e.run(provider=stub(e))
    assert sorted(r["candidate_id"] for r in e.rows()) == [9101, 9102, 9103]
    assert len(e.raw_all()) == 3
