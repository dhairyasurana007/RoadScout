"""Task 26b: verify_drive skips candidates whose verification failed transiently (network), aborts otherwise."""
from __future__ import annotations

import logging

import pytest

import rs26b_setup  # noqa: F401
from rs9_helpers import class_names, corrupt, make_verdict, rs9_env, stub  # noqa: F401

BASE = 9101


def exhausted(msg="boom-exhausted-unique"):
    from backend.providers import RetriesExhausted

    return RetriesExhausted(msg, 5)


def timed_out(msg="boom-timeout-unique"):
    from backend.providers import ProviderTimeoutError

    return ProviderTimeoutError(msg)


def cid(k):
    return BASE + k


def verification_error():
    from backend.verification import VerificationError

    return VerificationError


def fail_ks(*ks, kind=exhausted):
    return {k: kind() for k in ks}


# ------------------------------------------------------------------ skip and continue

@pytest.mark.parametrize("kind", [exhausted, timed_out], ids=["RetriesExhausted", "ProviderTimeoutError"])
def test_transiently_failed_candidate_is_skipped_and_the_rest_are_verified(rs9_env, kind):
    e = rs9_env
    e.add_candidates(4)
    p = stub(e, script=fail_ks(1, kind=kind))
    summary = e.run(provider=p)
    assert len(p.calls) == 4, "the loop must continue with the next candidate"
    got = e.by_candidate()
    assert sorted(got) == [cid(0), cid(2), cid(3)]
    assert summary.failed_candidate_ids == [cid(1)]
    assert summary.candidate_count == 3
    assert summary.accepted_count + summary.rejected_count == 3
    assert len(summary.verdicts) == 3
    assert sorted(v["candidate_id"] for v in summary.verdicts) == [cid(0), cid(2), cid(3)]


def test_skipped_candidate_is_never_accepted(rs9_env):
    from backend.verification import load_accepted_candidates, load_verdicts

    e = rs9_env
    e.add_candidates(4)
    e.run(provider=stub(e, script=fail_ks(2)))
    accepted_ids = [c["id"] for c in load_accepted_candidates("drive-a", db_path=str(e.db))]
    assert cid(2) not in accepted_ids
    assert sorted(accepted_ids) == [cid(0), cid(1), cid(3)]
    assert cid(2) not in {v["candidate_id"] for v in load_verdicts("drive-a", db_path=str(e.db))}
    assert cid(2) not in {r["candidate_id"] for r in e.raw_all()}


def test_failed_ids_are_in_candidate_order(rs9_env):
    e = rs9_env
    e.add_candidates(6)
    summary = e.run(provider=stub(e, script={4: exhausted(), 1: timed_out(), 5: exhausted()}))
    assert summary.failed_candidate_ids == [cid(1), cid(4), cid(5)]
    assert sorted(e.by_candidate()) == [cid(0), cid(2), cid(3)]


def test_skipping_the_first_and_the_last_candidate(rs9_env):
    e = rs9_env
    e.add_candidates(5)
    summary = e.run(provider=stub(e, script=fail_ks(0, 4)))
    assert summary.failed_candidate_ids == [cid(0), cid(4)]
    assert sorted(e.by_candidate()) == [cid(1), cid(2), cid(3)]


def test_two_failures_in_a_row_are_tolerated(rs9_env):
    e = rs9_env
    e.add_candidates(5)
    summary = e.run(provider=stub(e, script=fail_ks(1, 2)))
    assert summary.failed_candidate_ids == [cid(1), cid(2)]
    assert sorted(e.by_candidate()) == [cid(0), cid(3), cid(4)]


def test_a_mix_of_both_transient_kinds_counts_together(rs9_env):
    e = rs9_env
    e.add_candidates(5)
    summary = e.run(provider=stub(e, script={1: exhausted(), 2: timed_out()}))
    assert summary.failed_candidate_ids == [cid(1), cid(2)]


def test_no_failures_gives_an_empty_failed_list(rs9_env):
    e = rs9_env
    e.add_candidates(3)
    summary = e.run(provider=stub(e))
    assert summary.failed_candidate_ids == []
    assert isinstance(summary.failed_candidate_ids, list)
    assert summary.candidate_count == 3


def test_empty_drive_has_an_empty_failed_list(rs9_env):
    summary = rs9_env.run(provider=stub(rs9_env))
    assert summary.failed_candidate_ids == []
    assert summary.candidate_count == 0


def test_verified_rows_are_complete_and_correct_despite_skips(rs9_env):
    e = rs9_env
    e.add_candidates(4)
    names = class_names()
    p = stub(e, script={1: exhausted(), 2: lambda c, cp, pv: make_verdict(False, names[0], 0, None, 0.2)})
    summary = e.run(provider=p)
    got = e.accepted()
    assert got == {cid(0): True, cid(2): False, cid(3): True}
    assert summary.accepted_count == 2 and summary.rejected_count == 1


def test_rerun_after_a_skip_verifies_the_previously_skipped_candidate(rs9_env):
    e = rs9_env
    e.add_candidates(3)
    first = e.run(provider=stub(e, script=fail_ks(1)))
    assert first.failed_candidate_ids == [cid(1)]
    second = e.run(provider=stub(e))
    assert second.failed_candidate_ids == []
    assert sorted(e.by_candidate()) == [cid(0), cid(1), cid(2)]
    assert len(e.raw_all()) == 3


def test_rerun_with_a_different_skip_replaces_the_verdicts_in_one_go(rs9_env):
    e = rs9_env
    e.add_candidates(3)
    e.run(provider=stub(e))
    assert len(e.raw_all()) == 3
    summary = e.run(provider=stub(e, script=fail_ks(0)))
    assert summary.failed_candidate_ids == [cid(0)]
    assert sorted(e.by_candidate()) == [cid(1), cid(2)], "the drive's verdicts are replaced; no stale row"
    assert len(e.raw_all()) == 2


def test_other_drives_verdicts_are_untouched_by_a_skipping_run(rs9_env):
    e = rs9_env
    e.add_candidates(3, drive="drive-a", base_id=9101)
    e.add_candidates(2, drive="drive-b", base_id=9201)
    e.run("drive-b", provider=stub(e))
    before = e.by_candidate("drive-b")
    e.run("drive-a", provider=stub(e, script=fail_ks(0)))
    assert e.by_candidate("drive-b") == before


# ------------------------------------------------------------------ logging

def test_skip_is_logged_with_candidate_id_and_exception_type_only(rs9_env, caplog, tmp_path):
    e = rs9_env
    e.add_candidates(3)
    caplog.set_level(logging.DEBUG)
    e.run(provider=stub(e, script={1: exhausted("secret-message-9f2c")}))
    text = " | ".join(r.getMessage() for r in caplog.records if r.name.startswith("backend"))
    relevant = [r for r in caplog.records if str(cid(1)) in r.getMessage()]
    assert relevant, "the skipped candidate id must be logged"
    assert any("RetriesExhausted" in r.getMessage() for r in relevant)
    assert "secret-message-9f2c" not in caplog.text, "only the exception type is logged, not its message"
    assert str(tmp_path) not in text
    assert ".jpg" not in text


def test_timeout_skip_is_logged_with_its_type(rs9_env, caplog):
    e = rs9_env
    e.add_candidates(3)
    caplog.set_level(logging.DEBUG)
    e.run(provider=stub(e, script={2: timed_out("secret-message-t1")}))
    assert any(str(cid(2)) in r.getMessage() and "ProviderTimeoutError" in r.getMessage() for r in caplog.records)
    assert "secret-message-t1" not in " | ".join(r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ everything else still aborts

def _abort_errors():
    import backend.providers as bp

    return {
        "SpendCapExceeded": bp.SpendCapExceeded("cap hit unique"),
        "ProviderConfigError": bp.ProviderConfigError("missing META_API_KEY unique"),
        "ProviderError_permanent": bp.ProviderError("Meta API returned HTTP 401: unique"),
        "ProviderError_413": bp.ProviderError("Meta API returned HTTP 413: unique"),
        "TransientProviderError_not_exhausted": bp.TransientProviderError("blip unique"),
        "RuntimeError": RuntimeError("unique runtime"),
        "ValueError": ValueError("unique value"),
        "OSError": OSError("unique os"),
    }


@pytest.mark.parametrize("name", sorted(_abort_errors()))
def test_other_errors_still_abort_and_store_nothing(rs9_env, name):
    e = rs9_env
    e.add_candidates(4)
    err = _abort_errors()[name]
    p = stub(e, script={1: err})
    with pytest.raises(Exception) as ei:
        e.run(provider=p)
    assert str(cid(1)) in str(ei.value), "the error names the candidate: %s" % ei.value
    assert e.raw_all() == [], "nothing stored when the stage aborts"
    assert len(p.calls) == 2, "no more calls after an aborting error"


def test_abort_error_after_a_skip_stores_nothing(rs9_env):
    from backend.providers import SpendCapExceeded

    e = rs9_env
    e.add_candidates(4)
    p = stub(e, script={1: exhausted(), 2: SpendCapExceeded("cap")})
    with pytest.raises(Exception):
        e.run(provider=p)
    assert e.raw_all() == []


def test_spend_cap_error_keeps_its_original_error_visible(rs9_env):
    from backend.providers import SpendCapExceeded

    e = rs9_env
    e.add_candidates(3)
    err = SpendCapExceeded("cap hit")
    with pytest.raises(verification_error()) as ei:
        e.run(provider=stub(e, script={0: err}))
    chain, cur = [], ei.value
    while cur is not None and cur not in chain:
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__
    assert any(c is err for c in chain)


@pytest.mark.parametrize("bad", ["severity_four", "confidence_high", "unknown_class", "none"])
def test_malformed_verdict_still_aborts_even_after_a_skip(rs9_env, bad):
    e = rs9_env
    e.add_candidates(4)
    names = class_names()
    bads = {
        "severity_four": lambda c, cp, pv: corrupt(make_verdict(True, names[0], 2, "left", 0.9), severity=4),
        "confidence_high": lambda c, cp, pv: corrupt(make_verdict(True, names[0], 2, "left", 0.9), confidence=1.5),
        "unknown_class": lambda c, cp, pv: corrupt(make_verdict(True, names[0], 2, "left", 0.9), hazard_class="zzz"),
        "none": lambda c, cp, pv: None,
    }
    with pytest.raises(Exception) as ei:
        e.run(provider=stub(e, script={0: exhausted(), 2: bads[bad]}))
    assert str(cid(2)) in str(ei.value)
    assert e.raw_all() == []


# ------------------------------------------------------------------ outage guard

def test_three_failures_in_a_row_abort_the_stage(rs9_env):
    e = rs9_env
    e.add_candidates(7)
    p = stub(e, script=fail_ks(2, 3, 4))
    with pytest.raises(verification_error()):
        e.run(provider=p)
    assert e.raw_all() == []
    assert len(p.calls) == 5, "stop right after the third failure in a row, do not grind on"


def test_three_in_a_row_mixed_kinds(rs9_env):
    e = rs9_env
    e.add_candidates(6)
    p = stub(e, script={0: exhausted(), 1: timed_out(), 2: exhausted()})
    with pytest.raises(verification_error()):
        e.run(provider=p)
    assert e.raw_all() == []
    assert len(p.calls) == 3


def test_a_success_resets_the_consecutive_count(rs9_env):
    e = rs9_env
    e.add_candidates(8)
    # fail fail OK fail fail OK fail fail: never three in a row, 6 skipped, 2 verified
    p = stub(e, script=fail_ks(0, 1, 3, 4, 6, 7))
    summary = e.run(provider=p)
    assert summary.failed_candidate_ids == [cid(k) for k in (0, 1, 3, 4, 6, 7)]
    assert sorted(e.by_candidate()) == [cid(2), cid(5)]
    assert len(p.calls) == 8


def test_three_in_a_row_after_earlier_isolated_failures_aborts(rs9_env):
    e = rs9_env
    e.add_candidates(9)
    p = stub(e, script=fail_ks(0, 2, 5, 6, 7))
    with pytest.raises(verification_error()):
        e.run(provider=p)
    assert e.raw_all() == []
    assert len(p.calls) == 8


def test_two_in_a_row_at_the_very_end_is_fine(rs9_env):
    e = rs9_env
    e.add_candidates(5)
    summary = e.run(provider=stub(e, script=fail_ks(3, 4)))
    assert summary.failed_candidate_ids == [cid(3), cid(4)]


def test_every_candidate_failing_aborts_even_below_three(rs9_env):
    e = rs9_env
    e.add_candidates(2)
    with pytest.raises(verification_error()):
        e.run(provider=stub(e, script=fail_ks(0, 1)))
    assert e.raw_all() == []


def test_a_single_candidate_that_fails_aborts(rs9_env):
    e = rs9_env
    e.add_candidates(1)
    with pytest.raises(verification_error()):
        e.run(provider=stub(e, script=fail_ks(0)))
    assert e.raw_all() == []


def test_all_failing_with_more_than_three_aborts(rs9_env):
    e = rs9_env
    e.add_candidates(5)
    p = stub(e, default=exhausted())
    with pytest.raises(verification_error()):
        e.run(provider=p)
    assert e.raw_all() == []
    assert len(p.calls) == 3


def test_guard_error_is_a_provider_error_and_leaks_nothing(rs9_env):
    from backend.providers import ProviderError

    e = rs9_env
    e.add_candidates(4)
    with pytest.raises(ProviderError) as ei:
        e.run(provider=stub(e, default=exhausted("secret-guard-msg-77")))
    assert str(e.img_dir) not in str(ei.value)
    assert ".jpg" not in str(ei.value)


def test_guard_abort_keeps_the_previous_run_verdicts(rs9_env):
    e = rs9_env
    e.add_candidates(5)
    e.run(provider=stub(e))
    before = e.rows()
    assert len(before) == 5
    with pytest.raises(verification_error()):
        e.run(provider=stub(e, script=fail_ks(1, 2, 3)))
    assert e.rows() == before


# ------------------------------------------------------------------ summary dataclass

def test_summary_old_positional_construction_still_works():
    from backend.verification import VerificationSummary

    s = VerificationSummary("d", 0.5, "v1", 0, 0, 0, [])
    assert s.failed_candidate_ids == []
    s2 = VerificationSummary("d", 0.5, "v1", 3, 2, 1)
    assert s2.verdicts == [] and s2.failed_candidate_ids == []
    assert (s2.candidate_count, s2.accepted_count, s2.rejected_count) == (3, 2, 1)


def test_summary_failed_ids_keyword_and_independent_defaults():
    from backend.verification import VerificationSummary

    s = VerificationSummary("d", 0.5, "v1", 1, 1, 0, [], failed_candidate_ids=[4, 5])
    assert s.failed_candidate_ids == [4, 5]
    a = VerificationSummary("d", 0.5, "v1", 0, 0, 0)
    b = VerificationSummary("d", 0.5, "v1", 0, 0, 0)
    a.failed_candidate_ids.append(1)
    assert b.failed_candidate_ids == [], "the default list must not be shared"


def test_summary_failed_ids_is_the_last_field():
    import dataclasses

    from backend.verification import VerificationSummary

    names = [f.name for f in dataclasses.fields(VerificationSummary)]
    assert names[-1] == "failed_candidate_ids"
    assert names[:7] == ["drive_id", "min_confidence", "prompt_version", "candidate_count", "accepted_count",
                         "rejected_count", "verdicts"]
