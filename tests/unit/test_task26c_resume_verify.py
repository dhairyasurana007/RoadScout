"""Task 26c: an interrupted verifying stage keeps the verdicts obtained so far and resumes with the rest."""
from __future__ import annotations

import pytest

import rs26c_helpers as H
from rs10_helpers import DONE, FAILED, rs10_env, spy, status_of  # noqa: F401
from rs9_helpers import make_verdict, rs9_env, stub  # noqa: F401


def exhausted(msg="boom-26c-exhausted"):
    from backend.providers import RetriesExhausted

    return RetriesExhausted(msg, 5)


def baseline(e):
    did = e.upload()
    res, exc = H.run(e, did, spy())
    assert exc is None, exc
    assert status_of(e.get(did)) in DONE
    return did, H.snapshot(e, did)


# ------------------------------------------------------------------ the pipeline

@pytest.mark.parametrize("fail_at", [1, 2], ids=["fails-at-candidate-1", "fails-at-candidate-2"])
def test_failure_keeps_earlier_verdicts_and_resume_verifies_only_the_rest(rs10_env, fail_at):
    e = rs10_env
    did, clean = baseline(e)
    p1 = H.fail_verify_at(fail_at)
    H.run(e, did, p1)
    d = e.get(did)
    assert status_of(d) in FAILED
    assert "boom-verify-xyz" in (d.get("error") or "")
    assert len(p1.verify_calls) == fail_at + 1
    stored = e.verdicts(did)
    assert len(stored) == fail_at, "verdicts before the failing candidate are stored"
    assert e.counts(did)["candidates"] == 3, "the aggregated candidates survive the interruption"
    cand_ids = [c["id"] for c in e.candidates(did)]
    created = {v["candidate_id"]: v["created_at"] for v in stored}

    p2 = spy()
    res, exc = H.run(e, did, p2)
    assert exc is None, exc
    assert status_of(e.get(did)) in DONE
    assert len(p2.verify_calls) == 3 - fail_at, "the provider is called only for candidates without a verdict"
    assert [c["id"] for c in e.candidates(did)] == cand_ids, "aggregation is skipped on resume: same candidate rows"
    after = {v["candidate_id"]: v["created_at"] for v in e.verdicts(did)}
    for cid, ts in created.items():
        assert after[cid] == ts, "an existing verdict is kept, not re-created"
    assert len(after) == 3
    H.assert_same_snapshot(H.snapshot(e, did), clean)


def test_detection_is_not_repeated_when_only_verification_failed(rs10_env):
    e = rs10_env
    did = e.upload()
    H.run(e, did, H.fail_verify_at(1))
    p2 = spy()
    H.run(e, did, p2)
    assert p2.detect_calls == [], "detection finished earlier; no SAM calls on resume"
    assert len(p2.verify_calls) == 2


def test_resume_reports_the_stage_it_resumed_at(rs10_env):
    e = rs10_env
    did = e.upload()
    H.run(e, did, H.fail_verify_at(1))
    H.run(e, did, spy())
    d = e.get(did)
    assert d["resumed"] is True
    assert d["resumed_from"] == "verifying"


def test_two_verification_interruptions_then_success(rs10_env):
    e = rs10_env
    did, clean = baseline(e)
    H.run(e, did, H.fail_verify_at(1))
    p2 = H.fail_verify_at(0)           # resumed: candidate 1 fails again at once
    H.run(e, did, p2)
    assert len(p2.verify_calls) == 1 and len(e.verdicts(did)) == 1
    p3 = spy()
    H.run(e, did, p3)
    assert len(p3.verify_calls) == 2
    assert status_of(e.get(did)) in DONE
    H.assert_same_snapshot(H.snapshot(e, did), clean)


def test_network_skipped_candidate_gets_no_verdict_and_is_retried_on_the_next_resume(rs10_env):
    e = rs10_env
    did = e.upload()

    def hook(s, n):
        if n == 1:
            raise exhausted()

    p1 = spy(on_verify=hook)
    H.run(e, did, p1)
    d = e.get(did)
    assert status_of(d) in DONE and d["warning"]
    assert len(e.verdicts(did)) == 2
    # the drive finished "done": the next process call is a clean run, but a failed run in between resumes
    p2 = spy()
    H.run(e, did, p2)
    assert status_of(e.get(did)) in DONE
    assert len(e.verdicts(did)) == 3
    assert e.get(did)["warning"] is None


def test_tampered_detections_force_the_aggregation_to_be_redone(rs10_env):
    e = rs10_env
    did = e.upload()
    H.run(e, did, H.fail_verify_at(1))
    assert e.counts(did)["candidates"] == 3
    import sqlite3

    con = sqlite3.connect(str(e.db))
    try:
        with con:
            con.execute("DELETE FROM detections WHERE drive_id = ? AND chunk = 2", (did,))
    finally:
        con.close()
    p2 = spy()
    H.run(e, did, p2)
    assert status_of(e.get(did)) in DONE
    assert e.counts(did)["candidates"] == 2, "aggregation must be re-run when the detections it used changed"
    assert len(e.verdicts(did)) == 2


# ------------------------------------------------------------------ verify_drive directly (rs9 env)

def cid(k):
    return 9101 + k


def test_direct_resume_stores_each_verdict_as_obtained(rs9_env):
    e = rs9_env
    e.add_candidates(4)
    p = stub(e, script={2: RuntimeError("boom-direct")})
    with pytest.raises(Exception):
        e.run(provider=p, resume=True)
    assert sorted(e.by_candidate()) == [cid(0), cid(1)], "verdicts before the failure stay stored"


def test_direct_resume_calls_only_for_candidates_without_a_verdict(rs9_env):
    e = rs9_env
    e.add_candidates(4)
    with pytest.raises(Exception):
        e.run(provider=stub(e, script={2: RuntimeError("boom-direct")}), resume=True)
    created = {r["candidate_id"]: r["created_at"] for r in e.rows()}
    p2 = stub(e)
    summary = e.run(provider=p2, resume=True)
    assert [c["cand"]["id"] for c in p2.calls] == [cid(2), cid(3)]
    assert sorted(e.by_candidate()) == [cid(k) for k in range(4)]
    for r in e.rows():
        if r["candidate_id"] in created:
            assert r["created_at"] == created[r["candidate_id"]]
    assert summary.candidate_count == 4
    assert len(summary.verdicts) == 4


def test_direct_resume_with_everything_verified_makes_no_calls(rs9_env):
    e = rs9_env
    e.add_candidates(3)
    e.run(provider=stub(e), resume=True)
    before = e.rows()
    p2 = stub(e)
    summary = e.run(provider=p2, resume=True)
    assert p2.calls == []
    assert e.rows() == before
    assert summary.candidate_count == 3
    assert summary.failed_candidate_ids == []


def test_direct_resume_reverifies_when_the_prompt_version_changed(rs9_env):
    e = rs9_env
    e.add_candidates(3)
    e.run(provider=stub(e), resume=True)
    e.set_version("t26c-v2")
    p2 = stub(e)
    e.run(provider=p2, resume=True)
    assert len(p2.calls) == 3
    assert {r["prompt_version"] for r in e.rows()} == {"t26c-v2"}
    assert len(e.rows()) == 3


def test_direct_resume_retries_a_network_skipped_candidate_only(rs9_env):
    e = rs9_env
    e.add_candidates(4)
    s1 = e.run(provider=stub(e, script={1: exhausted()}), resume=True)
    assert s1.failed_candidate_ids == [cid(1)]
    assert sorted(e.by_candidate()) == [cid(0), cid(2), cid(3)]
    p2 = stub(e)
    s2 = e.run(provider=p2, resume=True)
    assert [c["cand"]["id"] for c in p2.calls] == [cid(1)]
    assert s2.failed_candidate_ids == []
    assert sorted(e.by_candidate()) == [cid(k) for k in range(4)]


def test_direct_without_resume_is_all_or_nothing_as_before(rs9_env):
    e = rs9_env
    e.add_candidates(4)
    with pytest.raises(Exception):
        e.run(provider=stub(e, script={2: RuntimeError("boom-direct")}))
    assert e.raw_all() == [], "without resume=True nothing is stored when the stage fails"


def test_direct_without_resume_replaces_old_verdicts_and_calls_for_every_candidate(rs9_env):
    e = rs9_env
    e.add_candidates(3)
    e.run(provider=stub(e))
    p2 = stub(e)
    e.run(provider=p2)
    assert len(p2.calls) == 3
    assert len(e.raw_all()) == 3


def test_direct_default_resume_is_off(rs9_env):
    import inspect

    from backend.verification import verify_drive

    p = inspect.signature(verify_drive).parameters["resume"]
    assert p.default is False


def test_direct_resume_failure_names_the_candidate_like_before(rs9_env):
    from backend.verification import VerificationError

    e = rs9_env
    e.add_candidates(3)
    with pytest.raises(VerificationError) as ei:
        e.run(provider=stub(e, script={1: RuntimeError("boom-direct")}), resume=True)
    assert str(cid(1)) in str(ei.value)
    assert ei.value.candidate_id == cid(1)


def test_direct_resume_malformed_verdict_stores_nothing_for_that_candidate(rs9_env):
    from rs9_helpers import corrupt

    e = rs9_env
    e.add_candidates(3)
    bad = lambda c, cp, pv: corrupt(make_verdict(True, "pothole", 2, "left", 0.9), severity=4)  # noqa: E731
    with pytest.raises(Exception):
        e.run(provider=stub(e, script={1: bad}), resume=True)
    assert cid(1) not in e.by_candidate()


def test_direct_resume_on_a_drive_without_candidates_is_a_no_op(rs9_env):
    e = rs9_env
    p = stub(e)
    summary = e.run(provider=p, resume=True)
    assert p.calls == [] and summary.candidate_count == 0
