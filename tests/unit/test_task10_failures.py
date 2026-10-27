"""Task 10: errors in any stage set status failed, record the stage and message, keep data consistent."""
from __future__ import annotations

import pytest

from rs10_helpers import (CHUNKS, DETECTIONS, DONE, FAILED, ERROR_KEYS, err_of, patch_everywhere,  # noqa: F401
                          rs10_env, sha_file, spy, status_of)

ZERO = {"detections": 0, "candidates": 0, "verdicts": 0, "all_verdicts": 0}


def assert_failed(e, did, *needles, any_of=()):
    d = e.get(did)
    assert status_of(d) in FAILED, d
    msg = err_of(d)
    assert msg.strip(), "a failed drive must carry a non-empty error message: %s" % d
    low = msg.lower()
    for n in needles:
        assert n.lower() in low, "error %r should mention %r" % (msg, n)
    if any_of:
        assert any(k in low for k in any_of), "error %r should name the failing stage (one of %s)" % (msg, any_of)
    # recorded on the drive itself (DB), not only computed for the response
    row = e.row(did)
    cols = [k for k in row if "error" in k.lower() and row[k]]
    assert cols, "no error column on the drives row: %s" % list(row)
    assert msg in {str(row[k]) for k in cols} or any(str(row[k]) in msg or msg in str(row[k]) for k in cols)
    # stable: reading again gives the same terminal answer
    d2 = e.get(did)
    assert status_of(d2) in FAILED and err_of(d2) == msg
    return msg


def test_detection_error_marks_failed_with_stage_and_message(rs10_env):
    e = rs10_env
    did = e.upload()
    p = spy(fail_detect_on=1)
    e.run(did, p)  # may raise or return; the recorded state is what matters
    assert_failed(e, did, "boom-detect-xyz", any_of=("detect",))
    assert e.counts(did) == ZERO, "a failed detection stage leaves no detections, candidates or verdicts"
    assert p.verify_calls == [], "later stages must not run after a failure"
    assert e.video_file(did).is_file()


def test_verification_error_marks_failed(rs10_env):
    e = rs10_env
    did = e.upload()
    p = spy(fail_verify_on=1)
    e.run(did, p)
    assert_failed(e, did, "boom-verify-xyz", any_of=("verif",))
    c = e.counts(did)
    assert c["verdicts"] == 0 and c["all_verdicts"] == 0, "no partial verdicts"
    assert c["candidates"] in (0, CHUNKS)
    assert e.orphan_verdicts() == []
    # whatever earlier-stage data remains is usable
    for cand in e.candidates(did):
        from pathlib import Path

        assert Path(cand["best_frame_path"]).is_file()
    assert e.video_file(did).is_file()


def test_aggregation_error_marks_failed_and_skips_verification(rs10_env):
    e = rs10_env
    did = e.upload()

    def boom(*a, **k):
        raise RuntimeError("boom-aggregate-xyz")

    patch_everywhere(e.mp, "backend.candidates", "aggregate_drive", boom)
    p = spy()
    e.run(did, p)
    assert_failed(e, did, "boom-aggregate-xyz", any_of=("aggregat", "candidate", "track", "geotag"))
    assert p.verify_calls == []
    c = e.counts(did)
    assert c["candidates"] == 0 and c["verdicts"] == 0


def test_preprocess_error_on_corrupt_video(rs10_env):
    e = rs10_env
    did = e.upload()
    path = e.video_file(did)
    path.write_bytes(b"this is definitely not a video" * 50)
    p = spy()
    e.run(did, p)
    assert_failed(e, did, any_of=("preprocess",))
    assert p.detect_calls == [] and p.verify_calls == []
    assert e.counts(did) == ZERO
    assert path.is_file(), "the (bad) upload is never deleted automatically"


def test_preprocess_error_on_missing_video_file(rs10_env):
    e = rs10_env
    did = e.upload()
    e.video_file(did).unlink()
    p = spy()
    e.run(did, p)
    assert_failed(e, did)
    assert p.detect_calls == []
    assert e.counts(did) == ZERO


def test_failed_drive_can_be_fixed_and_rerun(rs10_env):
    e = rs10_env
    did = e.upload()
    good = e.video_file(did).read_bytes()
    e.video_file(did).write_bytes(b"garbage")
    e.run(did, spy())
    assert_failed(e, did)
    e.video_file(did).write_bytes(good)
    p = spy()
    res, exc = e.run(did, p)
    assert exc is None
    d = e.get(did)
    assert status_of(d) in DONE
    assert err_of(d) == "", "a successful run clears the previous error"
    assert e.counts(did)["verdicts"] == CHUNKS


def test_failure_in_rerun_after_success_leaves_consistent_data(rs10_env):
    e = rs10_env
    did = e.upload()
    e.run(did, spy())
    assert e.counts(did)["verdicts"] == CHUNKS
    e.run(did, spy(fail_verify_on=0))
    assert_failed(e, did, "boom-verify-xyz")
    assert e.orphan_verdicts() == []
    c = e.counts(did)
    assert c["verdicts"] in (0, CHUNKS)
    assert c["all_verdicts"] == c["verdicts"]
    # and a clean run afterwards fully recovers
    e.run(did, spy())
    assert status_of(e.get(did)) in DONE
    assert err_of(e.get(did)) == ""
    assert e.counts(did) == {"detections": DETECTIONS, "candidates": CHUNKS, "verdicts": CHUNKS,
                             "all_verdicts": CHUNKS}


def test_failure_in_detection_during_rerun_keeps_candidates_and_verdicts_consistent(rs10_env):
    e = rs10_env
    did = e.upload()
    e.run(did, spy())
    e.run(did, spy(fail_detect_on=0))
    assert_failed(e, did, "boom-detect-xyz")
    assert e.orphan_verdicts() == []
    c = e.counts(did)
    # either the old results survive intact or all stages were cleared, never a mixture
    assert (c["detections"], c["candidates"], c["verdicts"]) in {
        (DETECTIONS, CHUNKS, CHUNKS), (0, 0, 0), (DETECTIONS, 0, 0), (0, CHUNKS, CHUNKS)}
    assert c["verdicts"] == c["all_verdicts"]


def test_one_drives_failure_does_not_affect_another(rs10_env):
    e = rs10_env
    bad, good = e.upload(), e.upload()
    e.run(bad, spy(fail_detect_on=0))
    e.run(good, spy())
    assert status_of(e.get(bad)) in FAILED
    assert status_of(e.get(good)) in DONE
    assert err_of(e.get(good)) == ""
    assert e.counts(good)["verdicts"] == CHUNKS
    assert e.counts(bad) == ZERO


def test_error_is_a_readable_string_without_secrets(rs10_env):
    e = rs10_env
    e.mp.setenv("META_API_KEY", "sk-test-secret-should-never-appear")
    did = e.upload()
    e.run(did, spy(fail_detect_on=0))
    msg = err_of(e.get(did))
    assert isinstance(msg, str) and 0 < len(msg) < 5000
    assert "sk-test-secret-should-never-appear" not in msg
    assert "Traceback" not in msg or len(msg) < 5000


def test_failed_drive_original_files_untouched(rs10_env):
    e = rs10_env
    did = e.upload()
    v, g = e.video_file(did), e.gpx_file(did)
    hv, hg = sha_file(v), sha_file(g)
    e.run(did, spy(fail_verify_on=2))
    assert v.is_file() and g.is_file()
    assert sha_file(v) == hv and sha_file(g) == hg


def test_interrupted_drive_is_marked_failed_on_startup_recovery(rs10_env):
    import importlib
    import sqlite3

    e = rs10_env
    did = e.upload()
    con = sqlite3.connect(str(e.db))
    con.execute("UPDATE drives SET status = 'processing' WHERE id = ?", (did,))
    con.commit()
    con.close()
    mod = importlib.import_module("backend.pipeline")
    mod.recover_interrupted()
    d = e.get(did)
    assert status_of(d) in FAILED
    assert "interrupted" in err_of(d).lower()
    # and it can be re-run afterwards
    e.run(did, spy())
    assert status_of(e.get(did)) in DONE
