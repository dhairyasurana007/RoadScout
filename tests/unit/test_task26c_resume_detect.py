"""Task 26c: an interrupted detecting stage resumes at the first chunk that is not recorded as done."""
from __future__ import annotations

import pytest

import rs26c_helpers as H
from rs10_helpers import DONE, FAILED, rs10_env, spy, status_of  # noqa: F401


def baseline(e, **upload_kw):
    """An uninterrupted run of a fresh drive: (drive id, snapshot, drive JSON)."""
    did = e.upload(**upload_kw)
    res, exc = H.run(e, did, spy())
    assert exc is None, exc
    d = e.get(did)
    assert status_of(d) in DONE, d
    return did, H.snapshot(e, did), d


def chunks_in_db(e, did):
    return sorted(r["chunk"] for r in e.q("SELECT DISTINCT chunk FROM detections WHERE drive_id = ?", (did,)))


@pytest.mark.parametrize("fail_at", [1, 2], ids=["fails-at-chunk-1", "fails-at-chunk-2"])
def test_failure_keeps_finished_chunks_and_resume_sends_only_the_rest(rs10_env, fail_at):
    e = rs10_env
    did, clean, _ = baseline(e)
    p1 = H.fail_detect_at(fail_at)
    H.run(e, did, p1)
    assert status_of(e.get(did)) in FAILED
    assert H.chunk_names(p1) == H.CHUNK_FILES[: fail_at + 1]
    assert chunks_in_db(e, did) == list(range(fail_at)), "chunks before the failing one are stored"
    assert e.counts(did)["candidates"] in (0, 3)

    p2 = spy()
    res, exc = H.run(e, did, p2)
    assert exc is None, exc
    d = e.get(did)
    assert status_of(d) in DONE
    assert H.chunk_names(p2) == H.CHUNK_FILES[fail_at:], "only the chunks not done are sent to the provider"
    H.assert_same_snapshot(H.snapshot(e, did), clean)


def test_resume_makes_no_duplicate_rows(rs10_env):
    e = rs10_env
    did = e.upload()
    H.run(e, did, H.fail_detect_at(2))
    H.run(e, did, spy())
    rows = e.q("SELECT * FROM detections WHERE drive_id = ?", (did,))
    assert rows
    assert H.duplicates(rows, ("chunk", "frame_index", "track_key", "class_prompt")) == []


def test_failure_on_the_first_chunk_stores_nothing_and_resume_sends_everything(rs10_env):
    e = rs10_env
    did = e.upload()
    H.run(e, did, H.fail_detect_at(0))
    assert status_of(e.get(did)) in FAILED
    assert e.counts(did)["detections"] == 0
    p2 = spy()
    H.run(e, did, p2)
    assert status_of(e.get(did)) in DONE
    assert H.chunk_names(p2) == H.CHUNK_FILES


def test_two_interruptions_then_success(rs10_env):
    e = rs10_env
    did, clean, _ = baseline(e)
    p1 = H.fail_detect_at(1)          # clean start: chunk 0 done, chunk 1 fails
    H.run(e, did, p1)
    p2 = H.fail_detect_at(1)          # resumed: chunk 1 done, chunk 2 fails
    H.run(e, did, p2)
    assert H.chunk_names(p2) == ["chunk_0001.mp4", "chunk_0002.mp4"]
    assert status_of(e.get(did)) in FAILED
    assert chunks_in_db(e, did) == [0, 1]
    p3 = spy()
    H.run(e, did, p3)
    assert H.chunk_names(p3) == ["chunk_0002.mp4"]
    assert status_of(e.get(did)) in DONE
    H.assert_same_snapshot(H.snapshot(e, did), clean)


def test_resumed_run_makes_every_verification_call_exactly_once(rs10_env):
    e = rs10_env
    did = e.upload()
    H.run(e, did, H.fail_detect_at(1))
    p2 = spy()
    H.run(e, did, p2)
    assert len(p2.verify_calls) == 3
    assert e.counts(did)["verdicts"] == 3


def test_detection_failure_after_resume_still_marks_the_drive_failed(rs10_env):
    e = rs10_env
    did = e.upload()
    H.run(e, did, H.fail_detect_at(1))
    p2 = H.fail_detect_at(0)
    H.run(e, did, p2)
    d = e.get(did)
    assert status_of(d) in FAILED
    assert "boom-detect-xyz" in (d.get("error") or "")
    assert d.get("failed_stage") in ("detecting", None) or "detect" in str(d.get("failed_stage"))


def test_stationary_counts_and_skipped_chunks_match_an_uninterrupted_run(rs10_env):
    e = rs10_env
    gpx = H.gpx_with_a_stop(still_seconds=5)
    did, clean, clean_json = baseline(e, gpx=gpx)
    assert clean_json["stationary_chunks_skipped"] >= 1, "the synthetic parked start must skip a chunk"
    # chunk 0 is skipped (never sent); of the sent chunks the first is chunk 1
    p1 = H.fail_detect_at(1)
    H.run(e, did, p1)
    assert H.chunk_names(p1) == ["chunk_0001.mp4", "chunk_0002.mp4"]
    assert status_of(e.get(did)) in FAILED
    p2 = spy()
    H.run(e, did, p2)
    d = e.get(did)
    assert status_of(d) in DONE
    assert H.chunk_names(p2) == ["chunk_0002.mp4"]
    assert H.stationary_counts(d) == H.stationary_counts(clean_json)
    H.assert_same_snapshot(H.snapshot(e, did), clean)


def test_resume_when_the_interrupted_chunk_is_the_only_one_left_sent(rs10_env):
    e = rs10_env
    gpx = H.gpx_with_a_stop(still_seconds=5)
    did, clean, clean_json = baseline(e, gpx=gpx)
    H.run(e, did, H.fail_detect_at(0))      # chunk 1 (first sent chunk) fails: nothing stored
    p2 = spy()
    H.run(e, did, p2)
    assert H.chunk_names(p2) == ["chunk_0001.mp4", "chunk_0002.mp4"], "a skipped (stationary) chunk is never sent"
    assert H.stationary_counts(e.get(did)) == H.stationary_counts(clean_json)


def test_drive_json_chunk_counts_after_resume(rs10_env):
    e = rs10_env
    did, _, clean_json = baseline(e)
    H.run(e, did, H.fail_detect_at(1))
    H.run(e, did, spy())
    d = e.get(did)
    for k in ("frames_total", "stationary_frames_skipped", "chunks_sent", "stationary_chunks_skipped"):
        assert d[k] == clean_json[k], (k, d[k], clean_json[k])


def test_privacy_fail_closed_after_a_resumed_run(rs10_env):
    """Every candidate image has a blurred copy after a resumed run."""
    from pathlib import Path

    e = rs10_env
    did = e.upload()
    H.run(e, did, H.fail_detect_at(1))
    H.run(e, did, spy())
    assert status_of(e.get(did)) in DONE
    cands = e.candidates(did)
    assert cands
    for c in cands:
        best = Path(c["best_frame_path"])
        assert best.with_name(best.stem + "_blur.jpg").is_file(), "no blurred copy for %s" % best.name
