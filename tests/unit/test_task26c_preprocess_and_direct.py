"""Task 26c: preprocessing is reused on resume when its work folder is complete; run_detection keeps its old
all-or-nothing behaviour unless resume=True."""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest

import rs26c_helpers as H
from rs10_helpers import DONE, FAILED, rs10_env, spy, status_of  # noqa: F401
from rs7_helpers import RecordingProvider, rs7_env, rs7_work  # noqa: F401


def interrupted(e, fail_at=1):
    did = e.upload()
    H.run(e, did, H.fail_detect_at(fail_at))
    assert status_of(e.get(did)) in FAILED
    return did


def mtimes(e):
    wd = H.work_dir(e)
    return {p.name: p.stat().st_mtime_ns for p in wd.iterdir() if p.is_file()}


# ------------------------------------------------------------------ preprocessing reuse

def test_complete_work_folder_is_reused_on_resume(rs10_env):
    e = rs10_env
    did = interrupted(e)
    before = mtimes(e)
    assert "frame_timestamps.json" in before and len([n for n in before if n.startswith("chunk_")]) == 3
    p2 = spy()
    H.run(e, did, p2)
    assert status_of(e.get(did)) in DONE
    after = mtimes(e)
    for name, mt in before.items():
        assert after.get(name) == mt, "%s was rewritten: ffmpeg must not run again" % name
    assert H.chunk_names(p2) == ["chunk_0001.mp4", "chunk_0002.mp4"]


def test_resume_after_a_failure_at_the_first_chunk_also_reuses_the_work_folder(rs10_env):
    e = rs10_env
    did = interrupted(e, fail_at=0)
    before = mtimes(e)
    p2 = spy()
    H.run(e, did, p2)
    assert status_of(e.get(did)) in DONE
    assert mtimes(e) == before or all(mtimes(e).get(k) == v for k, v in before.items())
    assert H.chunk_names(p2) == H.CHUNK_FILES


def test_missing_chunk_file_redoes_preprocessing(rs10_env):
    e = rs10_env
    did = interrupted(e)
    wd = H.work_dir(e)
    (wd / "chunk_0002.mp4").unlink()
    p2 = spy()
    res, exc = H.run(e, did, p2)
    assert exc is None, exc
    assert status_of(e.get(did)) in DONE
    assert (wd / "chunk_0002.mp4").is_file(), "preprocessing was redone, the chunk exists again"
    assert H.work_chunks(e) == H.CHUNK_FILES
    assert sorted(set(H.chunk_names(p2))) == H.chunk_names(p2)
    assert "chunk_0002.mp4" in H.chunk_names(p2)


def test_missing_timestamps_file_redoes_preprocessing(rs10_env):
    e = rs10_env
    did = interrupted(e)
    wd = H.work_dir(e)
    (wd / "frame_timestamps.json").unlink()
    p2 = spy()
    H.run(e, did, p2)
    assert status_of(e.get(did)) in DONE
    assert (wd / "frame_timestamps.json").is_file()
    json.loads((wd / "frame_timestamps.json").read_text(encoding="utf-8"))


def test_corrupt_timestamps_file_redoes_preprocessing(rs10_env):
    e = rs10_env
    did = interrupted(e)
    wd = H.work_dir(e)
    (wd / "frame_timestamps.json").write_text("{ this is not json", encoding="utf-8")
    p2 = spy()
    H.run(e, did, p2)
    assert status_of(e.get(did)) in DONE
    json.loads((wd / "frame_timestamps.json").read_text(encoding="utf-8"))
    assert len(H.work_chunks(e)) == 3


def test_deleted_work_folder_redoes_preprocessing(rs10_env):
    import shutil

    e = rs10_env
    did = interrupted(e)
    shutil.rmtree(H.work_dir(e))
    p2 = spy()
    res, exc = H.run(e, did, p2)
    assert exc is None, exc
    assert status_of(e.get(did)) in DONE
    assert H.work_chunks(e) == H.CHUNK_FILES
    assert e.counts(did)["verdicts"] == len(e.candidates(did)) > 0


def test_result_after_redoing_preprocessing_matches_an_uninterrupted_run(rs10_env):
    e = rs10_env
    did = e.upload()
    H.run(e, did, spy())
    clean = H.snapshot(e, did)
    H.run(e, did, H.fail_detect_at(1))
    (H.work_dir(e) / "chunk_0001.mp4").unlink()
    H.run(e, did, spy())
    assert status_of(e.get(did)) in DONE
    H.assert_same_snapshot(H.snapshot(e, did), clean)


# ------------------------------------------------------------------ run_detection directly (rs7 env)

def boom_at(chunk):
    def script(call_no, chunk_idx, prompts):
        if chunk_idx == chunk:
            raise RuntimeError("boom-direct-chunk-%d" % chunk)
        return None

    return script


def summary_fields(s):
    d = asdict(s)
    d.pop("drive_id", None)
    d.pop("stationary_note", None)    # names the drive id
    return d


def test_direct_default_is_all_or_nothing(rs7_env):
    e = rs7_env
    with pytest.raises(Exception):
        e.run(provider=RecordingProvider(script=boom_at(1)))
    assert e.raw_rows() == [], "without resume=True a failure stores nothing"


def test_direct_default_rerun_replaces_and_sends_every_chunk(rs7_env):
    e = rs7_env
    e.run(provider=RecordingProvider())
    rp = RecordingProvider()
    e.run(provider=rp)
    assert sorted(c["chunk"] for c in rp.calls) == [0, 1, 2]


def test_direct_resume_default_is_off():
    import inspect

    from backend.detection import run_detection

    assert inspect.signature(run_detection).parameters["resume"].default is False


def test_direct_resume_true_stores_each_chunk_when_it_finishes(rs7_env):
    e = rs7_env
    with pytest.raises(Exception):
        e.run(provider=RecordingProvider(script=boom_at(2)), resume=True)
    assert sorted({r["chunk"] for r in e.rows()}) == [0, 1]


def test_direct_resume_sends_only_unfinished_chunks_and_matches_a_clean_run(rs7_env):
    e = rs7_env
    e.add_drive("drive-b")
    clean = e.run("drive-b", provider=RecordingProvider())
    with pytest.raises(Exception):
        e.run(provider=RecordingProvider(script=boom_at(1)), resume=True)
    rp = RecordingProvider()
    resumed = e.run(provider=rp, resume=True)
    assert sorted(c["chunk"] for c in rp.calls) == [1, 2]
    assert summary_fields(resumed) == summary_fields(clean)
    from rs7_helpers import signature_rows

    a = [r[:7] for r in signature_rows(e.rows("drive-a"))]
    b = [r[:7] for r in signature_rows(e.rows("drive-b"))]
    assert a == b


def test_direct_resume_true_on_a_fresh_drive_behaves_like_a_normal_run(rs7_env):
    e = rs7_env
    rp = RecordingProvider()
    s = e.run(provider=rp, resume=True)
    assert sorted(c["chunk"] for c in rp.calls) == [0, 1, 2]
    assert s.chunk_count == 3
    assert len({r["chunk"] for r in e.rows()}) == 3


def test_direct_resume_when_everything_is_done_sends_nothing(rs7_env):
    e = rs7_env
    first = e.run(provider=RecordingProvider(), resume=True)
    rp = RecordingProvider()
    again = e.run(provider=rp, resume=True)
    assert rp.calls == []
    assert summary_fields(again) == summary_fields(first)
    assert len(e.rows()) == first.detection_count


def test_direct_resume_does_not_duplicate_rows(rs7_env):
    e = rs7_env
    with pytest.raises(Exception):
        e.run(provider=RecordingProvider(script=boom_at(2)), resume=True)
    e.run(provider=RecordingProvider(), resume=True)
    raw = e.raw_rows()
    keys = [(r["chunk"], r["frame_index"], r["track_key"], r["class_prompt"]) for r in raw]
    assert len(keys) == len(set(keys))


def test_direct_non_resume_run_clears_checkpoints_of_an_earlier_resume_run(rs7_env):
    """A plain run replaces everything; a later resume=True run must not think old chunks are still done."""
    e = rs7_env
    with pytest.raises(Exception):
        e.run(provider=RecordingProvider(script=boom_at(2)), resume=True)
    e.run(provider=RecordingProvider())          # plain run: replaces all rows
    rp = RecordingProvider()
    e.run(provider=rp, resume=True)
    # whichever way the implementation records it, the stored result is complete and consistent
    assert sorted({r["chunk"] for r in e.rows()}) == [0, 1, 2]
    raw = e.raw_rows()
    assert len(raw) == len({(r["chunk"], r["frame_index"], r["track_key"], r["class_prompt"]) for r in raw})
