"""Task 26c: when a failed drive resumes and when it starts clean; restart; the API; the new JSON fields."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

import rs26c_helpers as H
from rs10_helpers import DONE, FAILED, INITIAL, clip_bytes, rs10_env, spy, status_of  # noqa: F401


def interrupted(e, fail_at=1, **upload_kw):
    """A fresh drive whose first run failed in detection at chunk ``fail_at`` (checkpoints exist)."""
    did = e.upload(**upload_kw)
    p = H.fail_detect_at(fail_at)
    H.run(e, did, p)
    assert status_of(e.get(did)) in FAILED
    assert H.chunk_names(p) == H.CHUNK_FILES[: fail_at + 1]
    return did


def interrupted_via_config(e, fail_at=1):
    """Like ``interrupted`` but the failing run takes its provider from get_provider(), as the HTTP route does."""
    did = e.upload()
    p = H.fail_detect_at(fail_at)
    e.use_provider(p)
    H.run(e, did, None)
    assert status_of(e.get(did)) in FAILED
    assert H.chunk_names(p) == H.CHUNK_FILES[: fail_at + 1]
    return did


def assert_clean_start(e, did, p2, *, expect_chunks=None):
    d = e.get(did)
    assert status_of(d) in DONE, d
    names = H.chunk_names(p2)
    assert names == (expect_chunks if expect_chunks is not None else H.work_chunks(e)), names
    assert d["resumed"] is False and d["resumed_from"] is None
    rows = e.q("SELECT * FROM detections WHERE drive_id = ?", (did,))
    assert H.duplicates(rows, ("chunk", "frame_index", "track_key", "class_prompt")) == []


def assert_resumed(e, did, p2, remaining):
    d = e.get(did)
    assert status_of(d) in DONE, d
    assert H.chunk_names(p2) == remaining
    assert d["resumed"] is True
    assert d["resumed_from"] == "detecting"


# ------------------------------------------------------------------ resume decision

def test_failed_drive_with_checkpoints_resumes(rs10_env):
    e = rs10_env
    did = interrupted(e)
    p2 = spy()
    H.run(e, did, p2)
    assert_resumed(e, did, p2, ["chunk_0001.mp4", "chunk_0002.mp4"])


def test_resume_through_the_http_route(rs10_env):
    e = rs10_env
    did = interrupted_via_config(e)
    p2 = spy()
    e.use_provider(p2)
    r = H.http_process(e, did)
    assert r.status_code == 202, r.text
    e.wait(did)
    assert_resumed(e, did, p2, ["chunk_0001.mp4", "chunk_0002.mp4"])


def test_done_drive_always_runs_clean(rs10_env):
    e = rs10_env
    did = e.upload()
    H.run(e, did, spy())
    assert status_of(e.get(did)) in DONE
    p2 = spy()
    H.run(e, did, p2)
    assert_clean_start(e, did, p2, expect_chunks=H.CHUNK_FILES)
    assert len(p2.verify_calls) == 3


def test_done_drive_runs_clean_through_the_http_route(rs10_env):
    e = rs10_env
    did = e.upload()
    H.run(e, did, spy())
    p2 = spy()
    e.use_provider(p2)
    assert H.http_process(e, did).status_code == 202
    e.wait(did)
    assert_clean_start(e, did, p2, expect_chunks=H.CHUNK_FILES)


def test_uploaded_drive_runs_clean_with_resumed_false(rs10_env):
    e = rs10_env
    did = e.upload()
    d = e.get(did)
    assert d["resumed"] is False and d["resumed_from"] is None
    p = spy()
    H.run(e, did, p)
    assert_clean_start(e, did, p, expect_chunks=H.CHUNK_FILES)


def test_failed_drive_without_checkpoints_runs_clean(rs10_env):
    """A drive that failed in preprocessing has no checkpoints: the next run is clean."""
    e = rs10_env
    did = e.upload()
    path = e.video_file(did)
    good = path.read_bytes()
    path.write_bytes(b"this is definitely not a video" * 50)
    H.run(e, did, spy())
    assert status_of(e.get(did)) in FAILED
    path.write_bytes(good)
    p2 = spy()
    H.run(e, did, p2)
    d = e.get(did)
    assert status_of(d) in DONE
    assert d["resumed"] is False and d["resumed_from"] is None
    assert H.chunk_names(p2) == H.CHUNK_FILES


def test_resumed_flags_are_reset_by_the_next_clean_run(rs10_env):
    e = rs10_env
    did = interrupted(e)
    H.run(e, did, spy())
    assert e.get(did)["resumed"] is True
    p3 = spy()
    H.run(e, did, p3)                      # done -> clean
    assert_clean_start(e, did, p3, expect_chunks=H.CHUNK_FILES)


def test_resumed_flag_is_false_while_a_run_has_just_failed_clean(rs10_env):
    e = rs10_env
    did = e.upload()
    H.run(e, did, H.fail_detect_at(1))
    d = e.get(did)
    assert d["resumed"] is False and d["resumed_from"] is None


# ------------------------------------------------------------------ restart

def test_restart_argument_forces_a_clean_start(rs10_env):
    e = rs10_env
    did = interrupted(e)
    p2 = spy()
    res, exc = H.run(e, did, p2, restart=True)
    assert exc is None, exc
    assert_clean_start(e, did, p2, expect_chunks=H.CHUNK_FILES)
    assert len(p2.verify_calls) == 3


def test_restart_false_is_the_default_behaviour(rs10_env):
    e = rs10_env
    did = interrupted(e)
    p2 = spy()
    H.run(e, did, p2, restart=False)
    assert_resumed(e, did, p2, ["chunk_0001.mp4", "chunk_0002.mp4"])


def test_run_pipeline_signature_has_restart_default_false():
    import inspect

    from backend.pipeline import run_pipeline

    p = inspect.signature(run_pipeline).parameters["restart"]
    assert p.default is False


@pytest.mark.parametrize("value", ["true", "True", "1"])
def test_restart_query_parameter_forces_a_clean_start(rs10_env, value):
    e = rs10_env
    did = interrupted(e)
    p2 = spy()
    e.use_provider(p2)
    r = H.http_process(e, did, restart=value)
    assert r.status_code == 202, r.text
    e.wait(did)
    assert_clean_start(e, did, p2, expect_chunks=H.CHUNK_FILES)


def test_restart_false_query_parameter_resumes(rs10_env):
    e = rs10_env
    did = interrupted_via_config(e)
    p2 = spy()
    e.use_provider(p2)
    assert H.http_process(e, did, restart="false").status_code == 202
    e.wait(did)
    assert_resumed(e, did, p2, ["chunk_0001.mp4", "chunk_0002.mp4"])


def test_restart_on_a_done_drive_is_still_a_clean_run(rs10_env):
    e = rs10_env
    did = e.upload()
    H.run(e, did, spy())
    p2 = spy()
    e.use_provider(p2)
    assert H.http_process(e, did, restart="true").status_code == 202
    e.wait(did)
    assert_clean_start(e, did, p2, expect_chunks=H.CHUNK_FILES)


def test_restart_then_a_new_failure_resumes_from_the_new_checkpoints(rs10_env):
    e = rs10_env
    did = interrupted(e, fail_at=2)
    p2 = H.fail_detect_at(0)
    H.run(e, did, p2, restart=True)              # clean start: chunk 0 fails at once
    assert H.chunk_names(p2) == ["chunk_0000.mp4"]
    assert e.counts(did)["detections"] == 0, "the old run's chunks were discarded"
    p3 = spy()
    H.run(e, did, p3)
    assert H.chunk_names(p3) == H.CHUNK_FILES    # nothing was checkpointed by the restarted run


def test_unknown_drive_is_404_with_or_without_restart(rs10_env):
    e = rs10_env
    assert H.http_process(e, "no-such-drive").status_code == 404
    assert H.http_process(e, "no-such-drive", restart="true").status_code == 404


def test_processing_drive_is_409_with_or_without_restart(rs10_env):
    e = rs10_env
    did = interrupted(e)
    con = sqlite3.connect(str(e.db))
    try:
        with con:
            con.execute("UPDATE drives SET status = 'processing', stage = 'detecting' WHERE id = ?", (did,))
    finally:
        con.close()
    assert H.http_process(e, did).status_code == 409
    assert H.http_process(e, did, restart="true").status_code == 409
    assert status_of(e.get(did)) == "processing"


# ------------------------------------------------------------------ the fingerprint

def other_prompts(e):
    root = Path(__import__("backend").__file__).resolve().parent.parent
    data = json.loads((root / "config" / "hazard_prompts.json").read_text(encoding="utf-8"))
    first = next(iter(data["classes"]))
    data["classes"][first][0] = data["classes"][first][0] + " on the pavement"
    path = e.tmp / "other_prompts.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def change_prompts(e):
    e.mp.setenv("ROADSCOUT_HAZARD_PROMPTS", str(other_prompts(e)))


def change_fps(e):
    e.mp.setenv("ROADSCOUT_PREPROCESS_FPS", "4")


def change_chunk_seconds(e):
    e.mp.setenv("ROADSCOUT_CHUNK_SECONDS", "6")   # 2 chunks of 18 frames (the fake provider emits 12-frame tracks)


def change_height(e):
    e.mp.setenv("ROADSCOUT_PREPROCESS_MAX_HEIGHT", "240")


def change_offset(e, did):
    con = sqlite3.connect(str(e.db))
    try:
        with con:
            con.execute("UPDATE drives SET clock_offset_seconds = 5.0 WHERE id = ?", (did,))
    finally:
        con.close()


def touch_video(e, did):
    p = e.video_file(did)
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime + 3600))


def replace_video(e, did):
    p = e.video_file(did)
    p.write_bytes(clip_bytes() + b"\x00" * 1024)     # same pictures, different size


FINGERPRINT_CHANGES = {
    "prompts": lambda e, did: change_prompts(e),
    "fps": lambda e, did: change_fps(e),
    "chunk_seconds": lambda e, did: change_chunk_seconds(e),
    "max_height": lambda e, did: change_height(e),
    "clock_offset": change_offset,
    "video_mtime": touch_video,
    "video_replaced": replace_video,
}


@pytest.mark.parametrize("name", sorted(FINGERPRINT_CHANGES))
def test_a_changed_fingerprint_starts_clean(rs10_env, name):
    e = rs10_env
    did = interrupted(e)
    FINGERPRINT_CHANGES[name](e, did)
    p2 = spy()
    res, exc = H.run(e, did, p2)
    assert exc is None, exc
    assert_clean_start(e, did, p2)
    assert H.chunk_names(p2)[0] == "chunk_0000.mp4"
    assert e.counts(did)["verdicts"] == len(e.candidates(did))


def test_a_different_provider_name_starts_clean(rs10_env):
    e = rs10_env
    did = e.upload()
    p1 = H.fail_detect_at(1)
    p1.name = "provider-one"
    H.run(e, did, p1)
    assert status_of(e.get(did)) in FAILED
    p2 = spy()
    p2.name = "provider-two"
    H.run(e, did, p2)
    assert_clean_start(e, did, p2, expect_chunks=H.CHUNK_FILES)


def test_the_same_provider_name_on_a_new_instance_resumes(rs10_env):
    e = rs10_env
    did = e.upload()
    p1 = H.fail_detect_at(1)
    p1.name = "same-provider"
    H.run(e, did, p1)
    p2 = spy()
    p2.name = "same-provider"
    H.run(e, did, p2)
    assert_resumed(e, did, p2, ["chunk_0001.mp4", "chunk_0002.mp4"])


def test_prompts_are_fingerprinted_by_content_not_by_path(rs10_env):
    e = rs10_env
    root = Path(__import__("backend").__file__).resolve().parent.parent
    copy_a = e.tmp / "prompts_a.json"
    copy_b = e.tmp / "elsewhere" / "prompts_b.json"
    copy_b.parent.mkdir()
    shutil.copy(root / "config" / "hazard_prompts.json", copy_a)
    shutil.copy(root / "config" / "hazard_prompts.json", copy_b)
    e.mp.setenv("ROADSCOUT_HAZARD_PROMPTS", str(copy_a))
    did = interrupted(e)
    e.mp.setenv("ROADSCOUT_HAZARD_PROMPTS", str(copy_b))
    p2 = spy()
    H.run(e, did, p2)
    assert_resumed(e, did, p2, ["chunk_0001.mp4", "chunk_0002.mp4"])


def test_unchanged_settings_resume_even_with_unrelated_env_changes(rs10_env):
    e = rs10_env
    did = interrupted(e)
    e.mp.setenv("ROADSCOUT_VERIFY_MIN_CONFIDENCE", "0.5")
    e.mp.setenv("ROADSCOUT_SPEND_CAP_USD", "50")
    p2 = spy()
    H.run(e, did, p2)
    assert_resumed(e, did, p2, ["chunk_0001.mp4", "chunk_0002.mp4"])


def test_video_identity_is_size_and_mtime_not_its_path(rs10_env):
    """Reading the same file back (no change) keeps the fingerprint: resume works after a plain stat."""
    e = rs10_env
    did = interrupted(e)
    e.video_file(did).stat()
    p2 = spy()
    H.run(e, did, p2)
    assert_resumed(e, did, p2, ["chunk_0001.mp4", "chunk_0002.mp4"])


def test_clean_start_after_a_fingerprint_change_leaves_no_stale_chunks(rs10_env):
    e = rs10_env
    did = interrupted(e, fail_at=2)
    change_chunk_seconds(e)
    p2 = spy()
    H.run(e, did, p2)
    assert status_of(e.get(did)) in DONE
    chunk_values = sorted(r["chunk"] for r in e.q("SELECT DISTINCT chunk FROM detections WHERE drive_id = ?", (did,)))
    assert chunk_values == list(range(len(H.work_chunks(e)))) or set(chunk_values) <= set(range(len(H.work_chunks(e))))
    assert len(H.work_chunks(e)) == 2, "12 s at 6 s chunks"


# ------------------------------------------------------------------ clear_results and interruptions

def test_clear_results_removes_the_checkpoints(rs10_env):
    from backend.pipeline import clear_results

    e = rs10_env
    did = interrupted(e)
    clear_results(did, db_path=e.db)
    assert e.counts(did)["detections"] == 0
    p2 = spy()
    H.run(e, did, p2)
    assert_clean_start(e, did, p2, expect_chunks=H.CHUNK_FILES)


def test_server_stop_mid_run_marks_interrupted_and_the_next_call_resumes(rs10_env):
    from backend.pipeline import recover_interrupted

    e = rs10_env
    did = interrupted_via_config(e)
    con = sqlite3.connect(str(e.db))
    try:
        with con:   # what a killed server leaves behind
            con.execute("UPDATE drives SET status = 'processing', stage = 'detecting', error = NULL, "
                        "failed_stage = NULL WHERE id = ?", (did,))
    finally:
        con.close()
    recover_interrupted()
    d = e.get(did)
    assert status_of(d) in FAILED and "interrupted" in (d.get("error") or "")
    assert e.counts(did)["detections"] > 0, "the finished chunks survive the interruption"
    p2 = spy()
    e.use_provider(p2)
    assert H.http_process(e, did).status_code == 202
    e.wait(did)
    assert_resumed(e, did, p2, ["chunk_0001.mp4", "chunk_0002.mp4"])


# ------------------------------------------------------------------ JSON fields and migration

def test_new_drive_json_has_the_resume_fields(rs10_env):
    e = rs10_env
    did = e.upload()
    d = e.get(did)
    assert d["resumed"] is False
    assert d["resumed_from"] is None


def test_list_endpoint_has_the_resume_fields(rs10_env):
    e = rs10_env
    e.upload()
    body = e.client.get("/api/drives").json()
    items = body if isinstance(body, list) else body.get("drives", body.get("items", []))
    assert items
    for d in items:
        assert d["resumed"] is False and d["resumed_from"] is None


def test_resumed_fields_in_the_process_response_and_get(rs10_env):
    e = rs10_env
    did = interrupted_via_config(e)
    e.use_provider(spy())
    r = H.http_process(e, did)
    assert r.status_code == 202
    body = r.json()
    assert "resumed" in body and "resumed_from" in body
    e.wait(did)
    d = e.get(did)
    assert d["resumed"] is True and d["resumed_from"] == "detecting"


def _cols(path):
    con = sqlite3.connect(str(path))
    try:
        return {r[1]: r for r in con.execute("PRAGMA table_info(drives)")}
    finally:
        con.close()


def test_drives_table_has_the_new_columns(tmp_path):
    from backend.db import init_db

    path = tmp_path / "m.sqlite"
    init_db(path)
    cols = _cols(path)
    assert "resumed" in cols and "resumed_from" in cols


def test_migration_adds_the_columns_to_an_old_database(tmp_path):
    from backend.db import init_db

    path = tmp_path / "old.sqlite"
    init_db(path)
    con = sqlite3.connect(str(path))
    try:
        info = [(r[1], r[3]) for r in con.execute("PRAGMA table_info(drives)")]
        names = [c for c, nn in info if (nn or c in ("id", "status")) and c not in ("resumed", "resumed_from")]
        con.execute("INSERT INTO drives (%s) VALUES (%s)" % (",".join(names), ",".join("?" * len(names))),
                    ["old-1" if c == "id" else ("failed" if c == "status" else "x") for c in names])
        con.commit()
        try:
            con.execute("ALTER TABLE drives DROP COLUMN resumed")
            con.execute("ALTER TABLE drives DROP COLUMN resumed_from")
        except sqlite3.OperationalError as e:
            pytest.skip("this SQLite cannot drop a column (%s)" % e)
        con.commit()
    finally:
        con.close()
    assert "resumed" not in _cols(path)
    init_db(path)
    init_db(path)
    cols = _cols(path)
    assert "resumed" in cols and "resumed_from" in cols
    con = sqlite3.connect(str(path))
    try:
        assert con.execute("SELECT id, status FROM drives WHERE id = 'old-1'").fetchone() == ("old-1", "failed")
    finally:
        con.close()


def test_checkpoint_storage_is_created_by_init_db_and_idempotent(tmp_path):
    from backend.db import init_db

    path = tmp_path / "c.sqlite"
    init_db(path)
    con = sqlite3.connect(str(path))
    try:
        before = sorted(r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'"))
    finally:
        con.close()
    init_db(path)
    con = sqlite3.connect(str(path))
    try:
        after = sorted(r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'"))
    finally:
        con.close()
    assert before == after
