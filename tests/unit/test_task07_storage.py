"""Task 7: raw detections are stored in the `detections` table and read back unchanged."""
import json
import re
import sqlite3

import pytest

from rs7_helpers import (  # noqa: F401  (fixtures are imported for pytest)
    CHUNK_SECONDS,
    FPS,
    RecordingProvider,
    det,
    frame,
    result,
    rs7_env,
    rs7_work,
    signature_rows,
)

PER_CHUNK_FAKE = 12  # the fake "default" scenario: a 12-frame track


def _expected_fake_rows(env, rows):
    """What the fake provider returns per chunk, using the prompt the code actually sent."""
    from backend.providers import FakeProvider

    exp = {}
    prompts = {r["chunk"]: r["prompt"] for r in rows}
    for c in sorted(prompts):
        res = FakeProvider("default").detect(str(env.work_dir / ("chunk_%04d.mp4" % c)), [prompts[c]])
        exp[c] = [d for _, d in res.all_detections()], [f for f, _ in res.all_detections()]
    return exp


# ------------------------------------------------------------------ schema


def test_init_db_creates_the_detections_table_with_planned_columns(monkeypatch, tmp_path):
    monkeypatch.setenv("ROADSCOUT_DB", str(tmp_path / "schema.sqlite"))
    from backend.db import init_db

    init_db()
    init_db()  # safe to repeat
    conn = sqlite3.connect(str(tmp_path / "schema.sqlite"))
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(detections)")]
    finally:
        conn.close()
    assert cols, "init_db() did not create a detections table"
    low = [c.lower() for c in cols]
    for needed in ("id", "drive_id", "chunk", "mask"):
        assert needed in low, "column %r missing from detections: %s" % (needed, cols)
    assert any("timestamp" in c for c in low), "no frame timestamp column: %s" % cols
    assert any("prompt" in c or c == "class" for c in low), "no class prompt column: %s" % cols
    assert any("track" in c for c in low), "no track id column: %s" % cols
    assert any("box" in c or c in ("x1", "bbox") for c in low), "no box column: %s" % cols
    assert any("frame" in c and "timestamp" not in c for c in low), "no frame index column: %s" % cols


def test_running_detection_creates_the_table_even_if_init_db_was_not_rerun(rs7_env):
    conn = sqlite3.connect(str(rs7_env.db))
    try:
        conn.execute("DROP TABLE IF EXISTS detections")
        conn.commit()
    finally:
        conn.close()
    rs7_env.run(provider=RecordingProvider())
    assert len(rs7_env.rows()) == 3 * PER_CHUNK_FAKE


# ------------------------------------------------------------------ default (fake) behaviour


def test_fake_default_stores_one_row_per_frame_detection_per_chunk(rs7_env):
    rs7_env.run(provider=RecordingProvider())
    rows = rs7_env.rows()
    assert len(rows) == 3 * PER_CHUNK_FAKE
    for c in (0, 1, 2):
        assert sum(1 for r in rows if r["chunk"] == c) == PER_CHUNK_FAKE
    assert all(r["raw"]["drive_id"] == "drive-a" for r in rows)
    ids = [r["id"] for r in rows]
    assert None not in ids and len(set(ids)) == len(ids), "row ids must be unique and non-null"


def test_stored_boxes_masks_scores_prompts_read_back_unchanged(rs7_env):
    rs7_env.run(provider=RecordingProvider())
    rows = rs7_env.rows()
    exp = _expected_fake_rows(rs7_env, rows)
    for c in (0, 1, 2):
        mine = sorted((r for r in rows if r["chunk"] == c), key=lambda r: r["frame_index"])
        dets, _ = exp[c]
        assert len(mine) == len(dets)
        for r, d in zip(mine, dets):
            assert r["box"] == pytest.approx(tuple(d.box))
            assert r["mask"] == d.mask, "mask changed on the way through the database"
            assert r["score"] == pytest.approx(d.score)
            assert r["prompt"] == d.prompt


def test_global_timestamps_come_from_chunk_start_plus_frame_time(rs7_env):
    rs7_env.run(provider=RecordingProvider())
    ts = rs7_env.ts
    by_global = {f["index"]: f["timestamp"] for f in ts["frames"]}
    rows = rs7_env.rows()
    for r in rows:
        c = r["chunk"]
        ch = ts["chunks"][c]
        k = round((r["ts"] - ch["start_time"]) * FPS)
        assert 0 <= k < ch["frame_count"], "timestamp %s not inside chunk %d" % (r["ts"], c)
        assert r["ts"] == pytest.approx(ch["start_time"] + k / FPS, abs=1e-3)
        assert r["ts"] == pytest.approx(by_global[ch["start_frame"] + k], abs=1e-3)
    # chunk 2's frames really are in the later part of the drive, not chunk-relative
    assert min(r["ts"] for r in rows if r["chunk"] == 2) == pytest.approx(2 * CHUNK_SECONDS, abs=1e-3)
    assert max(r["ts"] for r in rows if r["chunk"] == 2) < 3 * CHUNK_SECONDS + 1e-6


def test_frame_index_is_consistent_per_chunk(rs7_env):
    """Frame index is either the global index (preferred) or the chunk-relative one, never mixed."""
    rs7_env.run(provider=RecordingProvider())
    ts = rs7_env.ts
    rows = rs7_env.rows()
    glob_ok = all(r["frame_index"] == ts["chunks"][r["chunk"]]["start_frame"] + round(
        (r["ts"] - ts["chunks"][r["chunk"]]["start_time"]) * FPS) for r in rows)
    rel_ok = all(r["frame_index"] == round((r["ts"] - ts["chunks"][r["chunk"]]["start_time"]) * FPS) for r in rows)
    assert glob_ok or rel_ok, "frame_index is neither the global nor the in-chunk index"


# ------------------------------------------------------------------ exact value round-trip with a scripted provider


def _scripted_result(chunk):
    if chunk == 0:
        return result(
            frame(0, [det(1, (10.5, 20.25, 60.75, 80.0), 0.123456, "pothole"),
                      det(2, (0, 0, 159, 89), 1.0, "debris on the road", mask=None)]),
            frame(1, []),
            frame(2, [det(1, (12, 22, 64, 84), 0.5, "pothole")]),
        )
    return None


def test_multiple_detections_per_frame_none_mask_and_float_precision_round_trip(rs7_env):
    from backend.providers import rle_from_box

    rec = RecordingProvider(script=lambda n, c, p: _scripted_result(c) if c == 0 else result())
    rs7_env.run(provider=rec)
    rows = sorted(rs7_env.rows(), key=lambda r: (r["frame_index"], r["box"]))
    assert len(rows) == 3, "frame 1 has no detections, so exactly 3 rows expected, got %d" % len(rows)
    assert all(r["chunk"] == 0 for r in rows)
    a, b, c = rows
    base = rows[0]["frame_index"]
    # frame 0: two detections
    assert a["frame_index"] == b["frame_index"] == base
    by_prompt = {r["prompt"]: r for r in (a, b)}
    p = by_prompt["pothole"]
    d = by_prompt["debris on the road"]
    assert p["box"] == (10.5, 20.25, 60.75, 80.0)
    assert p["score"] == pytest.approx(0.123456, abs=1e-9)
    assert p["mask"] == rle_from_box(160, 90, (10.5, 20.25, 60.75, 80.0))
    assert d["box"] == (0.0, 0.0, 159.0, 89.0)
    assert d["score"] == pytest.approx(1.0)
    assert d["mask"] is None, "a detection without a mask must read back as no mask"
    # frame 2 is two frames later
    assert c["frame_index"] == base + 2
    assert c["ts"] == pytest.approx(a["ts"] + 2 / FPS, abs=1e-3)
    assert c["box"] == (12.0, 22.0, 64.0, 84.0)


def test_unicode_and_odd_prompt_text_survives(rs7_env):
    weird = 'pothole "deep" / café — 100%'
    rec = RecordingProvider(script=lambda n, c, p: result(frame(0, [det(7, (1, 1, 5, 5), 0.9, weird)])) if c == 1 else result())
    rs7_env.run(provider=rec)
    rows = rs7_env.rows()
    assert len(rows) == 1 and rows[0]["prompt"] == weird and rows[0]["chunk"] == 1


# ------------------------------------------------------------------ empty results


def test_empty_result_stores_nothing_and_is_not_an_error(rs7_env):
    rec = RecordingProvider(script=lambda n, c, p: result())
    rs7_env.run(provider=rec)  # must not raise
    assert rs7_env.all_count() == 0
    assert len(rec.calls) >= 3, "every chunk is still sent even if nothing is found"


def test_frames_with_no_detections_store_nothing(rs7_env):
    rec = RecordingProvider(script=lambda n, c, p: result(frame(0, []), frame(1, []), frame(2, [])))
    rs7_env.run(provider=rec)
    assert rs7_env.all_count() == 0


def test_only_chunks_with_detections_get_rows(rs7_env):
    rec = RecordingProvider(
        script=lambda n, c, p: result(frame(0, [det(1, (5, 5, 50, 50), 0.8, "pothole")])) if c == 1 else result())
    rs7_env.run(provider=rec)
    rows = rs7_env.rows()
    assert [r["chunk"] for r in rows] == [1]


# ------------------------------------------------------------------ track ids across chunks


def _track_script(n, c, p):
    if c == 0:
        return result(
            frame(0, [det(1, (5, 5, 50, 50), 0.8, "pothole")]),
            frame(1, [det(1, (6, 6, 52, 52), 0.8, "pothole"), det(2, (60, 60, 90, 80), 0.7, "pothole")]),
            frame(2, [det(1, (7, 7, 54, 54), 0.8, "pothole"), det(2, (61, 61, 91, 81), 0.7, "pothole")]),
        )
    if c == 1:
        return result(frame(0, [det(1, (5, 5, 50, 50), 0.9, "pothole")]),
                      frame(1, [det(1, (6, 6, 52, 52), 0.9, "pothole")]))
    return result()


def test_same_raw_track_id_in_different_chunks_does_not_collide(rs7_env):
    rs7_env.run(provider=RecordingProvider(script=_track_script))
    rows = rs7_env.rows()
    assert len(rows) == 7  # chunk 0: 1 + 2 + 2, chunk 1: 2
    # raw ids: chunk0 has raw 1 (x3) and 2 (x2); chunk1 has raw 1 (x2)
    pairs = {}
    for r in rows:
        pairs.setdefault(r["chunk"], []).append(r)
    cols = sorted(rows[0]["tracks"])

    def raw_of(r, col):
        from rs7_helpers import _last_int

        return _last_int(r["tracks"][col])

    # 1. at least one track column keeps the provider's raw id readable
    raw_cols = [c for c in cols
                if all((raw_of(r, c) in (1, 2)) for r in rows)
                and {raw_of(r, c) for r in pairs[0]} == {1, 2}
                and {raw_of(r, c) for r in pairs[1]} == {1}]
    assert raw_cols, "no track column preserves the raw provider track ids (columns: %s)" % cols

    # 2. at least one track column is unique per (chunk, raw id) and stable inside a chunk
    def key_for(col):
        mapping = {}
        for r in rows:
            raw = [raw_of(rr, raw_cols[0]) for rr in [r]][0]
            mapping.setdefault((r["chunk"], raw), set()).add(str(r["tracks"][col]))
        return mapping

    good = []
    for col in cols:
        mapping = key_for(col)
        if all(len(v) == 1 for v in mapping.values()):
            values = [next(iter(v)) for v in mapping.values()]
            if len(set(values)) == len(values) == 3:
                good.append(col)
    assert good, "no track column namespaces track ids per chunk: (chunk 0, raw 1) and (chunk 1, raw 1) collide"


def test_track_continuity_inside_a_chunk_is_kept(rs7_env):
    rs7_env.run(provider=RecordingProvider(script=_track_script))
    rows = [r for r in rs7_env.rows() if r["chunk"] == 0]
    boxes_by_first_x = {}
    for r in rows:
        key = "t1" if r["box"][0] < 20 else "t2"
        boxes_by_first_x.setdefault(key, []).append(r)
    assert len(boxes_by_first_x["t1"]) == 3 and len(boxes_by_first_x["t2"]) == 2
    for col in rows[0]["tracks"]:
        vals1 = {str(r["tracks"][col]) for r in boxes_by_first_x["t1"]}
        vals2 = {str(r["tracks"][col]) for r in boxes_by_first_x["t2"]}
        if len(vals1) == 1 and len(vals2) == 1 and vals1 != vals2:
            return
    pytest.fail("the 3 sightings of one tracked object and the 2 of another are not distinguishable by a track column")


# ------------------------------------------------------------------ idempotency


def test_running_twice_does_not_double_insert(rs7_env):
    rs7_env.run(provider=RecordingProvider())
    first = signature_rows(rs7_env.rows())
    n1 = rs7_env.all_count()
    rs7_env.run(provider=RecordingProvider())
    second = signature_rows(rs7_env.rows())
    assert rs7_env.all_count() == n1 == 3 * PER_CHUNK_FAKE
    assert first == second, "rerun changed the stored detections"


def test_third_run_still_the_same(rs7_env):
    for _ in range(3):
        rs7_env.run(provider=RecordingProvider())
    assert rs7_env.all_count() == 3 * PER_CHUNK_FAKE


def test_rerun_of_one_drive_leaves_other_drives_alone(rs7_env):
    rs7_env.add_drive("drive-b")
    rs7_env.run("drive-a", provider=RecordingProvider())
    rs7_env.run("drive-b", provider=RecordingProvider(
        script=lambda n, c, p: result(frame(0, [det(3, (5, 5, 50, 50), 0.6, "pothole")])) if c == 0 else result()))
    b_before = signature_rows(rs7_env.rows("drive-b"))
    assert len(b_before) == 1
    rs7_env.run("drive-a", provider=RecordingProvider())
    assert rs7_env.all_count() == 3 * PER_CHUNK_FAKE + 1
    assert signature_rows(rs7_env.rows("drive-b")) == b_before
    assert len(rs7_env.rows("drive-a")) == 3 * PER_CHUNK_FAKE


def test_two_drives_over_the_same_chunks_both_get_full_rows(rs7_env):
    rs7_env.add_drive("drive-b")
    rs7_env.run("drive-a", provider=RecordingProvider())
    rs7_env.run("drive-b", provider=RecordingProvider())
    assert len(rs7_env.rows("drive-a")) == len(rs7_env.rows("drive-b")) == 3 * PER_CHUNK_FAKE


def test_rerun_after_empty_first_run_inserts_rows(rs7_env):
    rs7_env.run(provider=RecordingProvider(script=lambda n, c, p: result()))
    assert rs7_env.all_count() == 0
    rs7_env.run(provider=RecordingProvider())
    assert rs7_env.all_count() == 3 * PER_CHUNK_FAKE


# ------------------------------------------------------------------ provider errors


def _error_cases():
    from backend.providers import (
        ProviderConfigError,
        ProviderTimeoutError,
        RetriesExhausted,
        SpendCapExceeded,
        TransientProviderError,
    )

    return [
        ("retries_exhausted", lambda: RetriesExhausted("gave up after 3 attempts: HTTP 503", 3)),
        ("hard_config", lambda: ProviderConfigError("bad request: video too long")),
        ("spend_cap", lambda: SpendCapExceeded("spend cap reached")),
        ("timeout", lambda: ProviderTimeoutError("detect timed out")),
        ("transient", lambda: TransientProviderError("rate limited")),
    ]


@pytest.mark.parametrize("name", ["retries_exhausted", "hard_config", "spend_cap", "timeout", "transient"])
def test_provider_error_surfaces_and_leaves_no_rows_for_the_failing_chunk(rs7_env, name):
    from backend.providers import ProviderError

    make = dict(_error_cases())[name]

    def script(n, c, p):
        if c == 1:
            raise make()
        return None  # delegate to the fake provider

    rec = RecordingProvider(script=script)
    with pytest.raises(Exception) as ei:
        rs7_env.run(provider=rec)
    exc = ei.value
    chain = []
    e = exc
    while e is not None and e not in chain:
        chain.append(e)
        e = e.__cause__ or e.__context__
    assert any(isinstance(x, ProviderError) for x in chain), \
        "the provider error was swallowed or replaced by an unrelated exception: %r" % (exc,)
    text = " ".join(str(x) for x in chain).lower()
    assert re.search(r"chunk[_ ]?0*1\b", text) or "chunk_0001" in text, \
        "the error should say which chunk failed: %r" % (exc,)
    rows = rs7_env.rows()
    assert all(r["chunk"] != 1 for r in rows), "partial rows left for the failing chunk"
    assert all(r["chunk"] in (0, 2) for r in rows)


def test_failed_run_can_be_retried_cleanly(rs7_env):
    from backend.providers import RetriesExhausted

    def script(n, c, p):
        if c == 1:
            raise RetriesExhausted("boom", 3)
        return None

    with pytest.raises(Exception):
        rs7_env.run(provider=RecordingProvider(script=script))
    rs7_env.run(provider=RecordingProvider())  # the provider recovered
    rows = rs7_env.rows()
    assert len(rows) == 3 * PER_CHUNK_FAKE, "after a failed run and a good rerun every chunk is stored exactly once"


def test_verification_is_never_called_by_detection(rs7_env):
    rec = RecordingProvider()  # verify() raises NotImplementedError if called
    rs7_env.run(provider=rec)
    assert rs7_env.all_count() > 0
