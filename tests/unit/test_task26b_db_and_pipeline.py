"""Task 26b: the drives.warning column, its migration, and the pipeline warning for skipped candidates."""
from __future__ import annotations

import re
import sqlite3

import pytest

import rs26b_setup  # noqa: F401
from rs10_helpers import DONE, FAILED, rs10_env, spy, status_of  # noqa: F401


def exhausted(msg="boom-pipe-exhausted"):
    from backend.providers import RetriesExhausted

    return RetriesExhausted(msg, 5)


def timed_out(msg="boom-pipe-timeout"):
    from backend.providers import ProviderTimeoutError

    return ProviderTimeoutError(msg)


def failing_on(indexes, make=exhausted):
    """on_verify hook for the spy: raise a transient error on the given verify call numbers."""
    wanted = set(indexes)

    def hook(s, n):
        if n in wanted:
            raise make()

    return hook


def warning_of(d):
    assert "warning" in d, "the drive JSON must have a 'warning' key: %s" % sorted(d)
    return d["warning"]


# ------------------------------------------------------------------ db column and migration

def _cols(path):
    con = sqlite3.connect(str(path))
    try:
        return {r[1]: r for r in con.execute("PRAGMA table_info(drives)")}
    finally:
        con.close()


def test_drives_table_has_a_nullable_text_warning_column(tmp_path):
    from backend.db import init_db

    path = tmp_path / "w.sqlite"
    init_db(path)
    cols = _cols(path)
    assert "warning" in cols
    _cid, _name, ctype, notnull, default, _pk = cols["warning"]
    assert ctype.upper() == "TEXT"
    assert notnull == 0


def test_migration_adds_the_column_to_an_old_database_and_keeps_rows(tmp_path):
    from backend.db import init_db

    path = tmp_path / "old.sqlite"
    init_db(path)
    con = sqlite3.connect(str(path))
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(drives)")]
        values = {"id": "old-drive-1", "status": "done"}
        notnull = {r[1]: r[3] for r in con.execute("PRAGMA table_info(drives)")}
        names, params = [], []
        for c in cols:
            if c in values:
                names.append(c)
                params.append(values[c])
            elif notnull[c] and c != "warning":
                names.append(c)
                params.append("x" if c not in ("size_bytes",) else 1)
        con.execute("INSERT INTO drives (%s) VALUES (%s)" % (",".join(names), ",".join("?" * len(names))), params)
        con.commit()
        try:
            con.execute("ALTER TABLE drives DROP COLUMN warning")
        except sqlite3.OperationalError as e:
            pytest.skip("this SQLite cannot drop a column (%s)" % e)
        con.commit()
    finally:
        con.close()
    assert "warning" not in _cols(path)
    init_db(path)
    init_db(path)  # idempotent
    assert "warning" in _cols(path)
    con = sqlite3.connect(str(path))
    try:
        row = con.execute("SELECT id, status, warning FROM drives WHERE id = 'old-drive-1'").fetchone()
    finally:
        con.close()
    assert row == ("old-drive-1", "done", None)


def test_init_db_twice_keeps_one_warning_column(tmp_path):
    from backend.db import init_db

    path = tmp_path / "twice.sqlite"
    init_db(path)
    init_db(path)
    assert [c for c in _cols(path) if c == "warning"] == ["warning"]


def test_warning_value_survives_init_db(tmp_path):
    from backend.db import init_db

    path = tmp_path / "keep.sqlite"
    init_db(path)
    con = sqlite3.connect(str(path))
    try:
        cols = {r[1]: r[3] for r in con.execute("PRAGMA table_info(drives)")}
        names = [c for c, nn in cols.items() if nn and c != "warning"]
        con.execute("INSERT INTO drives (%s, warning) VALUES (%s, ?)" % (",".join(names), ",".join("?" * len(names))),
                    ["d1" if c == "id" else ("done" if c == "status" else "x") for c in names] + ["kept warning"])
        con.commit()
    finally:
        con.close()
    init_db(path)
    con = sqlite3.connect(str(path))
    try:
        assert con.execute("SELECT warning FROM drives").fetchone()[0] == "kept warning"
    finally:
        con.close()


# ------------------------------------------------------------------ drive JSON

def test_new_drive_json_has_warning_null(rs10_env):
    e = rs10_env
    did = e.upload()
    d = e.get(did)
    assert warning_of(d) is None


def test_drive_list_entries_have_the_warning_key(rs10_env):
    e = rs10_env
    e.upload()
    r = e.client.get("/api/drives")
    assert r.status_code == 200
    body = r.json()
    items = body if isinstance(body, list) else body.get("drives", body.get("items", []))
    assert items
    for d in items:
        assert "warning" in d and d["warning"] is None


# ------------------------------------------------------------------ pipeline

def test_clean_run_has_no_warning(rs10_env):
    e = rs10_env
    did = e.upload()
    p = spy()
    e.run(did, p)
    d = e.get(did)
    assert status_of(d) in DONE
    assert warning_of(d) is None
    assert e.row(did)["warning"] is None
    assert len(p.verify_calls) == 3


@pytest.mark.parametrize("make", [exhausted, timed_out], ids=["RetriesExhausted", "ProviderTimeoutError"])
def test_skipped_candidate_still_finishes_done_with_a_warning(rs10_env, make):
    e = rs10_env
    did = e.upload()
    p = spy(on_verify=failing_on([1], make))
    res, exc = e.run(did, p)
    assert exc is None, exc
    d = e.get(did)
    assert status_of(d) in DONE, d
    assert d.get("error") in (None, "")
    w = warning_of(d)
    assert isinstance(w, str) and w.strip()
    assert e.row(did)["warning"] == w
    assert len(p.verify_calls) == 3, "all three candidates were tried"
    c = e.counts(did)
    assert c["candidates"] == 3
    assert c["verdicts"] == 2, "the skipped candidate has no verdict"


def test_warning_text_says_how_many_were_skipped_and_what_to_do(rs10_env):
    e = rs10_env
    did = e.upload()
    e.run(did, spy(on_verify=failing_on([2])))
    w = warning_of(e.get(did))
    assert re.search(r"\b1 of 3\b", w), w
    low = w.lower()
    assert "could not be verified" in low
    assert "network" in low
    assert "skipped" in low
    assert "process the drive again" in low
    assert "\n" not in w.strip()


def test_two_skipped_of_three_counts_both(rs10_env):
    e = rs10_env
    did = e.upload()
    e.run(did, spy(on_verify=failing_on([0, 2])))
    d = e.get(did)
    assert status_of(d) in DONE
    assert re.search(r"\b2 of 3\b", warning_of(d)), d["warning"]
    assert e.counts(did)["verdicts"] == 1


def test_warning_has_no_paths_and_no_key(rs10_env):
    e = rs10_env
    did = e.upload()
    e.run(did, spy(on_verify=failing_on([1], lambda: exhausted("C:\\secret\\dir\\frame.jpg Bearer abc123"))))
    w = warning_of(e.get(did))
    assert "secret" not in w and "abc123" not in w and "Bearer" not in w
    assert str(e.tmp) not in w
    assert not re.search(r"[A-Za-z]:\\|/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+|\.jpg|\.mp4", w), w
    assert "key" not in w.lower()


def test_skipped_candidate_never_becomes_a_hazard_link(rs10_env):
    e = rs10_env
    did = e.upload()
    e.run(did, spy(on_verify=failing_on([1])))
    verdict_ids = {v["candidate_id"] for v in e.verdicts(did)}
    skipped = [c for c in e.candidates(did) if c["id"] not in verdict_ids]
    assert len(skipped) == 1
    assert skipped[0]["hazard_id"] is None


def test_rerun_without_skips_clears_the_old_warning(rs10_env):
    e = rs10_env
    did = e.upload()
    e.run(did, spy(on_verify=failing_on([1])))
    assert warning_of(e.get(did))
    e.run(did, spy())
    d = e.get(did)
    assert status_of(d) in DONE
    assert warning_of(d) is None
    assert e.row(did)["warning"] is None
    assert e.counts(did)["verdicts"] == 3


def test_new_run_resets_the_warning_as_soon_as_it_starts(rs10_env):
    e = rs10_env
    did = e.upload()
    e.run(did, spy(on_verify=failing_on([1])))
    assert warning_of(e.get(did))
    seen = []

    def hook(s, n):
        if n == 0:
            seen.append(e.row(did)["warning"])

    e.run(did, spy(on_verify=hook))
    assert seen == [None], "the old warning must be gone while the next run is still verifying"


def test_run_with_a_different_skip_replaces_the_text(rs10_env):
    e = rs10_env
    did = e.upload()
    e.run(did, spy(on_verify=failing_on([1])))
    assert re.search(r"\b1 of 3\b", warning_of(e.get(did)))
    e.run(did, spy(on_verify=failing_on([0, 1])))
    w = warning_of(e.get(did))
    assert re.search(r"\b2 of 3\b", w) and not re.search(r"\b1 of 3\b", w)


def test_clear_results_resets_the_warning(rs10_env):
    from backend.pipeline import clear_results

    e = rs10_env
    did = e.upload()
    e.run(did, spy(on_verify=failing_on([1])))
    assert warning_of(e.get(did))
    clear_results(did, db_path=e.db)
    assert e.row(did)["warning"] is None
    assert warning_of(e.get(did)) is None


def test_all_candidates_failing_fails_the_drive_without_a_warning(rs10_env):
    e = rs10_env
    did = e.upload()
    e.run(did, spy(on_verify=failing_on([0, 1, 2])))
    d = e.get(did)
    assert status_of(d) in FAILED
    assert "verif" in (d.get("error") or "").lower()
    assert warning_of(d) is None
    c = e.counts(did)
    assert c["verdicts"] == 0 and c["all_verdicts"] == 0


def test_failed_run_after_a_warning_run_has_no_stale_warning(rs10_env):
    e = rs10_env
    did = e.upload()
    e.run(did, spy(on_verify=failing_on([1])))
    assert warning_of(e.get(did))
    e.run(did, spy(on_verify=failing_on([0, 1, 2])))
    d = e.get(did)
    assert status_of(d) in FAILED
    assert warning_of(d) is None


def test_permanent_provider_error_still_fails_the_drive(rs10_env):
    from backend.providers import ProviderError

    e = rs10_env
    did = e.upload()

    def hook(s, n):
        if n == 1:
            raise ProviderError("Meta API returned HTTP 402: no credits")

    e.run(did, spy(on_verify=hook))
    d = e.get(did)
    assert status_of(d) in FAILED
    assert "402" in (d.get("error") or "")
    assert warning_of(d) is None
    assert e.counts(did)["all_verdicts"] == 1, "task 26c: the verdict obtained before the abort stays stored"


def test_spend_cap_error_still_fails_the_drive(rs10_env):
    from backend.providers import SpendCapExceeded

    e = rs10_env
    did = e.upload()

    def hook(s, n):
        if n == 1:
            raise SpendCapExceeded("cap reached")

    e.run(did, spy(on_verify=hook))
    d = e.get(did)
    assert status_of(d) in FAILED
    assert e.counts(did)["all_verdicts"] == 1, "task 26c: the verdict obtained before the abort stays stored"


def test_pipeline_result_carries_the_skipped_ids_when_it_exposes_the_summary(rs10_env):
    e = rs10_env
    did = e.upload()
    res, exc = e.run(did, spy(on_verify=failing_on([1])))
    assert exc is None
    summary = getattr(res, "verification", None)
    if summary is None:
        pytest.skip("the pipeline result does not expose the verification summary")
    assert len(summary.failed_candidate_ids) == 1
    verdict_ids = {v["candidate_id"] for v in e.verdicts(did)}
    assert summary.failed_candidate_ids[0] not in verdict_ids


def test_warning_via_http_process_route(rs10_env):
    e = rs10_env
    did = e.upload()
    e.use_provider(spy(on_verify=failing_on([1])))
    e.process_via_http(did)
    d = e.wait(did)
    assert status_of(d) in DONE
    assert warning_of(d) and re.search(r"\b1 of 3\b", d["warning"])
