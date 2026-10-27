"""Task 17 backend: user edits (class, severity) and status survive deduplication / pipeline re-runs."""
from datetime import timedelta

import pytest

from rs17_helpers import (FFMPEG, T0, VA, known_classes, offset, rs17_env, run_pipeline_for,  # noqa: F401
                          upload)

CLASSES = known_classes()


def one_hazard(env, *, cls="pothole", severity=1, conf=0.6, score=0.3):
    env.add_drive("A", T0)
    lat, lon = VA
    env.add_cand("A", lat, lon, 0.0, cls=cls, severity=severity, conf=conf, score=score,
                 frame_path=str(env.tmp / "fa.jpg"))
    env.dedup("A")
    hs = env.rows()
    assert len(hs) == 1
    return hs[0]


def test_class_edit_survives_plain_rerun(rs17_env):
    env = rs17_env
    h = one_hazard(env)
    env.ok(h["id"], hazard_class="damaged guardrail")
    env.dedup("A")
    env.dedup("A")
    rows = env.rows()
    assert len(rows) == 1 and rows[0]["id"] == h["id"]
    assert rows[0]["hazard_class"] == "damaged guardrail"
    assert env.by_id()[h["id"]]["hazard_class"] == "damaged guardrail"


def test_severity_edit_survives_plain_rerun(rs17_env):
    env = rs17_env
    h = one_hazard(env, severity=1)
    env.ok(h["id"], severity=3)
    env.dedup("A")
    r = env.rows()
    assert len(r) == 1 and r[0]["id"] == h["id"] and r[0]["severity"] == 3


def test_status_survives_rerun_after_approve_and_reject(rs17_env):
    env = rs17_env
    h = one_hazard(env)
    env.ok(h["id"], status="approved")
    env.dedup("A")
    assert env.row(h["id"])["status"] == "approved"
    env.ok(h["id"], status="rejected")
    env.dedup("A")
    assert env.row(h["id"])["status"] == "rejected"
    env.ok(h["id"], status="new")
    env.dedup("A")
    assert env.row(h["id"])["status"] == "new"


def test_edit_back_to_original_value_still_works(rs17_env):
    env = rs17_env
    h = one_hazard(env, cls="pothole", severity=1)
    env.ok(h["id"], hazard_class="damaged guardrail", severity=3)
    env.ok(h["id"], hazard_class="pothole", severity=1)
    env.dedup("A")
    r = env.row(h["id"])
    assert (r["hazard_class"], r["severity"]) == ("pothole", 1)


def test_new_sighting_updates_derived_fields_but_keeps_edits(rs17_env):
    env = rs17_env
    h = one_hazard(env, cls="pothole", severity=1, conf=0.6, score=0.3)
    env.ok(h["id"], hazard_class="damaged guardrail", severity=3)
    env.ok(h["id"], status="approved")
    before = env.row(h["id"])
    env.add_drive("B", T0 + timedelta(days=7))
    lat, lon = offset(*VA, north_m=3.0)
    pb = str(env.tmp / "fb.jpg")
    env.add_cand("B", lat, lon, 0.0, cls="pothole", severity=2, conf=0.95, score=0.99, frame_path=pb)
    env.dedup("B")
    rows = env.rows()
    assert len(rows) == 1, "second sighting must merge into the existing hazard"
    r = rows[0]
    assert r["id"] == h["id"]
    # edits and status are sticky
    assert r["hazard_class"] == "damaged guardrail"
    assert r["severity"] == 3
    assert r["status"] == "approved"
    # derived fields still move
    assert r["sightings"] == 2
    assert r["confidence"] >= before["confidence"]
    assert r["last_seen"] != before["last_seen"] or r["last_seen"] > before["last_seen"]
    assert r["photo_path"] == pb, "photo should follow the best-scoring candidate"
    # and a re-run of the first drive afterwards is stable too
    env.dedup("A")
    r2 = env.row(h["id"])
    assert (r2["hazard_class"], r2["severity"], r2["status"], r2["sightings"]) == ("damaged guardrail", 3, "approved", 2)


def test_class_edit_does_not_freeze_severity(rs17_env, tmp_path):
    """Editing only the class must leave severity derived: compare with an identical unedited control DB."""
    from backend.db import init_db

    env = rs17_env
    ctrl = tmp_path / "ctrl" / "c.sqlite"
    init_db(ctrl)

    def build(db, edit):
        e = env
        e.add_drive("A", T0, db=db)
        e.add_cand("A", VA[0], VA[1], 0.0, cls="pothole", severity=1, conf=0.6, score=0.3,
                   frame_path=str(tmp_path / "ca.jpg"), db=db)
        e.dedup("A", db=db)
        if edit:
            hid = e.rows()[0]["id"]
            e.ok(hid, hazard_class="damaged guardrail")
        e.add_drive("B", T0 + timedelta(days=7), db=db)
        lat, lon = offset(*VA, north_m=2.0)
        e.add_cand("B", lat, lon, 0.0, cls="pothole", severity=3, conf=0.97, score=0.99,
                   frame_path=str(tmp_path / "cb.jpg"), db=db)
        e.dedup("B", db=db)

    build(env.db, True)
    build(ctrl, False)
    import sqlite3
    con = sqlite3.connect(str(ctrl))
    ctrl_sev = con.execute("SELECT severity FROM hazards").fetchall()
    con.close()
    got = env.rows()
    assert len(got) == 1 and len(ctrl_sev) == 1
    assert got[0]["severity"] == ctrl_sev[0][0], "class-only edit froze severity"
    assert got[0]["hazard_class"] == "damaged guardrail"


def test_severity_edit_does_not_freeze_confidence_or_sightings(rs17_env):
    env = rs17_env
    h = one_hazard(env, conf=0.5, score=0.3)
    env.ok(h["id"], severity=3)
    env.add_drive("B", T0 + timedelta(days=3))
    env.add_drive("C", T0 + timedelta(days=6))
    for d in ("B", "C"):
        lat, lon = offset(*VA, north_m=1.0)
        env.add_cand(d, lat, lon, 0.0, cls="pothole", severity=1, conf=0.9, score=0.5,
                     frame_path=str(env.tmp / ("f%s.jpg" % d)))
        env.dedup(d)
    r = env.rows()
    assert len(r) == 1 and r[0]["severity"] == 3
    assert r[0]["sightings"] == 3
    assert r[0]["confidence"] > h["confidence"]


def test_edited_class_that_changes_clustering_does_not_crash(rs17_env):
    env = rs17_env
    h = one_hazard(env, cls="pothole", severity=1)
    env.ok(h["id"], hazard_class="damaged guardrail", severity=3)
    env.add_drive("B", T0 + timedelta(days=7))
    lat, lon = offset(*VA, north_m=2.0)
    env.add_cand("B", lat, lon, 0.0, cls="damaged guardrail", severity=2, conf=0.9, score=0.8,
                 frame_path=str(env.tmp / "fb.jpg"))
    env.dedup("B")
    env.dedup("A")
    env.dedup("B")
    rows = {r["id"]: r for r in env.rows()}
    assert h["id"] in rows, "the user-edited hazard vanished"
    assert rows[h["id"]]["hazard_class"] == "damaged guardrail"
    assert rows[h["id"]]["severity"] == 3
    assert 1 <= len(rows) <= 2
    assert sum(r["sightings"] for r in rows.values()) == 2
    cands = env.q("SELECT hazard_id FROM candidates")
    assert all(c["hazard_id"] in rows for c in cands), "candidate linked to a missing hazard"
    for hid in rows:
        assert env.q("SELECT COUNT(*) AS n FROM candidates WHERE hazard_id = ?", (hid,))[0]["n"] >= 1, "orphan hazard"
    assert {int(i["id"]) for i in env.hazards()} == set(rows)


def test_edit_to_class_of_neighbour_does_not_crash(rs17_env):
    env = rs17_env
    env.add_drive("A", T0)
    env.add_cand("A", VA[0], VA[1], 0.0, cls="pothole", frame_path=str(env.tmp / "f1.jpg"))
    lat, lon = offset(*VA, north_m=4.0)
    env.add_cand("A", lat, lon, 0.0, cls="debris in lane", frame_path=str(env.tmp / "f2.jpg"))
    env.dedup("A")
    rows = env.rows()
    assert len(rows) == 2
    pot = next(r for r in rows if r["hazard_class"] == "pothole")
    env.ok(pot["id"], hazard_class="debris in lane")
    env.dedup("A")
    env.dedup("A")
    rows2 = {r["id"]: r for r in env.rows()}
    assert pot["id"] in rows2 and 1 <= len(rows2) <= 2
    assert rows2[pot["id"]]["hazard_class"] == "debris in lane"


def test_unedited_hazard_is_still_recomputed(rs17_env):
    """Sticky is per hazard: an untouched sibling keeps being derived from its candidates."""
    env = rs17_env
    env.add_drive("A", T0)
    env.add_cand("A", VA[0], VA[1], 0.0, cls="pothole", severity=1, frame_path=str(env.tmp / "f1.jpg"))
    lat, lon = offset(*VA, north_m=500.0)
    env.add_cand("A", lat, lon, 0.0, cls="debris in lane", severity=2, frame_path=str(env.tmp / "f2.jpg"))
    env.dedup("A")
    rows = {r["hazard_class"]: r for r in env.rows()}
    env.ok(rows["pothole"]["id"], severity=3, hazard_class="faded lane markings")
    env.x("UPDATE hazards SET severity = 1 WHERE id = ?", (rows["debris in lane"]["id"],))
    env.dedup("A")
    after = {r["id"]: r for r in env.rows()}
    assert after[rows["pothole"]["id"]]["severity"] == 3
    assert after[rows["pothole"]["id"]]["hazard_class"] == "faded lane markings"
    sib = after[rows["debris in lane"]["id"]]
    assert (sib["hazard_class"], sib["severity"]) == ("debris in lane", 2), "untouched hazard not recomputed"


# ------------------------------------------------------------------ real pipeline runs (fake provider, 24 s clip)

@pytest.fixture()
def processed(rs17_env):
    if not FFMPEG:
        pytest.skip("ffmpeg not on PATH")
    env = rs17_env
    did = upload(env.client)
    res = run_pipeline_for(did, env.db)
    assert res.status == "done", getattr(res, "error", res)
    if not env.rows():
        pytest.skip("the fake provider produced no hazards")
    return env, did


def other_class(cls):
    return next(c for c in CLASSES if c != cls)


@pytest.fixture()
def processed2(processed):
    """Two identical drives processed: every hazard has two sightings, so re-running one drive keeps it alive."""
    env, d1 = processed
    d2 = upload(env.client)
    assert run_pipeline_for(d2, env.db).status == "done"
    return env, d1, d2


def test_pipeline_rerun_keeps_all_review_work(processed2):
    env, d1, d2 = processed2
    before = env.rows()
    plan = {}
    for i, h in enumerate(before):
        fields = {"hazard_class": other_class(h["hazard_class"]), "severity": h["severity"] % 3 + 1,
                  "status": ("approved", "rejected")[i % 2]}
        env.ok(h["id"], **fields)
        plan[h["id"]] = fields
    assert run_pipeline_for(d1, env.db).status == "done"
    after = {r["id"]: r for r in env.rows()}
    assert set(after) == set(plan), "hazard ids changed across a pipeline re-run"
    for hid, f in plan.items():
        assert after[hid]["hazard_class"] == f["hazard_class"]
        assert after[hid]["severity"] == f["severity"]
        assert after[hid]["status"] == f["status"]
        assert after[hid]["sightings"] == next(b for b in before if b["id"] == hid)["sightings"]
    got = env.by_id()
    for hid, f in plan.items():
        assert got[hid]["hazard_class"] == f["hazard_class"] and got[hid]["status"] == f["status"]


def test_pipeline_second_identical_drive_adds_sighting_but_keeps_edits(processed):
    env, did = processed
    h = env.rows()[0]
    new_cls, new_sev = other_class(h["hazard_class"]), h["severity"] % 3 + 1
    env.ok(h["id"], hazard_class=new_cls, severity=new_sev, status="approved")
    d2 = upload(env.client)
    assert run_pipeline_for(d2, env.db).status == "done"
    r = env.row(h["id"])
    assert r is not None, "edited hazard vanished after a second drive"
    assert (r["hazard_class"], r["severity"], r["status"]) == (new_cls, new_sev, "approved")
    assert r["sightings"] >= 2


def test_untouched_hazards_are_recomputed_on_pipeline_rerun(processed2):
    env, d1, d2 = processed2
    rows = env.rows()
    if len(rows) < 2:
        pytest.skip("need two hazards")
    edited, plain = rows[0], rows[1]
    env.ok(edited["id"], severity=edited["severity"] % 3 + 1)
    env.x("UPDATE hazards SET severity = ? WHERE id = ?", (plain["severity"] % 3 + 1, plain["id"]))
    assert run_pipeline_for(d1, env.db).status == "done"
    assert env.row(plain["id"])["severity"] == plain["severity"]
    assert env.row(edited["id"])["severity"] == edited["severity"] % 3 + 1


# ------------------------------------------------------------------ schema upgrade

def test_init_db_upgrades_an_existing_database(rs17_env):
    """Dropping the columns added after task 14 and running init_db again restores them without data loss."""
    from backend.db import init_db

    env = rs17_env
    base = {"id", "hazard_class", "lat", "lon", "direction", "severity", "confidence", "sightings",
            "candidate_count", "first_seen", "last_seen", "photo_path", "crop_path", "status", "created_at",
            "updated_at"}
    extra = [c for c in env.columns() if c not in base]
    if not extra:
        pytest.skip("no extra hazards columns were added; nothing to migrate")
    hid = env.add_hazard(hazard_class="pothole", severity=2, status="approved")
    import sqlite3

    con = sqlite3.connect(str(env.db))
    try:
        for c in extra:
            con.execute("ALTER TABLE hazards DROP COLUMN %s" % c)
        con.commit()
    except sqlite3.OperationalError as e:
        con.close()
        pytest.skip("cannot drop column (%s)" % e)
    con.close()
    assert set(env.columns()) == base
    init_db()
    init_db()
    assert set(extra) <= set(env.columns())
    assert env.row(hid)["status"] == "approved"
    assert env.ok(hid, severity=3)["severity"] == 3


# ------------------------------------------------------------------ transient re-run state must not erase an edit

def _two_drive_hazard(env, db, sev_a, sev_b, tag):
    from datetime import timedelta as td

    env.add_drive("A", T0, db=db)
    env.add_drive("B", T0 + td(days=7), db=db)
    ca = env.add_cand("A", VA[0], VA[1], 0.0, cls="pothole", severity=sev_a, conf=0.7, score=0.9,
                      frame_path=str(env.tmp / ("fa_%s.jpg" % tag)), db=db)
    lat, lon = offset(*VA, north_m=2.0)
    env.add_cand("B", lat, lon, 0.0, cls="pothole", severity=sev_b, conf=0.7, score=0.4,
                 frame_path=str(env.tmp / ("fb_%s.jpg" % tag)), db=db)
    env.dedup("A", db=db)
    env.dedup("B", db=db)
    return ca


def _sev(db):
    import sqlite3

    con = sqlite3.connect(str(db))
    try:
        return [r[0] for r in con.execute("SELECT severity FROM hazards ORDER BY id")]
    finally:
        con.close()


def test_edit_survives_a_rerun_whose_intermediate_state_has_the_edited_value_as_automatic(rs17_env, tmp_path):
    """A pipeline re-run first clears the drive's candidates and dedups (hazard supported by the other drive only),
    then re-adds them. If the automatic severity of that intermediate state equals the user's edited value, the
    edit must not be forgotten: after the final dedup the user's value still wins."""
    from backend.db import init_db

    env = rs17_env
    chosen = None
    for sev_a, sev_b in ((1, 3), (3, 1), (2, 3), (3, 2), (1, 2), (2, 1)):
        ctrl_both = tmp_path / ("both_%d%d.sqlite" % (sev_a, sev_b))
        ctrl_b = tmp_path / ("b_%d%d.sqlite" % (sev_a, sev_b))
        init_db(ctrl_both)
        init_db(ctrl_b)
        _two_drive_hazard(env, ctrl_both, sev_a, sev_b, "c%d%d" % (sev_a, sev_b))
        env.add_drive("B", T0, db=ctrl_b)
        env.add_cand("B", VA[0], VA[1], 0.0, cls="pothole", severity=sev_b, conf=0.7, score=0.4,
                     frame_path=str(tmp_path / "x.jpg"), db=ctrl_b)
        env.dedup("B", db=ctrl_b)
        both, only_b = _sev(ctrl_both), _sev(ctrl_b)
        if len(both) == 1 and len(only_b) == 1 and both[0] != only_b[0]:
            chosen = (sev_a, sev_b, both[0], only_b[0])
            break
    if chosen is None:
        pytest.skip("severity aggregation made the two-drive and one-drive results equal for every pair tried")
    sev_a, sev_b, both_sev, b_only = chosen

    ca = _two_drive_hazard(env, env.db, sev_a, sev_b, "main")
    hid = env.rows()[0]["id"]
    assert env.rows()[0]["severity"] == both_sev
    env.ok(hid, severity=b_only)              # the user picks exactly what drive B alone would say
    assert env.row(hid)["severity"] == b_only

    # re-run of drive A: its candidates vanish, dedup runs, then they come back
    env.x("DELETE FROM verdicts WHERE drive_id = 'A'")
    env.x("DELETE FROM candidates WHERE drive_id = 'A'")
    env.dedup("A")
    assert env.row(hid) is not None and env.row(hid)["severity"] == b_only
    env.add_cand("A", VA[0], VA[1], 0.0, cls="pothole", severity=sev_a, conf=0.7, score=0.9,
                 frame_path=str(env.tmp / "fa_again.jpg"))
    env.dedup("A")
    r = env.row(hid)
    assert r is not None, "hazard id changed"
    assert r["severity"] == b_only, "the user's severity edit was lost (%s -> %s) after a re-run" % (b_only, r["severity"])
