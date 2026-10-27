"""Task 17 backend: review work is parked when a drive's candidates are cleared and adopted by the new cluster."""
import math
from datetime import timedelta

import pytest

from rs17_helpers import (FFMPEG, M_PER_DEG, T0, VA, gpx_bytes as _gpx_default, known_classes, offset,  # noqa: F401
                          rs17_env, run_pipeline_for, upload)
from test_task17_sticky import other_class, processed  # noqa: F401  (fixture re-use)

DECIDED = ("approved", "rejected", "reported")


def park(env, drive="A"):
    """What clearing a drive's results does: delete its candidates, then the quiet hazard refresh."""
    from backend.dedup import refresh_hazards_quietly

    env.x("DELETE FROM verdicts WHERE drive_id = ?", (drive,))
    env.x("DELETE FROM candidates WHERE drive_id = ?", (drive,))
    refresh_hazards_quietly(str(env.db))


def members(env, hid):
    return env.q("SELECT COUNT(*) AS n FROM candidates WHERE hazard_id = ?", (hid,))[0]["n"]


def make(env, *, cls="pothole", heading=0.0, north=0.0, status="approved", sev=None, hcls=None, drive="A",
         tag="a"):
    """One hazard from one candidate at VA + north metres, then reviewed. Returns its row."""
    if not env.q("SELECT id FROM drives WHERE id = ?", (drive,)):
        env.add_drive(drive, T0)
    lat, lon = offset(*VA, north_m=north)
    env.add_cand(drive, lat, lon, heading, cls=cls, severity=2, frame_path=str(env.tmp / ("f_%s.jpg" % tag)))
    env.dedup(drive)
    hid = max(r["id"] for r in env.rows())
    fields = {}
    if status in ("approved", "rejected"):
        fields["status"] = status
    if sev is not None:
        fields["severity"] = sev
    if hcls is not None:
        fields["hazard_class"] = hcls
    if fields:
        env.ok(hid, **fields)
    if status in ("reported", "possibly_fixed"):
        env.x("UPDATE hazards SET status = ? WHERE id = ?", (status, hid))
    return env.row(hid)


def readd(env, *, cls="pothole", heading=0.0, north=0.0, east=0.0, drive="A", tag="b", severity=3, run=True):
    lat, lon = offset(*VA, north_m=north, east_m=east)
    env.add_cand(drive, lat, lon, heading, cls=cls, severity=severity, conf=0.9, score=0.95,
                 frame_path=str(env.tmp / ("f_%s.jpg" % tag)))
    if run:
        env.dedup(drive)
    return lat, lon


# ------------------------------------------------------------------ parking

@pytest.mark.parametrize("status", DECIDED)
def test_decided_hazard_is_parked_without_candidates(rs17_env, status):
    env = rs17_env
    h = make(env, status=status)
    park(env)
    r = env.row(h["id"])
    assert r is not None, "a hazard with review work was deleted when its candidates were cleared"
    assert r["status"] == status
    assert members(env, h["id"]) == 0


def test_edit_only_hazard_is_parked(rs17_env):
    env = rs17_env
    h = make(env, status="new", sev=3)
    assert env.row(h["id"])["status"] == "new"
    park(env)
    r = env.row(h["id"])
    assert r is not None and r["severity"] == 3


def test_class_edit_only_hazard_is_parked(rs17_env):
    env = rs17_env
    h = make(env, status="new", hcls="damaged guardrail")
    park(env)
    assert env.row(h["id"]) is not None


def test_undecided_hazard_is_deleted_immediately(rs17_env):
    env = rs17_env
    make(env, status="new")
    assert len(env.rows()) == 1
    park(env)
    assert env.rows() == []


def test_undecided_orphan_is_never_resurrected(rs17_env):
    env = rs17_env
    make(env, status="new")
    park(env)
    readd(env, north=1.0)
    rows = env.rows()
    assert len(rows) == 1 and rows[0]["status"] == "new" and not rows[0]["class_edited"] and not rows[0]["severity_edited"]


def test_parked_hazards_do_not_break_the_api(rs17_env):
    env = rs17_env
    h = make(env, status="approved")
    park(env)
    items = env.by_id()
    assert h["id"] in items and items[h["id"]]["status"] == "approved"
    r = env.client.get("/api/hazards/%s/photo" % h["id"])
    assert r.status_code < 500
    assert env.ok(h["id"], severity=3)["severity"] == 3


# ------------------------------------------------------------------ adoption: distance

@pytest.mark.parametrize("status", DECIDED)
def test_adopted_within_10m_keeps_id_status_and_edits(rs17_env, status):
    env = rs17_env
    h = make(env, status=status, sev=3, hcls="damaged guardrail", north=0.0)
    park(env)
    lat, lon = readd(env, cls="pothole", north=10.0)
    rows = env.rows()
    assert len(rows) == 1, rows
    r = rows[0]
    assert r["id"] == h["id"], "decided hazard was not adopted by the new cluster (new id %s)" % r["id"]
    assert r["status"] == status
    assert (r["hazard_class"], r["severity"]) == ("damaged guardrail", 3)
    assert r["class_edited"] and r["severity_edited"]
    assert r["created_at"] == h["created_at"]
    assert r["reviewed_at"] == h["reviewed_at"]
    # everything else is recomputed from the new member
    assert abs(r["lat"] - lat) < 1e-7 and abs(r["lon"] - lon) < 1e-7
    assert r["sightings"] == 1
    assert r["photo_path"] == str(env.tmp / "f_b.jpg")
    assert env.q("SELECT hazard_id FROM candidates") == [{"hazard_id": h["id"]}]
    assert env.by_id()[h["id"]]["status"] == status


def test_adopted_hazard_keeps_decision_through_further_reruns(rs17_env):
    env = rs17_env
    h = make(env, status="rejected", sev=1)
    park(env)
    readd(env, north=5.0)
    for _ in range(3):
        env.dedup("A")
    r = env.rows()
    assert len(r) == 1 and r[0]["id"] == h["id"] and r[0]["status"] == "rejected" and r[0]["severity"] == 1


def test_not_adopted_at_100m(rs17_env):
    env = rs17_env
    h = make(env, status="approved", sev=3, hcls="damaged guardrail")
    park(env)
    readd(env, north=100.0)
    rows = env.rows()
    assert len(rows) == 1, "the unadopted decided orphan must be deleted in that run"
    r = rows[0]
    assert r["status"] == "new" and not r["class_edited"] and not r["severity_edited"]
    assert r["hazard_class"] == "pothole" and r["severity"] == 3  # automatic values of the new candidate
    assert r["reviewed_at"] is None
    assert abs(r["lat"] - offset(*VA, north_m=100.0)[0]) < 1e-6


@pytest.mark.parametrize("north,adopted", [(0.0, True), (23.0, True), (27.0, False), (60.0, False)])
def test_distance_threshold_is_the_dedup_distance(rs17_env, north, adopted):
    env = rs17_env
    h = make(env, status="approved")
    park(env)
    readd(env, north=north)
    rows = env.rows()
    assert len(rows) == 1
    assert (rows[0]["status"] == "approved") is adopted
    if adopted:
        assert rows[0]["id"] == h["id"]


def test_adoption_distance_uses_the_dedup_distance_parameter(rs17_env):
    env = rs17_env
    h = make(env, status="approved")
    park(env)
    readd(env, north=40.0, run=False)
    import inspect

    from rs17_helpers import DBP, FUNC_NAMES
    import importlib

    mod = importlib.import_module("backend.dedup")
    f = next(getattr(mod, n) for n in FUNC_NAMES if callable(getattr(mod, n, None)))
    sig = inspect.signature(f)
    dist = next((n for n in ("distance_m", "max_distance_m", "radius_m") if n in sig.parameters), None)
    if dist is None:
        pytest.skip("no distance parameter on the dedup function")
    db = next((n for n in DBP if n in sig.parameters), "db_path")
    f("A", **{db: str(env.db), dist: 60.0})
    rows = env.rows()
    assert len(rows) == 1 and rows[0]["id"] == h["id"] and rows[0]["status"] == "approved"


# ------------------------------------------------------------------ adoption: class

def test_class_mismatch_is_not_adopted(rs17_env):
    env = rs17_env
    make(env, status="approved", cls="pothole")
    park(env)
    readd(env, cls="debris in lane", north=3.0)
    rows = env.rows()
    assert len(rows) == 1
    assert rows[0]["hazard_class"] == "debris in lane" and rows[0]["status"] == "new"


def test_edited_orphan_is_matched_by_its_automatic_class(rs17_env):
    env = rs17_env
    h = make(env, status="approved", cls="pothole", hcls="damaged guardrail")
    park(env)
    readd(env, cls="pothole", north=4.0)
    rows = env.rows()
    assert len(rows) == 1 and rows[0]["id"] == h["id"]
    assert rows[0]["hazard_class"] == "damaged guardrail" and rows[0]["class_edited"]
    assert rows[0]["status"] == "approved"


def test_edited_orphan_is_not_matched_by_its_edited_class(rs17_env):
    env = rs17_env
    make(env, status="approved", cls="pothole", hcls="damaged guardrail")
    park(env)
    readd(env, cls="damaged guardrail", north=4.0)
    rows = env.rows()
    assert len(rows) == 1
    assert rows[0]["status"] == "new" and not rows[0]["class_edited"] and rows[0]["hazard_class"] == "damaged guardrail"


# ------------------------------------------------------------------ adoption: heading

@pytest.mark.parametrize("old,new,adopted", [
    (0.0, 0.0, True), (0.0, 44.0, True), (0.0, 46.0, False), (0.0, 90.0, False), (0.0, 180.0, False),
    (350.0, 10.0, True), (10.0, 350.0, True), (350.0, 40.0, False), (90.0, 270.0, False),
])
def test_heading_tolerance_decides_adoption(rs17_env, old, new, adopted):
    env = rs17_env
    h = make(env, status="approved", heading=old)
    park(env)
    readd(env, heading=new, north=2.0)
    rows = env.rows()
    assert len(rows) == 1
    if adopted:
        assert rows[0]["id"] == h["id"] and rows[0]["status"] == "approved"
    else:
        assert rows[0]["status"] == "new"


def test_null_direction_on_the_orphan_is_not_adopted(rs17_env):
    env = rs17_env
    h = make(env, status="approved", heading=0.0)
    park(env)
    env.x("UPDATE hazards SET direction = NULL WHERE id = ?", (h["id"],))
    readd(env, heading=0.0, north=1.0)
    rows = env.rows()
    assert len(rows) == 1 and rows[0]["status"] == "new"


# ------------------------------------------------------------------ adoption: pairing

def test_two_orphans_one_cluster_lowest_id_wins_a_tie(rs17_env):
    env = rs17_env
    h1 = make(env, status="approved", heading=0.0, tag="h1")
    h2 = make(env, status="rejected", heading=80.0, tag="h2")
    assert h1["id"] < h2["id"] and len(env.rows()) == 2
    park(env)
    assert {r["id"] for r in env.rows()} == {h1["id"], h2["id"]}
    readd(env, heading=40.0, north=3.0)  # within 45 degrees of both
    rows = env.rows()
    assert len(rows) == 1
    assert rows[0]["id"] == h1["id"] and rows[0]["status"] == "approved"


def test_two_orphans_one_cluster_nearest_wins_over_lower_id(rs17_env):
    env = rs17_env
    h1 = make(env, status="approved", heading=0.0, north=0.0, tag="h1")
    h2 = make(env, status="rejected", heading=80.0, north=8.0, tag="h2")
    park(env)
    readd(env, heading=40.0, north=9.0)
    rows = env.rows()
    assert len(rows) == 1
    assert rows[0]["id"] == h2["id"] and rows[0]["status"] == "rejected"


def test_two_orphans_two_clusters_each_adopts_its_own(rs17_env):
    env = rs17_env
    h1 = make(env, status="approved", heading=0.0, tag="h1", sev=3)
    h2 = make(env, status="rejected", heading=80.0, tag="h2", sev=1)
    park(env)
    readd(env, heading=0.0, north=2.0, tag="n1", run=False)
    readd(env, heading=80.0, north=2.0, tag="n2", run=False)
    env.dedup("A")
    by = {r["id"]: r for r in env.rows()}
    assert set(by) == {h1["id"], h2["id"]}
    assert (by[h1["id"]]["status"], by[h1["id"]]["severity"]) == ("approved", 3)
    assert (by[h2["id"]]["status"], by[h2["id"]]["severity"]) == ("rejected", 1)


def test_extra_cluster_beyond_the_orphans_gets_a_new_hazard(rs17_env):
    env = rs17_env
    h = make(env, status="approved", heading=0.0)
    park(env)
    readd(env, heading=0.0, north=2.0, tag="n1", run=False)
    readd(env, heading=0.0, north=500.0, tag="n2", run=False)
    env.dedup("A")
    rows = {r["id"]: r for r in env.rows()}
    assert len(rows) == 2 and rows[h["id"]]["status"] == "approved"
    assert sorted(r["status"] for r in rows.values()) == ["approved", "new"]
    assert abs(rows[h["id"]]["lat"] - offset(*VA, north_m=2.0)[0]) < 1e-6


def test_orphan_in_a_different_place_does_not_block_other_adoptions(rs17_env):
    env = rs17_env
    far = make(env, status="rejected", north=2000.0, tag="far")
    near = make(env, status="approved", north=0.0, tag="near")
    park(env)
    readd(env, north=3.0)
    rows = env.rows()
    assert len(rows) == 1 and rows[0]["id"] == near["id"] and rows[0]["status"] == "approved"
    assert far["id"] not in {r["id"] for r in rows}


def test_orphan_with_other_drive_hazard_alive_is_untouched(rs17_env):
    """A hazard that still has members from another drive is not an orphan and is never parked or adopted away."""
    env = rs17_env
    env.add_drive("B", T0 + timedelta(days=3))
    lat, lon = offset(*VA, north_m=1000.0)
    env.add_cand("B", lat, lon, 0.0, cls="pothole", frame_path=str(env.tmp / "fb.jpg"))
    env.dedup("B")
    hb = env.rows()[0]
    env.ok(hb["id"], status="approved", severity=3)
    h = make(env, status="rejected")
    park(env)
    readd(env, north=2.0)
    by = {r["id"]: r for r in env.rows()}
    assert by[hb["id"]]["status"] == "approved" and by[hb["id"]]["severity"] == 3 and members(env, hb["id"]) == 1
    assert by[h["id"]]["status"] == "rejected"


# ------------------------------------------------------------------ real pipeline runs (fake provider, 24 s clip)

def gpx_shifted(east_m=0.0, north_m=0.0):
    lon0 = -77.30 + east_m / (M_PER_DEG * math.cos(math.radians(38.88)))
    pts = []
    for s in range(61):
        t = (T0 + timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%SZ")
        pts.append('<trkpt lat="%.8f" lon="%.8f"><ele>100</ele><time>%s</time></trkpt>'
                   % (38.88 + (s * 12.0 + north_m) / M_PER_DEG, lon0, t))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="rs17" '
            'xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>' + "".join(pts)
            + "</trkseg></trk></gpx>\n").encode("utf-8")


def rewrite_gpx(env, did, **shift):
    from pathlib import Path

    p = env.q("SELECT gpx_path FROM drives WHERE id = ?", (did,))[0]["gpx_path"]
    for cand in (Path(p), env.tmp / "storage" / p, env.tmp / p):
        if cand.is_file():
            cand.write_bytes(gpx_shifted(**shift))
            return
    pytest.skip("cannot locate the stored GPX file (%s)" % p)


def review_all(env):
    plan = {}
    for i, h in enumerate(env.rows()):
        f = {"hazard_class": other_class(h["hazard_class"]), "severity": h["severity"] % 3 + 1,
             "status": ("approved", "rejected")[i % 2]}
        env.ok(h["id"], **f)
        plan[h["id"]] = f
    return plan


def assert_plan_kept(env, plan):
    after = {r["id"]: r for r in env.rows()}
    assert set(after) == set(plan), "hazard ids changed: %s -> %s" % (sorted(plan), sorted(after))
    for hid, f in plan.items():
        for k in ("hazard_class", "severity", "status"):
            assert after[hid][k] == f[k], "%s of hazard %s changed" % (k, hid)
    for hid in after:
        assert members(env, hid) >= 1, "hazard %s has no candidates" % hid


def test_single_drive_pipeline_rerun_keeps_the_review_work(processed):
    env, did = processed
    plan = review_all(env)
    assert run_pipeline_for(did, env.db).status == "done"
    assert_plan_kept(env, plan)
    assert run_pipeline_for(did, env.db).status == "done"
    assert_plan_kept(env, plan)


def test_single_drive_rerun_after_small_gps_shift_still_adopts(processed):
    env, did = processed
    before = {r["id"]: r for r in env.rows()}
    plan = review_all(env)
    rewrite_gpx(env, did, east_m=10.0)
    assert run_pipeline_for(did, env.db).status == "done"
    after = {r["id"]: r for r in env.rows()}
    moved = [abs(after[i]["lon"] - before[i]["lon"]) * M_PER_DEG * math.cos(math.radians(38.88))
             for i in after if i in before]
    if not moved or max(moved) < 5.0:
        pytest.skip("the re-run did not re-read the rewritten GPX; cannot test a position change")
    assert_plan_kept(env, plan)


def test_single_drive_rerun_after_big_gps_shift_replaces_hazards(processed):
    env, did = processed
    n = len(env.rows())
    review_all(env)
    rewrite_gpx(env, did, east_m=100.0)
    assert run_pipeline_for(did, env.db).status == "done"
    rows = env.rows()
    if any(r["status"] != "new" for r in rows):
        pytest.skip("positions did not change; the rewritten GPX was not re-read")
    assert len(rows) == n
    assert all(not r["class_edited"] and not r["severity_edited"] and r["reviewed_at"] is None for r in rows)
    for r in rows:
        assert members(env, r["id"]) >= 1


def test_failed_run_leaves_parked_hazards_and_the_next_run_reconciles(processed):
    from backend.providers import FakeProvider
    from backend.pipeline import run_pipeline

    env, did = processed
    rows = env.rows()
    if len(rows) < 2:
        pytest.skip("need two hazards")
    decided, undecided = rows[0], rows[1]
    env.ok(decided["id"], status="approved", severity=decided["severity"] % 3 + 1)
    keep = env.row(decided["id"])

    class Boom(FakeProvider):
        def detect(self, *a, **kw):
            raise RuntimeError("detector down")

    res = run_pipeline(did, provider=Boom("default"), db_path=str(env.db))
    assert res.status == "failed"
    after = {r["id"]: r for r in env.rows()}
    assert decided["id"] in after, "the reviewed hazard disappeared when the run failed"
    assert after[decided["id"]]["status"] == "approved"
    assert members(env, decided["id"]) == 0
    assert undecided["id"] not in after or after[undecided["id"]]["status"] != "new" or members(env, undecided["id"]) > 0
    assert all(r["status"] != "new" or r["class_edited"] or r["severity_edited"] for r in after.values()), \
        "an undecided hazard survived without candidates"
    # the parked hazard is visible and harmless
    assert decided["id"] in env.by_id()
    assert env.client.get("/api/hazards/%s/photo" % decided["id"]).status_code < 500

    assert run_pipeline_for(did, env.db).status == "done"
    final = {r["id"]: r for r in env.rows()}
    assert decided["id"] in final
    assert final[decided["id"]]["status"] == "approved"
    assert final[decided["id"]]["severity"] == keep["severity"]
    for hid in final:
        assert members(env, hid) >= 1, "hazard %s still has no candidates after the next successful run" % hid
    assert len(final) == len(rows)


def test_failed_run_then_unadoptable_success_removes_the_parked_hazard(processed):
    """If the next successful run finds nothing near a parked hazard, that hazard is deleted in that run."""
    from backend.providers import FakeProvider
    from backend.pipeline import run_pipeline

    env, did = processed
    rows = env.rows()
    review_all(env)

    class Boom(FakeProvider):
        def detect(self, *a, **kw):
            raise RuntimeError("detector down")

    assert run_pipeline(did, provider=Boom("default"), db_path=str(env.db)).status == "failed"
    assert env.rows(), "reviewed hazards should be parked"
    rewrite_gpx(env, did, east_m=100.0)
    assert run_pipeline_for(did, env.db).status == "done"
    out = env.rows()
    if any(r["status"] != "new" for r in out):
        pytest.skip("positions did not change; the rewritten GPX was not re-read")
    for r in out:
        assert members(env, r["id"]) >= 1
    assert len(out) == len(rows)
