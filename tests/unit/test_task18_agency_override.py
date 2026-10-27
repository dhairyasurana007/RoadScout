"""Task 18: the user can change the agency; the override persists and survives dedup re-runs."""
import pytest

from rs18_helpers import (DC_PT, MD_PT, MDOT_URL, VA_PT, VDOT_HOST, T0, category_blob, find_key, form_urls,  # noqa: F401
                          known_classes, nospace, offset, rep_agency, rep_user_chosen, rs18_env, topic_for)

CLASSES = known_classes()


def test_an_agency_route_exists(rs18_env):
    assert rs18_env.agency_route() is not None, \
        "no way to change the agency (expected PATCH /api/hazards/{id} with an 'agency' field)"


def agency_cols(env, hid, value):
    return [c for c, v in env.row(hid).items() if isinstance(v, str) and v.strip().lower() == value.lower()]


def test_override_va_hazard_to_mdot(rs18_env):
    env = rs18_env
    hid, _ = env.seed(hazard_class="pothole")
    assert rep_agency(env.report(hid)) == "VDOT"
    env.ok_agency(hid, "MDOT SHA")
    body = env.report(hid)
    assert rep_agency(body) == "MDOT SHA"
    assert rep_user_chosen(body) is True, "report must mark the agency as user-chosen"
    assert nospace(topic_for("pothole")) in nospace(category_blob(body))
    assert any(MDOT_URL in u for u in form_urls(body)) and not any(VDOT_HOST in u for u in form_urls(body))


def test_override_md_hazard_to_vdot(rs18_env):
    env = rs18_env
    hid, _ = env.seed(lat=MD_PT[0], lon=MD_PT[1], hazard_class="damaged guardrail")
    assert rep_agency(env.report(hid)) == "MDOT SHA"
    env.ok_agency(hid, "VDOT")
    body = env.report(hid)
    assert rep_agency(body) == "VDOT"
    assert rep_user_chosen(body) is True
    assert any(VDOT_HOST in u for u in form_urls(body)) and not any(MDOT_URL in u for u in form_urls(body))


@pytest.mark.parametrize("cls", CLASSES)
def test_category_mapping_switches_with_the_agency(rs18_env, cls):
    env = rs18_env
    hid, _ = env.seed(hazard_class=cls)
    vdot_cat = nospace(category_blob(env.report(hid)))
    env.ok_agency(hid, "MDOT SHA")
    mdot_cat = nospace(category_blob(env.report(hid)))
    assert nospace(topic_for(cls)) in mdot_cat
    env.ok_agency(hid, "VDOT")
    assert nospace(category_blob(env.report(hid))) == vdot_cat


def test_vdot_categories_are_not_just_the_mdot_topics(rs18_env):
    env = rs18_env
    differ = 0
    for i, cls in enumerate(CLASSES):
        hid, _ = env.seed(name="d%d" % i, hazard_class=cls)
        if nospace(topic_for(cls)) not in nospace(category_blob(env.report(hid))):
            differ += 1
    assert differ >= 1, "VDOT categories are identical to the MDOT SHA topics for every class"


def test_override_to_none(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    env.ok_agency(hid, "none")
    r = env.get_report(hid)
    assert r.status_code < 500
    if r.status_code == 200:
        body = r.json()
        assert rep_agency(body) is None
        assert rep_user_chosen(body) in (True, None)
        assert not form_urls(body)
    else:
        assert r.status_code in (404, 409, 422)


def test_override_hazard_outside_any_jurisdiction(rs18_env):
    env = rs18_env
    hid, _ = env.seed(lat=DC_PT[0], lon=DC_PT[1])
    env.ok_agency(hid, "VDOT")
    r = env.get_report(hid)
    assert r.status_code == 200, "user chose VDOT for a DC hazard; a report is expected"
    body = r.json()
    assert rep_agency(body) == "VDOT" and rep_user_chosen(body) is True


def test_null_resets_to_automatic(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    env.ok_agency(hid, "MDOT SHA")
    assert rep_agency(env.report(hid)) == "MDOT SHA"
    env.ok_agency(hid, None)
    body = env.report(hid)
    assert rep_agency(body) == "VDOT"
    assert rep_user_chosen(body) is False
    assert not agency_cols(env, hid, "MDOT SHA"), "the stored override was not cleared"


def test_auto_string_resets_or_is_refused(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    env.ok_agency(hid, "MDOT SHA")
    r = env.set_agency(hid, "auto")
    assert r.status_code in (200, 422)
    if r.status_code == 200:
        body = env.report(hid)
        assert rep_agency(body) == "VDOT" and rep_user_chosen(body) is False
    else:
        assert rep_agency(env.report(hid)) == "MDOT SHA"


def test_reset_none_override_on_dc_hazard(rs18_env):
    env = rs18_env
    hid, _ = env.seed(lat=DC_PT[0], lon=DC_PT[1])
    env.ok_agency(hid, "MDOT SHA")
    assert rep_agency(env.report(hid)) == "MDOT SHA"
    env.ok_agency(hid, None)
    r = env.get_report(hid)
    if r.status_code == 200:
        assert rep_agency(r.json()) is None


def test_override_is_persisted_in_sqlite_and_cleared_by_reset(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    before_cols = set(env.columns())
    env.ok_agency(hid, "MDOT SHA")
    assert agency_cols(env, hid, "MDOT SHA"), "no hazards column holds the chosen agency: %s" % env.row(hid)
    env.ok_agency(hid, "VDOT")
    assert agency_cols(env, hid, "VDOT"), "explicit VDOT choice on a VDOT hazard is still remembered as user-chosen"
    assert rep_user_chosen(env.report(hid)) is True
    env.ok_agency(hid, None)
    assert not agency_cols(env, hid, "VDOT") and not agency_cols(env, hid, "MDOT SHA")
    assert before_cols <= set(env.columns())


def test_override_none_is_persisted(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    before = env.row(hid)
    env.ok_agency(hid, "none")
    after = env.row(hid)
    changed = [k for k in after if k not in ("updated_at", "reviewed_at") and after[k] != before.get(k)]
    assert changed, "choosing 'no agency' left the row unchanged; nothing is remembered"
    env.ok_agency(hid, None)
    again = env.row(hid)
    assert {k: v for k, v in again.items() if k not in ("updated_at", "reviewed_at")} == \
           {k: v for k, v in before.items() if k not in ("updated_at", "reviewed_at")}


def test_override_does_not_touch_other_hazards(rs18_env):
    env = rs18_env
    a, _ = env.seed(name="a")
    b, _ = env.seed(name="b")
    env.ok_agency(a, "MDOT SHA")
    assert rep_agency(env.report(b)) == "VDOT"
    assert rep_user_chosen(env.report(b)) is False


def test_override_does_not_change_review_fields(rs18_env):
    env = rs18_env
    hid, _ = env.seed(status="approved", severity=2)
    before = env.row(hid)
    env.ok_agency(hid, "MDOT SHA")
    after = env.row(hid)
    for k in ("status", "hazard_class", "severity", "lat", "lon", "direction", "sightings", "photo_path", "first_seen",
              "last_seen", "confidence"):
        assert after[k] == before[k], k


def test_override_response_and_list_stay_consistent(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    r = env.ok_agency(hid, "MDOT SHA")
    if env.agency_route() != env.generic_route():
        return
    body = r.json()
    if isinstance(body, dict) and "id" not in body:
        body = next((v for v in body.values() if isinstance(v, dict) and "id" in v), body)
    item = {int(h["id"]): h for h in env.hazards()}[hid]
    assert set(body) == set(item), "update response and list item must carry the same keys"


# ------------------------------------------------------------------ validation

@pytest.mark.parametrize("bad", ["banana", "", " ", 5, True, ["VDOT"], {"a": 1}, "DC", "vdot; DROP TABLE hazards"])
def test_invalid_agency_is_422_and_changes_nothing(rs18_env, bad):
    env = rs18_env
    hid, _ = env.seed()
    env.ok_agency(hid, "MDOT SHA")
    before = env.row(hid)
    r = env.set_agency(hid, bad)
    assert r.status_code == 422, "agency=%r -> %s" % (bad, r.status_code)
    assert env.row(hid) == before
    assert rep_agency(env.report(hid)) == "MDOT SHA"


def test_unknown_hazard_is_404(rs18_env):
    env = rs18_env
    assert rs18_env.set_agency(424242, "VDOT").status_code == 404


def test_lowercase_or_padded_values_are_either_accepted_or_422(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    for v in ("vdot", "mdot sha", " VDOT", "MDOT_SHA"):
        r = env.set_agency(hid, v)
        assert r.status_code in (200, 422), (v, r.status_code)
        if r.status_code == 200:
            assert rep_agency(env.report(hid)) in ("VDOT", "MDOT SHA")


def test_combined_update_is_atomic(rs18_env):
    env = rs18_env
    g = env.generic_route()
    if env.agency_route() != g:
        pytest.skip("the agency has its own route")
    hid, _ = env.seed(status="new")
    r = getattr(env.client, g[0])(env.sub(g[1], hid), json={"status": "approved", "agency": "banana"})
    assert r.status_code == 422
    assert env.row(hid)["status"] == "new"
    r = getattr(env.client, g[0])(env.sub(g[1], hid), json={"status": "approved", "agency": "MDOT SHA"})
    assert r.status_code == 200
    assert env.row(hid)["status"] == "approved"
    assert rep_agency(env.report(hid)) == "MDOT SHA"


def test_status_only_update_does_not_clear_the_override(rs18_env):
    env = rs18_env
    hid, _ = env.seed(status="approved")
    env.ok_agency(hid, "MDOT SHA")
    for st in ("rejected", "approved"):
        assert env.set_status(hid, st).status_code == 200
    body = env.report(hid)
    assert rep_agency(body) == "MDOT SHA" and rep_user_chosen(body) is True


def test_class_and_severity_edits_do_not_clear_the_override(rs18_env):
    env = rs18_env
    hid, _ = env.seed(status="approved")
    env.ok_agency(hid, "MDOT SHA")
    g = env.generic_route()
    r = getattr(env.client, g[0])(env.sub(g[1], hid), json={"hazard_class": "damaged guardrail", "severity": 1})
    assert r.status_code == 200
    assert rep_agency(env.report(hid)) == "MDOT SHA"


# ------------------------------------------------------------------ survives dedup re-runs

def make_hazard_via_dedup(env, drive="A", north=0.0, heading=0.0, cls="pothole"):
    if not env.q("SELECT id FROM drives WHERE id = ?", (drive,)):
        env.add_drive(drive, T0)
    lat, lon = offset(*VA_PT, north_m=north)
    env.add_cand(drive, lat, lon, heading, cls=cls, severity=2)
    env.dedup(drive)
    return max(r["id"] for r in env.rows())


def test_override_survives_dedup_reruns(rs18_env):
    env = rs18_env
    hid = make_hazard_via_dedup(env)
    assert env.set_status(hid, "approved").status_code == 200
    env.ok_agency(hid, "MDOT SHA")
    for _ in range(3):
        env.dedup("A")
        rows = env.rows()
        assert len(rows) == 1 and rows[0]["id"] == hid
        body = env.report(hid)
        assert rep_agency(body) == "MDOT SHA" and rep_user_chosen(body) is True


def test_override_survives_a_rerun_with_a_new_nearby_candidate(rs18_env):
    env = rs18_env
    hid = make_hazard_via_dedup(env)
    assert env.set_status(hid, "approved").status_code == 200
    env.ok_agency(hid, "none")
    before = env.row(hid)
    lat, lon = offset(*VA_PT, north_m=3.0)
    env.add_cand("A", lat, lon, 0.0, cls="pothole", severity=3, conf=0.9, score=0.9)
    env.dedup("A")
    rows = env.rows()
    assert len(rows) == 1 and rows[0]["id"] == hid
    r = env.get_report(hid)
    if r.status_code == 200:
        assert rep_agency(r.json()) is None
    changed = [k for k in ("agency", "agency_override", "agency_choice") if k in before]
    for k in changed:
        assert rows[0][k] == before[k]


def test_override_survives_parking_and_adoption(rs18_env):
    from backend.dedup import refresh_hazards_quietly

    env = rs18_env
    hid = make_hazard_via_dedup(env)
    assert env.set_status(hid, "approved").status_code == 200
    env.ok_agency(hid, "MDOT SHA")
    env.x("DELETE FROM verdicts WHERE drive_id = 'A'")
    env.x("DELETE FROM candidates WHERE drive_id = 'A'")
    refresh_hazards_quietly(str(env.db))
    assert env.row(hid) is not None
    lat, lon = offset(*VA_PT, north_m=4.0)
    env.add_cand("A", lat, lon, 0.0, cls="pothole", severity=2)
    env.dedup("A")
    rows = env.rows()
    assert len(rows) == 1 and rows[0]["id"] == hid
    body = env.report(hid)
    assert rep_agency(body) == "MDOT SHA" and rep_user_chosen(body) is True


def test_unadopted_rerun_does_not_leak_override_to_a_new_hazard(rs18_env):
    from backend.dedup import refresh_hazards_quietly

    env = rs18_env
    hid = make_hazard_via_dedup(env)
    assert env.set_status(hid, "approved").status_code == 200
    env.ok_agency(hid, "MDOT SHA")
    env.x("DELETE FROM verdicts WHERE drive_id = 'A'")
    env.x("DELETE FROM candidates WHERE drive_id = 'A'")
    refresh_hazards_quietly(str(env.db))
    lat, lon = offset(*VA_PT, north_m=200.0)
    env.add_cand("A", lat, lon, 0.0, cls="pothole", severity=2)
    env.dedup("A")
    rows = env.rows()
    assert len(rows) == 1 and rows[0]["status"] == "new"
    assert not agency_cols(env, rows[0]["id"], "MDOT SHA")
    assert env.set_status(rows[0]["id"], "approved").status_code == 200
    body = env.report(rows[0]["id"])
    assert rep_agency(body) == "VDOT" and rep_user_chosen(body) is False


def test_auto_agency_recomputed_after_dedup_moves_the_hazard(rs18_env):
    env = rs18_env
    hid = make_hazard_via_dedup(env)
    assert env.set_status(hid, "approved").status_code == 200
    assert rep_agency(env.report(hid)) == "VDOT"
    env.x("UPDATE candidates SET lat = ?, lon = ? WHERE hazard_id = ?", (MD_PT[0], MD_PT[1], hid))
    env.dedup("A")
    rows = env.rows()
    assert len(rows) == 1
    if abs(rows[0]["lat"] - MD_PT[0]) > 1e-3:
        pytest.skip("dedup did not move the hazard")
    assert rep_agency(env.report(rows[0]["id"])) == "MDOT SHA"
