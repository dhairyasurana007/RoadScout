"""Task 18: the prefilled report endpoint (GET /api/hazards/{id}/report)."""
import re
import warnings

import pytest

from rs18_helpers import (DC_PT, MD_PT, MDOT_URL, PHILLY_PT, VA_PT, VDOT_HOST, all_strings, category_blob,  # noqa: F401
                          cls_norm, compass_tokens, dump, expected_compass, find_key, form_urls, has_submitted_note,
                          known_classes, map_links, nospace, norm_agency, rep_agency, rep_compass, rep_description,
                          rep_photo_url, rep_text, rep_user_chosen, rs18_env, topic_for)

CLASSES = known_classes()


def test_five_known_classes_map_to_mdot_topics():
    assert len(CLASSES) == 5
    for c in CLASSES:
        topic_for(c)


def test_a_report_route_exists(rs18_env):
    assert rs18_env.report_path() is not None, "no GET report route (expected GET /api/hazards/{id}/report)"


# ------------------------------------------------------------------ basic shape, Virginia

def test_virginia_report_shape(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    body = env.report(hid)
    assert str(find_key(body, "hazard_id", "id")) == str(hid)
    assert rep_agency(body) == "VDOT"
    assert rep_user_chosen(body) is False, "report must say the agency is the automatic default (e.g. agency_source: 'auto')"
    assert category_blob(body).strip()
    assert rep_description(body).strip()
    assert any(VDOT_HOST in u for u in form_urls(body)), "VDOT form URL (https://my.vdot.virginia.gov/) missing"
    assert not any(MDOT_URL in u for u in form_urls(body))


def test_vdot_field_names_are_marked_unverified(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    low = dump(env.report(hid)).lower()
    assert re.search(r"unverified|best[ _-]?effort|not verified|unconfirmed", low), \
        "VDOT form field names are unverified; the report must say so (e.g. 'unverified' / 'best effort')"


@pytest.mark.parametrize("cls", CLASSES)
def test_vdot_category_exists_for_every_class_and_is_distinct(rs18_env, cls):
    env = rs18_env
    hid, _ = env.seed(hazard_class=cls)
    body = env.report(hid)
    assert rep_agency(body) == "VDOT"
    assert len(category_blob(body).strip()) >= 3


def test_vdot_categories_differ_between_classes(rs18_env):
    env = rs18_env
    seen = {}
    for i, cls in enumerate(CLASSES):
        hid, _ = env.seed(name="c%d" % i, hazard_class=cls)
        seen[cls] = nospace(category_blob(env.report(hid)))
    assert len(set(seen.values())) == 5, seen


# ------------------------------------------------------------------ basic shape, Maryland

def test_maryland_report_shape(rs18_env):
    env = rs18_env
    hid, _ = env.seed(lat=MD_PT[0], lon=MD_PT[1])
    body = env.report(hid)
    assert rep_agency(body) == "MDOT SHA"
    assert rep_user_chosen(body) is False
    assert any(MDOT_URL in u for u in form_urls(body)), "MDOT SHA form URL missing (%s)" % MDOT_URL
    assert not any(VDOT_HOST in u for u in form_urls(body))


@pytest.mark.parametrize("cls", CLASSES)
def test_mdot_category_is_the_form_topic(rs18_env, cls):
    env = rs18_env
    hid, _ = env.seed(hazard_class=cls, lat=MD_PT[0], lon=MD_PT[1])
    body = env.report(hid)
    assert nospace(topic_for(cls)) in nospace(category_blob(body)), \
        "category %r does not contain the MDOT SHA Topic %r" % (category_blob(body), topic_for(cls))


def test_mdot_report_states_state_roads_only(rs18_env):
    env = rs18_env
    hid, _ = env.seed(lat=MD_PT[0], lon=MD_PT[1])
    body = env.report(hid)
    assert re.search(r"state[ -]+(road|highway)", (dump(body) + rep_text(body)).lower()), \
        "MDOT SHA covers state roads only; the report must say so (county/city streets go to the county or city)"


def test_mdot_form_without_photo_upload_is_noted_or_photo_still_offered(rs18_env):
    env = rs18_env
    hid, _ = env.seed(lat=MD_PT[0], lon=MD_PT[1])
    body = env.report(hid)
    assert rep_photo_url(body), "the blurred photo URL is still given (for download) even though the MDOT form has no upload"
    low = dump(body).lower()
    if not re.search(r"no photo|cannot upload|can.t upload|does not (accept|allow|support).{0,20}(photo|upload)|"
                     r"no (image|upload)|not upload|without (a )?photo|photo upload", low):
        warnings.warn("MDOT SHA form has no photo upload; the report does not mention it (informative)")


# ------------------------------------------------------------------ description / location / photo

def test_description_content(rs18_env):
    env = rs18_env
    hid, _ = env.seed(hazard_class="pothole", severity=3, sightings=4, direction=135.0)
    d = rep_description(env.report(hid)).lower()
    assert "pothole" in d, d
    assert "4" in d, "sightings missing: " + d
    assert re.search(r"severity|severe|major|serious|high|3", d), d
    for day, fmts in (("18", ("2026-10-18", "oct 18", "october 18", "10/18/2026", "18 oct", "oct. 18")),
                      ("22", ("2026-10-22", "oct 22", "october 22", "10/22/2026", "22 oct", "oct. 22"))):
        assert any(f in d for f in fmts), "first/last seen date %s missing in: %s" % (day, d)
    assert re.search(r"south[ -]?east|\bse\b|southeast|135", d), "direction of travel missing: " + d
    assert len(d) < 1200, "description should be short"


def test_description_has_no_filesystem_paths(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    d = rep_description(env.report(hid))
    d2 = d.replace("https://", "").replace("http://", "")
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]|/home/|/tmp/|\\\\|storage|\.jpg|\.jpeg|zqblur|zqorig", d2)


def test_description_follows_class_and_severity(rs18_env):
    env = rs18_env
    h1, _ = env.seed(name="x1", hazard_class="debris in lane", severity=1)
    d = rep_description(env.report(h1)).lower()
    assert "debris" in d
    assert "pothole" not in d


def test_model_reason_is_included_when_available(rs18_env):
    env = rs18_env
    env.add_drive("rsn")
    env.add_cand("rsn", *VA_PT, 135.0, cls="pothole", severity=3)
    env.dedup("rsn")
    rows = env.rows()
    assert len(rows) == 1
    hid = rows[0]["id"]
    env.x("UPDATE verdicts SET reason = ?, report_description = ?", ("zqreason flooded pit by the lane edge",
                                                                       "zqrepdesc deep hole in right lane"))
    r = env.set_status(hid, "approved")
    assert r.status_code == 200
    body = env.report(hid)
    blob = (rep_description(body) + " " + rep_text(body)).lower()
    assert "zqreason" in blob or "zqrepdesc" in blob, \
        "the model's reason / report_description from the verdict must be in the prefilled description"


def test_location_block(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    body = env.report(hid)
    lat = find_key(body, "lat", "latitude")
    lon = find_key(body, "lon", "lng", "longitude")
    assert abs(float(lat) - VA_PT[0]) < 1e-4 and abs(float(lon) - VA_PT[1]) < 1e-4
    links = map_links(body)
    assert links, "no map link (OpenStreetMap URL expected)"
    assert any("38.851" in u and "77.303" in u for u in links), links
    urls = [m for u in links for m in re.findall(r"https?://\S+", u)]
    assert urls and all(u.startswith("https://") for u in urls)


@pytest.mark.parametrize("deg", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 350.0, 10.0, 22.0, 23.0, 359.9])
def test_compass_from_heading(rs18_env, deg):
    env = rs18_env
    hid, _ = env.seed(direction=deg)
    body = env.report(hid)
    c = rep_compass(body)
    assert c is not None, "report has no compass direction string (e.g. location.compass: 'SE')"
    assert expected_compass(deg) in compass_tokens(c), "heading %s -> %r (expected %s)" % (deg, c, expected_compass(deg))


def test_missing_direction_does_not_break_the_report(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    env.x("UPDATE hazards SET direction = NULL WHERE id = ?", (hid,))
    body = env.report(hid)
    assert rep_description(body)
    assert "none" not in rep_description(body).lower().replace("nonetheless", "")
    assert "nan" not in rep_text(body).lower().split()


def test_photo_url_is_relative_and_serves_the_blurred_bytes(rs18_env):
    env = rs18_env
    hid, info = env.seed()
    body = env.report(hid)
    url = rep_photo_url(body)
    assert isinstance(url, str) and url.startswith("/api/"), url
    assert "://" not in url
    r = env.client.get(url)
    assert r.status_code == 200
    assert r.headers["content-type"].split(";")[0].strip().lower() == "image/jpeg"
    assert r.content == info["blur"].read_bytes()
    assert r.content != info["orig"].read_bytes()


def test_hazard_without_photo_still_reports(rs18_env):
    env = rs18_env
    hid, _ = env.seed(with_photo=False)
    body = env.report(hid)
    url = rep_photo_url(body)
    if url:
        assert env.client.get(url).status_code == 404
    assert rep_agency(body) == "VDOT"


def test_plain_text_version(rs18_env):
    env = rs18_env
    hid, _ = env.seed(hazard_class="pothole", lat=MD_PT[0], lon=MD_PT[1])
    body = env.report(hid)
    t = rep_text(body)
    assert "\n" in t
    assert "MDOT" in t
    assert nospace(topic_for("pothole")) in nospace(t)
    assert "39.084" in t and "77.153" in t
    assert rep_description(body).strip()[:30] in t
    assert any(u in t for u in ("openstreetmap.org",)), "text version lacks the map link"
    assert MDOT_URL in t, "text version lacks the form URL"


def test_text_version_for_vdot(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    t = rep_text(env.report(hid))
    assert "VDOT" in t and VDOT_HOST in t
    assert "38.851" in t and "77.303" in t


def test_nothing_is_submitted_flag(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    body = env.report(hid)
    assert has_submitted_note(body), \
        "report must clearly say nothing is submitted (e.g. 'submitted': false plus a 'not submitted automatically' note)"


def test_report_is_deterministic_and_read_only(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    before = env.rows()
    a = env.report(hid)
    b = env.report(hid)
    assert a == b
    assert env.rows() == before


def test_report_follows_edits(rs18_env):
    env = rs18_env
    hid, _ = env.seed(hazard_class="pothole", severity=1)
    before = nospace(category_blob(env.report(hid)))
    g = env.generic_route()
    assert g is not None
    r = getattr(env.client, g[0])(env.sub(g[1], hid), json={"hazard_class": "damaged guardrail", "severity": 3})
    assert r.status_code == 200, r.text
    body = env.report(hid)
    assert "guardrail" in rep_description(body).lower()
    assert nospace(category_blob(body)) != before, "category did not follow the class edit"


def test_auto_agency_follows_the_location_not_a_stale_value(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    assert rep_agency(env.report(hid)) == "VDOT"
    env.x("UPDATE hazards SET lat = ?, lon = ? WHERE id = ?", (MD_PT[0], MD_PT[1], hid))
    body = env.report(hid)
    assert rep_agency(body) == "MDOT SHA"
    assert rep_user_chosen(body) is False


# ------------------------------------------------------------------ status gating / errors

@pytest.mark.parametrize("status", ["new", "rejected"])
def test_non_approved_hazards_get_no_report(rs18_env, status):
    env = rs18_env
    hid, _ = env.seed(status=status)
    r = env.get_report(hid)
    assert r.status_code in (409, 422), "%s hazard -> %s" % (status, r.status_code)


def test_possibly_fixed_hazard_gets_no_report_or_a_clean_error(rs18_env):
    env = rs18_env
    hid, _ = env.seed(status="possibly_fixed")
    r = env.get_report(hid)
    assert r.status_code in (200, 409, 422)


def test_reported_hazard_report_is_available_or_cleanly_refused(rs18_env):
    env = rs18_env
    hid, _ = env.seed(status="reported")
    r = env.get_report(hid)
    assert r.status_code in (200, 409, 422), r.status_code
    if r.status_code == 200:
        assert rep_agency(r.json()) == "VDOT"
    else:
        warnings.warn("report of a 'reported' hazard is refused with %s" % r.status_code)


def test_unknown_hazard_is_404(rs18_env):
    env = rs18_env
    assert env.get_report(987654).status_code == 404


@pytest.mark.parametrize("bad", ["abc", "-1", "0", "1.5"])
def test_bad_ids(rs18_env, bad):
    env = rs18_env
    r = env.get_report(bad)
    assert r.status_code in (404, 422)


def test_status_change_unlocks_and_locks_the_report(rs18_env):
    env = rs18_env
    hid, _ = env.seed(status="new")
    assert env.get_report(hid).status_code in (409, 422)
    assert env.set_status(hid, "approved").status_code == 200
    assert env.get_report(hid).status_code == 200
    assert env.set_status(hid, "rejected").status_code == 200
    assert env.get_report(hid).status_code in (409, 422)


# ------------------------------------------------------------------ no agency (other jurisdictions)

@pytest.mark.parametrize("name,pt", [("DC", DC_PT), ("Philadelphia", PHILLY_PT), ("Atlantic", (38.0, -74.0))])
def test_hazard_in_other_jurisdiction_has_no_agency(rs18_env, name, pt):
    env = rs18_env
    hid, _ = env.seed(lat=pt[0], lon=pt[1])
    r = env.get_report(hid)
    assert r.status_code < 500
    if r.status_code == 200:
        body = r.json()
        assert rep_agency(body) is None, "%s hazard must have agency null/none: %r" % (name, find_key(body, "agency"))
        assert rep_user_chosen(body) in (False, None)
        assert not form_urls(body), "no agency means no agency form URL"
        notes = [s for s in all_strings(body, keys=False) if re.search(r"no agency|no .*agency|outside|jurisdiction|"
                                                                      r"not covered|choose|select", s, re.I)]
        assert notes, "an explanatory note for the no-agency case is expected"
    else:
        warnings.warn("no-agency hazard report answers %s" % r.status_code)
        assert r.status_code in (404, 409, 422)


def test_no_agency_hazard_still_in_the_list_and_not_an_error(rs18_env):
    env = rs18_env
    hid, _ = env.seed(lat=DC_PT[0], lon=DC_PT[1])
    assert any(int(h["id"]) == hid for h in env.hazards())


# ------------------------------------------------------------------ hazards list stays consistent

def test_list_still_has_the_core_keys(rs18_env):
    env = rs18_env
    hid, _ = env.seed()
    items = {int(h["id"]): h for h in env.hazards()}
    for k in ("id", "hazard_class", "lat", "lon", "direction", "severity", "confidence", "sightings", "first_seen",
              "last_seen", "status"):
        assert k in items[hid]
    # the update response and the list item keep the same shape (task 17 invariant)
    g = env.generic_route()
    r = getattr(env.client, g[0])(env.sub(g[1], hid), json={"severity": 2})
    assert r.status_code == 200
    body = r.json()
    body = body.get("hazard", body) if isinstance(body, dict) and "id" not in body else body
    items = {int(h["id"]): h for h in env.hazards()}
    assert set(body) == set(items[hid])
