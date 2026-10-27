"""Task 16 (map view), backend: GET /api/hazards lists the deduplicated hazards of all drives for the map."""
from __future__ import annotations

import math
import re
import sqlite3

import pytest

from rs16_helpers import (REQUIRED_KEYS, RAW_PATH_KEYS, STATUSES, fwd, rs16_env, rs16_syn)  # noqa: F401


def test_empty_database_gives_empty_list(rs16_env):
    r = rs16_env.client.get(rs16_env.list_route())
    assert r.status_code == 200
    assert r.json() == []


def test_returns_every_hazard_of_every_drive_and_status(rs16_syn):
    env, data = rs16_syn
    items = env.hazards()
    assert sorted(int(i["id"]) for i in items) == sorted(d["id"] for d in data.values())
    assert {i["status"] for i in items} == set(STATUSES)


def test_items_carry_the_required_fields_with_the_stored_values(rs16_syn):
    env, data = rs16_syn
    rows = {r["id"]: r for r in env.hazards_rows()}
    for it in env.hazards():
        for k in REQUIRED_KEYS:
            assert k in it, "missing key %r in %s" % (k, sorted(it))
        row = rows[int(it["id"])]
        assert isinstance(it["id"], int)
        assert it["hazard_class"] == row["hazard_class"]
        assert it["lat"] == pytest.approx(row["lat"])
        assert it["lon"] == pytest.approx(row["lon"])
        if row["direction"] is None:
            assert it["direction"] is None
        else:
            assert it["direction"] == pytest.approx(row["direction"])
        assert it["severity"] == row["severity"]
        assert it["confidence"] == pytest.approx(row["confidence"])
        assert it["sightings"] == row["sightings"]
        assert isinstance(it["sightings"], int)
        assert it["first_seen"] == row["first_seen"]
        assert it["last_seen"] == row["last_seen"]
        assert it["status"] == row["status"]
        assert isinstance(it["lat"], float) and isinstance(it["lon"], float)
        assert math.isfinite(it["lat"]) and math.isfinite(it["lon"])


def test_json_is_strict_and_a_list_of_objects(rs16_syn):
    env, _ = rs16_syn
    r = env.client.get(env.list_route())
    assert r.headers["content-type"].split(";")[0].strip() == "application/json"
    body = r.json()
    assert isinstance(body, list) and all(isinstance(i, dict) for i in body)
    assert "NaN" not in r.text and "Infinity" not in r.text


def test_photo_url_is_a_relative_api_url_or_null(rs16_syn):
    env, data = rs16_syn
    by = env.by_id()
    for name, d in data.items():
        url = by[d["id"]].get("photo_url")
        if d["pair"] is None:
            assert not url or url.startswith("/api/")
            continue
        assert isinstance(url, str) and url.startswith("/api/"), (name, url)
        assert "://" not in url and ".." not in url and "\\" not in url
        assert not url.lower().split("?")[0].endswith((".jpg", ".jpeg", ".png"))
    for it in by.values():
        cu = it.get("crop_url")
        if cu:
            assert cu.startswith("/api/") and "://" not in cu and ".." not in cu


def test_photo_urls_from_the_list_serve_the_blurred_bytes(rs16_syn):
    env, data = rs16_syn
    by = env.by_id()
    for name, d in data.items():
        if d["pair"] is None:
            continue
        r = env.client.get(by[d["id"]]["photo_url"])
        assert r.status_code == 200, name
        assert r.headers["content-type"].split(";")[0].strip() == "image/jpeg"
        assert r.content == d["pair"]["blur"].read_bytes()
        assert r.content != d["pair"]["orig"].read_bytes()


def test_crop_url_serves_the_blurred_crop_when_offered(rs16_syn):
    env, data = rs16_syn
    by = env.by_id()
    for name in ("h1", "h4"):
        d = data[name]
        cu = by[d["id"]].get("crop_url")
        if not cu:
            pytest.skip("crop_url is optional and not offered")
        r = env.client.get(cu)
        assert r.status_code == 200
        assert r.content == d["pair"]["blur_crop"].read_bytes()
        assert r.content != d["pair"]["orig_crop"].read_bytes()
    # hazards without a crop never offer a working crop link
    for name in ("h2", "h3", "h5"):
        cu = by[data[name]["id"]].get("crop_url")
        if cu:
            assert env.client.get(cu).status_code == 404


def test_hazard_without_photo_has_no_working_photo_link(rs16_syn):
    env, data = rs16_syn
    it = env.by_id()[data["h5"]["id"]]
    url = it.get("photo_url")
    if url:
        assert env.client.get(url).status_code == 404


def test_hazard_with_missing_photo_file_does_not_break_the_list(rs16_syn):
    env, data = rs16_syn
    data["h1"]["pair"]["blur"].unlink()
    r = env.client.get(env.list_route())
    assert r.status_code == 200
    by = {int(i["id"]): i for i in r.json()}
    assert data["h1"]["id"] in by
    url = by[data["h1"]["id"]].get("photo_url")
    if url:
        assert env.client.get(url).status_code == 404
    # the others are unaffected
    assert env.client.get(by[data["h2"]["id"]]["photo_url"]).status_code == 200


def test_json_exposes_no_filesystem_path(rs16_syn):
    env, data = rs16_syn
    r = env.client.get(env.list_route())
    body = r.text
    assert str(env.tmp) not in body and fwd(env.tmp) not in body
    assert str(env.tmp).replace("\\", "\\\\") not in body
    for d in data.values():
        if d["pair"]:
            for k in ("orig", "blur", "orig_crop", "blur_crop"):
                p = d["pair"][k]
                if p is not None:
                    assert p.name not in body, "file name leaked: %s" % p.name
    for it in r.json():
        for k, v in it.items():
            assert k.lower() not in RAW_PATH_KEYS, "raw path field %r exposed" % k
            if isinstance(v, str) and not v.startswith("/api/"):
                assert not re.search(r"\.(jpe?g|png|mp4|gpx|sqlite|db)\b", v.lower()), (k, v)
                assert "\\" not in v and not re.match(r"^[A-Za-z]:", v) and not v.startswith("/")


def test_null_lat_lon_cannot_be_pinned(rs16_syn):
    """The schema may forbid NULL coordinates (then there is nothing to test). Otherwise the hazard must be
    excluded from the list or carry null coordinates (never fake ones such as 0,0 and never a 500)."""
    env, data = rs16_syn
    try:
        hid = env.x(
            "INSERT INTO hazards (hazard_class, lat, lon, direction, severity, confidence, sightings, "
            "candidate_count, first_seen, last_seen, photo_path, crop_path, status, created_at, updated_at) "
            "VALUES ('pothole', NULL, NULL, NULL, 1, 0.5, 1, 1, '2026-10-20T12:00:00+00:00', "
            "'2026-10-20T12:00:00+00:00', NULL, NULL, 'new', 'x', 'x')")
    except sqlite3.IntegrityError:
        pytest.skip("hazards.lat/lon are NOT NULL, a hazard without coordinates cannot exist")
    r = env.client.get(env.list_route())
    assert r.status_code == 200
    by = {int(i["id"]): i for i in r.json()}
    if hid in by:
        assert by[hid]["lat"] is None and by[hid]["lon"] is None
    for did, d in ((data["h1"]["id"], data["h1"]),):
        assert did in by and by[did]["lat"] == pytest.approx(d["lat"])


def test_order_is_deterministic(rs16_syn):
    env, data = rs16_syn
    # identical sort keys on purpose
    for i in range(3):
        env.add_hazard(hazard_class="pothole", lat=38.9 + i * 0.001, lon=-77.4, confidence=0.5, severity=1,
                       first_seen="2026-10-20T12:00:00+00:00", last_seen="2026-10-20T12:00:00+00:00")
    first = [i["id"] for i in env.hazards()]
    for _ in range(4):
        assert [i["id"] for i in env.hazards()] == first
    assert len(first) == len(set(first)) == 8


def test_order_follows_a_sensible_documented_key(rs16_syn):
    """Ascending id is the preferred order; other natural orders (by time, descending id) are tolerated."""
    env, _ = rs16_syn
    items = env.hazards()
    ids = [i["id"] for i in items]

    def monotone(vals):
        return vals == sorted(vals) or vals == sorted(vals, reverse=True)

    keys = [ids, [i["last_seen"] for i in items], [i["first_seen"] for i in items],
            [i["severity"] for i in items], [i["confidence"] for i in items]]
    assert any(monotone(k) for k in keys), "order of %s follows no natural key" % ids


def test_reflects_database_changes_without_caching(rs16_syn):
    env, data = rs16_syn
    hid = data["h1"]["id"]
    assert env.by_id()[hid]["status"] == "new"
    env.x("UPDATE hazards SET status = 'approved' WHERE id = ?", (hid,))
    assert env.by_id()[hid]["status"] == "approved"
    env.x("DELETE FROM hazards WHERE id = ?", (hid,))
    assert hid not in env.by_id()
    new_id = env.add_hazard(lat=38.95, lon=-77.5)
    assert new_id in env.by_id()


def test_many_hazards_all_returned(rs16_env):
    env = rs16_env
    ids = [env.add_hazard(lat=38.0 + i * 0.0005, lon=-77.0 - i * 0.0005) for i in range(300)]
    got = [int(i["id"]) for i in env.hazards()]
    assert sorted(got) == sorted(ids)


def test_get_does_not_modify_anything(rs16_syn):
    env, _ = rs16_syn
    before = (env.q("SELECT * FROM hazards ORDER BY id"), env.q("SELECT * FROM candidates ORDER BY id"))
    env.hazards()
    after = (env.q("SELECT * FROM hazards ORDER BY id"), env.q("SELECT * FROM candidates ORDER BY id"))
    assert before == after


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_list_route_is_get_only(rs16_syn, method):
    env, _ = rs16_syn
    before = env.hazards_rows()
    r = getattr(env.client, method)(env.list_route())
    assert r.status_code in (404, 405, 422), r.status_code
    assert not (200 <= r.status_code < 300)
    assert env.hazards_rows() == before


def test_openapi_lists_only_get_for_the_list_route(rs16_env):
    ops = rs16_env.paths()[rs16_env.list_route()]
    assert set(ops) == {"get"}, sorted(ops)


def test_existing_per_drive_route_still_works(rs16_syn):
    env, _ = rs16_syn
    assert env.client.get("/api/drives/nope/hazards").status_code == 404


# ------------------------------------------------------------------ optional filters (only when offered)

def test_status_filter_when_offered(rs16_syn):
    env, data = rs16_syn
    if "status" not in env.list_params():
        pytest.skip("no status filter offered")
    for st in STATUSES:
        got = env.hazards(status=st)
        want = sorted(d["id"] for d in data.values() if d["status"] == st)
        assert sorted(int(i["id"]) for i in got) == want, st
        assert all(i["status"] == st for i in got)
    r = env.client.get(env.list_route(), params={"status": "bogus-status"})
    assert r.status_code in (400, 422) or (r.status_code == 200 and r.json() == [])


def test_status_filter_unfiltered_still_returns_all(rs16_syn):
    env, data = rs16_syn
    if "status" not in env.list_params():
        pytest.skip("no status filter offered")
    assert len(env.hazards()) == len(data)


def test_drive_filter_when_offered(rs16_syn):
    env, data = rs16_syn
    name = next((n for n in ("drive_id", "drive") if n in env.list_params()), None)
    if name is None:
        pytest.skip("no drive filter offered")
    a = env.hazards(**{name: "synA"})
    assert sorted(int(i["id"]) for i in a) == sorted(data[n]["id"] for n in ("h1", "h2"))
    b = env.hazards(**{name: "synB"})
    assert sorted(int(i["id"]) for i in b) == sorted(data[n]["id"] for n in ("h3", "h4"))
    r = env.client.get(env.list_route(), params={name: "no-such-drive"})
    assert r.status_code == 404 or (r.status_code == 200 and r.json() == [])
