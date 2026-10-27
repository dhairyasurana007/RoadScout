"""Task 16 (map view), backend: hazards produced by the real (fake-provider) pipeline appear on the map API
with working, blurred-only photo links."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from rs16_helpers import (DONE, FFMPEG, FFPROBE, RAW_PATH_KEYS, fwd, rs16_env, status_of)  # noqa: F401

pytestmark = pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not on PATH")


@pytest.fixture()
def processed(rs16_env):
    env = rs16_env
    assert "META_API_KEY" not in os.environ
    did = env.process_ok()
    return env, did


def norm(p):
    return os.path.normcase(os.path.abspath(str(p)))


def test_processed_drive_yields_hazards_on_the_map_api(processed):
    env, did = processed
    rows = env.hazards_rows()
    assert rows, "the fake-provider pipeline produced no hazards"
    items = env.hazards()
    assert sorted(int(i["id"]) for i in items) == sorted(r["id"] for r in rows)
    by = {int(i["id"]): i for i in items}
    for r in rows:
        it = by[r["id"]]
        assert it["hazard_class"] == r["hazard_class"]
        assert it["lat"] == pytest.approx(r["lat"]) and it["lon"] == pytest.approx(r["lon"])
        assert it["sightings"] == r["sightings"] and it["status"] == r["status"] == "new"
        assert it["severity"] == r["severity"]
        assert it["confidence"] == pytest.approx(r["confidence"])
        assert 38.5 < it["lat"] < 39.5 and -78 < it["lon"] < -76.5  # the synthetic GPX is near 38.88, -77.30


def test_every_listed_photo_is_the_blurred_image_not_the_original(processed):
    env, did = processed
    cands = env.cands(did)
    originals = []
    for c in cands:
        for k in ("best_frame_path", "crop_path"):
            if c.get(k):
                originals.append(Path(c[k]))
    assert originals
    originals_bytes = [p.read_bytes() for p in originals if p.is_file()]
    assert originals_bytes
    storage = env.storage.resolve()
    n = 0
    for it in env.hazards():
        row = next(r for r in env.hazards_rows() if r["id"] == it["id"])
        assert it["photo_url"] and it["photo_url"].startswith("/api/"), it
        r = env.client.get(it["photo_url"])
        assert r.status_code == 200
        assert r.headers["content-type"].split(";")[0].strip() == "image/jpeg"
        f = env.resolve(row["photo_path"])
        assert f is not None and r.content == f.read_bytes()
        assert r.content not in originals_bytes, "an unblurred candidate frame was served"
        assert norm(f) not in {norm(p) for p in originals}
        assert Path(f).resolve().is_relative_to(storage)
        assert "candidates" in Path(f).resolve().relative_to(storage).parts
        n += 1
        if it.get("crop_url"):
            rc = env.client.get(it["crop_url"])
            assert rc.status_code == 200
            assert rc.content not in originals_bytes
    assert n


def test_photo_route_by_id_matches_the_listed_url(processed):
    env, did = processed
    for it in env.hazards():
        a = env.client.get(it["photo_url"])
        b = env.photo(it["id"])
        assert a.status_code == b.status_code == 200
        assert a.content == b.content


def test_pointing_a_pipeline_hazard_at_its_unblurred_frame_is_refused(processed):
    env, did = processed
    h = env.hazards_rows()[0]
    cand = next(c for c in env.cands(did) if c["hazard_id"] == h["id"])
    orig = Path(cand["best_frame_path"])
    assert orig.is_file()
    env.x("UPDATE hazards SET photo_path = ? WHERE id = ?", (str(orig), h["id"]))
    r = env.photo(h["id"])
    assert r.status_code == 404
    assert r.content != orig.read_bytes()


def test_list_json_has_no_paths_or_unblurred_file_names(processed):
    env, did = processed
    r = env.client.get(env.list_route())
    body = r.text
    assert str(env.tmp) not in body and fwd(env.tmp) not in body
    for c in env.cands(did):
        for k in ("best_frame_path", "crop_path", "blurred_frame_path", "blurred_crop_path"):
            if c.get(k):
                assert Path(c[k]).name not in body
    for it in r.json():
        assert not (set(k.lower() for k in it) & RAW_PATH_KEYS)


def test_map_api_agrees_with_the_per_drive_list(processed):
    env, did = processed
    per_drive = env.client.get("/api/drives/%s/hazards" % did).json()
    assert per_drive
    # every accepted candidate sits in some deduplicated hazard that the map API lists
    hz_ids = {int(i["id"]) for i in env.hazards()}
    linked = {c["hazard_id"] for c in env.cands(did) if c["hazard_id"] is not None}
    assert linked and linked <= hz_ids


def test_second_drive_updates_the_map_without_duplicates_or_broken_photos(processed):
    env, did = processed
    did2 = env.upload()
    env.run(did2)
    assert status_of(env.get_drive(did2)) in DONE
    rows = env.hazards_rows()
    items = env.hazards()
    assert sorted(int(i["id"]) for i in items) == sorted(r["id"] for r in rows)
    assert len({int(i["id"]) for i in items}) == len(items)
    by = {int(i["id"]): i for i in items}
    for r in rows:
        assert by[r["id"]]["sightings"] == r["sightings"]
        assert env.client.get(by[r["id"]]["photo_url"]).status_code == 200


def test_rerunning_the_pipeline_keeps_all_listed_photos_working(processed):
    env, did = processed
    env.run(did)
    items = env.hazards()
    assert items
    assert sorted(int(i["id"]) for i in items) == sorted(r["id"] for r in env.hazards_rows())
    for it in items:
        assert env.client.get(it["photo_url"]).status_code == 200


def test_process_endpoint_end_to_end_then_map(rs16_env):
    env = rs16_env
    did = env.upload()
    route = re.sub(r"\{[^}/]+\}", did, env.process_route())
    r = env.client.post(route)
    assert r.status_code in (200, 201, 202), r.text
    d = env.wait(did)
    assert status_of(d) in DONE, d
    items = env.hazards()
    assert items
    for it in items:
        assert env.client.get(it["photo_url"]).status_code == 200


def test_status_set_in_the_db_shows_on_the_map(processed):
    env, did = processed
    h = env.hazards_rows()[0]
    env.x("UPDATE hazards SET status = 'approved' WHERE id = ?", (h["id"],))
    assert env.by_id()[h["id"]]["status"] == "approved"
    if "status" in env.list_params():
        assert [int(i["id"]) for i in env.hazards(status="approved")] == [h["id"]]


def test_blurred_photos_survive_the_delete_video_button(processed):
    env, did = processed
    route = env.delete_video_route()
    if route is None:
        pytest.skip("no delete-video route")
    method, path = route
    r = getattr(env.client, method)(re.sub(r"\{[^}/]+\}", did, path))
    assert r.status_code in (200, 202, 204), r.text
    items = env.hazards()
    assert items
    for it in items:
        resp = env.client.get(it["photo_url"])
        assert resp.status_code == 200
        row = next(x for x in env.hazards_rows() if x["id"] == it["id"])
        assert resp.content == env.resolve(row["photo_path"]).read_bytes()


def test_deleting_a_drive_artifact_never_turns_into_a_500(processed):
    """Fail closed: if a blurred file vanishes, the list still works and its link 404s."""
    env, did = processed
    row = env.hazards_rows()[0]
    f = env.resolve(row["photo_path"])
    f.unlink()
    r = env.client.get(env.list_route())
    assert r.status_code == 200
    it = next(i for i in r.json() if i["id"] == row["id"])
    if it.get("photo_url"):
        assert env.client.get(it["photo_url"]).status_code == 404
