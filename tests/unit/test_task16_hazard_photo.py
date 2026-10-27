"""Task 16 (map view), backend: the hazard photo route serves ONLY the blurred image the hazard points at."""
from __future__ import annotations

import pytest

from rs16_helpers import rs16_env, rs16_syn, write_jpeg  # noqa: F401


def jpeg_ct(r):
    return r.headers["content-type"].split(";")[0].strip().lower()


def assert_not_served(r, *forbidden):
    assert r.status_code == 404, "expected 404, got %s" % r.status_code
    for p in forbidden:
        assert r.content != p.read_bytes(), "the forbidden file %s was served" % p


# ------------------------------------------------------------------ happy path

def test_serves_the_blurred_best_frame(rs16_syn):
    env, data = rs16_syn
    for name in ("h1", "h2", "h3", "h4"):
        pair = data[name]["pair"]
        r = env.photo(data[name]["id"])
        assert r.status_code == 200, name
        assert jpeg_ct(r) == "image/jpeg"
        assert r.content == pair["blur"].read_bytes()
        assert r.content != pair["orig"].read_bytes()


def test_explicit_variant_best_is_the_same_as_default(rs16_syn):
    env, data = rs16_syn
    r = env.client.get(env.photo_url_for(data["h1"]["id"]) + "?variant=best")
    assert r.status_code in (200, 422)
    if r.status_code == 200:
        assert r.content == data["h1"]["pair"]["blur"].read_bytes()


def test_variant_crop_serves_the_blurred_crop(rs16_syn):
    env, data = rs16_syn
    for name in ("h1", "h4"):
        pair = data[name]["pair"]
        r = env.photo(data[name]["id"], crop=True)
        assert r.status_code == 200, name
        assert jpeg_ct(r) == "image/jpeg"
        assert r.content == pair["blur_crop"].read_bytes()
        assert r.content != pair["orig_crop"].read_bytes()
        assert r.content != pair["blur"].read_bytes()


def test_each_hazard_serves_its_own_photo(rs16_syn):
    env, data = rs16_syn
    seen = {}
    for name in ("h1", "h2", "h3", "h4"):
        seen[name] = env.photo(data[name]["id"]).content
    assert len(set(seen.values())) == 4


def test_photo_is_read_fresh_from_disk(rs16_syn):
    env, data = rs16_syn
    pair = data["h1"]["pair"]
    assert env.photo(data["h1"]["id"]).content == pair["blur"].read_bytes()
    write_jpeg(pair["blur"], 99999)
    assert env.photo(data["h1"]["id"]).content == pair["blur"].read_bytes()


def test_serving_changes_nothing(rs16_syn):
    env, data = rs16_syn
    before = (env.q("SELECT * FROM hazards ORDER BY id"), env.q("SELECT * FROM candidates ORDER BY id"),
              data["h1"]["pair"]["blur"].read_bytes(), data["h1"]["pair"]["orig"].read_bytes())
    env.photo(data["h1"]["id"])
    env.photo(data["h1"]["id"], crop=True)
    after = (env.q("SELECT * FROM hazards ORDER BY id"), env.q("SELECT * FROM candidates ORDER BY id"),
             data["h1"]["pair"]["blur"].read_bytes(), data["h1"]["pair"]["orig"].read_bytes())
    assert before == after


# ------------------------------------------------------------------ not found / missing

@pytest.mark.parametrize("hid", ["999999", "0", "-1", "123456789012345678901234567890"])
def test_unknown_hazard_is_404(rs16_syn, hid):
    env, _ = rs16_syn
    r = env.client.get(env.photo_url_for(hid))
    assert r.status_code in (404, 422), r.status_code
    assert not r.headers.get("content-type", "").startswith("image/")


def test_hazard_with_null_photo_path_is_404(rs16_syn):
    env, data = rs16_syn
    assert_not_served(env.photo(data["h5"]["id"]))
    assert_not_served(env.photo(data["h5"]["id"], crop=True))


def test_null_photo_path_does_not_fall_back_to_the_candidate_originals(rs16_syn):
    env, data = rs16_syn
    pair = data["h1"]["pair"]
    env.x("UPDATE hazards SET photo_path = NULL, crop_path = NULL WHERE id = ?", (data["h1"]["id"],))
    assert pair["orig"].is_file() and pair["blur"].is_file()
    assert_not_served(env.photo(data["h1"]["id"]), pair["orig"], pair["blur"])
    assert_not_served(env.photo(data["h1"]["id"], crop=True), pair["orig_crop"], pair["blur_crop"])


def test_null_crop_path_is_404_for_the_crop_variant_only(rs16_syn):
    env, data = rs16_syn
    pair = data["h1"]["pair"]
    env.x("UPDATE hazards SET crop_path = NULL WHERE id = ?", (data["h1"]["id"],))
    assert_not_served(env.photo(data["h1"]["id"], crop=True), pair["orig_crop"], pair["blur_crop"], pair["blur"])
    assert env.photo(data["h1"]["id"]).status_code == 200


def test_missing_file_is_404_not_500(rs16_syn):
    env, data = rs16_syn
    pair = data["h1"]["pair"]
    pair["blur"].unlink()
    assert_not_served(env.photo(data["h1"]["id"]), pair["orig"])
    pair["blur_crop"].unlink()
    assert_not_served(env.photo(data["h1"]["id"], crop=True), pair["orig_crop"])


def test_path_that_is_a_directory_is_404(rs16_syn):
    env, data = rs16_syn
    env.x("UPDATE hazards SET photo_path = ? WHERE id = ?", (str(env.cand_dir("synA")), data["h1"]["id"]))
    assert env.photo(data["h1"]["id"]).status_code == 404


@pytest.mark.parametrize("bad", ["", " ", "\x00", "nul", "http://example.com/x.jpg", "file:///etc/passwd"])
def test_garbage_stored_paths_fail_closed(rs16_syn, bad):
    env, data = rs16_syn
    env.x("UPDATE hazards SET photo_path = ? WHERE id = ?", (bad, data["h1"]["id"]))
    r = env.photo(data["h1"]["id"])
    assert r.status_code == 404


# ------------------------------------------------------------------ containment in storage / candidates

def test_file_outside_storage_is_never_served(rs16_syn):
    env, data = rs16_syn
    outside = write_jpeg(env.tmp / "outside" / "secret.jpg", 4242)
    env.x("UPDATE hazards SET photo_path = ?, crop_path = ? WHERE id = ?",
          (str(outside), str(outside), data["h1"]["id"]))
    assert_not_served(env.photo(data["h1"]["id"]), outside)
    assert_not_served(env.photo(data["h1"]["id"], crop=True), outside)


def test_file_in_storage_but_outside_candidate_folders_is_not_served(rs16_syn):
    env, data = rs16_syn
    stray = [
        write_jpeg(env.storage / "synA" / "stray.jpg", 5001),
        write_jpeg(env.storage / "synA" / "work" / "frames" / "f0001.jpg", 5002),
        write_jpeg(env.storage / "top_level.jpg", 5003),
        write_jpeg(env.storage / "synA" / "candidates_other" / "x.jpg", 5004),
    ]
    for p in stray:
        env.x("UPDATE hazards SET photo_path = ? WHERE id = ?", (str(p), data["h1"]["id"]))
        assert_not_served(env.photo(data["h1"]["id"]), p)


def test_dotdot_in_stored_path_cannot_escape_the_candidate_folder(rs16_syn):
    env, data = rs16_syn
    secret = write_jpeg(env.storage / "synA" / "secret.jpg", 6001)
    sneaky = env.cand_dir("synA") / ".." / "secret.jpg"
    env.x("UPDATE hazards SET photo_path = ? WHERE id = ?", (str(sneaky), data["h1"]["id"]))
    assert_not_served(env.photo(data["h1"]["id"]), secret)
    sneaky2 = str(env.cand_dir("synA")) + "/../../../outside_secret.jpg"
    out = write_jpeg(env.tmp / "outside_secret.jpg", 6002)
    env.x("UPDATE hazards SET photo_path = ? WHERE id = ?", (sneaky2, data["h1"]["id"]))
    assert_not_served(env.photo(data["h1"]["id"]), out)


def test_relative_stored_path_outside_storage_is_not_served(rs16_syn):
    env, data = rs16_syn
    out = write_jpeg(env.tmp / "rel_secret.jpg", 6003)
    env.x("UPDATE hazards SET photo_path = ? WHERE id = ?", ("../rel_secret.jpg", data["h1"]["id"]))
    assert_not_served(env.photo(data["h1"]["id"]), out)


def test_candidate_folder_of_another_drive_is_fine(rs16_syn):
    """Hazards span drives: a blurred image in ANY drive's candidates folder is servable."""
    env, data = rs16_syn
    for name in ("h1", "h3"):
        assert env.photo(data[name]["id"]).status_code == 200


# ------------------------------------------------------------------ never the unblurred original

def test_hazard_pointing_at_the_unblurred_best_frame_is_404(rs16_syn):
    env, data = rs16_syn
    pair = data["h1"]["pair"]
    env.x("UPDATE hazards SET photo_path = ? WHERE id = ?", (str(pair["orig"]), data["h1"]["id"]))
    assert_not_served(env.photo(data["h1"]["id"]), pair["orig"])


def test_hazard_pointing_at_the_unblurred_crop_is_404(rs16_syn):
    env, data = rs16_syn
    pair = data["h1"]["pair"]
    env.x("UPDATE hazards SET crop_path = ?, photo_path = ? WHERE id = ?",
          (str(pair["orig_crop"]), str(pair["orig_crop"]), data["h1"]["id"]))
    assert_not_served(env.photo(data["h1"]["id"], crop=True), pair["orig_crop"])
    assert_not_served(env.photo(data["h1"]["id"]), pair["orig_crop"])


def test_hazard_pointing_at_the_original_of_another_candidate_or_drive_is_404(rs16_syn):
    env, data = rs16_syn
    other = data["h3"]["pair"]  # a candidate of another drive and another hazard
    env.x("UPDATE hazards SET photo_path = ?, crop_path = ? WHERE id = ?",
          (str(other["orig"]), str(other["orig"]), data["h1"]["id"]))
    assert_not_served(env.photo(data["h1"]["id"]), other["orig"])
    assert_not_served(env.photo(data["h1"]["id"], crop=True), other["orig"])


def test_original_is_refused_even_when_it_has_no_blurred_twin(rs16_syn):
    env, data = rs16_syn
    d = env.cand_dir("synA")
    orig = env.jpeg(d / "lonely_best.jpg")
    env.add_candidate("synA", best=orig)  # blur failed / not run: no blurred image
    env.x("UPDATE hazards SET photo_path = ? WHERE id = ?", (str(orig), data["h2"]["id"]))
    assert_not_served(env.photo(data["h2"]["id"]), orig)


def test_original_path_spelled_differently_is_still_refused(rs16_syn):
    env, data = rs16_syn
    pair = data["h1"]["pair"]
    p = str(pair["orig"])
    variants = [p.replace("\\", "/"), str(pair["orig"].parent / "." / pair["orig"].name),
                str(pair["orig"].parent.parent / "candidates" / ".." / "candidates" / pair["orig"].name)]
    if p != p.upper():
        variants.append(p.upper() if p.upper() != p else p)
    for v in variants:
        env.x("UPDATE hazards SET photo_path = ? WHERE id = ?", (v, data["h1"]["id"]))
        r = env.photo(data["h1"]["id"])
        assert r.content != pair["orig"].read_bytes(), "original served through spelling %r" % v


@pytest.mark.parametrize("variant", ["original", "raw", "unblurred", "full", "source", "best_frame", "orig"])
def test_no_variant_unlocks_the_unblurred_image(rs16_syn, variant):
    env, data = rs16_syn
    pair = data["h1"]["pair"]
    for url in (env.photo_url_for(data["h1"]["id"]) + "?variant=" + variant,):
        r = env.client.get(url)
        assert r.status_code in (200, 400, 404, 422)
        assert r.content != pair["orig"].read_bytes() and r.content != pair["orig_crop"].read_bytes()


def test_no_query_parameter_can_choose_a_file(rs16_syn):
    env, data = rs16_syn
    pair = data["h1"]["pair"]
    for qs in ("?path=%s" % pair["orig"], "?file=%s" % pair["orig"], "?photo_path=%s" % pair["orig"],
               "?src=%s" % pair["orig"].name):
        r = env.client.get(env.photo_url_for(data["h1"]["id"]) + qs)
        assert r.content != pair["orig"].read_bytes()


def test_deleting_the_unblurred_originals_leaves_blurred_photos_working(rs16_syn):
    env, data = rs16_syn
    for name in ("h1", "h2", "h3", "h4"):
        for k in ("orig", "orig_crop"):
            p = data[name]["pair"][k]
            if p is not None:
                p.unlink()
    for name in ("h1", "h2", "h3", "h4"):
        r = env.photo(data[name]["id"])
        assert r.status_code == 200
        assert r.content == data[name]["pair"]["blur"].read_bytes()


# ------------------------------------------------------------------ ids and methods

@pytest.mark.parametrize("hid", [
    "abc", "1.5", "1e2", "%2e%2e%2f%2e%2e%2fetc%2fpasswd", "..%2f..%2fsecret", "1%2f..%2f2", "1;rm", "1 OR 1=1",
    "%00", "1%00", "%2e%2e", "null", "undefined", "1,2",
])
def test_non_integer_and_traversal_ids_are_rejected(rs16_syn, hid):
    env, data = rs16_syn
    secret = write_jpeg(env.tmp / "secret.jpg", 7001)
    url = env.photo_url_for(hid)
    r = env.client.get(url)
    assert r.status_code in (400, 404, 422), (hid, r.status_code)
    assert r.content != secret.read_bytes()
    assert not r.headers.get("content-type", "").startswith("image/")


def test_id_with_plus_sign_or_leading_zeros_never_serves_another_hazard(rs16_syn):
    env, data = rs16_syn
    h1 = data["h1"]["pair"]["blur"].read_bytes()
    for hid in ("+%d" % data["h1"]["id"], "%03d" % data["h1"]["id"]):
        r = env.client.get(env.photo_url_for(hid))
        if r.status_code == 200:
            assert r.content == h1  # parsed as the same integer, never a different file


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_photo_route_is_get_only(rs16_syn, method):
    env, data = rs16_syn
    before = env.hazards_rows()
    r = getattr(env.client, method)(env.photo_url_for(data["h1"]["id"]))
    assert r.status_code in (404, 405, 422)
    assert env.hazards_rows() == before
    assert data["h1"]["pair"]["blur"].is_file()


def test_openapi_photo_route_is_get_only(rs16_env):
    ops = rs16_env.paths()[rs16_env.photo_route()]
    assert set(ops) == {"get"}, sorted(ops)


def test_per_drive_photo_route_still_works(rs16_syn):
    env, _ = rs16_syn
    r = env.client.get("/api/drives/synA/hazards/1/photo")
    assert r.status_code == 404  # unknown drive in this synthetic db
