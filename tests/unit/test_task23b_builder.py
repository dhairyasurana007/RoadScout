"""Task 23b: build_demo_dataset - a privacy-safe read-only demo folder built from a project database."""
from __future__ import annotations

import dataclasses
import inspect
import re
import sqlite3
from pathlib import Path

import pytest

from rs23b_helpers import (DRIVE_LETTER, IMG_EXT, PATH_COLS, World, all_files, all_text_values, dump, rows, sha,
                           tables, tree_hash)


@pytest.fixture()
def world(tmp_path):
    return World(tmp_path)


@pytest.fixture()
def built(world, tmp_path):
    out = tmp_path / "out" / "demo"
    report = world.build(out)
    return world, out, report


def rel(prefix, drive, name):
    return f"{prefix}/storage/{drive}/candidates/{name}"


# ------------------------------------------------------------------ the report

def test_signature():
    from backend.demo_dataset import build_demo_dataset

    p = inspect.signature(build_demo_dataset).parameters
    assert p["drive_ids"].default is None
    assert tuple(p["statuses"].default) == ("approved", "reported")
    assert p["path_prefix"].default == "demo"
    assert p["overwrite"].default is False


def test_report_counts(built):
    world, out, r = built
    from backend.demo_dataset import DemoBuildReport

    assert isinstance(r, DemoBuildReport) and dataclasses.is_dataclass(r)
    assert r.hazards_kept == 2
    assert r.candidates_kept == 3
    assert r.drives_kept == 2
    assert r.files_copied == 3
    assert r.bytes_copied == sum(p.stat().st_size for p in world.photo_files)


def test_unservable_hazards_are_skipped_with_a_reason(built):
    world, out, r = built
    skipped = dict((int(h), reason) for h, reason in r.hazards_skipped)
    assert world.skipped_unservable <= set(skipped)
    assert all(isinstance(v, str) and v.strip() for v in skipped.values())
    assert not (world.kept & set(skipped))
    for h in world.skipped_unservable:
        assert not re.search(r"[A-Za-z]:[\\/]|/Users/", skipped[h]), "a reason must not carry a local path"


def test_reasons_differ_in_a_useful_way(built):
    world, out, r = built
    skipped = dict((int(h), reason) for h, reason in r.hazards_skipped)
    assert len({skipped[h] for h in world.skipped_unservable}) >= 3


# ------------------------------------------------------------------ what is kept

def test_only_the_kept_hazards_with_all_their_columns(built):
    world, out, r = built
    src = {h["id"]: h for h in rows(world.db, "SELECT * FROM hazards")}
    got = {h["id"]: h for h in rows(out / "demo.sqlite", "SELECT * FROM hazards")}
    assert set(got) == world.kept
    for hid, row in got.items():
        for col, val in src[hid].items():
            if col in ("photo_path", "crop_path"):
                continue
            assert row[col] == val, (hid, col)


def test_hazard_photo_paths_are_relative_with_forward_slashes(built):
    world, out, r = built
    got = {h["id"]: h for h in rows(out / "demo.sqlite", "SELECT * FROM hazards")}
    assert got[1]["photo_path"] == rel("demo", "d1", "c101_best_blur.jpg")
    assert got[1]["crop_path"] == rel("demo", "d1", "c101_crop_blur.jpg")
    assert got[2]["photo_path"] == rel("demo", "d2", "c201_best_blur.jpg")
    assert got[2]["crop_path"] is None


def test_other_statuses_and_other_candidates_are_not_copied(built):
    world, out, r = built
    cand_ids = {c["id"] for c in rows(out / "demo.sqlite", "SELECT id FROM candidates")}
    assert cand_ids == world.kept_candidates
    verdict_ids = {v["candidate_id"] for v in rows(out / "demo.sqlite", "SELECT candidate_id FROM verdicts")}
    assert verdict_ids == world.kept_candidates
    drives = {d["id"] for d in rows(out / "demo.sqlite", "SELECT id FROM drives")}
    assert drives == world.kept_drives


def test_kept_candidates_and_verdicts_keep_their_data(built):
    world, out, r = built
    src = {c["id"]: c for c in rows(world.db, "SELECT * FROM candidates")}
    got = {c["id"]: c for c in rows(out / "demo.sqlite", "SELECT * FROM candidates")}
    for cid, row in got.items():
        for col, val in src[cid].items():
            if col in PATH_COLS:
                continue
            if col in ("blur_status", "blur_note") and row["blurred_frame_path"] is None:
                continue        # no blurred copy was published for this candidate, so it is no longer 'ok'
            assert row[col] == val, (cid, col)
    sv = {v["candidate_id"]: v for v in rows(world.db, "SELECT * FROM verdicts")}
    gv = {v["candidate_id"]: v for v in rows(out / "demo.sqlite", "SELECT * FROM verdicts")}
    for cid, row in gv.items():
        assert row == sv[cid]


def test_unblurred_columns_hold_no_real_path(built, tmp_path):
    world, out, r = built
    for c in rows(out / "demo.sqlite", "SELECT * FROM candidates"):
        for col in ("best_frame_path", "crop_path"):
            v = c[col]
            if v is None:
                continue
            assert not Path(v).is_absolute() and not DRIVE_LETTER.search(v)
            assert str(world.storage) not in v
            assert not (out / v).is_file() and not (Path(tmp_path) / v).is_file()


def test_candidate_blurred_columns_are_relative_when_present(built):
    world, out, r = built
    got = {c["id"]: c for c in rows(out / "demo.sqlite", "SELECT * FROM candidates")}
    assert got[101]["blurred_frame_path"] == rel("demo", "d1", "c101_best_blur.jpg")
    assert got[101]["blurred_crop_path"] == rel("demo", "d1", "c101_crop_blur.jpg")
    assert got[201]["blurred_frame_path"] == rel("demo", "d2", "c201_best_blur.jpg")
    for c in got.values():
        for col in ("blurred_frame_path", "blurred_crop_path"):
            v = c[col]
            if v is not None:
                assert v.startswith("demo/storage/") and "\\" not in v and not DRIVE_LETTER.search(v)


def test_candidate_without_a_copied_photo_has_no_dangling_absolute_path(built):
    world, out, r = built
    c102 = rows(out / "demo.sqlite", "SELECT * FROM candidates WHERE id = 102")[0]
    for col in PATH_COLS:
        v = c102[col]
        assert v is None or (not Path(v).is_absolute() and str(world.storage) not in v)


def test_kept_drives_are_scrubbed_and_keep_their_facts(built):
    world, out, r = built
    src = {d["id"]: d for d in rows(world.db, "SELECT * FROM drives")}
    for d in rows(out / "demo.sqlite", "SELECT * FROM drives"):
        s = src[d["id"]]
        assert d["video_filename"] is None
        assert d["video_deleted_at"], "the UI then says the raw video is gone"
        assert d["error"] is None and d["warning"] is None
        for col in ("video_path", "gpx_path"):
            v = d[col]
            assert v is None or (not Path(v).is_absolute() and not DRIVE_LETTER.search(v) and str(world.storage) not in v)
        for col in ("started_at", "started_at_source", "status", "duration_seconds", "created_at",
                    "clock_offset_seconds"):
            assert d[col] == s[col], (d["id"], col)


def test_tables_the_demo_must_not_carry_are_empty(built):
    world, out, r = built
    for t in ("detections", "detection_chunks", "pipeline_runs", "exports"):
        assert rows(out / "demo.sqlite", f"SELECT COUNT(*) AS n FROM {t}")[0]["n"] == 0, t


def test_app_meta_is_copied(built):
    world, out, r = built
    assert rows(out / "demo.sqlite", "SELECT key, value FROM app_meta ORDER BY key") == \
        rows(world.db, "SELECT key, value FROM app_meta ORDER BY key")


def test_schema_is_the_projects_own(built, tmp_path):
    world, out, r = built
    from backend.db import init_db

    fresh = tmp_path / "fresh.sqlite"
    init_db(fresh)
    assert tables(out / "demo.sqlite") == tables(fresh)
    for t in tables(fresh):
        a = rows(fresh, f"PRAGMA table_info({t})")
        b = rows(out / "demo.sqlite", f"PRAGMA table_info({t})")
        assert a == b, t
    assert rows(out / "demo.sqlite", "PRAGMA integrity_check")[0]["integrity_check"] == "ok"


def test_the_output_database_opens_with_the_projects_helpers(built):
    world, out, r = built
    from backend.db import init_db

    before = dump(out / "demo.sqlite")
    init_db(out / "demo.sqlite")
    assert dump(out / "demo.sqlite") == before


# ------------------------------------------------------------------ what is copied

def test_only_the_allowed_files_exist_and_they_are_the_blurred_photos(built):
    world, out, r = built
    files = all_files(out)
    assert files == sorted([
        "demo.sqlite",
        "storage/d1/candidates/c101_best_blur.jpg",
        "storage/d1/candidates/c101_crop_blur.jpg",
        "storage/d2/candidates/c201_best_blur.jpg",
    ])


def test_copies_are_byte_identical_to_the_blurred_sources(built):
    world, out, r = built
    for p in world.photo_files:
        d = p.parent.parent.name
        assert (out / "storage" / d / "candidates" / p.name).read_bytes() == p.read_bytes()


def test_no_original_video_gpx_frame_or_export_is_copied(built):
    world, out, r = built
    names = {Path(f).name for f in all_files(out)}
    for bad in ("video.mov", "track.gpx", "chunk_0000.mp4", "export_1.mp4", "stray_unreferenced.jpg",
                "c101_best.jpg", "c101_crop.jpg", "c102_best.jpg", "c102_best_blur.jpg", "c301_best.jpg"):
        assert bad not in names, bad
    assert not [f for f in all_files(out) if f.lower().endswith((".mov", ".mp4", ".gpx", ".m4v"))]


def test_no_original_name_or_content_is_in_the_output(built):
    world, out, r = built
    original_names = {p.name for p in world.originals}
    original_hashes = {sha(p) for p in world.originals}
    for f in all_files(out):
        assert Path(f).name not in original_names
        if f != "demo.sqlite":
            assert sha(out / f) not in original_hashes


def test_nothing_unexpected_is_left_next_to_the_database(built):
    world, out, r = built
    assert sorted(p.name for p in out.iterdir()) == ["demo.sqlite", "storage"]


# ------------------------------------------------------------------ the privacy scan holds on the result

def test_no_text_in_the_database_leaks_a_local_path(built, tmp_path):
    world, out, r = built
    needles = {str(world.storage), str(world.storage).replace("\\", "/"), str(world.db), str(world.db).replace("\\", "/"),
               str(world.storage.resolve()), str(tmp_path)}
    for table, col, v in all_text_values(out / "demo.sqlite"):
        for n in needles:
            assert n not in v, (table, col)
        assert not DRIVE_LETTER.search(v), (table, col, v)
        assert "/Users/" not in v and "\\Users\\" not in v, (table, col)


def test_every_image_is_referenced_by_the_database(built):
    world, out, r = built
    refs = set()
    for t, c, v in all_text_values(out / "demo.sqlite"):
        if c in PATH_COLS or c in ("photo_path",):
            refs.add(v.replace("\\", "/"))
    for f in all_files(out):
        if f.lower().endswith(IMG_EXT):
            assert f"demo/{f}" in refs, f


def test_every_kept_hazard_photo_exists_in_the_output(built):
    world, out, r = built
    for h in rows(out / "demo.sqlite", "SELECT * FROM hazards"):
        for col in ("photo_path", "crop_path"):
            if h[col]:
                assert (out.parent / h[col]).is_file() or (out / h[col].split("/", 1)[1]).is_file()


# ------------------------------------------------------------------ options

def test_path_prefix(world, tmp_path):
    out = tmp_path / "o"
    world.build(out, path_prefix="site/demo")
    got = {h["id"]: h for h in rows(out / "demo.sqlite", "SELECT * FROM hazards")}
    assert got[1]["photo_path"] == rel("site/demo", "d1", "c101_best_blur.jpg")
    assert "\\" not in got[1]["photo_path"]
    c = rows(out / "demo.sqlite", "SELECT * FROM candidates WHERE id = 101")[0]
    assert c["blurred_frame_path"] == rel("site/demo", "d1", "c101_best_blur.jpg")


def test_drive_ids_restrict_the_hazards(world, tmp_path):
    out = tmp_path / "o"
    r = world.build(out, drive_ids=["d1"])
    assert {h["id"] for h in rows(out / "demo.sqlite", "SELECT id FROM hazards")} == {1}
    assert {d["id"] for d in rows(out / "demo.sqlite", "SELECT id FROM drives")} == {"d1"}
    assert r.hazards_kept == 1 and r.drives_kept == 1 and r.files_copied == 2
    assert not (out / "storage" / "d2").exists()


def test_drive_ids_with_two_drives(world, tmp_path):
    out = tmp_path / "o"
    r = world.build(out, drive_ids=["d1", "d2"])
    assert r.hazards_kept == 2


def test_statuses_option(world, tmp_path):
    out = tmp_path / "o"
    r = world.build(out, statuses=("approved", "reported", "rejected", "new"))
    assert {h["id"] for h in rows(out / "demo.sqlite", "SELECT id FROM hazards")} == {1, 2, 4, 5}
    assert r.hazards_kept == 4
    got = {c["id"] for c in rows(out / "demo.sqlite", "SELECT id FROM candidates")}
    assert {104, 202} <= got and 103 not in got


def test_only_approved(world, tmp_path):
    out = tmp_path / "o"
    r = world.build(out, statuses=("approved",))
    assert {h["id"] for h in rows(out / "demo.sqlite", "SELECT id FROM hazards")} == {1}
    assert {d["id"] for d in rows(out / "demo.sqlite", "SELECT id FROM drives")} == {"d1"}


def test_a_status_that_is_servable_but_not_listed_is_never_copied(world, tmp_path):
    out = tmp_path / "o"
    world.build(out, statuses=("reported",))
    assert not (out / "storage" / "d1").exists()
    assert {h["id"] for h in rows(out / "demo.sqlite", "SELECT id FROM hazards")} == {2}


def test_nothing_to_keep_gives_an_empty_but_valid_demo(world, tmp_path):
    out = tmp_path / "o"
    r = world.build(out, statuses=("possibly_fixed",))
    assert r.hazards_kept == 0 and r.files_copied == 0 and r.bytes_copied == 0
    assert all_files(out) == ["demo.sqlite"]
    assert rows(out / "demo.sqlite", "SELECT COUNT(*) AS n FROM hazards")[0]["n"] == 0


# ------------------------------------------------------------------ the source is never touched

def test_source_database_and_storage_are_unchanged(world, tmp_path):
    before_db = world.fingerprint()
    before_tree = tree_hash(world.storage)
    world.build(tmp_path / "o")
    assert world.fingerprint() == before_db
    assert tree_hash(world.storage) == before_tree
    assert not list(world.db.parent.glob("*-wal")) and not list(world.db.parent.glob("*-journal"))


def test_missing_source_database_is_an_error_and_leaves_nothing(world, tmp_path):
    from backend.demo_dataset import build_demo_dataset

    out = tmp_path / "o"
    with pytest.raises(Exception):
        build_demo_dataset(str(tmp_path / "nope.sqlite"), str(world.storage), str(out))
    assert not out.exists() or not any(out.iterdir())
    assert not (tmp_path / "nope.sqlite").exists()


def test_deterministic(world, tmp_path):
    r1 = world.build(tmp_path / "a")
    r2 = world.build(tmp_path / "b")
    a1, a2 = dataclasses.asdict(r1), dataclasses.asdict(r2)
    a1.pop("out_dir", None)
    a2.pop("out_dir", None)
    assert a1 == a2
    assert dump(tmp_path / "a" / "demo.sqlite") == dump(tmp_path / "b" / "demo.sqlite")
    assert tree_hash(tmp_path / "a" / "storage") == tree_hash(tmp_path / "b" / "storage")


def test_works_with_pathlib_arguments(world, tmp_path):
    from backend.demo_dataset import build_demo_dataset

    r = build_demo_dataset(world.db, world.storage, tmp_path / "o")
    assert r.hazards_kept == 2


# ------------------------------------------------------------------ existing output folders

def test_non_empty_output_is_refused_and_untouched(world, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "keep.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(Exception):
        world.build(out)
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]


def test_an_existing_empty_output_folder_is_fine(world, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    assert world.build(out).hazards_kept == 2


def test_a_previous_demo_is_refused_without_overwrite(world, tmp_path):
    out = tmp_path / "o"
    world.build(out)
    before = tree_hash(out)
    with pytest.raises(Exception):
        world.build(out)
    assert tree_hash(out) == before


def test_overwrite_replaces_only_the_database_and_storage(world, tmp_path):
    out = tmp_path / "o"
    world.build(out, statuses=("approved", "reported", "rejected"))
    (out / "README.txt").write_text("hand written", encoding="utf-8")
    (out / "notes").mkdir()
    (out / "notes" / "n.txt").write_text("also mine", encoding="utf-8")
    r = world.build(out, overwrite=True)
    assert r.hazards_kept == 2
    assert (out / "README.txt").read_text(encoding="utf-8") == "hand written"
    assert (out / "notes" / "n.txt").read_text(encoding="utf-8") == "also mine"
    assert {h["id"] for h in rows(out / "demo.sqlite", "SELECT id FROM hazards")} == {1, 2}
    stored = {f for f in all_files(out) if f.startswith("storage/")}
    assert stored == {"storage/d1/candidates/c101_best_blur.jpg", "storage/d1/candidates/c101_crop_blur.jpg",
                      "storage/d2/candidates/c201_best_blur.jpg"}, "the earlier build's extra photo is gone"


def test_overwrite_on_a_fresh_folder_just_builds(world, tmp_path):
    assert world.build(tmp_path / "fresh", overwrite=True).hazards_kept == 2


# ------------------------------------------------------------------ the privacy scan fails closed

def test_a_verdict_text_with_a_local_path_aborts_the_build(world, tmp_path):
    con = world.conn()
    with con:
        con.execute("UPDATE verdicts SET reason = ? WHERE candidate_id = 101",
                    (f"see {world.storage}/d1/c101_best.jpg",))
    con.close()
    out = tmp_path / "o"
    with pytest.raises(Exception):
        world.build(out)
    assert not out.exists() or not any(out.iterdir()), "no partial output is left"


@pytest.mark.parametrize("leak", ["saved at C:\\Users\\me\\x.jpg", "see /Users/me/clip", "D:/footage/clip.mov",
                                  "open \\Users\\me\\frame"])
def test_local_path_shapes_in_text_abort_the_build(world, tmp_path, leak):
    con = world.conn()
    with con:
        con.execute("UPDATE verdicts SET report_description = ? WHERE candidate_id = 201", (leak,))
    con.close()
    out = tmp_path / "o"
    with pytest.raises(Exception):
        world.build(out)
    assert not out.exists() or not any(out.iterdir())


def test_a_path_in_a_hazard_column_aborts_the_build(world, tmp_path):
    con = world.conn()
    with con:
        con.execute("UPDATE hazards SET first_seen = ? WHERE id = 1", (str(world.db),))
    con.close()
    out = tmp_path / "o"
    with pytest.raises(Exception):
        world.build(out)
    assert not out.exists() or not any(out.iterdir())


def test_a_blurred_file_that_is_byte_identical_to_an_original_is_never_published(world, tmp_path):
    """The photo file differs by path from every original, but its bytes equal one: it must not be copied."""
    clone = world.cand_dir("d2") / "c201_best_blur.jpg"
    clone.write_bytes(world.files[201]["best"].read_bytes())
    out = tmp_path / "o"
    try:
        r = world.build(out)
    except Exception:
        assert not out.exists() or not any(out.iterdir()), "no partial output is left"
        return
    original_hashes = {sha(p) for p in world.originals}
    assert not [f for f in all_files(out) if f != "demo.sqlite" and sha(out / f) in original_hashes]
    assert 2 in {int(h) for h, _ in r.hazards_skipped}


def test_failed_overwrite_does_not_destroy_unrelated_files(world, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "keep.txt").write_text("mine", encoding="utf-8")
    con = world.conn()
    with con:
        con.execute("UPDATE verdicts SET reason = ? WHERE candidate_id = 101", ("C:\\Users\\leak",))
    con.close()
    with pytest.raises(Exception):
        world.build(out, overwrite=True)
    assert (out / "keep.txt").read_text(encoding="utf-8") == "mine"
    assert not (out / "demo.sqlite").exists()


# ------------------------------------------------------------------ fail closed on the photo itself

def test_a_candidate_original_is_not_servable_even_through_another_spelling(world, tmp_path):
    """photo_path spelled with '..' resolves to an original: still refused."""
    original = world.files[101]["best"]
    sneaky = world.storage / "d1" / "candidates" / ".." / "candidates" / original.name
    con = world.conn()
    with con:
        con.execute("UPDATE hazards SET photo_path = ? WHERE id = 1", (str(sneaky),))
    con.close()
    out = tmp_path / "o"
    r = world.build(out)
    assert 1 not in {h["id"] for h in rows(out / "demo.sqlite", "SELECT id FROM hazards")}
    assert 1 in {int(h) for h, _ in r.hazards_skipped}
    assert original.name not in {Path(f).name for f in all_files(out)}


def test_a_hazard_whose_crop_is_an_original_is_not_published_with_that_crop(world, tmp_path):
    original_crop = world.files[101]["crop"]
    con = world.conn()
    with con:
        con.execute("UPDATE hazards SET crop_path = ? WHERE id = 1", (str(original_crop),))
    con.close()
    out = tmp_path / "o"
    try:
        world.build(out)
    except Exception:
        assert not out.exists() or not any(out.iterdir())
        return
    assert original_crop.name not in {Path(f).name for f in all_files(out)}
    originals = {sha(p) for p in world.originals}
    assert not [f for f in all_files(out) if f != "demo.sqlite" and sha(out / f) in originals]
    for h in rows(out / "demo.sqlite", "SELECT * FROM hazards"):
        assert h["crop_path"] is None or h["crop_path"].endswith("_blur.jpg")


def test_a_symlink_or_odd_extension_is_not_trusted(world, tmp_path):
    odd = world.cand_dir("d1") / "c101_weird.txt"
    odd.write_bytes(b"not an image")
    con = world.conn()
    with con:
        con.execute("UPDATE hazards SET photo_path = ? WHERE id = 1", (str(odd),))
    con.close()
    out = tmp_path / "o"
    try:
        world.build(out)
    except Exception:
        return
    assert "c101_weird.txt" not in {Path(f).name for f in all_files(out)}


def test_png_photos_are_allowed(world, tmp_path):
    png = world.cand_dir("d2") / "c201_best_blur.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nBLURRED-PNG")
    con = world.conn()
    with con:
        con.execute("UPDATE hazards SET photo_path = ? WHERE id = 2", (str(png),))
    con.close()
    out = tmp_path / "o"
    world.build(out)
    h = rows(out / "demo.sqlite", "SELECT * FROM hazards WHERE id = 2")[0]
    assert h["photo_path"] == rel("demo", "d2", "c201_best_blur.png")
    assert (out / "storage" / "d2" / "candidates" / "c201_best_blur.png").is_file()
