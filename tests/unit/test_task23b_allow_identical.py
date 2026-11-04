"""Task 23b follow-up: allow_identical, the human-reviewed override for a 'blurred' file that equals its original."""
from __future__ import annotations

import importlib
import inspect
import os
import re
from pathlib import Path

import pytest

from rs23b_helpers import (DRIVE_LETTER, IMG_EXT, PATH_COLS, World, all_files, all_text_values, rows, run_cli,
                           tree_hash)


@pytest.fixture()
def world(tmp_path):
    return World(tmp_path)


def rel(prefix, drive, name):
    return f"{prefix}/storage/{drive}/candidates/{name}"


def make_identical(world, cid=201, crop=False):
    """The published 'blurred' file has exactly the bytes of the original (nothing in the frame needed blurring)."""
    key = "crop" if crop else "best"
    published = world.files[cid]["blurred_crop" if crop else "blurred"]
    published.write_bytes(world.files[cid][key].read_bytes())
    return published


def pairs(items):
    return [(int(h), str(n)) for h, n in items]


def items(body):
    return body if isinstance(body, list) else body.get("hazards", body.get("items", []))


def args(world, out, *extra):
    return ["--db", world.db, "--storage", world.storage, "--out", out, *extra]


# ------------------------------------------------------------------ the library

def test_allow_identical_is_off_by_default():
    from backend.demo_dataset import build_demo_dataset

    assert inspect.signature(build_demo_dataset).parameters["allow_identical"].default is False


def test_default_report_has_an_empty_identical_copies_list(world, tmp_path):
    r = world.build(tmp_path / "o")
    assert r.identical_copies == []


def test_default_is_strict_a_byte_identical_file_fails_the_build(world, tmp_path):
    make_identical(world)
    out = tmp_path / "o"
    with pytest.raises(Exception):
        world.build(out)
    assert not out.exists() or not any(out.iterdir()), "nothing is left behind"


def test_default_is_strict_for_an_identical_crop_too(world, tmp_path):
    make_identical(world, 101, crop=True)
    out = tmp_path / "o"
    with pytest.raises(Exception):
        world.build(out)
    assert not out.exists() or not any(out.iterdir())


def test_allow_identical_builds_and_lists_the_copy(world, tmp_path):
    make_identical(world)
    out = tmp_path / "o"
    r = world.build(out, allow_identical=True)
    assert pairs(r.identical_copies) == [(2, "c201_best_blur.jpg")]
    assert r.hazards_kept == 2 and r.files_copied == 3 and r.candidates_kept == 3 and r.drives_kept == 2
    assert all_files(out) == sorted([
        "demo.sqlite", "storage/d1/candidates/c101_best_blur.jpg", "storage/d1/candidates/c101_crop_blur.jpg",
        "storage/d2/candidates/c201_best_blur.jpg"])
    assert (out / "storage" / "d2" / "candidates" / "c201_best_blur.jpg").read_bytes() == \
        world.files[201]["best"].read_bytes()


def test_allow_identical_publishes_at_the_blurred_path_never_the_originals(world, tmp_path):
    make_identical(world)
    out = tmp_path / "o"
    world.build(out, allow_identical=True)
    assert "c201_best.jpg" not in {Path(f).name for f in all_files(out)}
    h = rows(out / "demo.sqlite", "SELECT * FROM hazards WHERE id = 2")[0]
    assert h["photo_path"] == rel("demo", "d2", "c201_best_blur.jpg")
    c = rows(out / "demo.sqlite", "SELECT * FROM candidates WHERE id = 201")[0]
    assert c["blurred_frame_path"] == rel("demo", "d2", "c201_best_blur.jpg")
    assert c["best_frame_path"] != str(world.files[201]["best"])


def test_allow_identical_lists_every_identical_file_of_a_hazard(world, tmp_path):
    make_identical(world, 101)
    make_identical(world, 101, crop=True)
    r = world.build(tmp_path / "o", allow_identical=True)
    assert sorted(pairs(r.identical_copies)) == [(1, "c101_best_blur.jpg"), (1, "c101_crop_blur.jpg")]


def test_allow_identical_changes_nothing_else_in_the_output(world, tmp_path):
    normal = tmp_path / "normal"
    world.build(normal)
    make_identical(world)
    ident = tmp_path / "ident"
    world.build(ident, allow_identical=True)
    for t in ("hazards", "candidates", "verdicts", "app_meta"):
        assert rows(ident / "demo.sqlite", f"SELECT * FROM {t} ORDER BY rowid") == \
            rows(normal / "demo.sqlite", f"SELECT * FROM {t} ORDER BY rowid"), t
    a = rows(ident / "demo.sqlite", "SELECT * FROM drives ORDER BY id")
    b = rows(normal / "demo.sqlite", "SELECT * FROM drives ORDER BY id")
    for x, y in zip(a, b):
        x.pop("video_deleted_at")
        y.pop("video_deleted_at")
        assert x == y
    for t in ("detections", "detection_chunks", "pipeline_runs", "exports"):
        assert rows(ident / "demo.sqlite", f"SELECT COUNT(*) AS n FROM {t}")[0]["n"] == 0


def test_allow_identical_output_passes_the_same_privacy_checks(world, tmp_path):
    make_identical(world)
    out = tmp_path / "o"
    world.build(out, allow_identical=True)
    needles = {str(world.storage), str(world.storage).replace("\\", "/"), str(world.db), str(tmp_path)}
    for table, col, v in all_text_values(out / "demo.sqlite"):
        for n in needles:
            assert n not in v
        assert not DRIVE_LETTER.search(v) and "/Users/" not in v and "\\Users\\" not in v
    refs = {v for t, c, v in all_text_values(out / "demo.sqlite") if c in PATH_COLS or c == "photo_path"}
    for f in all_files(out):
        if f.lower().endswith(IMG_EXT):
            assert f"demo/{f}" in refs
    names = {Path(f).name for f in all_files(out)}
    for bad in ("video.mov", "track.gpx", "chunk_0000.mp4", "export_1.mp4", "stray_unreferenced.jpg", "c201_best.jpg"):
        assert bad not in names


def test_allow_identical_with_nothing_identical_lists_nothing(world, tmp_path):
    r = world.build(tmp_path / "o", allow_identical=True)
    assert r.identical_copies == []
    assert r.hazards_kept == 2 and r.files_copied == 3


def test_allow_identical_does_not_unlock_a_photo_path_that_is_an_original(world, tmp_path):
    con = world.conn()
    with con:
        con.execute("UPDATE hazards SET photo_path = ? WHERE id = 1", (str(world.files[101]["best"]),))
    con.close()
    out = tmp_path / "o"
    r = world.build(out, allow_identical=True)
    assert 1 not in {h["id"] for h in rows(out / "demo.sqlite", "SELECT id FROM hazards")}
    assert 1 in {int(h) for h, _ in r.hazards_skipped}
    assert "c101_best.jpg" not in {Path(f).name for f in all_files(out)}


def test_allow_identical_does_not_unlock_the_sneaky_original_spelling(world, tmp_path):
    original = world.files[101]["best"]
    sneaky = world.storage / "d1" / "candidates" / ".." / "candidates" / original.name
    con = world.conn()
    with con:
        con.execute("UPDATE hazards SET photo_path = ? WHERE id = 1", (str(sneaky),))
    con.close()
    out = tmp_path / "o"
    r = world.build(out, allow_identical=True)
    assert 1 in {int(h) for h, _ in r.hazards_skipped}
    assert original.name not in {Path(f).name for f in all_files(out)}


@pytest.mark.parametrize("leak", ["saved at C:\\Users\\me\\x.jpg", "see /Users/me/clip"])
def test_allow_identical_does_not_unlock_a_local_path_in_text(world, tmp_path, leak):
    make_identical(world)
    con = world.conn()
    with con:
        con.execute("UPDATE verdicts SET report_description = ? WHERE candidate_id = 201", (leak,))
    con.close()
    out = tmp_path / "o"
    with pytest.raises(Exception):
        world.build(out, allow_identical=True)
    assert not out.exists() or not any(out.iterdir())


def test_allow_identical_does_not_unlock_a_path_in_a_hazard_column(world, tmp_path):
    con = world.conn()
    with con:
        con.execute("UPDATE hazards SET first_seen = ? WHERE id = 1", (str(world.db),))
    con.close()
    out = tmp_path / "o"
    with pytest.raises(Exception):
        world.build(out, allow_identical=True)
    assert not out.exists() or not any(out.iterdir())


def test_allow_identical_still_refuses_a_non_empty_output_folder(world, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "keep.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(Exception):
        world.build(out, allow_identical=True)
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]


def test_allow_identical_with_overwrite(world, tmp_path):
    out = tmp_path / "o"
    world.build(out)
    make_identical(world)
    r = world.build(out, allow_identical=True, overwrite=True)
    assert pairs(r.identical_copies) == [(2, "c201_best_blur.jpg")]


def test_allow_identical_is_reproducible(world, tmp_path):
    make_identical(world)
    r1 = world.build(tmp_path / "a", allow_identical=True)
    r2 = world.build(tmp_path / "b", allow_identical=True)
    assert pairs(r1.identical_copies) == pairs(r2.identical_copies)
    assert tree_hash(tmp_path / "a" / "storage") == tree_hash(tmp_path / "b" / "storage")


def test_allow_identical_and_skip_identical_together(world, tmp_path):
    make_identical(world)
    out = tmp_path / "o"
    try:
        r = world.build(out, allow_identical=True, skip_identical=True)
    except ValueError:
        assert not out.exists() or not any(out.iterdir())
        return
    published = (out / "storage" / "d2" / "candidates" / "c201_best_blur.jpg").exists()
    listed = 2 in {h for h, _ in pairs(r.identical_copies)}
    skipped = 2 in {int(h) for h, _ in r.hazards_skipped}
    assert (published and listed and not skipped) or (not published and skipped and not listed)


def test_source_is_untouched_by_an_allow_identical_build(world, tmp_path):
    make_identical(world)
    before_db, before_tree = world.fingerprint(), tree_hash(world.storage)
    world.build(tmp_path / "o", allow_identical=True)
    assert world.fingerprint() == before_db and tree_hash(world.storage) == before_tree


# ------------------------------------------------------------------ served by the demo app

@pytest.fixture()
def identical_app(world, tmp_path, monkeypatch):
    make_identical(world)
    out = tmp_path / "demo"
    report = world.build(out, allow_identical=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROADSCOUT_DEMO", "1")
    monkeypatch.setenv("ROADSCOUT_DB", str(out / "demo.sqlite"))
    monkeypatch.setenv("ROADSCOUT_STORAGE_DIR", str(out / "storage"))
    monkeypatch.setenv("ROADSCOUT_COST_LOG", "off")
    monkeypatch.setenv("ROADSCOUT_PROVIDER", "fake")
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("ROADSCOUT_STATIC_DIR", raising=False)
    import backend.main as main

    main = importlib.reload(main)
    from fastapi.testclient import TestClient

    yield world, out, TestClient(main.app), report
    os.environ.pop("ROADSCOUT_DEMO", None)
    importlib.reload(main)


def test_the_identical_copy_is_served_by_the_demo_app(identical_app):
    world, out, c, report = identical_app
    got = {h["id"]: h for h in items(c.get("/api/hazards").json())}
    assert set(got) == world.kept
    r = c.get(got[2]["photo_url"])
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/jpeg")
    assert r.content == world.files[201]["best"].read_bytes()
    assert c.get(got[1]["photo_url"]).status_code == 200
    assert c.patch("/api/hazards/2", json={"status": "rejected"}).status_code == 403


def test_the_candidate_route_serves_the_identical_copy_from_its_blurred_column(identical_app):
    world, out, c, report = identical_app
    assert c.get("/api/drives/d2/hazards/201/photo").status_code == 200


# ------------------------------------------------------------------ the command line

def test_cli_allow_identical_builds_and_mentions_the_unchanged_image(world, tmp_path):
    make_identical(world)
    out = tmp_path / "o"
    cp = run_cli(args(world, out, "--allow-identical"))
    assert cp.returncode == 0, cp.stderr
    low = cp.stdout.lower()
    assert "identical" in low or "unchanged" in low
    assert "c201_best_blur.jpg" in cp.stdout or re.search(r"\b2\b", cp.stdout)
    assert (out / "storage" / "d2" / "candidates" / "c201_best_blur.jpg").is_file()
    assert "Traceback" not in cp.stdout + cp.stderr


def test_cli_without_the_flag_fails_on_an_identical_file(world, tmp_path):
    make_identical(world)
    out = tmp_path / "o"
    cp = run_cli(args(world, out))
    assert cp.returncode == 1
    assert cp.stderr.strip() and "Traceback" not in cp.stderr
    assert not out.exists() or not any(out.iterdir())


def test_cli_allow_identical_on_a_normal_set_works_and_lists_no_copy(world, tmp_path):
    out = tmp_path / "o"
    cp = run_cli(args(world, out, "--allow-identical"))
    assert cp.returncode == 0, cp.stderr
    assert len([f for f in all_files(out) if f.startswith("storage/")]) == 3
    assert "c201_best_blur.jpg" not in cp.stdout


def test_cli_help_mentions_allow_identical():
    cp = run_cli(["--help"])
    assert cp.returncode == 0 and "--allow-identical" in cp.stdout


def test_cli_both_override_flags_do_not_crash(world, tmp_path):
    make_identical(world)
    cp = run_cli(args(world, tmp_path / "o", "--allow-identical", "--skip-identical"))
    assert cp.returncode in (0, 1, 2)
    assert "Traceback" not in cp.stdout + cp.stderr


def test_cli_allow_identical_still_fails_on_a_path_leak(world, tmp_path):
    make_identical(world)
    con = world.conn()
    with con:
        con.execute("UPDATE verdicts SET reason = ? WHERE candidate_id = 101", ("see C:\\Users\\me\\clip",))
    con.close()
    out = tmp_path / "o"
    cp = run_cli(args(world, out, "--allow-identical"))
    assert cp.returncode == 1
    assert not out.exists() or not any(out.iterdir())
