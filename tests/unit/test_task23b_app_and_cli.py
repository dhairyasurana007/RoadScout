"""Task 23b: the built folder is served correctly by the read-only demo app; the command line."""
from __future__ import annotations

import importlib
import os
import re
import sqlite3
from pathlib import Path

import pytest

from rs23b_helpers import ROOT, World, all_files, rows, run_cli, tree_hash

DEMO_BODY = {"detail": "This is a read-only demo."}


@pytest.fixture()
def world(tmp_path):
    return World(tmp_path)


@pytest.fixture()
def demo_app(world, tmp_path, monkeypatch):
    """Build into <tmp>/demo and serve it the way the Docker image does (relative paths resolve from the cwd)."""
    out = tmp_path / "demo"
    world.build(out)
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

    yield world, out, TestClient(main.app)
    os.environ.pop("ROADSCOUT_DEMO", None)
    importlib.reload(main)


def items(body):
    return body if isinstance(body, list) else body.get("hazards", body.get("items", []))


# ------------------------------------------------------------------ served by the app

def test_hazard_list_has_exactly_the_kept_hazards(demo_app):
    world, out, c = demo_app
    r = c.get("/api/hazards")
    assert r.status_code == 200
    got = items(r.json())
    assert {h["id"] for h in got} == world.kept
    for h in got:
        assert h.get("photo_url"), "the photo of every demo hazard must be servable"


def test_photos_are_served_as_the_blurred_bytes(demo_app):
    world, out, c = demo_app
    for h in items(c.get("/api/hazards").json()):
        r = c.get(h["photo_url"])
        assert r.status_code == 200 and r.headers["content-type"].startswith("image/jpeg")
        assert b"ORIGINAL" not in r.content
        assert b"BLURRED-BEST" in r.content
    crop = c.get("/api/hazards/1/photo?variant=crop")
    assert crop.status_code == 200 and b"BLURRED-CROP-101" in crop.content


def test_the_other_hazards_are_not_reachable(demo_app):
    world, out, c = demo_app
    for hid in (4, 5, 6, 7, 8, 9, 10, 987):
        assert c.get(f"/api/hazards/{hid}/photo").status_code == 404, hid


def test_report_for_an_approved_hazard(demo_app):
    world, out, c = demo_app
    r = c.get("/api/hazards/1/report")
    assert r.status_code == 200
    assert "pothole" in r.text.lower()
    txt = c.get("/api/hazards/1/report.txt")
    assert txt.status_code == 200


def test_candidate_photo_route_serves_only_the_blurred_copy(demo_app):
    world, out, c = demo_app
    r = c.get("/api/drives/d1/hazards/101/photo")
    assert r.status_code == 200 and b"BLURRED-BEST-101" in r.content


def test_drives_say_the_raw_video_is_gone(demo_app):
    world, out, c = demo_app
    ds = c.get("/api/drives").json()
    ds = ds if isinstance(ds, list) else ds.get("drives", ds.get("items", []))
    assert {d["id"] for d in ds} == world.kept_drives
    for d in ds:
        flags = {k: v for k, v in d.items() if "delet" in k.lower()}
        assert flags and any(bool(v) for v in flags.values()), d
        assert not d.get("error") and not d.get("warning")


def test_mutations_are_forbidden(demo_app):
    world, out, c = demo_app
    r = c.patch("/api/hazards/1", json={"status": "rejected"})
    assert r.status_code == 403 and r.json() == DEMO_BODY
    assert c.post("/api/drives/d1/process").status_code == 403
    assert c.get("/api/demo").json() == {"demo": True}


def test_the_app_never_wrote_to_the_demo_database(demo_app):
    world, out, c = demo_app
    before = (out / "demo.sqlite").read_bytes()
    c.get("/api/hazards")
    c.get("/api/hazards/1/report")
    c.patch("/api/hazards/1", json={"status": "rejected"})
    assert (out / "demo.sqlite").read_bytes() == before


# ------------------------------------------------------------------ command line

def args(world, out, *extra):
    return ["--db", world.db, "--storage", world.storage, "--out", out, *extra]


def test_cli_builds_and_prints_a_summary(world, tmp_path):
    out = tmp_path / "o"
    cp = run_cli(args(world, out))
    assert cp.returncode == 0, cp.stderr
    assert cp.stdout.strip()
    assert re.search(r"\b2\b", cp.stdout), "the summary shows the hazards kept"
    assert "Traceback" not in cp.stdout + cp.stderr
    assert {h["id"] for h in rows(out / "demo.sqlite", "SELECT id FROM hazards")} == world.kept
    assert len([f for f in all_files(out) if f.startswith("storage/")]) == 3


def test_cli_summary_mentions_skipped_hazards(world, tmp_path):
    cp = run_cli(args(world, tmp_path / "o"))
    low = cp.stdout.lower()
    assert "skip" in low


def test_cli_does_not_modify_the_source(world, tmp_path):
    before_db = world.fingerprint()
    before_tree = tree_hash(world.storage)
    cp = run_cli(args(world, tmp_path / "o"))
    assert cp.returncode == 0, cp.stderr
    assert world.fingerprint() == before_db and tree_hash(world.storage) == before_tree


def test_cli_reads_the_default_db_and_storage_from_the_environment(world, tmp_path):
    out = tmp_path / "o"
    cp = run_cli(["--out", out], env_over={"ROADSCOUT_DB": str(world.db), "ROADSCOUT_STORAGE_DIR": str(world.storage)})
    assert cp.returncode == 0, cp.stderr
    assert {h["id"] for h in rows(out / "demo.sqlite", "SELECT id FROM hazards")} == world.kept


def test_cli_drive_option(world, tmp_path):
    out = tmp_path / "o"
    cp = run_cli(args(world, out, "--drive", "d1"))
    assert cp.returncode == 0, cp.stderr
    assert {h["id"] for h in rows(out / "demo.sqlite", "SELECT id FROM hazards")} == {1}


def test_cli_drive_option_repeated(world, tmp_path):
    out = tmp_path / "o"
    cp = run_cli(args(world, out, "--drive", "d1", "--drive", "d2"))
    assert cp.returncode == 0, cp.stderr
    assert {h["id"] for h in rows(out / "demo.sqlite", "SELECT id FROM hazards")} == world.kept


def test_cli_refuses_a_non_empty_out_dir_without_force(world, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "keep.txt").write_text("mine", encoding="utf-8")
    cp = run_cli(args(world, out))
    assert cp.returncode in (1, 2)
    assert cp.stderr.strip() and "Traceback" not in cp.stderr
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]


def test_cli_force_replaces_the_dataset_and_keeps_other_files(world, tmp_path):
    out = tmp_path / "o"
    assert run_cli(args(world, out)).returncode == 0
    (out / "README.txt").write_text("mine", encoding="utf-8")
    cp = run_cli(args(world, out, "--force"))
    assert cp.returncode == 0, cp.stderr
    assert (out / "README.txt").read_text(encoding="utf-8") == "mine"
    assert (out / "demo.sqlite").is_file()


def test_cli_privacy_scan_failure_exits_1_and_leaves_nothing(world, tmp_path):
    con = world.conn()
    with con:
        con.execute("UPDATE verdicts SET reason = ? WHERE candidate_id = 101", ("see C:\\Users\\me\\clip",))
    con.close()
    out = tmp_path / "o"
    cp = run_cli(args(world, out))
    assert cp.returncode == 1
    assert cp.stderr.strip() and "Traceback" not in cp.stderr
    assert not out.exists() or not any(out.iterdir())


def test_cli_missing_database_fails_and_leaves_nothing(world, tmp_path):
    out = tmp_path / "o"
    cp = run_cli(["--db", tmp_path / "nope.sqlite", "--storage", world.storage, "--out", out])
    assert cp.returncode in (1, 2)
    assert cp.stderr.strip() and "Traceback" not in cp.stderr
    assert not out.exists() or not any(out.iterdir())


def test_cli_bad_arguments_exit_2(world, tmp_path):
    assert run_cli(["--no-such-option"]).returncode == 2
    assert run_cli(["--db"]).returncode == 2
    assert run_cli(["--out"]).returncode == 2


def test_cli_help_lists_the_options():
    cp = run_cli(["--help"])
    assert cp.returncode == 0
    for opt in ("--db", "--storage", "--out", "--drive", "--force"):
        assert opt in cp.stdout


def test_cli_needs_no_key_and_no_provider(world, tmp_path):
    cp = run_cli(args(world, tmp_path / "o"), env_over={"META_API_KEY": "", "ROADSCOUT_PROVIDER": "live"})
    assert cp.returncode == 0, cp.stderr
    assert "META_API_KEY" not in cp.stdout + cp.stderr


def test_cli_output_is_deterministic(world, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert run_cli(args(world, a)).returncode == 0
    assert run_cli(args(world, b)).returncode == 0
    assert tree_hash(a / "storage") == tree_hash(b / "storage")
    from rs23b_helpers import dump

    assert dump(a / "demo.sqlite") == dump(b / "demo.sqlite")
