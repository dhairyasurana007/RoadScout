"""Task 8: one candidate per tracked object, track fields, short-track filtering, idempotence."""
from __future__ import annotations

import pytest

from rs8_helpers import (  # noqa: F401  (fixtures come from the helpers module)
    FPC, MIN_ENV, W, H, load_module, make_spec, rs8_cache, rs8_env, ts_of,
)
from backend.providers import rle_area, rle_from_box

BOX = (60, 40, 100, 70)


def keyed(cands):
    return {(c["cls"], c["track_key"]): c for c in cands}


def test_one_candidate_per_track_with_fields(rs8_env):
    e = rs8_env
    e.use_spec(make_spec())
    # track 1: frames 1,2,3,6 (gap) with growing masks; track 2 in the same chunk
    sizes = {1: (60, 40, 90, 60), 2: (60, 40, 100, 70), 3: (55, 40, 105, 70), 6: (60, 40, 95, 65)}
    for f, b in sizes.items():
        e.add_det(0, f, 1, b)
    e.add_track(0, [0, 1, 2, 3], 2, (10, 10, 30, 30), hazard_class="debris in lane")
    e.run()
    cs = e.cands()
    assert len(cs) == 2
    a = next(c for c in cs if c["cls"] == "pothole")
    b = next(c for c in cs if c["cls"] == "debris in lane")
    assert a["frames"] == 4
    assert a["first_ts"] == pytest.approx(ts_of(0, 1), abs=1e-6)
    assert a["last_ts"] == pytest.approx(ts_of(0, 6), abs=1e-6)
    assert b["frames"] == 4
    assert b["first_ts"] == pytest.approx(ts_of(0, 0), abs=1e-6)
    assert b["last_ts"] == pytest.approx(ts_of(0, 3), abs=1e-6)
    assert a["track_key"] == "0:1" and b["track_key"] == "0:2"
    assert a["hazard_id"] is None and b["hazard_id"] is None
    assert all(c["raw"]["drive_id"] == "drive-a" for c in cs)
    if a["mask_area"] is not None:
        biggest = max(rle_area(rle_from_box(W, H, bx)) for bx in sizes.values())
        assert a["mask_area"] == biggest
    else:
        pytest.fail("candidate row has no column holding the largest mask area (name containing 'mask' and 'area')")


def test_largest_mask_area_is_the_max_over_frames(rs8_env):
    e = rs8_env
    e.use_spec(make_spec())
    widths = [20, 50, 30, 40]
    for i, w in enumerate(widths):
        e.add_det(0, i, 1, (10, 10, 10 + w, 40))
    e.run()
    (c,) = e.cands()
    assert c["mask_area"] == rle_area(rle_from_box(W, H, (10, 10, 60, 40)))


def test_null_masks_do_not_crash_and_count_as_frames(rs8_env):
    e = rs8_env
    e.use_spec(make_spec())
    e.add_det(0, 0, 1, BOX, mask=None)
    e.add_det(0, 1, 1, BOX)
    e.add_det(0, 2, 1, BOX, mask=None)
    e.add_det(0, 3, 1, BOX, mask=None)
    e.run()
    (c,) = e.cands()
    assert c["frames"] == 4
    assert c["mask_area"] == rle_area(rle_from_box(W, H, BOX))


def test_same_track_id_in_different_chunks_are_different_objects(rs8_env):
    e = rs8_env
    e.use_spec(make_spec())
    for ch in range(3):
        e.add_track(ch, range(4), 1, BOX)
    e.run()
    cs = e.cands()
    assert len(cs) == 3
    assert sorted(c["track_key"] for c in cs) == ["0:1", "1:1", "2:1"]
    assert sorted(round(c["first_ts"], 4) for c in cs) == [round(ts_of(ch, 0), 4) for ch in range(3)]
    assert len({c["best_path"] for c in cs}) == 3, "each candidate needs its own best-frame file"
    assert len({c["crop_path"] for c in cs}) == 3, "each candidate needs its own crop file"


def test_same_track_key_with_two_classes_gives_two_candidates(rs8_env):
    e = rs8_env
    e.use_spec(make_spec())
    e.add_track(0, range(4), 1, BOX, hazard_class="pothole")
    e.add_track(0, range(4), 1, (10, 10, 40, 30), hazard_class="debris in lane")
    e.run()
    cs = e.cands()
    assert len(cs) == 2
    assert {c["cls"] for c in cs} == {"pothole", "debris in lane"}
    assert len({c["best_path"] for c in cs}) == 2
    assert len({c["crop_path"] for c in cs}) == 2


# ----------------------------------------------------------------------------- minimum track length


def lengths_setup(e, lengths=(1, 2, 3, 4)):
    e.use_spec(make_spec(n_chunks=1))
    for tid, n in enumerate(lengths, start=1):
        e.add_track(0, range(n), tid, (10 * tid, 10, 10 * tid + 8, 30))


def kept_lengths(e):
    return sorted(c["frames"] for c in e.cands())


def test_min_frames_param_drops_short_and_keeps_exact_minimum(rs8_env):
    e = rs8_env
    lengths_setup(e)
    e.run(min_frames=3)
    assert kept_lengths(e) == [3, 4]


@pytest.mark.parametrize("minimum,expected", [(1, [1, 2, 3, 4]), (2, [2, 3, 4]), (4, [4]), (5, [])])
def test_min_frames_boundaries(rs8_env, minimum, expected):
    e = rs8_env
    lengths_setup(e)
    e.run(min_frames=minimum)
    assert kept_lengths(e) == expected


def test_min_frames_env_var(rs8_env):
    e = rs8_env
    lengths_setup(e)
    e.mp.setenv(MIN_ENV, "4")
    e.run()
    assert kept_lengths(e) == [4]
    e.mp.setenv(MIN_ENV, "2")
    e.run()
    assert kept_lengths(e) == [2, 3, 4]


def test_min_frames_param_overrides_env(rs8_env):
    e = rs8_env
    lengths_setup(e)
    e.mp.setenv(MIN_ENV, "4")
    e.run(min_frames=1)
    assert kept_lengths(e) == [1, 2, 3, 4]


def test_default_minimum_drops_two_frame_track_and_keeps_eight(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    e.add_track(0, range(2), 1, (10, 10, 20, 30))
    e.add_track(0, range(8), 2, (40, 10, 50, 30))
    e.run()
    assert kept_lengths(e) == [8]


def test_frames_seen_counts_frames_not_gaps(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    for f in (0, 2, 4, 7):
        e.add_det(0, f, 1, BOX)
    e.run(min_frames=4)
    (c,) = e.cands()
    assert c["frames"] == 4
    assert c["first_ts"] == pytest.approx(ts_of(0, 0), abs=1e-6)
    assert c["last_ts"] == pytest.approx(ts_of(0, 7), abs=1e-6)


# ----------------------------------------------------------------------------- empty and idempotent


def test_drive_without_detections_gives_no_candidates(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    e.run()
    assert e.cands() == []


def test_all_tracks_too_short_gives_no_candidates(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    e.add_track(0, range(2), 1, BOX)
    e.add_det(0, 5, 2, BOX)
    e.run(min_frames=3)
    assert e.cands() == []


def test_rerun_replaces_the_drives_candidates(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=2))
    e.add_track(0, range(4), 1, BOX)
    e.add_track(1, range(5), 1, BOX)
    e.run()
    first = e.cands()
    assert len(first) == 2
    e.run()
    second = e.cands()
    assert len(second) == 2, "re-running must not duplicate candidates"
    strip = lambda cs: sorted((c["track_key"], c["cls"], c["frames"], round(c["first_ts"], 6),
                                round(c["last_ts"], 6)) for c in cs)
    assert strip(first) == strip(second)


def test_rerun_reflects_changed_detections(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=2))
    e.add_track(0, range(4), 1, BOX)
    e.add_track(1, range(5), 1, BOX)
    e.run()
    assert len(e.cands()) == 2
    e.delete_dets("chunk = 1")
    e.run()
    cs = e.cands()
    assert [c["track_key"] for c in cs] == ["0:1"]
    e.delete_dets()
    e.run()
    assert e.cands() == []


def test_rerun_leaves_other_drives_alone(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    e.add_drive("drive-b")
    e.add_track(0, range(4), 1, BOX, drive="drive-a")
    e.add_track(0, range(5), 1, BOX, drive="drive-b")
    e.run("drive-b")
    before = [(c["track_key"], c["frames"], c["id"]) for c in e.cands("drive-b")]
    assert len(before) == 1
    e.run("drive-a")
    e.run("drive-a")
    assert [(c["track_key"], c["frames"], c["id"]) for c in e.cands("drive-b")] == before
    assert len(e.cands("drive-a")) == 1


def test_chunks_argument_forms_are_equivalent(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    e.add_track(0, range(4), 1, BOX)
    e.run(chunks=e.work)  # folder
    a = [(c["track_key"], c["frames"]) for c in e.cands()]
    e.run(chunks=e.work / "frame_timestamps.json")  # json file
    b = [(c["track_key"], c["frames"]) for c in e.cands()]
    assert a == b == [("0:1", 4)]


def test_load_candidates_matches_table(rs8_env):
    e = rs8_env
    e.use_spec(make_spec(n_chunks=1))
    e.add_track(0, range(4), 1, BOX)
    e.add_track(0, range(5), 2, (10, 10, 30, 30))
    e.run()
    mod = load_module()
    assert callable(getattr(mod, "load_candidates", None)), "backend.candidates.load_candidates is missing"
    rows = mod.load_candidates("drive-a", db_path=e.db)
    assert len(rows) == 2
    assert mod.load_candidates("nope", db_path=e.db) == []
