"""Task 14: idempotency, re-runs after candidates change, status preservation, determinism."""
import random
from datetime import timedelta

import pytest

from dd14_helpers import (T_A, T_B, T_C, VA, add_cand, add_drive, cand_rows, dedup, dedup_all, drop_drive_candidates,
                          execute, hazard_of, hazards, make_db, offset, parse_time, partition, rows, same_hazard)


@pytest.fixture
def db(tmp_path, monkeypatch):
    d = make_db(tmp_path, monkeypatch)
    add_drive(d, "A", T_A)
    add_drive(d, "B", T_B)
    add_drive(d, "C", T_C)
    return d


def snapshot(db):
    return ([tuple(sorted(h.items())) for h in hazards(db)],
            [(c["id"], c["hazard_id"]) for c in cand_rows(db)])


def assert_no_orphans(db):
    for h in hazards(db):
        n = rows(db, "SELECT COUNT(*) AS n FROM candidates WHERE hazard_id = ?", (h["id"],))[0]["n"]
        assert n >= 1, "hazard %s has no candidates" % h["id"]


# ------------------------------------------------------------------ idempotency

def test_rerun_same_drive_is_idempotent(db):
    add_cand(db, "A", VA[0], VA[1], 0.0)
    lat, lon = offset(VA[0], VA[1], 7.0)
    add_cand(db, "A", lat, lon, 0.0, best_ts=30.0)
    dedup(db, "A")
    first = snapshot(db)
    dedup(db, "A")
    dedup(db, "A")
    assert snapshot(db) == first
    assert len(hazards(db)) == 1


def test_rerun_all_drives_is_idempotent(db):
    far = offset(VA[0], VA[1], 300.0)
    add_cand(db, "A", VA[0], VA[1], 0.0)
    add_cand(db, "B", VA[0], VA[1], 0.0)
    add_cand(db, "B", far[0], far[1], 0.0, best_ts=40.0)
    add_cand(db, "C", far[0], far[1], 0.0)
    dedup_all(db, ["A", "B", "C"])
    first = snapshot(db)
    assert len(hazards(db)) == 2
    dedup_all(db, ["A", "B", "C"])
    dedup_all(db, ["C", "B", "A"])
    assert snapshot(db) == first


def test_incremental_second_drive_joins_existing_hazard(db):
    a = add_cand(db, "A", VA[0], VA[1], 0.0)
    dedup(db, "A")
    (h1,) = hazards(db)
    b = add_cand(db, "B", VA[0], VA[1], 0.0)
    dedup(db, "B")
    (h2,) = hazards(db)
    assert same_hazard(db, a, b)
    assert h2["sightings"] == 2
    assert h2["id"] == h1["id"], "the existing hazard should be updated, not replaced"


def test_hazard_ids_are_stable_across_reruns(db):
    add_cand(db, "A", VA[0], VA[1], 0.0)
    add_cand(db, "B", VA[0], VA[1], 0.0)
    dedup_all(db, ["A", "B"])
    ids = [h["id"] for h in hazards(db)]
    dedup_all(db, ["A", "B"])
    assert [h["id"] for h in hazards(db)] == ids


# ------------------------------------------------------------------ status preservation

@pytest.mark.parametrize("status", ["approved", "rejected", "reported", "possibly_fixed"])
def test_user_status_survives_a_new_sighting(db, status):
    add_cand(db, "A", VA[0], VA[1], 0.0)
    dedup(db, "A")
    execute(db, "UPDATE hazards SET status = ?", (status,))
    add_cand(db, "B", VA[0], VA[1], 0.0)
    dedup(db, "B")
    (h,) = hazards(db)
    assert h["status"] == status
    assert h["sightings"] == 2


@pytest.mark.parametrize("status", ["approved", "rejected", "reported"])
def test_user_status_survives_plain_reruns(db, status):
    add_cand(db, "A", VA[0], VA[1], 0.0)
    add_cand(db, "B", VA[0], VA[1], 0.0)
    dedup_all(db, ["A", "B"])
    execute(db, "UPDATE hazards SET status = ?", (status,))
    dedup_all(db, ["A", "B"])
    dedup_all(db, ["B", "A"])
    (h,) = hazards(db)
    assert h["status"] == status


def test_status_of_unrelated_hazards_is_untouched(db):
    far = offset(VA[0], VA[1], 800.0)
    add_cand(db, "A", VA[0], VA[1], 0.0)
    add_cand(db, "A", far[0], far[1], 0.0, best_ts=50.0)
    dedup(db, "A")
    first = min(hazards(db), key=lambda h: h["lat"])
    execute(db, "UPDATE hazards SET status = 'reported' WHERE id = ?", (first["id"],))
    add_cand(db, "B", far[0], far[1], 0.0)
    dedup(db, "B")
    by_id = {h["id"]: h for h in hazards(db)}
    assert by_id[first["id"]]["status"] == "reported"
    assert len(by_id) == 2
    assert sorted(h["status"] for h in by_id.values()) == ["new", "reported"]


# ------------------------------------------------------------------ candidates replaced / changed

def test_replacing_a_drives_candidates_removes_orphan_hazard(db):
    add_cand(db, "A", VA[0], VA[1], 0.0)
    dedup(db, "A")
    assert len(hazards(db)) == 1
    drop_drive_candidates(db, "A")          # what re-running aggregation does
    dedup(db, "A")
    assert hazards(db) == []


def test_replacing_with_candidates_elsewhere_moves_the_hazard(db):
    add_cand(db, "A", VA[0], VA[1], 0.0)
    dedup(db, "A")
    execute(db, "UPDATE hazards SET status = 'approved'")
    drop_drive_candidates(db, "A")
    lat, lon = offset(VA[0], VA[1], 600.0)
    c = add_cand(db, "A", lat, lon, 0.0)
    dedup(db, "A")
    hs = hazards(db)
    assert len(hs) == 1
    assert abs(hs[0]["lat"] - lat) < 1e-6
    assert hazard_of(db, c) == hs[0]["id"]
    assert_no_orphans(db)


def test_replacing_with_same_place_does_not_duplicate(db):
    add_cand(db, "A", VA[0], VA[1], 0.0)
    dedup(db, "A")
    drop_drive_candidates(db, "A")
    c = add_cand(db, "A", VA[0], VA[1], 0.0)
    dedup(db, "A")
    hs = hazards(db)
    assert len(hs) == 1
    assert hs[0]["sightings"] == 1
    assert hazard_of(db, c) == hs[0]["id"]


def test_dropping_one_drive_keeps_hazard_seen_by_another(db):
    pb = str(db.parent / "frame_b.jpg")
    add_cand(db, "A", VA[0], VA[1], 0.0, score=0.9, conf=0.9, severity=3)
    b = add_cand(db, "B", VA[0], VA[1], 0.0, score=0.5, conf=0.6, severity=2, frame_path=pb)
    dedup_all(db, ["A", "B"])
    (h0,) = hazards(db)
    assert h0["sightings"] == 2
    execute(db, "UPDATE hazards SET status = 'reported' WHERE id = ?", (h0["id"],))
    drop_drive_candidates(db, "A")
    dedup(db, "A")
    (h,) = hazards(db)
    assert h["id"] == h0["id"]
    assert h["status"] == "reported"
    assert h["sightings"] == 1
    assert h["photo_path"] == pb
    assert h["severity"] == 2
    assert h["confidence"] <= 0.6 + 1e-6
    assert hazard_of(db, b) == h["id"]
    fs, ls = parse_time(h["first_seen"]), parse_time(h["last_seen"])
    assert T_B <= fs <= T_B + timedelta(hours=1) and T_B <= ls <= T_B + timedelta(hours=1)


def test_candidate_losing_acceptance_leaves_its_hazard(db):
    a = add_cand(db, "A", VA[0], VA[1], 0.0)
    b = add_cand(db, "B", VA[0], VA[1], 0.0)
    dedup_all(db, ["A", "B"])
    execute(db, "UPDATE verdicts SET accepted = 0 WHERE candidate_id = ?", (a,))
    dedup(db, "A")
    (h,) = hazards(db)
    assert hazard_of(db, a) is None
    assert hazard_of(db, b) == h["id"]
    assert h["sightings"] == 1


def test_only_candidate_losing_acceptance_deletes_hazard(db):
    a = add_cand(db, "A", VA[0], VA[1], 0.0)
    dedup(db, "A")
    execute(db, "UPDATE verdicts SET accepted = 0 WHERE candidate_id = ?", (a,))
    dedup(db, "A")
    assert hazards(db) == []
    assert hazard_of(db, a) is None


def test_candidate_moving_away_splits_the_hazard(db):
    a = add_cand(db, "A", VA[0], VA[1], 0.0)
    b = add_cand(db, "B", VA[0], VA[1], 0.0)
    dedup_all(db, ["A", "B"])
    far = offset(VA[0], VA[1], 500.0)
    execute(db, "UPDATE candidates SET lat = ?, lon = ? WHERE id = ?", (far[0], far[1], b))
    dedup_all(db, ["A", "B"])
    assert len(hazards(db)) == 2
    assert not same_hazard(db, a, b)
    assert_no_orphans(db)


def test_drive_with_no_accepted_candidates_leaves_table_untouched(db):
    add_cand(db, "A", VA[0], VA[1], 0.0)
    add_cand(db, "B", VA[0], VA[1], 0.0)
    dedup_all(db, ["A", "B"])
    execute(db, "UPDATE hazards SET status = 'approved'")
    add_cand(db, "C", VA[0], VA[1], 0.0, accepted=False, conf=0.1)
    add_cand(db, "C", VA[0], VA[1], 0.0, geotag="no_gps")
    before = snapshot(db)
    dedup(db, "C")
    assert snapshot(db) == before


def test_drive_with_no_candidates_at_all_leaves_table_untouched(db):
    add_cand(db, "A", VA[0], VA[1], 0.0)
    dedup(db, "A")
    before = snapshot(db)
    dedup(db, "C")
    dedup(db, "unknown")
    assert snapshot(db) == before


def test_dedup_on_empty_database_creates_nothing(db):
    dedup(db, "A")
    assert hazards(db) == []


# ------------------------------------------------------------------ chains and determinism

def _chain(db, drive, offsets, order):
    ids = {}
    for i in order:
        lat, lon = offset(VA[0], VA[1], offsets[i])
        ids[i] = add_cand(db, drive, lat, lon, 0.0, best_ts=10.0 + i)
    return ids


def test_tight_clump_becomes_one_hazard(db):
    for n in (0.0, 2.0, -3.0, 4.0, 1.0):
        lat, lon = offset(VA[0], VA[1], n)
        add_cand(db, "A", lat, lon, 0.0, best_ts=10.0 + n)
    dedup(db, "A")
    assert len(hazards(db)) == 1
    assert hazards(db)[0]["sightings"] in (1, 5)


def test_two_clumps_become_two_hazards(db):
    for base_n in (0.0, 300.0):
        for d in ("A", "B"):
            lat, lon = offset(VA[0], VA[1], base_n + 3.0)
            add_cand(db, d, lat, lon, 0.0)
    dedup_all(db, ["A", "B"])
    hs = hazards(db)
    assert len(hs) == 2
    assert all(h["sightings"] == 2 for h in hs)


def test_chain_is_sane(db):
    """Candidates 0, 20, 40 m apart: either one hazard (single linkage) or two (anchored), never lost."""
    offsets = [0.0, 20.0, 40.0]
    ids = _chain(db, "A", offsets, [0, 1, 2])
    dedup(db, "A")
    hs = hazards(db)
    assert 1 <= len(hs) <= 2
    for cid in ids.values():
        assert hazard_of(db, cid) is not None
    assert_no_orphans(db)
    # the end points are 40 m apart; the middle one joins at least one of them
    assert same_hazard(db, ids[0], ids[1]) or same_hazard(db, ids[1], ids[2])


def test_chain_is_deterministic_and_order_independent(tmp_path, monkeypatch):
    offsets = [0.0, 20.0, 40.0, 60.0]
    results = []
    for k, order in enumerate(([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1])):
        d = make_db(tmp_path / ("r%d" % k), monkeypatch)
        add_drive(d, "A", T_A)
        ids = _chain(d, "A", offsets, order)
        dedup(d, "A")
        first = partition(d, list(ids.values()))
        dedup(d, "A")
        assert partition(d, list(ids.values())) == first      # stable under re-runs
        inv = {v: k2 for k2, v in ids.items()}
        results.append({frozenset(inv[c] for c in grp) for grp in first})
        for cid in ids.values():
            assert hazard_of(d, cid) is not None
    assert results[0] == results[1] == results[2], results


def test_shuffled_insertion_order_gives_same_groups(tmp_path, monkeypatch):
    rnd = random.Random(14)
    points = [(n, e) for n in (0.0, 10.0, 200.0, 210.0) for e in (0.0, 8.0)]
    outcomes = []
    for k in range(3):
        d = make_db(tmp_path / ("s%d" % k), monkeypatch)
        add_drive(d, "A", T_A)
        order = list(range(len(points)))
        rnd.shuffle(order)
        ids = {}
        for i in order:
            lat, lon = offset(VA[0], VA[1], points[i][0], points[i][1])
            ids[i] = add_cand(d, "A", lat, lon, 0.0, best_ts=10.0 + i)
        dedup(d, "A")
        inv = {v: k2 for k2, v in ids.items()}
        outcomes.append({frozenset(inv[c] for c in grp) for grp in partition(d, list(ids.values()))})
        assert len(hazards(d)) == 2
    assert outcomes[0] == outcomes[1] == outcomes[2]


def test_drive_processing_order_does_not_change_simple_result(tmp_path, monkeypatch):
    outcomes = []
    for k, order in enumerate((["A", "B", "C"], ["C", "B", "A"], ["B", "C", "A"])):
        d = make_db(tmp_path / ("o%d" % k), monkeypatch)
        for did, t in (("A", T_A), ("B", T_B), ("C", T_C)):
            add_drive(d, did, t)
        near = {did: add_cand(d, did, *offset(VA[0], VA[1], 2.0 * i), 0.0) for i, did in enumerate("ABC")}
        far = add_cand(d, "C", *offset(VA[0], VA[1], 900.0), 0.0, best_ts=60.0)
        dedup_all(d, order)
        hs = hazards(d)
        assert len(hs) == 2
        outcomes.append(sorted(h["sightings"] for h in hs))
        assert same_hazard(d, near["A"], near["B"]) and same_hazard(d, near["B"], near["C"])
        assert not same_hazard(d, near["A"], far)
    assert outcomes[0] == outcomes[1] == outcomes[2]
