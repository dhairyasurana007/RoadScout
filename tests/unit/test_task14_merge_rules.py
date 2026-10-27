"""Task 14: which candidates merge and which do not."""
import pytest

from dd14_helpers import (ENV_DIST, ENV_HEAD, MD, NOVA, T_A, T_B, VA, add_cand, add_drive, ang_diff, dedup,
                          dedup_all, haversine, hazard_of, hazards, make_db, offset, same_hazard)

BASES = [pytest.param(VA, id="richmond-va"), pytest.param(MD, id="baltimore-md"), pytest.param(NOVA, id="nova-va")]


@pytest.fixture
def db(tmp_path, monkeypatch):
    d = make_db(tmp_path, monkeypatch)
    add_drive(d, "A", T_A)
    add_drive(d, "B", T_B)
    return d


def pair(db, base, north=0.0, east=0.0, *, ha=0.0, hb=0.0, da="A", dbid="B", cls_a="pothole", cls_b="pothole"):
    lat, lon = base
    la, lo = offset(lat, lon, north, east)
    a = add_cand(db, da, lat, lon, ha, cls=cls_a)
    b = add_cand(db, dbid, la, lo, hb, cls=cls_b)
    return a, b


# ------------------------------------------------------------------ eligibility

def test_rejected_candidate_is_not_deduplicated(db):
    c = add_cand(db, "A", VA[0], VA[1], 0.0, is_hazard=True, accepted=False, conf=0.2)
    dedup(db, "A")
    assert hazards(db) == []
    assert hazard_of(db, c) is None


def test_non_hazard_verdict_is_not_deduplicated(db):
    c = add_cand(db, "A", VA[0], VA[1], 0.0, is_hazard=False, accepted=False)
    dedup(db, "A")
    assert hazards(db) == []
    assert hazard_of(db, c) is None


def test_candidate_without_verdict_is_not_deduplicated(db):
    c = add_cand(db, "A", VA[0], VA[1], 0.0, with_verdict=False)
    dedup(db, "A")
    assert hazards(db) == []
    assert hazard_of(db, c) is None


@pytest.mark.parametrize("status", ["no_gps", "out_of_range"])
def test_ungeotagged_candidate_is_not_deduplicated(db, status):
    c = add_cand(db, "A", VA[0], VA[1], 0.0, geotag=status)
    dedup(db, "A")
    assert hazards(db) == []
    assert hazard_of(db, c) is None


def test_rejected_neighbour_does_not_join_or_count(db):
    good = add_cand(db, "A", VA[0], VA[1], 0.0)
    lat, lon = offset(VA[0], VA[1], 5.0)
    bad = add_cand(db, "B", lat, lon, 0.0, accepted=False, conf=0.1)
    lat, lon = offset(VA[0], VA[1], 6.0)
    nogeo = add_cand(db, "B", lat, lon, 0.0, geotag="no_gps")
    dedup_all(db, ["A", "B"])
    hs = hazards(db)
    assert len(hs) == 1
    assert hs[0]["sightings"] == 1
    assert hazard_of(db, good) == hs[0]["id"]
    assert hazard_of(db, bad) is None and hazard_of(db, nogeo) is None


def test_accepted_candidate_is_linked_to_its_hazard(db):
    c = add_cand(db, "A", VA[0], VA[1], 0.0)
    dedup(db, "A")
    (h,) = hazards(db)
    assert hazard_of(db, c) == h["id"]


# ------------------------------------------------------------------ distance

@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("north,east", [(22.0, 0.0), (0.0, 22.0), (-22.0, 0.0), (0.0, -22.0), (15.0, 15.0)])
def test_just_inside_threshold_merges_across_drives(db, base, north, east):
    lat, lon = offset(base[0], base[1], north, east)
    assert haversine(base[0], base[1], lat, lon) < 24.0
    a, b = pair(db, base, north, east)
    dedup_all(db, ["A", "B"])
    assert same_hazard(db, a, b)
    assert len(hazards(db)) == 1


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("north,east", [(28.0, 0.0), (0.0, 28.0), (-28.0, 0.0), (0.0, -30.0), (20.0, 20.0)])
def test_just_outside_threshold_stays_separate_across_drives(db, base, north, east):
    lat, lon = offset(base[0], base[1], north, east)
    assert haversine(base[0], base[1], lat, lon) > 26.0
    a, b = pair(db, base, north, east)
    dedup_all(db, ["A", "B"])
    assert not same_hazard(db, a, b)
    assert hazard_of(db, a) is not None and hazard_of(db, b) is not None
    assert len(hazards(db)) == 2


def test_east_west_distance_uses_true_metres_at_maryland_latitude(db):
    """24 m east at 39.3 N is ~0.00028 deg of longitude; treating degrees as equal-size would get it wrong."""
    a, b = pair(db, MD, 0.0, 24.0, ha=90.0, hb=90.0)
    dedup_all(db, ["A", "B"])
    assert same_hazard(db, a, b)


def test_thirty_metres_east_is_not_merged_at_maryland_latitude(db):
    a, b = pair(db, MD, 0.0, 30.0, ha=90.0, hb=90.0)
    dedup_all(db, ["A", "B"])
    assert not same_hazard(db, a, b)


def test_identical_position_merges(db):
    a, b = pair(db, VA)
    dedup_all(db, ["A", "B"])
    assert same_hazard(db, a, b)


def test_within_one_drive_merges_nearby(db):
    lat, lon = offset(VA[0], VA[1], 10.0)
    a = add_cand(db, "A", VA[0], VA[1], 0.0, best_ts=10.0)
    b = add_cand(db, "A", lat, lon, 0.0, best_ts=14.0)
    dedup(db, "A")
    assert same_hazard(db, a, b)
    assert len(hazards(db)) == 1


def test_within_one_drive_far_apart_stay_separate(db):
    lat, lon = offset(VA[0], VA[1], 300.0)
    a = add_cand(db, "A", VA[0], VA[1], 0.0, best_ts=10.0)
    b = add_cand(db, "A", lat, lon, 0.0, best_ts=40.0)
    dedup(db, "A")
    assert not same_hazard(db, a, b)
    assert len(hazards(db)) == 2


def test_distance_parameter_widens_and_narrows(db):
    a, b = pair(db, VA, 40.0)
    dedup_all(db, ["A", "B"], distance_m=60.0)
    assert same_hazard(db, a, b)


def test_distance_parameter_narrows(tmp_path, monkeypatch):
    d = make_db(tmp_path, monkeypatch)
    add_drive(d, "A", T_A)
    add_drive(d, "B", T_B)
    a, b = pair(d, VA, 15.0)
    dedup_all(d, ["A", "B"], distance_m=10.0)
    assert not same_hazard(d, a, b)
    assert len(hazards(d)) == 2


def test_distance_env_var_widens(db, monkeypatch):
    monkeypatch.setenv(ENV_DIST, "50")
    a, b = pair(db, VA, 40.0)
    dedup_all(db, ["A", "B"])
    assert same_hazard(db, a, b)


def test_distance_env_var_narrows(db, monkeypatch):
    monkeypatch.setenv(ENV_DIST, "10")
    a, b = pair(db, VA, 15.0)
    dedup_all(db, ["A", "B"])
    assert not same_hazard(db, a, b)


def test_distance_parameter_beats_env_var(db, monkeypatch):
    monkeypatch.setenv(ENV_DIST, "10")
    a, b = pair(db, VA, 30.0)
    dedup_all(db, ["A", "B"], distance_m=50.0)
    assert same_hazard(db, a, b)


def test_distance_env_var_read_at_call_time(db, monkeypatch):
    a, b = pair(db, VA, 15.0)
    monkeypatch.setenv(ENV_DIST, "5")
    dedup_all(db, ["A", "B"])
    assert not same_hazard(db, a, b)


# ------------------------------------------------------------------ class

def test_different_classes_never_merge(db):
    a, b = pair(db, VA, 0.0, cls_a="pothole", cls_b="debris in lane")
    dedup_all(db, ["A", "B"])
    assert not same_hazard(db, a, b)
    hs = hazards(db)
    assert len(hs) == 2
    assert {h["cls"] for h in hs} == {"pothole", "debris in lane"}


def test_different_classes_in_same_drive_never_merge(db):
    a = add_cand(db, "A", VA[0], VA[1], 0.0, cls="pothole")
    b = add_cand(db, "A", VA[0], VA[1], 0.0, cls="debris in lane")
    dedup(db, "A")
    assert not same_hazard(db, a, b)
    assert len(hazards(db)) == 2


def test_class_is_recorded_on_the_hazard(db):
    add_cand(db, "A", VA[0], VA[1], 0.0, cls="debris in lane")
    dedup(db, "A")
    (h,) = hazards(db)
    assert h["cls"] == "debris in lane"


# ------------------------------------------------------------------ heading

@pytest.mark.parametrize("ha,hb", [(0.0, 30.0), (0.0, 40.0), (90.0, 90.0), (359.0, 1.0), (350.0, 20.0),
                                   (10.0, 350.0), (180.0, 150.0), (270.0, 300.0)])
def test_similar_headings_merge(db, ha, hb):
    a, b = pair(db, VA, 5.0, ha=ha, hb=hb)
    dedup_all(db, ["A", "B"])
    assert ang_diff(ha, hb) < 45
    assert same_hazard(db, a, b)


@pytest.mark.parametrize("ha,hb", [(0.0, 180.0), (90.0, 270.0), (0.0, 90.0), (0.0, 60.0), (0.0, 50.0),
                                   (340.0, 40.0), (180.0, 0.0), (45.0, 225.0), (359.0, 181.0)])
def test_different_headings_never_merge(db, ha, hb):
    a, b = pair(db, VA, 5.0, ha=ha, hb=hb)
    dedup_all(db, ["A", "B"])
    assert not same_hazard(db, a, b)
    assert len(hazards(db)) == 2


def test_opposite_directions_same_spot_stay_separate_within_a_drive(db):
    a = add_cand(db, "A", VA[0], VA[1], 10.0)
    b = add_cand(db, "A", VA[0], VA[1], 190.0)
    dedup(db, "A")
    assert not same_hazard(db, a, b)
    assert len(hazards(db)) == 2


def test_heading_tolerance_parameter_widens(db):
    a, b = pair(db, VA, 5.0, ha=0.0, hb=90.0)
    dedup_all(db, ["A", "B"], heading_tolerance_deg=100.0)
    assert same_hazard(db, a, b)


def test_heading_tolerance_parameter_narrows(db):
    a, b = pair(db, VA, 5.0, ha=0.0, hb=30.0)
    dedup_all(db, ["A", "B"], heading_tolerance_deg=20.0)
    assert not same_hazard(db, a, b)


def test_heading_tolerance_env_var(db, monkeypatch):
    monkeypatch.setenv(ENV_HEAD, "100")
    a, b = pair(db, VA, 5.0, ha=0.0, hb=90.0)
    dedup_all(db, ["A", "B"])
    assert same_hazard(db, a, b)


def test_heading_tolerance_env_var_narrows(db, monkeypatch):
    monkeypatch.setenv(ENV_HEAD, "20")
    a, b = pair(db, VA, 5.0, ha=0.0, hb=30.0)
    dedup_all(db, ["A", "B"])
    assert not same_hazard(db, a, b)


def test_heading_param_beats_env_var(db, monkeypatch):
    monkeypatch.setenv(ENV_HEAD, "10")
    a, b = pair(db, VA, 5.0, ha=0.0, hb=30.0)
    dedup_all(db, ["A", "B"], heading_tolerance_deg=45.0)
    assert same_hazard(db, a, b)


def test_missing_heading_does_not_crash(db):
    """A geotagged candidate with an unknown heading must not break deduplication."""
    from dd14_helpers import execute

    a = add_cand(db, "A", VA[0], VA[1], 0.0)
    execute(db, "UPDATE candidates SET heading = NULL WHERE id = ?", (a,))
    b = add_cand(db, "B", VA[0], VA[1], 0.0)
    dedup_all(db, ["A", "B"])
    assert hazard_of(db, b) is not None
    assert len(hazards(db)) in (1, 2)
