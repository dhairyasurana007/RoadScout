"""Task 18: the pure agency-from-location function (Virginia -> VDOT, Maryland -> MDOT SHA, elsewhere -> none)."""
import math
import warnings

import pytest

from rs18_helpers import agency_fn  # noqa: F401

VA = [("Centreville", 38.84, -77.43), ("Arlington", 38.88, -77.10), ("Richmond", 37.54, -77.43),
      ("Fairfax City", 38.85, -77.30), ("Alexandria west", 38.80, -77.10), ("Roanoke", 37.27, -79.94),
      ("Norfolk", 36.85, -76.29), ("Fredericksburg", 38.30, -77.46), ("Winchester", 39.18, -78.16)]
MD = [("Rockville", 39.08, -77.15), ("Bethesda", 38.98, -77.10), ("Baltimore", 39.29, -76.61),
      ("Silver Spring north", 39.00, -77.01), ("Frederick", 39.41, -77.41), ("Annapolis", 38.98, -76.49),
      ("Salisbury", 38.36, -75.60), ("Columbia", 39.20, -76.86)]
NONE = [("DC Mall", 38.90, -77.03), ("DC north", 38.93, -77.03), ("DC Capitol", 38.8899, -77.0091),
        ("Philadelphia", 39.95, -75.17), ("Raleigh", 35.78, -78.64), ("Atlantic Ocean", 38.0, -74.0),
        ("Wilmington DE", 39.74, -75.55), ("Pittsburgh", 40.44, -79.99), ("Charleston WV", 38.35, -81.63),
        ("New York", 40.71, -74.00), ("Null island", 0.0, 0.0),
        ("Gulf of Mexico", 25.0, -90.0), ("London", 51.5, -0.12)]
# well known but narrow / coastal places a coarse outline may miss: informative only
SOFT = [("Hagerstown", 39.64, -77.72, "MDOT SHA"), ("Cumberland", 39.65, -78.76, "MDOT SHA"),
        ("Ocean City", 38.38, -75.07, "MDOT SHA"), ("Chincoteague", 37.93, -75.38, "VDOT"),
        ("Abingdon VA", 36.71, -81.98, "VDOT"), ("Silver Spring (300 m from DC)", 38.99, -77.03, "MDOT SHA"),
        ("Harpers Ferry WV", 39.32, -77.78, None), ("Martinsburg WV", 39.46, -77.96, None)]


@pytest.mark.parametrize("name,lat,lon", VA)
def test_virginia_is_vdot(agency_fn, name, lat, lon):
    assert agency_fn(lat, lon) == "VDOT", name


@pytest.mark.parametrize("name,lat,lon", MD)
def test_maryland_is_mdot_sha(agency_fn, name, lat, lon):
    assert agency_fn(lat, lon) == "MDOT SHA", name


@pytest.mark.parametrize("name,lat,lon", NONE)
def test_other_jurisdictions_have_no_agency(agency_fn, name, lat, lon):
    assert agency_fn(lat, lon) is None, name


def test_soft_edge_places_are_reported_not_enforced(agency_fn):
    wrong = []
    for name, lat, lon, want in SOFT:
        if agency_fn(lat, lon) != want:
            wrong.append("%s (%s,%s): expected %s got %s" % (name, lat, lon, want, agency_fn(lat, lon)))
    if wrong:
        warnings.warn("coarse boundary, informative only: " + "; ".join(wrong))


def test_deterministic(agency_fn):
    pts = [(lat, lon) for _, lat, lon in VA + MD + NONE]
    first = [agency_fn(*p) for p in pts]
    for _ in range(5):
        assert [agency_fn(*p) for p in pts] == first


def test_result_is_a_plain_value_per_call(agency_fn):
    """No hidden state: interleaving calls does not change answers."""
    a = [agency_fn(38.85, -77.30), agency_fn(39.08, -77.15), agency_fn(38.90, -77.03)]
    b = [agency_fn(38.90, -77.03), agency_fn(39.08, -77.15), agency_fn(38.85, -77.30)][::-1]
    assert a == b == ["VDOT", "MDOT SHA", None]


def test_integer_and_numpy_style_inputs(agency_fn):
    assert agency_fn(39, -77) in ("MDOT SHA", None, "VDOT")  # must not crash on ints
    import numpy as np

    assert agency_fn(np.float64(38.85), np.float64(-77.30)) == "VDOT"
    assert agency_fn("38.85", "-77.30") in ("VDOT", None)  # numeric strings: either way, no crash


BAD = [(float("nan"), -77.3), (38.85, float("nan")), (float("nan"), float("nan")), (None, -77.3), (38.85, None),
       (None, None), (91.0, -77.3), (-91.0, -77.3), (38.85, 181.0), (38.85, -181.0), (1e9, 1e9),
       (float("inf"), -77.3), (38.85, float("-inf")), ("abc", "def"), ("", ""), ([], {}), (True, False)]


@pytest.mark.parametrize("lat,lon", BAD)
def test_bad_input_means_no_agency_without_crashing(agency_fn, lat, lon):
    assert agency_fn(lat, lon) is None


def test_swapped_lat_lon_is_not_virginia(agency_fn):
    # (lon, lat) order mistakes must not produce an agency: -77.3 is a valid latitude (Antarctica), 38.85 a valid lon.
    assert agency_fn(-77.30, 38.85) is None


def test_potomac_crossing_area_does_not_crash(agency_fn):
    """I-495 American Legion Bridge area: either answer is tolerated, but it must be a clean answer."""
    for lat, lon in [(38.9667, -77.1486), (38.9660, -77.1500), (38.9650, -77.1530), (38.9700, -77.1400)]:
        assert agency_fn(lat, lon) in ("VDOT", "MDOT SHA", None)


def test_border_points_return_clean_answers(agency_fn):
    # Mason-Dixon-ish / Potomac / DC edges / state lines: no crash, a valid value.
    for lat, lon in [(39.7210, -77.5), (38.4, -77.0), (38.7916, -77.0399), (38.9958, -77.0402), (37.0, -75.9),
                     (36.5430, -76.0), (38.0270, -75.2400), (39.7220, -75.7880)]:
        assert agency_fn(lat, lon) in ("VDOT", "MDOT SHA", None)


def test_many_calls_are_fast(agency_fn):
    import time

    t = time.time()
    for i in range(2000):
        agency_fn(36.0 + (i % 50) * 0.08, -83.0 + (i % 40) * 0.2)
    assert time.time() - t < 10.0, "agency lookup is too slow for per-request use"


def test_northern_neighbours_of_virginia_are_not_virginia(agency_fn):
    # West Virginia, Tennessee, North Carolina, Kentucky just outside Virginia, > 15 km from the line
    for lat, lon in [(38.00, -81.00), (36.20, -82.80), (35.90, -79.00), (37.00, -83.50)]:
        assert agency_fn(lat, lon) is None, (lat, lon)


def test_delaware_and_pennsylvania_are_not_maryland(agency_fn):
    for lat, lon in [(39.20, -75.50), (39.90, -76.50), (40.20, -77.00), (39.82, -77.50)]:
        assert agency_fn(lat, lon) is None, (lat, lon)


def test_exact_float_types(agency_fn):
    assert math.isfinite(38.85)
    r = agency_fn(38.85, -77.30)
    assert r == "VDOT"
