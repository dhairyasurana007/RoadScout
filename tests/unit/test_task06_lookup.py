"""Task 6 hidden tests: interpolation, clock offset, speed, heading."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

import rs6_helpers as H
from rs6_helpers import Pt, T0, angdiff, build, haversine, write_gpx


def near(a, b, tol):
    return abs(a - b) <= tol


# ------------------------------------------------------------------ exact points / interpolation
def test_exact_trackpoint_times_return_that_point(tmp_path):
    pts = H.circle(n=21)
    f = write_gpx(tmp_path / "c.gpx", pts)
    lk = build(f)
    for p in pts:
        r = lk(p.t)
        assert near(r.lat, p.lat, 1e-7) and near(r.lon, p.lon, 1e-7), (p.t, r)
        assert H.finite(r)


def test_midpoint_linear_interpolation(tmp_path):
    pts = [Pt(0, 38.0, -77.0), Pt(10, 38.001, -77.002), Pt(20, 38.003, -77.003)]
    lk = build(write_gpx(tmp_path / "a.gpx", pts))
    r = lk(2.5)
    assert near(r.lat, 38.00025, 1e-7) and near(r.lon, -77.0005, 1e-7)
    r = lk(15)
    assert near(r.lat, 38.002, 1e-7) and near(r.lon, -77.0025, 1e-7)


def test_fractional_second_trackpoints_and_lookup(tmp_path):
    pts = [Pt(0.0, 38.0, -77.0), Pt(0.5, 38.0001, -77.0), Pt(1.0, 38.0002, -77.0)]
    lk = build(write_gpx(tmp_path / "f.gpx", pts))
    assert near(lk(0.25).lat, 38.00005, 1e-8)
    assert near(lk(0.5).lat, 38.0001, 1e-8)
    assert near(lk(0.75).lat, 38.00015, 1e-8)
    # 0.1 m per 0.5s is not the point; speed from positions: 11.12 m / 0.5 s ~ 22 m/s
    assert near(lk(0.25).speed, 0.0001 * H.M_PER_DEG / 0.5, 0.5)


def test_lookup_is_stateless_any_order(tmp_path):
    pts = H.line(45, 8, n=60)
    lk = build(write_gpx(tmp_path / "l.gpx", pts))
    ts = [3.3, 50.1, 7.7, 7.7, 0.0, 58.9, 12.25, 3.3]
    first = [lk(t) for t in ts]
    second = [lk(t) for t in reversed(ts)][::-1]
    assert first == second


def test_return_values_are_plain_finite_floats(tmp_path):
    lk = build(write_gpx(tmp_path / "a.gpx", H.line(90, 10)))
    r = lk(4.2)
    assert all(isinstance(x, float) for x in r)
    assert H.finite(r)


# ------------------------------------------------------------------ clock offset
def test_zero_offset_video_start_is_gpx_time(tmp_path):
    pts = H.circle(n=31)
    lk = build(write_gpx(tmp_path / "c.gpx", pts), start=T0, offset=0.0)
    r = lk(7)
    assert near(r.lat, pts[7].lat, 1e-7) and near(r.lon, pts[7].lon, 1e-7)


def test_positive_offset_added_to_video_time(tmp_path):
    pts = H.circle(n=31)
    f = write_gpx(tmp_path / "c.gpx", pts)
    lk = build(f, start=T0, offset=5.0)
    r = lk(2)  # gpx time = 0 + 2 + 5 = 7
    assert near(r.lat, pts[7].lat, 1e-7) and near(r.lon, pts[7].lon, 1e-7)


def test_negative_offset(tmp_path):
    pts = H.circle(n=31)
    f = write_gpx(tmp_path / "c.gpx", pts)
    lk = build(f, start=T0, offset=-3.0)
    r = lk(10)  # gpx time 7
    assert near(r.lat, pts[7].lat, 1e-7) and near(r.lon, pts[7].lon, 1e-7)


def test_fractional_offset(tmp_path):
    pts = [Pt(0, 38.0, -77.0), Pt(10, 38.001, -77.0)]
    lk = build(write_gpx(tmp_path / "o.gpx", pts), start=T0, offset=2.5)
    r = lk(2.5)  # gpx 5.0 -> halfway
    assert near(r.lat, 38.0005, 1e-8)


def test_offset_changes_result_vs_no_offset(tmp_path):
    f = write_gpx(tmp_path / "c.gpx", H.line(0, 10, n=30))
    a = build(f, offset=0.0)(10)
    b = build(f, offset=4.0)(10)
    assert haversine(a.lat, a.lon, b.lat, b.lon) == pytest.approx(40.0, rel=0.02)


def test_large_offset_within_a_day(tmp_path):
    # video started an hour before the gpx recording; offset +3600 aligns them.
    pts = H.line(0, 5, n=20)
    f = write_gpx(tmp_path / "x.gpx", pts)
    vs = T0 - timedelta(hours=1)
    lk = build(f, start=vs, offset=3600.0)
    r = lk(4)
    assert near(r.lat, pts[4].lat, 1e-7)


def test_video_start_not_equal_first_point(tmp_path):
    if not H.supports_video_start():
        pytest.skip("interface has no video start parameter; video time assumed = first point")
    pts = H.circle(n=61)
    f = write_gpx(tmp_path / "c.gpx", pts)
    lk = build(f, start=T0 + timedelta(seconds=30), offset=0.0)
    r = lk(0)
    assert near(r.lat, pts[30].lat, 1e-7) and near(r.lon, pts[30].lon, 1e-7)
    r = lk(10.5)
    mid_lat = (pts[40].lat + pts[41].lat) / 2
    assert near(r.lat, mid_lat, 1e-7)


def test_video_start_and_offset_combine(tmp_path):
    if not H.supports_video_start():
        pytest.skip("no video start parameter")
    pts = H.circle(n=61)
    f = write_gpx(tmp_path / "c.gpx", pts)
    lk = build(f, start=T0 + timedelta(seconds=20), offset=-5.0)
    r = lk(3)  # 20 + 3 - 5 = 18
    assert near(r.lat, pts[18].lat, 1e-7)


def test_video_start_in_non_utc_timezone_equivalent(tmp_path):
    if not H.supports_video_start():
        pytest.skip("no video start parameter")
    pts = H.circle(n=61)
    f = write_gpx(tmp_path / "c.gpx", pts)
    est = timezone(timedelta(hours=-4))
    start_est = (T0 + timedelta(seconds=15)).astimezone(est)
    r = build(f, start=start_est, offset=0.0)(5)
    assert near(r.lat, pts[20].lat, 1e-7)


def test_offset_pushing_outside_range_is_clamped_or_error(tmp_path):
    pts = H.line(0, 10, n=11)
    lk = build(write_gpx(tmp_path / "x.gpx", pts), offset=-1000.0)
    r = H.lookup_or_oor(lk, 0)
    if r != "OOR":
        assert near(r.lat, pts[0].lat, 1e-7) and H.finite(r)


# ------------------------------------------------------------------ speed
@pytest.mark.parametrize("speed", [3.0, 13.4, 30.0])
def test_speed_from_positions_when_no_tags(tmp_path, speed):
    lk = build(write_gpx(tmp_path / "s.gpx", H.line(30, speed, n=15)))
    for t in (0.5, 5.0, 9.3, 13.0):
        assert lk(t).speed == pytest.approx(speed, rel=0.02)


def test_speed_from_positions_uneven_time_spacing(tmp_path):
    p0 = (38.0, -77.0)
    p1 = H.step(*p0, 0, 100.0)
    p2 = H.step(*p1, 0, 100.0)
    pts = [Pt(0, *p0), Pt(5, *p1), Pt(25, *p2)]  # 20 m/s then 5 m/s
    lk = build(write_gpx(tmp_path / "u.gpx", pts))
    assert lk(2.5).speed == pytest.approx(20.0, rel=0.03)
    assert lk(15).speed == pytest.approx(5.0, rel=0.03)


def test_speed_tags_used_when_present(tmp_path):
    # positions imply 10 m/s but tags say 20 m/s: tags win
    pts = H.line(0, 10, n=11, tags=False)
    pts = [Pt(p.t, p.lat, p.lon, 20.0) for p in pts]
    lk = build(write_gpx(tmp_path / "t.gpx", pts))
    for t in (0, 3, 3.5, 10):
        assert lk(t).speed == pytest.approx(20.0, rel=0.01)


def test_speed_tags_interpolated(tmp_path):
    pts = [Pt(0, 38.0, -77.0, 10.0), Pt(10, 38.001, -77.0, 20.0)]
    lk = build(write_gpx(tmp_path / "t.gpx", pts))
    assert lk(0).speed == pytest.approx(10.0, abs=0.05)
    assert lk(5).speed == pytest.approx(15.0, abs=0.05)
    assert lk(10).speed == pytest.approx(20.0, abs=0.05)


def test_speed_tag_zero_is_respected(tmp_path):
    pts = H.line(0, 10, n=6)
    pts = [Pt(p.t, p.lat, p.lon, 0.0) for p in pts]
    lk = build(write_gpx(tmp_path / "t.gpx", pts))
    assert lk(2.5).speed == pytest.approx(0.0, abs=0.01)


def test_speed_tags_in_extensions_element(tmp_path):
    # Many loggers write <extensions><speed>. Softer requirement than a direct <speed> child.
    pts = [Pt(p.t, p.lat, p.lon, 25.0) for p in H.line(0, 10, n=11)]
    f = tmp_path / "e.gpx"
    f.write_text(H.gpx_xml([pts], speed_in="extensions"), encoding="utf-8")
    r = build(f)(5)
    assert r.speed == pytest.approx(25.0, rel=0.01)


def test_speed_is_in_metres_per_second_not_kmh_or_mph(tmp_path):
    lk = build(write_gpx(tmp_path / "s.gpx", H.line(0, 10, n=11)))
    s = lk(5).speed
    assert 9.7 < s < 10.3


def test_speed_nonnegative(tmp_path):
    lk = build(write_gpx(tmp_path / "s.gpx", H.circle(n=41)))
    assert all(lk(t).speed >= 0 for t in [0, 0.3, 5, 12.5, 30, 39.9])


def test_accelerating_speed_between_neighbours(tmp_path):
    p = (38.0, -77.0)
    pts = [Pt(0, *p)]
    for i, d in enumerate([5, 10, 15, 20, 25], start=1):
        p = H.step(*p, 90, d)
        pts.append(Pt(i, *p))
    lk = build(write_gpx(tmp_path / "a.gpx", pts))
    s = [lk(t + 0.5).speed for t in range(5)]
    assert all(4.5 <= x <= 26 for x in s)
    assert s[-1] > s[0]


# ------------------------------------------------------------------ heading
@pytest.mark.parametrize("heading", [0, 45, 90, 135, 180, 225, 270, 315, 359.0, 1.0])
def test_heading_straight_lines(tmp_path, heading):
    lk = build(write_gpx(tmp_path / "h.gpx", H.line(heading, 12, n=12, lat=38.9, lon=-77.0)))
    for t in (0.5, 4.2, 10.5):
        h = lk(t).heading
        assert 0 <= h < 360
        assert angdiff(h, heading) < 1.0, (heading, t, h)


def test_heading_zero_is_north_and_clockwise(tmp_path):
    n = build(write_gpx(tmp_path / "n.gpx", H.line(0, 10)))(5).heading
    e = build(write_gpx(tmp_path / "e.gpx", H.line(90, 10)))(5).heading
    assert angdiff(n, 0) < 1.0 and angdiff(e, 90) < 1.0


def test_heading_on_clockwise_circle_increases(tmp_path):
    pts = H.circle(radius_m=200, n=61, speed=10)
    lk = build(write_gpx(tmp_path / "c.gpx", pts))
    for t in range(0, 59, 4):
        mid = t + 0.5
        exp = H.bearing(pts[t].lat, pts[t].lon, pts[t + 1].lat, pts[t + 1].lon)
        assert angdiff(lk(mid).heading, exp) < 1.5, (t, lk(mid).heading, exp)


def test_heading_corner_each_segment(tmp_path):
    a = (38.0, -77.0)
    b = H.step(*a, 0, 100)
    c = H.step(*b, 90, 100)
    lk = build(write_gpx(tmp_path / "k.gpx", [Pt(0, *a), Pt(10, *b), Pt(20, *c)]))
    assert angdiff(lk(5).heading, 0) < 1.0
    assert angdiff(lk(15).heading, 90) < 1.0
    corner = lk(10).heading
    assert angdiff(corner, 0) < 1.5 or angdiff(corner, 90) < 1.5 or 0 <= corner <= 90


def test_heading_wraparound_through_north(tmp_path):
    a = (38.0, -77.0)
    b = H.step(*a, 355, 100)
    c = H.step(*b, 5, 100)
    lk = build(write_gpx(tmp_path / "w.gpx", [Pt(0, *a), Pt(10, *b), Pt(20, *c)]))
    assert angdiff(lk(5).heading, 355) < 1.0
    assert angdiff(lk(15).heading, 5) < 1.0


def test_heading_always_in_range(tmp_path):
    lk = build(write_gpx(tmp_path / "c.gpx", H.circle(n=81)))
    for i in range(0, 800):
        h = lk(i / 10).heading
        assert 0 <= h < 360 and math.isfinite(h)


# ------------------------------------------------------------------ stationary
def _stop_and_go(tmp_path, name="s.gpx", heading_before=0, heading_after=90):
    pts = H.line(heading_before, 10, n=6)  # t 0..5, moving
    last = pts[-1]
    stop = [Pt(6 + i, last.lat, last.lon) for i in range(10)]  # t 6..15 same place
    p = (last.lat, last.lon)
    go = []
    for i in range(1, 6):
        p = H.step(*p, heading_after, 10)
        go.append(Pt(15 + i, *p))
    return write_gpx(tmp_path / name, pts + stop + go), pts, go


def test_stationary_points_speed_zero_and_heading_finite(tmp_path):
    f, _, _ = _stop_and_go(tmp_path)
    lk = build(f)
    for t in (7.5, 10, 12.3, 14.5):
        r = lk(t)
        assert H.finite(r), (t, r)
        assert r.speed < 0.5
        assert 0 <= r.heading < 360


def test_stationary_keeps_last_heading(tmp_path):
    f, _, _ = _stop_and_go(tmp_path)
    lk = build(f)
    for t in (7.5, 10, 14.5):
        assert angdiff(lk(t).heading, 0) < 1.5, (t, lk(t).heading)


def test_heading_resumes_after_stop(tmp_path):
    f, _, _ = _stop_and_go(tmp_path)
    lk = build(f)
    assert angdiff(lk(17.5).heading, 90) < 1.5


def test_track_starting_stationary_is_finite(tmp_path):
    p = (38.0, -77.0)
    pts = [Pt(i, *p) for i in range(5)]
    q = p
    for i in range(5, 10):
        q = H.step(*q, 180, 8)
        pts.append(Pt(i, *q))
    lk = build(write_gpx(tmp_path / "ss.gpx", pts))
    r = lk(1.5)
    assert H.finite(r) and 0 <= r.heading < 360 and r.speed < 0.5
    assert angdiff(lk(7.5).heading, 180) < 1.5


def test_fully_stationary_track_is_finite(tmp_path):
    pts = [Pt(i, 38.0, -77.0) for i in range(10)]
    r = build(write_gpx(tmp_path / "z.gpx", pts))(4.5)
    assert H.finite(r) and r.speed < 0.5 and 0 <= r.heading < 360


def test_stationary_with_gps_jitter_does_not_blow_up(tmp_path):
    pts = [Pt(i, 38.0 + (i % 2) * 1e-7, -77.0) for i in range(20)]
    r = build(write_gpx(tmp_path / "j.gpx", pts))(7.5)
    assert H.finite(r) and 0 <= r.heading < 360
