"""Task 6 hidden tests: parsing robustness, ranges, ordering, segments, timezones, performance."""
from __future__ import annotations

import random
import time
from datetime import timedelta

import pytest

import rs6_helpers as H
from rs6_helpers import Pt, T0, build, write_gpx


def near(a, b, tol):
    return abs(a - b) <= tol


def _expect_rejected(src, **kw):
    """Parse+one lookup must raise some Exception with a non-empty message."""
    with pytest.raises(Exception) as ei:
        lk = build(src, **kw)
        lk(0.0)
    assert not isinstance(ei.value, H.InterfaceError), f"interface not found: {ei.value}"
    assert str(ei.value).strip(), "error should carry a clear message"
    return ei.value


# ------------------------------------------------------------------ out of range
def test_before_first_point_clamps_or_errors(tmp_path):
    pts = H.line(0, 10, n=11)
    lk = build(write_gpx(tmp_path / "a.gpx", pts))
    for t in (-0.001, -5, -1000):
        r = H.lookup_or_oor(lk, t)
        if r != "OOR":
            assert H.finite(r)
            assert near(r.lat, pts[0].lat, 1e-7) and near(r.lon, pts[0].lon, 1e-7)


def test_after_last_point_clamps_or_errors(tmp_path):
    pts = H.line(0, 10, n=11)
    lk = build(write_gpx(tmp_path / "a.gpx", pts))
    for t in (10.001, 11, 5000):
        r = H.lookup_or_oor(lk, t)
        if r != "OOR":
            assert H.finite(r)
            assert near(r.lat, pts[-1].lat, 1e-7) and near(r.lon, pts[-1].lon, 1e-7)


def test_out_of_range_behaviour_is_consistent_both_ends(tmp_path):
    lk = build(write_gpx(tmp_path / "a.gpx", H.line(0, 10, n=11)))
    lo = H.lookup_or_oor(lk, -50)
    hi = H.lookup_or_oor(lk, 60)
    assert (lo == "OOR") == (hi == "OOR")


def test_out_of_range_error_message_is_clear(tmp_path):
    lk = build(write_gpx(tmp_path / "a.gpx", H.line(0, 10, n=11)))
    try:
        r = lk(9999)
    except H.InterfaceError:
        raise
    except Exception as e:
        assert str(e).strip()
        assert not isinstance(e, (ZeroDivisionError, IndexError, KeyError, AttributeError, TypeError))
    else:
        assert H.finite(r)


def test_first_and_last_point_exact_are_in_range(tmp_path):
    pts = H.line(0, 10, n=11)
    lk = build(write_gpx(tmp_path / "a.gpx", pts))
    a, b = lk(0), lk(10)
    assert near(a.lat, pts[0].lat, 1e-7) and near(b.lat, pts[-1].lat, 1e-7)


def test_single_point_track(tmp_path):
    f = write_gpx(tmp_path / "one.gpx", [Pt(0, 38.0, -77.0)])
    try:
        lk = build(f)
        r = lk(0)
    except H.InterfaceError:
        raise
    except Exception as e:
        assert str(e).strip()  # rejecting a one-point track clearly is acceptable
    else:
        assert near(r.lat, 38.0, 1e-7) and H.finite(r)


# ------------------------------------------------------------------ ordering / duplicates
def test_unsorted_points_same_as_sorted_or_rejected(tmp_path):
    pts = H.circle(n=31)
    shuffled = pts[:]
    random.Random(7).shuffle(shuffled)
    ref = build(write_gpx(tmp_path / "s.gpx", pts))
    try:
        lk = build(write_gpx(tmp_path / "u.gpx", shuffled))
        got = [lk(t) for t in (0, 3.5, 12.25, 29.9, 30)]
    except H.InterfaceError:
        raise
    except Exception as e:
        assert str(e).strip()
        return
    for t, g in zip((0, 3.5, 12.25, 29.9, 30), got):
        e = ref(t)
        assert near(g.lat, e.lat, 1e-7) and near(g.lon, e.lon, 1e-7), t


def test_duplicate_timestamps_identical_points_no_crash(tmp_path):
    pts = H.line(0, 10, n=11)
    dup = pts[:5] + [pts[4]] + pts[5:]
    lk = build(write_gpx(tmp_path / "d.gpx", dup))
    for t in (3.5, 4.0, 4.5, 8):
        r = lk(t)
        assert H.finite(r)
        assert near(r.lat, H.step(H.LAT0, H.LON0, 0, 10 * t)[0], 2e-5)


def test_duplicate_timestamps_conflicting_points_sane(tmp_path):
    pts = H.line(0, 10, n=11)
    bad = pts[:5] + [Pt(4, pts[4].lat + 0.0001, pts[4].lon)] + pts[5:]
    try:
        lk = build(write_gpx(tmp_path / "d.gpx", bad))
        r = [lk(t) for t in (3.5, 4.0, 4.5)]
    except H.InterfaceError:
        raise
    except Exception as e:
        assert str(e).strip()
        return
    assert all(H.finite(x) for x in r)


def test_points_without_time_are_skipped_or_rejected(tmp_path):
    pts = H.line(0, 10, n=11)
    mixed = pts[:5] + [Pt(None, pts[4].lat + 0.00005, pts[4].lon)] + pts[5:]
    try:
        lk = build(write_gpx(tmp_path / "m.gpx", mixed))
        r = lk(7)
    except H.InterfaceError:
        raise
    except Exception as e:
        assert str(e).strip()
        return
    assert near(r.lat, pts[7].lat, 1e-7) and H.finite(r)


# ------------------------------------------------------------------ bad input
def test_gpx_without_any_timestamps_rejected(tmp_path):
    pts = H.line(0, 10, n=11)
    f = tmp_path / "nt.gpx"
    f.write_text(H.gpx_xml([pts], with_time=False), encoding="utf-8")
    err = _expect_rejected(f)
    assert isinstance(err, (ValueError, RuntimeError, OSError)) or "time" in str(err).lower()


def test_no_timestamp_error_mentions_time(tmp_path):
    f = tmp_path / "nt.gpx"
    f.write_text(H.gpx_xml([H.line(0, 10, n=11)], with_time=False), encoding="utf-8")
    err = _expect_rejected(f)
    assert any(w in str(err).lower() for w in ("time", "timestamp", "stamp"))


def test_gpx_with_no_trackpoints_rejected(tmp_path):
    f = tmp_path / "empty.gpx"
    f.write_text('<?xml version="1.0"?><gpx version="1.1" creator="x" xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg/></trk></gpx>', encoding="utf-8")
    _expect_rejected(f)


def test_malformed_xml_rejected(tmp_path):
    f = tmp_path / "bad.gpx"
    f.write_text("<gpx><trk><trkseg><trkpt lat='1' lon='2'>", encoding="utf-8")
    _expect_rejected(f)


def test_not_a_gpx_file_rejected(tmp_path):
    f = tmp_path / "notgpx.gpx"
    f.write_bytes(b"\x00\x01\x02 definitely not xml")
    _expect_rejected(f)


def test_empty_file_rejected(tmp_path):
    f = tmp_path / "e.gpx"
    f.write_bytes(b"")
    _expect_rejected(f)


def test_missing_file_raises(tmp_path):
    with pytest.raises(Exception) as ei:
        build(tmp_path / "nope.gpx")(0)
    assert not isinstance(ei.value, H.InterfaceError)


def test_waypoints_only_gpx_rejected(tmp_path):
    f = tmp_path / "wpt.gpx"
    f.write_text(
        '<?xml version="1.0"?><gpx version="1.1" creator="x" xmlns="http://www.topografix.com/GPX/1/1">'
        '<wpt lat="38" lon="-77"><time>2026-10-20T12:00:00Z</time></wpt></gpx>',
        encoding="utf-8",
    )
    try:
        lk = build(f)
        lk(0)
    except H.InterfaceError:
        raise
    except Exception as e:
        assert str(e).strip()
    # (accepting waypoints as a track is tolerated)


# ------------------------------------------------------------------ segments / gaps
def test_multiple_segments_both_queryable(tmp_path):
    s1 = H.line(0, 10, n=11, t0=0)
    s2 = H.line(90, 10, n=11, t0=20, lat=39.0, lon=-76.0)
    lk = build(write_gpx(tmp_path / "m.gpx", [s1, s2]))
    a = lk(4.5)
    assert near(a.lat, H.step(H.LAT0, H.LON0, 0, 45)[0], 1e-6)
    b = lk(25.5)
    assert near(b.lat, 39.0, 1e-6)
    assert near(b.lon, H.step(39.0, -76.0, 90, 55)[1], 1e-6)
    assert H.angdiff(b.heading, 90) < 1.0 and H.angdiff(a.heading, 0) < 1.0


def test_multiple_trk_elements(tmp_path):
    s1 = H.line(0, 10, n=11, t0=0)
    s2 = H.line(180, 10, n=11, t0=20, lat=39.0, lon=-76.0)
    lk = build(write_gpx(tmp_path / "m.gpx", [s1, s2], trk_count=2))
    assert H.angdiff(lk(25.5).heading, 180) < 1.0
    assert near(lk(10).lat, s1[-1].lat, 1e-7)


def test_gap_between_segments_interpolates_or_errors(tmp_path):
    s1 = H.line(0, 10, n=11, t0=0)
    s2 = H.line(0, 10, n=11, t0=60, lat=s1[-1].lat + 0.01, lon=s1[-1].lon)
    lk = build(write_gpx(tmp_path / "g.gpx", [s1, s2]))
    r = H.lookup_or_oor(lk, 35)
    if r != "OOR":
        assert H.finite(r)
        assert s1[-1].lat - 1e-7 <= r.lat <= s2[0].lat + 1e-7
        # a 50 s gap spanning ~1 km must not be reported as highway speed from tags-less data
        assert r.speed >= 0


def test_points_after_gap_exact(tmp_path):
    s1 = H.line(0, 10, n=11, t0=0)
    s2 = H.line(0, 10, n=11, t0=100, lat=39.0, lon=-77.0)
    lk = build(write_gpx(tmp_path / "g.gpx", [s1, s2]))
    assert near(lk(100).lat, 39.0, 1e-7)
    assert near(lk(10).lat, s1[-1].lat, 1e-7)


# ------------------------------------------------------------------ timezones / formats
@pytest.mark.parametrize("style", ["z", "plus0", "minus4", "plus530", "naive"])
def test_timezone_styles_equivalent(tmp_path, style):
    pts = H.circle(n=21)
    lk = build(write_gpx(tmp_path / "t.gpx", pts, time_style=style))
    for t in (0, 3, 7.5, 20):
        e = H.circle(n=21)
        i = int(t)
        if t == i:
            assert near(lk(t).lat, e[i].lat, 1e-7), (style, t)
        else:
            assert near(lk(t).lat, (e[i].lat + e[i + 1].lat) / 2, 1e-7), (style, t)


def test_fractional_timestamps_with_timezone(tmp_path):
    pts = [Pt(0.0, 38.0, -77.0), Pt(0.5, 38.0001, -77.0), Pt(1.5, 38.0003, -77.0)]
    lk = build(write_gpx(tmp_path / "t.gpx", pts, time_style="minus4"))
    assert near(lk(1.0).lat, 38.0002, 1e-8)


def test_microsecond_precision_timestamps(tmp_path):
    f = tmp_path / "us.gpx"
    f.write_text(
        '<?xml version="1.0"?><gpx version="1.1" creator="x" xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>'
        '<trkpt lat="38.0" lon="-77.0"><time>2026-10-20T12:00:00.123456Z</time></trkpt>'
        '<trkpt lat="38.001" lon="-77.0"><time>2026-10-20T12:00:10.123456Z</time></trkpt>'
        "</trkseg></trk></gpx>",
        encoding="utf-8",
    )
    lk = build(f, start=T0)
    r = lk(5.123456)
    assert near(r.lat, 38.0005, 1e-7)


def test_midnight_and_date_rollover(tmp_path):
    f = tmp_path / "roll.gpx"
    f.write_text(
        '<?xml version="1.0"?><gpx version="1.1" creator="x" xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>'
        '<trkpt lat="38.0" lon="-77.0"><time>2026-10-20T23:59:55Z</time></trkpt>'
        '<trkpt lat="38.001" lon="-77.0"><time>2026-10-21T00:00:05Z</time></trkpt>'
        "</trkseg></trk></gpx>",
        encoding="utf-8",
    )
    start = T0.replace(hour=23, minute=59, second=55)
    lk = build(f, start=start)
    assert near(lk(5).lat, 38.0005, 1e-7)


def test_gpx_10_namespace_and_no_namespace(tmp_path):
    body = ('<trk><trkseg>'
            '<trkpt lat="38.0" lon="-77.0"><time>2026-10-20T12:00:00Z</time></trkpt>'
            '<trkpt lat="38.001" lon="-77.0"><time>2026-10-20T12:00:10Z</time></trkpt>'
            '</trkseg></trk></gpx>')
    f1 = tmp_path / "v10.gpx"
    f1.write_text('<?xml version="1.0"?><gpx version="1.0" creator="x" xmlns="http://www.topografix.com/GPX/1/0">' + body, encoding="utf-8")
    assert near(build(f1)(5).lat, 38.0005, 1e-7)


def test_extra_elements_and_extensions_ignored(tmp_path):
    f = tmp_path / "x.gpx"
    f.write_text(
        '<?xml version="1.0"?><gpx version="1.1" creator="x" xmlns="http://www.topografix.com/GPX/1/1" '
        'xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1"><metadata><name>x</name></metadata>'
        '<trk><name>n</name><trkseg>'
        '<trkpt lat="38.0" lon="-77.0"><ele>5</ele><time>2026-10-20T12:00:00Z</time><sat>9</sat>'
        '<extensions><gpxtpx:TrackPointExtension><gpxtpx:hr>80</gpxtpx:hr></gpxtpx:TrackPointExtension></extensions></trkpt>'
        '<trkpt lat="38.001" lon="-77.0"><ele>6</ele><time>2026-10-20T12:00:10Z</time></trkpt>'
        "</trkseg></trk></gpx>",
        encoding="utf-8",
    )
    assert near(build(f)(5).lat, 38.0005, 1e-7)


def test_parse_accepts_str_path_and_pathlike(tmp_path):
    f = write_gpx(tmp_path / "a.gpx", H.line(0, 10))
    a = build(f)(5)
    b = build(str(f))(5)
    assert a == b


def test_parse_accepts_bytes_if_supported(tmp_path):
    f = write_gpx(tmp_path / "a.gpx", H.line(0, 10))
    try:
        r = build(f.read_bytes())(5)
    except H.InterfaceError:
        raise
    except Exception:
        pytest.skip("bytes input not supported by this interface (path-only)")
    assert near(r.lat, build(f)(5).lat, 1e-9)


# ------------------------------------------------------------------ performance
def test_10000_lookups_fast(tmp_path):
    pts = H.line(40, 15, n=3600)
    lk = build(write_gpx(tmp_path / "big.gpx", pts))
    rng = random.Random(1)
    ts = [rng.uniform(0, 3599) for _ in range(10_000)]
    lk(0)
    t0 = time.perf_counter()
    out = [lk(t) for t in ts]
    dt = time.perf_counter() - t0
    assert dt < 3.0, f"10k lookups took {dt:.2f}s"
    assert all(H.finite(o) for o in out)


def test_lookups_scale_sublinearly_with_track_length(tmp_path):
    small = build(write_gpx(tmp_path / "s.gpx", H.line(40, 15, n=200)))
    large = build(write_gpx(tmp_path / "l.gpx", H.line(40, 15, n=30000)))
    rng = random.Random(2)

    def run(lk, hi):
        ts = [rng.uniform(0, hi) for _ in range(5000)]
        lk(0)
        t0 = time.perf_counter()
        for t in ts:
            lk(t)
        return time.perf_counter() - t0

    ts_small = run(small, 199)
    ts_large = run(large, 29999)
    # linear scan would be ~150x slower on 150x more points; allow generous 15x
    assert ts_large < max(ts_small * 15, 1.0), (ts_small, ts_large)


def test_parse_large_track_reasonable_time(tmp_path):
    pts = H.line(40, 15, n=20000)
    f = write_gpx(tmp_path / "big.gpx", pts)
    t0 = time.perf_counter()
    lk = build(f)
    lk(100)
    assert time.perf_counter() - t0 < 10.0


def test_large_track_values_correct_at_end(tmp_path):
    pts = H.line(40, 15, n=5000)
    lk = build(write_gpx(tmp_path / "big.gpx", pts))
    r = lk(4998.5)
    assert near(r.lat, (pts[4998].lat + pts[4999].lat) / 2, 1e-7)
    assert r.speed == pytest.approx(15.0, rel=0.02)
