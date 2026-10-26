"""Drive upload endpoint: accepts video + GPX, creates a drive record, reads the start time."""
from datetime import timedelta

import pytest

from rs4_fixtures import *  # noqa: F401,F403
from rs4_helpers import (
    CLIP_KINDS,
    DEFAULT_CREATION_TIME,
    db_rows,
    drive_id,
    effective_key,
    list_drives,
    make_gpx,
    offset_key,
    ok_status,
    parse_ts,
    secs,
    sha,
    started_key,
    storage_hashes,
    unwrap_drive,
    upload,
)

META_TIME = parse_ts(DEFAULT_CREATION_TIME)
KNOWN_INITIAL_STATUSES = {"uploaded", "queued", "pending", "created", "new", "received", "ready"}


def _vid(data, name="clip.mp4", ctype="video/mp4"):
    return (name, data, ctype)


def _gpx(data=None, name="track.gpx", ctype="application/gpx+xml"):
    return (name, data if data is not None else make_gpx(), ctype)


# ------------------------------------------------------------------ happy path


def test_upload_creates_drive_with_id_and_status(rs4_client, rs4_api, rs4_clip):
    r = upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264")), _gpx())
    assert ok_status(r.status_code), r.text
    assert r.headers["content-type"].startswith("application/json")
    d = unwrap_drive(r.json())
    assert drive_id(d) not in (None, "")
    status = d.get("status")
    assert isinstance(status, str) and status, d
    assert status == status.lower()
    assert status in KNOWN_INITIAL_STATUSES, f"unexpected initial status {status!r}"


@pytest.mark.parametrize("kind", sorted(CLIP_KINDS))
def test_accepts_common_containers_and_codecs(kind, rs4_client, rs4_api, rs4_clip):
    ext = CLIP_KINDS[kind][0]
    data = rs4_clip(kind)
    r = upload(rs4_client, rs4_api, _vid(data, f"clip.{ext}", "application/octet-stream"), _gpx())
    assert ok_status(r.status_code), f"{kind}: {r.status_code} {r.text}"
    assert drive_id(unwrap_drive(r.json()))


def test_accepts_iphone_style_hevc_mov_with_octet_stream_type(rs4_client, rs4_api, rs4_clip):
    data = rs4_clip("mov-hevc")
    r = upload(rs4_client, rs4_api, ("IMG_0001.MOV", data, "application/octet-stream"), _gpx())
    assert ok_status(r.status_code), r.text


def test_accepts_dashcam_style_uppercase_names(rs4_client, rs4_api, rs4_clip):
    data = rs4_clip("mp4-h264")
    r = upload(rs4_client, rs4_api, ("20260314_123456_A.MP4", data, "video/mp4"),
               ("TRACK.GPX", make_gpx(), "text/xml"))
    assert ok_status(r.status_code), r.text


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_accepts_gpx_1_0_and_1_1(version, rs4_client, rs4_api, rs4_clip):
    r = upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264")), _gpx(make_gpx(version)))
    assert ok_status(r.status_code), r.text


def test_content_type_is_not_trusted_over_content(rs4_client, rs4_api, rs4_clip):
    """A real video sent as application/octet-stream and a real GPX as text/plain still work."""
    r = upload(rs4_client, rs4_api, ("movie", rs4_clip("mp4-h264"), "application/octet-stream"),
               ("log.xml", make_gpx(), "text/plain"))
    assert ok_status(r.status_code), r.text


def test_two_uploads_get_distinct_ids(rs4_client, rs4_api, rs4_clip):
    a = upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264", "red")), _gpx())
    b = upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264", "green")), _gpx())
    assert ok_status(a.status_code) and ok_status(b.status_code)
    ida, idb = drive_id(unwrap_drive(a.json())), drive_id(unwrap_drive(b.json()))
    assert ida != idb


def test_uploads_with_same_filename_do_not_overwrite_each_other(
        rs4_client, rs4_api, rs4_clip, rs4_env):
    v1, v2 = rs4_clip("mp4-h264", "red"), rs4_clip("mp4-h264", "blue")
    assert sha(v1) != sha(v2)
    assert ok_status(upload(rs4_client, rs4_api, _vid(v1, "clip.mp4"), _gpx()).status_code)
    assert ok_status(upload(rs4_client, rs4_api, _vid(v2, "clip.mp4"), _gpx()).status_code)
    hashes = storage_hashes(rs4_env)
    assert sha(v1) in hashes and sha(v2) in hashes, "second upload clobbered the first"
    assert len(db_rows(rs4_env)) == 2


def test_hostile_filenames_stay_inside_storage(rs4_client, rs4_api, rs4_clip, rs4_env, tmp_path):
    v = rs4_clip("mp4-h264")
    r = upload(rs4_client, rs4_api, ("../../evil.mp4", v, "video/mp4"),
               ("..\\..\\evil.gpx", make_gpx(), "application/gpx+xml"))
    # Either rejected or sanitized, but never written outside the storage dir.
    assert r.status_code < 500, r.text
    for stray in ("evil.mp4", "evil.gpx"):
        assert not (tmp_path / stray).exists()
        assert not (tmp_path.parent / stray).exists()
    if ok_status(r.status_code):
        assert sha(v) in storage_hashes(rs4_env)


def test_unicode_and_spaces_in_filenames(rs4_client, rs4_api, rs4_clip):
    r = upload(rs4_client, rs4_api, ("my drive éè 1.mp4", rs4_clip("mp4-h264"), "video/mp4"),
               ("my track ü.gpx", make_gpx(), "application/gpx+xml"))
    assert ok_status(r.status_code), r.text


def test_upload_works_without_meta_api_key(rs4_client, rs4_api, rs4_clip, monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    assert ok_status(upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264")), _gpx()).status_code)


# ----------------------------------------------------- start time from metadata


@pytest.mark.parametrize("kind,ext", [("mp4-h264", "mp4"), ("mov-hevc", "mov")])
def test_start_time_is_read_from_video_metadata(kind, ext, rs4_client, rs4_api, rs4_clip):
    data = rs4_clip(kind, "red", "2026-03-14T12:34:56Z")
    r = upload(rs4_client, rs4_api, _vid(data, f"clip.{ext}"), _gpx())
    assert ok_status(r.status_code), r.text
    d = unwrap_drive(r.json())
    key = started_key(d)
    assert key, f"no started_at-like key in {d}"
    got = parse_ts(d[key])
    assert got is not None, d
    assert abs(secs(got, META_TIME)) <= 1.5, f"started_at {got} != metadata {META_TIME}"


def test_start_time_follows_the_metadata_value(rs4_client, rs4_api, rs4_clip):
    data = rs4_clip("mp4-h264", "green", "2025-12-31T23:59:30Z")
    r = upload(rs4_client, rs4_api, _vid(data), _gpx(make_gpx(start="2025-12-31T23:59:30Z")))
    d = unwrap_drive(r.json())
    got = parse_ts(d[started_key(d)])
    assert abs(secs(got, parse_ts("2025-12-31T23:59:30Z"))) <= 1.5


def test_start_time_is_exposed_as_timezone_unambiguous_iso(rs4_client, rs4_api, rs4_clip):
    d = unwrap_drive(upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264")), _gpx()).json())
    value = d[started_key(d)]
    assert isinstance(value, str), "started_at should be an ISO-8601 string"
    assert value.endswith("Z") or "+" in value[10:] or "-" in value[10:], (
        f"started_at {value!r} carries no UTC offset")


def test_missing_creation_time_does_not_break_upload(rs4_client, rs4_api, rs4_clip):
    data = rs4_clip("mp4-h264", "yellow", None)
    r = upload(rs4_client, rs4_api, _vid(data), _gpx())
    assert ok_status(r.status_code), r.text
    d = unwrap_drive(r.json())
    key = started_key(d)
    if key and d[key] is not None:
        assert parse_ts(d[key]) is not None


def test_missing_creation_time_with_manual_offset_still_stores_offset(rs4_client, rs4_api, rs4_clip):
    data = rs4_clip("avi-mpeg4", "yellow", None)
    r = upload(rs4_client, rs4_api, _vid(data, "clip.avi"), _gpx(), offset=7)
    assert ok_status(r.status_code), r.text
    d = unwrap_drive(r.json())
    assert float(d[offset_key(d)]) == pytest.approx(7.0)


# ------------------------------------------------------------- clock offset


def test_offset_defaults_to_zero(rs4_client, rs4_api, rs4_clip):
    d = unwrap_drive(upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264")), _gpx()).json())
    key = offset_key(d)
    assert key, f"no clock offset key in {d}"
    assert d[key] in (0, 0.0, None) or float(d[key]) == 0.0


@pytest.mark.parametrize("offset", [12.5, -3, 3600, 0.25, "0"])
def test_manual_offset_is_stored(offset, rs4_client, rs4_api, rs4_clip):
    r = upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264")), _gpx(), offset=offset)
    assert ok_status(r.status_code), r.text
    d = unwrap_drive(r.json())
    assert float(d[offset_key(d)]) == pytest.approx(float(offset))


@pytest.mark.parametrize("offset", [42.0, -17.5])
def test_manual_offset_is_applied_to_the_start_time(offset, rs4_client, rs4_api, rs4_clip):
    """started_at stays the metadata time and an effective/aligned time = metadata + offset,
    or started_at itself is already shifted by the offset. Either way the shift is applied."""
    r = upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264")), _gpx(), offset=offset)
    d = unwrap_drive(r.json())
    shifted = META_TIME + timedelta(seconds=offset)
    eff = effective_key(d)
    if eff:
        assert abs(secs(parse_ts(d[eff]), shifted)) <= 1.5, d
    else:
        got = parse_ts(d[started_key(d)])
        assert min(abs(secs(got, META_TIME)), abs(secs(got, shifted))) <= 1.5, d


def test_offset_persists_on_get_by_id(rs4_client, rs4_api, rs4_clip):
    created = unwrap_drive(upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264")), _gpx(),
                                  offset=-8.5).json())
    got = rs4_client.get(f"{rs4_api.base}/{drive_id(created)}")
    assert got.status_code == 200
    d = unwrap_drive(got.json())
    assert float(d[offset_key(d)]) == pytest.approx(-8.5)


@pytest.mark.parametrize("bad", ["abc", "1,5", "12s", "nan", "inf", "-inf", "1e999999"])
def test_invalid_offset_is_rejected_without_creating_a_drive(
        bad, rs4_client, rs4_api, rs4_clip, rs4_env):
    r = upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264")), _gpx(), offset=bad)
    assert 400 <= r.status_code < 500, f"offset {bad!r}: {r.status_code} {r.text}"
    assert list_drives(rs4_client, rs4_api) == []
    assert db_rows(rs4_env) == []
    assert storage_hashes(rs4_env) == {}


# ----------------------------------------------------------- rejection paths


def _assert_rejected_cleanly(r, rs4_client, rs4_api, rs4_env):
    assert 400 <= r.status_code < 500, f"{r.status_code} {r.text[:300]}"
    assert list_drives(rs4_client, rs4_api) == [], "a drive was created for a rejected upload"
    assert db_rows(rs4_env) == [], "a drives row was left behind"
    assert storage_hashes(rs4_env) == {}, "files were left in storage for a rejected upload"


def test_missing_video_is_rejected(rs4_client, rs4_api, rs4_env):
    _assert_rejected_cleanly(upload(rs4_client, rs4_api, None, _gpx()), rs4_client, rs4_api, rs4_env)


def test_missing_gpx_is_rejected(rs4_client, rs4_api, rs4_clip, rs4_env):
    r = upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264")), None)
    _assert_rejected_cleanly(r, rs4_client, rs4_api, rs4_env)


def test_empty_request_is_rejected(rs4_client, rs4_api, rs4_env):
    r = rs4_client.post(rs4_api.path)
    _assert_rejected_cleanly(r, rs4_client, rs4_api, rs4_env)


@pytest.mark.parametrize("content", [
    pytest.param(b"", id="empty"),
    pytest.param(b"this is definitely not a video, just some text\n" * 50, id="text"),
    pytest.param(bytes(range(256)) * 64, id="binary"),
    pytest.param(b"<html><body>video?</body></html>", id="html"),
])
def test_non_video_content_is_rejected_even_with_video_name(
        content, rs4_client, rs4_api, rs4_env):
    r = upload(rs4_client, rs4_api, ("clip.mp4", content, "video/mp4"), _gpx())
    _assert_rejected_cleanly(r, rs4_client, rs4_api, rs4_env)


def test_gpx_sent_in_the_video_slot_is_rejected(rs4_client, rs4_api, rs4_env):
    r = upload(rs4_client, rs4_api, ("clip.mp4", make_gpx(), "video/mp4"), _gpx())
    _assert_rejected_cleanly(r, rs4_client, rs4_api, rs4_env)


def test_video_sent_in_the_gpx_slot_is_rejected(rs4_client, rs4_api, rs4_clip, rs4_env):
    v = rs4_clip("mp4-h264")
    r = upload(rs4_client, rs4_api, _vid(v), ("track.gpx", v, "application/gpx+xml"))
    _assert_rejected_cleanly(r, rs4_client, rs4_api, rs4_env)


@pytest.mark.parametrize("content", [
    pytest.param(b"", id="empty"),
    pytest.param(b"hello, not a track", id="text"),
    pytest.param(b"\x00\x01\x02\x03" * 100, id="binary"),
    pytest.param(b'<?xml version="1.0"?><html><body>nope</body></html>', id="html"),
    pytest.param(b'<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document/></kml>', id="kml"),
    pytest.param(b'<gpx version="1.1"><trk><trkseg><trkpt lat="1" lon="2"', id="truncated"),
])
def test_non_gpx_content_is_rejected_even_with_gpx_name(
        content, rs4_client, rs4_api, rs4_clip, rs4_env):
    r = upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264")), ("track.gpx", content, "application/gpx+xml"))
    _assert_rejected_cleanly(r, rs4_client, rs4_api, rs4_env)


def test_rejected_upload_does_not_disturb_existing_drives(rs4_client, rs4_api, rs4_clip, rs4_env):
    good = upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264")), _gpx())
    gid = drive_id(unwrap_drive(good.json()))
    bad = upload(rs4_client, rs4_api, ("x.mp4", b"junk" * 100, "video/mp4"), _gpx())
    assert 400 <= bad.status_code < 500
    drives = list_drives(rs4_client, rs4_api)
    assert [str(drive_id(d)) for d in drives] == [str(gid)]
    assert len(db_rows(rs4_env)) == 1
    assert len(storage_hashes(rs4_env)) == 2  # the good drive's video and GPX only


# ------------------------------------------------------------------- get/list


def test_list_is_empty_before_any_upload(rs4_client, rs4_api):
    assert list_drives(rs4_client, rs4_api) == []


def test_get_drive_by_id_returns_the_created_drive(rs4_client, rs4_api, rs4_clip):
    created = unwrap_drive(upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264")), _gpx(),
                                  offset=2).json())
    r = rs4_client.get(f"{rs4_api.base}/{drive_id(created)}")
    assert r.status_code == 200, r.text
    d = unwrap_drive(r.json())
    assert str(drive_id(d)) == str(drive_id(created))
    assert d["status"] == created["status"]
    assert d[started_key(d)] == created[started_key(created)]
    assert float(d[offset_key(d)]) == pytest.approx(2.0)


def test_get_unknown_drive_is_404(rs4_client, rs4_api):
    assert rs4_client.get(f"{rs4_api.base}/999999").status_code == 404
    assert rs4_client.get(f"{rs4_api.base}/does-not-exist-0000").status_code in (404, 422)


def test_list_contains_every_uploaded_drive(rs4_client, rs4_api, rs4_clip):
    ids = []
    for color in ("red", "green", "blue"):
        r = upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264", color)), _gpx())
        ids.append(str(drive_id(unwrap_drive(r.json()))))
    listed = [str(drive_id(d)) for d in list_drives(rs4_client, rs4_api)]
    assert sorted(listed) == sorted(ids)


def test_list_items_have_status_and_start_fields(rs4_client, rs4_api, rs4_clip):
    upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264")), _gpx())
    (item,) = list_drives(rs4_client, rs4_api)
    assert item.get("status")
    assert started_key(item)
    assert offset_key(item)


def test_drives_survive_a_fresh_client(rs4_app, rs4_client, rs4_api, rs4_clip):
    from fastapi.testclient import TestClient

    created = unwrap_drive(upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264")), _gpx()).json())
    with TestClient(rs4_app) as other:
        r = other.get(f"{rs4_api.base}/{drive_id(created)}")
        assert r.status_code == 200
        assert len(list_drives(other, rs4_api)) == 1


def test_response_does_not_leak_secrets(rs4_client, rs4_api, rs4_clip, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "sentinel-not-a-real-key-123456")
    r = upload(rs4_client, rs4_api, _vid(rs4_clip("mp4-h264")), _gpx())
    assert "sentinel-not-a-real-key-123456" not in r.text
    assert "sentinel-not-a-real-key-123456" not in rs4_client.get(rs4_api.base).text
