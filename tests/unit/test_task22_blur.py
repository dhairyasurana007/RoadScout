"""Task 22: the face / licence plate blur on the exported segment, the provider contract, fail closed, no leftovers."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

import rs22_helpers as H
from rs22_helpers import (FACE, PLATE, REF, SRC_DUR, SRC_FPS, SRC_H, SRC_W, StubProvider, assert_no_paths,
                          decode_frames, lap_var, luma, probe, rs22_env, rs22_std)  # noqa: F401

pytestmark = pytest.mark.filterwarnings("ignore")


def _ratios(frame, i, half):
    """(face_var, plate_var, ref_var) of the Laplacian variance in the given half (0 = raw, 1 = annotated); the
    reference always comes from the raw half, in a region that is neither masked nor annotated."""
    g = luma(frame)
    off = SRC_W * half
    face = lap_var(g[:, off:off + SRC_W], FACE)
    plate = lap_var(g[:, off:off + SRC_W], PLATE)
    ref = lap_var(g[:, :SRC_W], REF)
    return face, plate, ref


# ------------------------------------------------------------------ provider contract

def test_provider_is_asked_once_with_the_short_segment_and_the_blur_prompts(rs22_std):
    s = rs22_std
    assert len(s.stub.calls) == 1, "the provider must be called once per segment, got %d" % len(s.stub.calls)
    c = s.stub.calls[0]
    assert c["exists"], "the provider must receive an existing video file"
    assert c["suffix"] in H.VIDEO_EXT, c["path"]
    assert set(c["prompts"]) == {"face", "license plate"}, c["prompts"]
    assert "probe_error" not in c, c
    assert 7.0 <= c["dur"] <= 9.0, "the segment sent to the provider must be the short window, got %.2f s" % c["dur"]
    assert c["dur"] < SRC_DUR - 4, "the whole drive must never be sent to the provider"
    assert os.path.normcase(os.path.abspath(c["path"])) != os.path.normcase(
        os.path.abspath(str(s.env.video_file(s.did)))), "the original upload must not be sent to the provider"


def test_segment_sent_to_the_provider_is_gone_after_the_export(rs22_std):
    c = rs22_std.stub.calls[0]
    assert not Path(c["path"]).exists(), "an unblurred intermediate segment was left behind: %s" % Path(c["path"]).name


def test_only_the_final_mp4_is_left_in_the_exports_folder_and_no_stray_video_in_storage(rs22_std):
    s = rs22_std
    files = s.env.export_files(s.did)
    assert [p.suffix.lower() for p in files if p.suffix.lower() != ".json"] == [".mp4"], [p.name for p in files]
    for p in files:
        assert not re.search(r"tmp|temp|part|raw|segment|unblurred|intermediate", p.name, re.I), p.name
    assert not s.env.stray_videos([s.env.video_file(s.did)]), s.env.stray_videos([s.env.video_file(s.did)])
    assert [p.name for p in s.env.storage_files() if p.suffix.lower() in H.IMG_EXT] == []


def test_exported_clip_lives_in_a_drive_keyed_folder_under_storage(rs22_std):
    s = rs22_std
    f = s.video
    assert f.parent == s.env.storage / s.did / "exports", f
    served = s.env.client.get(s.env.video_url(s.rec))
    assert served.status_code == 200 and served.content == f.read_bytes()


def test_original_upload_is_not_modified(rs22_env):
    env = rs22_env
    env.use_provider(StubProvider("dense"))
    did = env.add_drive("d22a")
    hid = env.seed(did)
    before = env.uploads_sha([did])
    size = env.video_file(did).stat().st_size
    env.export_ok(hid)
    assert env.uploads_sha([did]) == before
    assert env.video_file(did).stat().st_size == size


# ------------------------------------------------------------------ blur in both halves

def test_face_and_plate_regions_are_blurred_in_both_halves(rs22_std):
    s = rs22_std
    fr = s.frames
    n = fr.shape[0]
    for i in (8, n // 2, n - 8):
        for half in (0, 1):
            face, plate, ref = _ratios(fr[i], i, half)
            assert ref > 400, "test clip lost its detail (ref %.0f): codec or test problem" % ref
            assert face < 0.3 * ref, "frame %d half %d: face region not blurred (%.0f vs %.0f)" % (i, half, face, ref)
            assert plate < 0.3 * ref, "frame %d half %d: plate region not blurred (%.0f vs %.0f)" % (i, half, plate, ref)


def test_blur_is_local_the_rest_of_the_raw_half_keeps_its_detail(rs22_std):
    s = rs22_std
    fr = s.frames
    g = luma(fr[fr.shape[0] // 2])
    ref = lap_var(g[:, :SRC_W], REF)
    other = lap_var(g[:, :SRC_W], (180, 20, 224, 70))
    assert other > 0.6 * ref, "blur spread outside the masks (%.0f vs %.0f)" % (other, ref)
    other_r = lap_var(g[:, SRC_W:], (200, 5, 244, 45))
    assert other_r > 0.4 * ref


def test_blur_only_in_the_frames_the_masks_cover(rs22_env):
    env = rs22_env
    stub = env.use_provider(StubProvider("first_half"))
    did = env.add_drive("d22a")
    hid = env.seed(did, best_t=9.0, others=())
    env.export_ok(hid)
    fr = decode_frames(env.export_videos(did)[0])
    n = fr.shape[0]
    early, late = int(n * 0.2), int(n * 0.8)
    for half in (0, 1):
        face, plate, ref = _ratios(fr[early], early, half)
        assert face < 0.3 * ref and plate < 0.3 * ref, "covered frame %d not blurred in half %d" % (early, half)
    face, plate, ref = _ratios(fr[late], late, 0)
    assert face > 0.6 * ref and plate > 0.6 * ref, "frame %d has no mask and must stay sharp" % late
    face, plate, ref = _ratios(fr[late], late, 1)
    assert face > 0.5 * ref and plate > 0.5 * ref


def test_masks_of_another_resolution_are_scaled_to_the_segment(rs22_env):
    env = rs22_env
    env.use_provider(StubProvider("half"))
    did = env.add_drive("d22a")
    hid = env.seed(did, best_t=9.0, others=())
    env.export_ok(hid)
    fr = decode_frames(env.export_videos(did)[0])
    i = fr.shape[0] // 2
    for half in (0, 1):
        face, plate, ref = _ratios(fr[i], i, half)
        assert face < 0.3 * ref and plate < 0.3 * ref, (half, face, plate, ref)


def test_no_faces_or_plates_found_is_not_an_error_and_nothing_is_blurred(rs22_env):
    env = rs22_env
    stub = env.use_provider(StubProvider("empty"))
    did = env.add_drive("d22a")
    hid = env.seed(did, best_t=9.0, others=())
    rec = env.export_ok(hid)
    assert len(stub.calls) == 1
    fr = decode_frames(env.export_videos(did)[0])
    i = fr.shape[0] // 2
    face, plate, ref = _ratios(fr[i], i, 0)
    assert face > 0.6 * ref and plate > 0.6 * ref


def test_works_with_the_fake_provider_without_key_or_network(rs22_env):
    env = rs22_env
    assert not os.environ.get("META_API_KEY")
    did = env.add_drive("d22a")
    hid = env.seed(did)
    env.add_det(did, 8.0, H.OVERLAY_BOX)
    rec = env.export_ok(hid)
    f = env.export_videos(did)[0]
    assert vstreams_ok(f)
    assert str(rec.get("status")).lower() in H.OK_STATUS


def vstreams_ok(f):
    v = H.vstreams(probe(f))
    return len(v) == 1 and v[0]["codec_name"] == "h264" and int(v[0]["width"]) == 2 * SRC_W


# ------------------------------------------------------------------ fail closed

FAIL_MODES = ["raise", "none", "string", "bad_counts", "bad_size", "spend"]


@pytest.mark.parametrize("mode", FAIL_MODES)
def test_fail_closed_no_export_file_and_a_clear_error(rs22_env, mode):
    env = rs22_env
    stub = env.use_provider(StubProvider(mode))
    did = env.add_drive("d22a")
    hid = env.seed(did)
    before = env.uploads_sha([did])
    r, rec = env.export_resp(hid)
    failed_record = rec is not None and str(rec.get("status", "")).strip().lower() in H.FAILED_STATUS
    assert r.status_code >= 400 or failed_record, \
        "provider mode %r must fail closed, got %s %s" % (mode, r.status_code, r.text[:300])
    assert not (r.status_code in (200, 201, 202) and rec is not None
                and str(rec.get("status", "")).strip().lower() in H.OK_STATUS), "unblurred export reported as ok"
    assert env.export_videos(did) == [], "an export video exists after a failed blur: %s" % env.export_videos(did)
    assert env.stray_videos([env.video_file(did)]) == []
    # the unblurred segment must be gone, too
    for c in stub.calls:
        assert not Path(c["path"]).exists(), "unblurred intermediate segment left behind"
    # the error message is clear and holds no path
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"detail": r.text}
    msg = " ".join(H.all_strings(body)).strip()
    assert len(msg) >= 5, "no error message"
    if r.status_code >= 400:
        assert_no_paths({"detail": body.get("detail") if isinstance(body, dict) else body}, env)
    elif rec is not None:
        assert_no_paths(rec, env)
        assert rec.get("error") or rec.get("message") or rec.get("detail"), "failed record without an error text"
    assert "traceback" not in msg.lower()
    assert str(env.tmp) not in msg
    # nothing downloadable and nothing ok in the listing
    ok = [e for e in env.list_for(hid) if str(e.get("status", "")).strip().lower() in H.OK_STATUS]
    assert ok == [], ok
    assert env.uploads_sha([did]) == before


def test_spend_cap_error_mentions_the_cap(rs22_env):
    env = rs22_env
    env.use_provider(StubProvider("spend"))
    did = env.add_drive("d22a")
    hid = env.seed(did)
    r, rec = env.export_resp(hid)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"detail": r.text}
    text = " ".join(H.all_strings(body)).lower()
    if rec is not None:
        text += " " + " ".join(H.all_strings(rec)).lower()
    assert re.search(r"cap|limit|budget|spend", text), text[:300]
    assert env.export_videos(did) == []


def test_a_failed_export_does_not_block_a_later_good_one_and_leaves_nothing_stale(rs22_env):
    env = rs22_env
    bad = env.use_provider(StubProvider("raise"))
    did = env.add_drive("d22a")
    hid = env.seed(did)
    r, rec = env.export_resp(hid)
    assert env.export_videos(did) == []
    good = env.use_provider(StubProvider("dense"))
    rec2 = env.export_ok(hid)
    assert len(env.export_videos(did)) == 1
    assert [p for p in env.export_files(did) if p.suffix.lower() not in (".mp4", ".json")] == []
    served = env.client.get(env.video_url(rec2))
    assert served.status_code == 200 and len(served.content) > 1000


def test_no_blur_call_means_no_export_the_provider_failure_is_not_ignored(rs22_env):
    """A provider that returns frames whose masks are all unusable must not produce an unblurred clip."""
    env = rs22_env
    env.use_provider(StubProvider("bad_counts"))
    did = env.add_drive("d22a")
    hid = env.seed(did)
    r, rec = env.export_resp(hid)
    assert env.export_videos(did) == []
    ok = [e for e in env.list_for(hid) if str(e.get("status", "")).strip().lower() in H.OK_STATUS]
    assert ok == []
