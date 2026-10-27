"""Task 22 privacy: every frame blurred inside the returned masks, no unblurred leftovers anywhere (storage tree, system
temp dir, repo), the whole drive is never sent to the provider."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

import rs22_helpers as H
from rs22_helpers import (FACE, PLATE, REF, SRC_DUR, SRC_W, StubProvider, decode_frames, lap_var, luma, rs22_env)  # noqa: F401

pytestmark = pytest.mark.filterwarnings("ignore")


def _walk(root, skip=()):
    out = set()
    for dp, dn, fn in os.walk(str(root), onerror=lambda e: None):
        dn[:] = [d for d in dn if d not in skip and not d.startswith("pytest-of-") and not d.startswith("rs22")]
        for f in fn:
            out.add(os.path.join(dp, f))
        for d in dn:
            out.add(os.path.join(dp, d) + os.sep)
    return out


def _all_frames_blurred(fr):
    bad = []
    for i in range(fr.shape[0]):
        g = luma(fr[i])
        ref = lap_var(g[:, :SRC_W], REF)
        for half in (0, 1):
            sub = g[:, SRC_W * half:SRC_W * (half + 1)]
            for name, box in (("face", FACE), ("plate", PLATE)):
                if lap_var(sub, box) >= 0.3 * ref:
                    bad.append((i, half, name))
    return bad


@pytest.mark.parametrize("mode", ["dense", "sparse", "half"])
def test_every_frame_of_both_halves_is_blurred_inside_every_mask(rs22_env, mode):
    env = rs22_env
    env.use_provider(StubProvider(mode))
    did = env.add_drive("d22a")
    hid = env.seed(did)
    for t in (7.0, 7.5, 8.0, 8.5):
        env.add_det(did, t, H.OVERLAY_BOX)
    env.export_ok(hid)
    fr = decode_frames(env.export_videos(did)[0])
    bad = _all_frames_blurred(fr)
    assert not bad, "unblurred pixels inside a returned mask in %d (frame, half, mask) cases, first: %s" % (len(bad), bad[:6])


@pytest.mark.parametrize("mode", ["dense", "raise", "none", "string", "bad_counts", "bad_size", "spend", "empty"])
def test_no_unblurred_segment_is_left_in_storage_temp_dir_or_repo(rs22_env, mode):
    env = rs22_env
    tmpdir = Path(tempfile.gettempdir())
    skip = {".git", "node_modules", "__pycache__", ".pytest_cache", "frontend"}
    before_tmp = _walk(tmpdir)
    before_repo = _walk(H.ROOT, skip)
    stub = env.use_provider(StubProvider(mode))
    did = env.add_drive("d22a")
    hid = env.seed(did)
    r, rec = env.export_resp(hid)
    after_tmp = _walk(tmpdir)
    after_repo = _walk(H.ROOT, skip)
    new_tmp = sorted(after_tmp - before_tmp)
    new_repo = sorted(after_repo - before_repo)
    assert not new_tmp, "new files/dirs left in the system temp dir: %s" % new_tmp[:10]
    assert not new_repo, "new files/dirs left in the repo tree: %s" % new_repo[:10]
    for c in stub.calls:
        assert not Path(c["path"]).exists()
        assert c["dur"] < SRC_DUR - 4, "whole drive (or most of it) sent to the provider: %.2f s" % c["dur"]
    vids = [p for p in env.storage_files() if p.suffix.lower() in H.VIDEO_EXT]
    ok = mode in ("dense", "empty")
    allowed = {str(env.video_file(did))} | ({str(env.exports_dir(did) / p.name) for p in env.export_videos(did)} if ok else set())
    assert {str(p) for p in vids} <= allowed, [str(p) for p in vids]
    if not ok:
        assert env.export_files(did) == [] or all(p.suffix == ".json" for p in env.export_files(did))
    assert not [p for p in env.storage_files() if p.suffix.lower() in H.IMG_EXT]


def test_whole_drive_is_never_sent_even_with_a_huge_window(rs22_env):
    env = rs22_env
    env.mp.setenv(H.export_env("before"), "500")
    env.mp.setenv(H.export_env("after"), "500")
    stub = env.use_provider(StubProvider("empty"))
    did = env.add_drive("d22b", clip=H.clip_b_bytes(), duration=40.0)
    hid = env.seed(did, best_t=20.0, others=())
    env.export_ok(hid)
    assert len(stub.calls) == 1 and stub.calls[0]["dur"] <= 20.6, stub.calls
