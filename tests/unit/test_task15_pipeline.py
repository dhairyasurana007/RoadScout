"""Task 15: the blurring stage of the pipeline (prompts, saved paths, ordering, accepted-only, failures, reruns)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from rs15_helpers import (BAD_STATUS, CHUNKS, DONE, FAILED, OK_STATUS, face_plate_boxes, is_image_path, lap_var,
                          load_gray, make_env, make_stub, norm_prompt, outside_boxes, rs15_done, rs15_env,  # noqa: F401
                          same_file, sha, status_of, Env)

BAD_RESULTS = {"none": None, "string": "garbage", "number": 42, "object": object()}


def norm_path(p):
    return os.path.normcase(os.path.abspath(str(p)))


def blur_ok(c):
    return str(c.get("blur_status") or "").strip().lower() in OK_STATUS


# ------------------------------------------------------------------ schema / stage list

def test_candidates_table_has_blur_columns(rs15_env):
    cols = set(rs15_env.columns("candidates"))
    assert {"blurred_frame_path", "blurred_crop_path", "blur_status"} <= cols, sorted(cols)


def test_stage_list_has_blurring_between_verifying_and_deduplicating():
    import backend.pipeline as p

    stages = list(getattr(p, "STAGES", ()))
    assert "blurring" in stages, "STAGES = %r" % (stages,)
    assert stages.index("blurring") == stages.index("verifying") + 1
    assert stages.index("deduplicating") == stages.index("blurring") + 1
    # task 19 may append one fix-check stage after deduplicating
    assert stages.index("deduplicating") >= len(stages) - 2


# ------------------------------------------------------------------ what the provider is asked

def test_provider_asked_for_face_and_license_plate_on_saved_images_not_video(rs15_done):
    env, did, stub = rs15_done
    assert len(stub.video_calls) == CHUNKS
    assert stub.image_calls, "the provider was never called on a saved image"
    by_image = {}
    for c in stub.image_calls:
        by_image.setdefault(norm_path(c["path"]), set()).update(norm_prompt(p) for p in c["prompts"])
    for path, prompts in by_image.items():
        assert {"face", "license plate"} <= prompts, (path, prompts)
        assert is_image_path(path)
    # the images sent are exactly the saved candidate images of accepted hazards (best frame and crop)
    saved = set()
    for c in env.accepted(did):
        for k in ("best_frame_path", "crop_path"):
            if c.get(k):
                saved.add(norm_path(env.path_of(c[k])))
    assert saved
    assert set(by_image) <= saved, "blur asked about files that are not saved accepted-candidate images"
    best = {norm_path(env.path_of(c["best_frame_path"])) for c in env.accepted(did)}
    assert best <= set(by_image), "every accepted best frame must be sent to the blur"
    # nothing under the drive's candidates folder
    for path in by_image:
        assert did in Path(path).as_posix()
        assert "candidates" in Path(path).parts


def test_blur_uses_no_video_calls_and_comes_after_verification(rs15_done):
    env, did, stub = rs15_done
    ev = stub.events
    assert ev.count("video") == CHUNKS
    first_image = ev.index("image")
    assert "video" not in ev[first_image:], "a video chunk was sent to the provider during blurring"
    last_verify = len(ev) - 1 - ev[::-1].index("verify")
    assert last_verify < first_image, "blurring started before every candidate was verified"


def test_blurring_stage_is_visible_before_deduplication(rs15_env):
    env = rs15_env
    seen = {}

    def on_image(stub, n, path):
        if n == 0:
            did = env._did
            d = env.q("SELECT stage, status FROM drives WHERE id = ?", (did,))[0]
            seen.update(stage=d["stage"], status=d["status"],
                        verdicts=env.q("SELECT COUNT(*) n FROM verdicts WHERE drive_id = ?", (did,))[0]["n"],
                        cands=env.q("SELECT COUNT(*) n FROM candidates WHERE drive_id = ?", (did,))[0]["n"],
                        linked=env.q("SELECT COUNT(*) n FROM candidates WHERE drive_id = ? AND hazard_id IS NOT NULL",
                                     (did,))[0]["n"],
                        hazards=env.q("SELECT COUNT(*) n FROM hazards")[0]["n"])

    stub = make_stub(on_image=on_image)
    did = env.upload()
    env._did = did
    env.process(stub, did=did)
    assert seen, "no blur call was made"
    assert seen["stage"] == "blurring"
    assert seen["status"] == "processing"
    assert seen["cands"] > 0 and seen["verdicts"] == seen["cands"], "blurring must start after verification"
    assert seen["linked"] == 0 and seen["hazards"] == 0, "blurring must run before deduplication"
    d = env.get(did)
    assert status_of(d) in DONE and d.get("stage") == "done"


# ------------------------------------------------------------------ results of the blur

def test_blurred_copies_saved_next_to_candidates_and_recorded(rs15_done):
    env, did, stub = rs15_done
    from PIL import Image

    acc = env.accepted(did)
    assert len(acc) == CHUNKS
    for c in acc:
        assert blur_ok(c), c
        assert c["blurred_frame_path"]
        orig = env.path_of(c["best_frame_path"])
        bl = env.path_of(c["blurred_frame_path"])
        assert bl.is_file() and orig.is_file()
        assert not same_file(bl, orig), "the blurred copy must be a separate file"
        assert bl.suffix.lower() in (".jpg", ".jpeg")
        assert "candidates" in bl.parts and did in bl.as_posix(), "blurred images live under the drive's candidates folder"
        with Image.open(orig) as a, Image.open(bl) as b:
            assert b.format == "JPEG" and a.size == b.size
        if c.get("crop_path"):
            assert c["blurred_crop_path"]
            oc, bc = env.path_of(c["crop_path"]), env.path_of(c["blurred_crop_path"])
            assert bc.is_file() and not same_file(bc, oc)
            with Image.open(oc) as a, Image.open(bc) as b:
                assert a.size == b.size
    # nothing leaked outside the test data dir
    for p in env.blurred_paths(did):
        assert env.tmp.resolve() in p.resolve().parents


def test_unblurred_originals_are_never_modified(rs15_done):
    env, did, stub = rs15_done
    seen = {norm_path(c["path"]): c["sha"] for c in stub.image_calls}
    assert seen
    for path, digest in seen.items():
        assert Path(path).is_file(), "the saved candidate image disappeared (it is kept until Delete video)"
        assert sha(path) == digest, "the unblurred original was modified by the blur stage"


def test_masks_are_blurred_and_the_rest_untouched(rs15_done):
    env, did, stub = rs15_done
    checked = 0
    for c in env.accepted(did):
        orig = load_gray(env.path_of(c["best_frame_path"]))
        out = load_gray(env.path_of(c["blurred_frame_path"]))
        h, w = orig.shape
        boxes = face_plate_boxes(w, h)
        for name, box in boxes.items():
            vo, vb = lap_var(orig, box, 10), lap_var(out, box, 10)
            assert vo > 500, "synthetic frame should be high contrast in the %s area (var %.1f)" % (name, vo)
            assert vb < 0.5 * vo, "%s mask not blurred (laplacian var %.1f -> %.1f)" % (name, vo, vb)
            checked += 1
        m = outside_boxes(orig.shape, list(boxes.values()), 24)
        d = abs(out - orig)[m]
        assert d.mean() < 8.0, "pixels well outside the masks changed (mean abs diff %.2f)" % d.mean()
    assert checked >= 2


def test_no_detections_gives_an_unchanged_but_recorded_copy(rs15_env):
    from rs15_helpers import empty_image_result

    env = rs15_env
    stub = make_stub(image_result=empty_image_result)
    did = env.process_ok(stub)
    acc = env.accepted(did)
    assert len(acc) == CHUNKS
    for c in acc:
        assert blur_ok(c) and c["blurred_frame_path"]
        orig = load_gray(env.path_of(c["best_frame_path"]))
        out = load_gray(env.path_of(c["blurred_frame_path"]))
        assert out.shape == orig.shape
        assert abs(out - orig).mean() < 6.0, "no faces or plates found, the copy should be unchanged"
        r = env.photo(did, c["id"])
        assert r.status_code == 200, "a frame with nothing to blur must still be viewable (as its copy)"


def test_hazards_point_at_blurred_images(rs15_done):
    env, did, stub = rs15_done
    hs = env.hazards_rows()
    assert hs, "no hazards after the pipeline"
    cands = env.cands(did)
    unblurred = {norm_path(env.path_of(c[k])) for c in cands for k in ("best_frame_path", "crop_path") if c.get(k)}
    blurred_frames = {norm_path(env.path_of(c["blurred_frame_path"])) for c in cands if c.get("blurred_frame_path")}
    blurred_crops = {norm_path(env.path_of(c["blurred_crop_path"])) for c in cands if c.get("blurred_crop_path")}
    cols = env.columns("hazards")
    for h in hs:
        assert h["photo_path"], h
        p = norm_path(env.path_of(h["photo_path"]))
        assert p not in unblurred, "hazards.photo_path points at an UNBLURRED image"
        assert p in blurred_frames or p in blurred_crops, "hazards.photo_path must be a blurred candidate image"
        assert Path(p).is_file()
        if "crop_path" in cols and h["crop_path"]:
            cp = norm_path(env.path_of(h["crop_path"]))
            assert cp not in unblurred, "hazards.crop_path points at an UNBLURRED image"
            assert cp in blurred_crops


# ------------------------------------------------------------------ only accepted candidates

def test_only_accepted_candidates_are_blurred(rs15_env):
    env = rs15_env
    stub = make_stub(reject_verify={1})
    did = env.process_ok(stub)
    acc, rej = env.accepted(did), env.rejected(did)
    assert len(acc) == CHUNKS - 1 and len(rej) == 1
    sent = {norm_path(c["path"]) for c in stub.image_calls}
    assert sent
    for c in rej:
        for k in ("best_frame_path", "crop_path"):
            if c.get(k):
                assert norm_path(env.path_of(c[k])) not in sent, "a rejected candidate's frame was sent to the provider"
        assert not c.get("blurred_frame_path") and not c.get("blurred_crop_path")
        assert not blur_ok(c)
        assert env.photo(did, c["id"]).status_code == 404
    for c in acc:
        assert blur_ok(c) and c["blurred_frame_path"]


def test_false_positive_scenario_makes_no_blur_calls(rs15_env):
    env = rs15_env
    stub = make_stub(scenario="false_positive")
    did = env.process_ok(stub)
    assert stub.image_calls == []
    assert [c for c in env.cands(did) if c.get("blurred_frame_path")] == []
    assert env.hazards_rows() == []


def test_empty_scenario_makes_no_blur_calls(rs15_env):
    env = rs15_env
    stub = make_stub(scenario="empty")
    did = env.process_ok(stub)
    assert stub.image_calls == [] and env.cands(did) == []


# ------------------------------------------------------------------ failures fail closed

def assert_fail_closed(env, did):
    """Every candidate is either blurred (status ok, files there) or has no servable photo at all."""
    bad = 0
    for c in env.cands(did):
        for crop in (False, True):
            r = env.photo(did, c["id"], crop=crop)
            assert r.status_code < 500, r.text
            if blur_ok(c) and c.get("blurred_frame_path"):
                continue
            bad += 1
            assert r.status_code == 404, (
                "candidate %s (blur_status=%r) is not blurred but its photo is served (%s, crop=%s)"
                % (c["id"], c.get("blur_status"), r.status_code, crop))
    for h in env.hazards_rows():
        unbl = {norm_path(env.path_of(c[k])) for c in env.cands(did) for k in ("best_frame_path", "crop_path")
                if c.get(k)}
        if h.get("photo_path"):
            assert norm_path(env.path_of(h["photo_path"])) not in unbl
    return bad


@pytest.mark.parametrize("n", [0, 1, 2])
def test_provider_failure_fails_closed(rs15_env, n):
    env = rs15_env
    stub = make_stub(fail_image_on=n)
    did = env.process(stub)
    d = env.get(did)
    cands = env.cands(did)
    if status_of(d) in FAILED:
        assert d.get("failed_stage") == "blurring", d
        err = str(d.get("error") or "")
        assert "blurring" in err.lower(), err
        assert "boom-blur-xyz" in err, err
        assert d.get("stage") != "done"
    else:
        # the other acceptable reading: the run finished but the failure is recorded on the candidates
        assert any(str(c.get("blur_status") or "").lower() in BAD_STATUS for c in cands), \
            "provider failure was swallowed: drive %s, statuses %s" % (status_of(d), [c.get("blur_status") for c in cands])
    assert any(not blur_ok(c) for c in cands), "a failed blur must leave at least one candidate not ok"
    failed = [c for c in cands if str(c.get("blur_status") or "").lower() in BAD_STATUS]
    for c in failed:
        assert not c.get("blurred_frame_path") and not c.get("blurred_crop_path"), \
            "a failed candidate must not keep (partial) blurred paths"
    assert_fail_closed(env, did)
    # the unblurred images are still not served anywhere, and the drive's raw files are untouched
    assert env.video_file(did).is_file()


@pytest.mark.parametrize("kind", sorted(BAD_RESULTS))
def test_garbage_result_fails_closed(rs15_env, kind):
    env = rs15_env
    value = BAD_RESULTS[kind]
    stub = make_stub(image_result=lambda s, p, pr: value)
    did = env.process(stub)
    d = env.get(did)
    cands = env.cands(did)
    assert not any(blur_ok(c) for c in env.accepted(did)), \
        "garbage from the provider (%s) was treated as 'nothing to blur'; the unblurred frame would be served" % kind
    if status_of(d) in FAILED:
        assert d.get("failed_stage") == "blurring" and "blurring" in str(d.get("error") or "").lower(), d
    assert_fail_closed(env, did)
    assert cands


def test_failed_blur_never_creates_hazard_with_unblurred_photo(rs15_env):
    env = rs15_env
    stub = make_stub(fail_image_on=0)
    did = env.process(stub)
    unbl = {norm_path(p) for p in env.unblurred_paths(did)}
    for h in env.hazards_rows():
        assert not h.get("photo_path") or norm_path(env.path_of(h["photo_path"])) not in unbl


def test_drive_can_recover_with_a_clean_rerun_after_blur_failure(rs15_env):
    env = rs15_env
    did = env.process(make_stub(fail_image_on=0))
    assert any(not blur_ok(c) for c in env.cands(did))
    env.process(make_stub(), did=did)
    d = env.get(did)
    assert status_of(d) in DONE and not d.get("error") and not d.get("failed_stage")
    acc = env.accepted(did)
    assert len(acc) == CHUNKS and all(blur_ok(c) for c in acc)


# ------------------------------------------------------------------ reruns

def test_rerun_replaces_blurred_images_without_stale_files(rs15_env):
    env = rs15_env
    did = env.process_ok(make_stub())
    first = {norm_path(p) for p in env.blurred_paths(did)}
    assert first
    env.process(make_stub(), did=did)
    assert status_of(env.get(did)) in DONE
    now = {norm_path(p) for p in env.blurred_paths(did)}
    assert now
    for p in first:
        assert p in now or not Path(p).exists(), "stale blurred image left behind after a rerun: %s" % p
    # every blurred file in the drive's candidates folder is referenced by the DB
    folder = Path(env.path_of(env.cands(did)[0]["blurred_frame_path"])).parent
    referenced = now | {norm_path(p) for p in env.unblurred_paths(did)}
    for f in folder.rglob("*"):
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            assert norm_path(f) in referenced or norm_path(f) not in first, "stale file %s" % f


def test_rerun_with_nothing_found_removes_old_blurred_images(rs15_env):
    env = rs15_env
    did = env.process_ok(make_stub())
    first = [Path(p) for p in env.blurred_paths(did)]
    assert first and all(p.is_file() for p in first)
    env.process(make_stub(scenario="empty"), did=did)
    assert status_of(env.get(did)) in DONE
    assert env.cands(did) == []
    leftover = [p for p in first if p.exists()]
    assert leftover == [], "blurred images of candidates that no longer exist were left behind: %s" % leftover
    assert env.hazards_rows() == []


def test_rerun_after_failure_leaves_no_blurred_images_of_old_candidates(rs15_env):
    env = rs15_env
    did = env.process_ok(make_stub())
    first = [Path(p) for p in env.blurred_paths(did)]
    env.process(make_stub(fail_video_on=0), did=did)  # fails in detection: old candidates are cleared
    assert status_of(env.get(did)) in FAILED
    ids = {c["id"] for c in env.cands(did)}
    for p in first:
        if p.exists():
            # still referenced by a surviving candidate row, or the rerun kept the old results intact
            refs = {norm_path(x) for x in env.blurred_paths(did)}
            assert norm_path(p) in refs, "orphan blurred image %s (candidates now %s)" % (p, ids)


# ------------------------------------------------------------------ fake provider / http

def test_default_fake_provider_path_works_without_key_or_network(rs15_env):
    env = rs15_env
    assert "META_API_KEY" not in os.environ
    did = env.upload()
    res, exc = env.run(did)  # provider=None -> get_provider() -> fake
    assert exc is None, repr(exc)
    d = env.get(did)
    assert status_of(d) in DONE, d
    acc = env.accepted(did)
    assert acc
    for c in acc:
        assert blur_ok(c) and env.path_of(c["blurred_frame_path"]).is_file()
        assert env.photo(did, c["id"]).status_code == 200


def test_process_endpoint_runs_the_blur_stage(rs15_env):
    env = rs15_env
    stub = make_stub()
    env.use_provider(stub)
    did = env.upload()
    r = env.post_process(did)
    assert r.status_code in (200, 201, 202), r.text
    d = env.wait(did)
    assert status_of(d) in DONE, d
    assert stub.image_calls
    for c in env.accepted(did):
        assert blur_ok(c)
        assert env.photo(did, c["id"]).status_code == 200
