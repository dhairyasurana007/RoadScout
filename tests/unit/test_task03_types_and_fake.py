"""Task 3: internal result types, provider interface and the fake provider."""
import json

import pytest

from rs3_helpers import (  # noqa: F401  (fixtures imported on purpose)
    PROMPTS, all_dets, canon, clean_env, clip, dets_of, find, frame_image,
    frames_of, frames_per_track, get, no_network, normalize, track_id,
    verdict_field,
)

SCENARIOS = ["default", "empty", "short_track", "false_positive"]


def fake(scenario=None):
    cls = find("FakeProvider")
    return cls() if scenario is None else cls(scenario=scenario)


# ------------------------------------------------------------- interface

def test_package_imports_without_key_or_network():
    find("ModelProvider")
    find("FakeProvider")
    find("FixtureProvider")
    find("LiveProvider")


def test_all_providers_implement_the_interface():
    base = find("ModelProvider")
    for key in ("FakeProvider", "FixtureProvider", "LiveProvider"):
        cls = find(key)
        assert issubclass(cls, base), f"{key} must subclass ModelProvider"
    for name in ("detect", "verify"):
        assert callable(getattr(base, name, None)), f"ModelProvider.{name} missing"


def test_exception_hierarchy():
    pe = find("ProviderError")
    tr = find("TransientProviderError")
    to = find("ProviderTimeoutError")
    cap = find("SpendCapExceeded")
    assert issubclass(pe, Exception)
    assert issubclass(tr, pe)
    assert issubclass(to, pe)
    assert issubclass(cap, Exception)
    # the hard cap must never be retried as if it were a flaky network
    assert not issubclass(cap, tr)


# ------------------------------------------------------------ result types

def test_detection_result_has_per_frame_boxes_masks_track_ids(clip):
    res = fake().detect(clip, PROMPTS)
    assert isinstance(res, find("DetectionResult"))
    frames = frames_of(res)
    assert frames, "default scenario must contain frames"
    dets = all_dets(res)
    assert dets, "default scenario must contain detections"
    for f in frames:
        assert isinstance(get(f, "frame_index", "index", "frame"), int)
        ts = get(f, "timestamp", "timestamp_s", "time_s", "t")
        assert isinstance(ts, (int, float)) and ts >= 0
    for d in dets:
        box = get(d, "box", "bbox")
        assert len(box) == 4 and all(isinstance(x, (int, float)) for x in box)
        x1, y1, x2, y2 = box
        assert x2 > x1 and y2 > y1, "box must be (x1, y1, x2, y2) with positive size"
        assert get(d, "mask") is not None
        assert isinstance(track_id(d), int)
        score = get(d, "score", "confidence")
        assert 0.0 <= score <= 1.0


def test_frame_timestamps_and_indexes_are_ordered(clip):
    frames = frames_of(fake().detect(clip, PROMPTS))
    idx = [get(f, "frame_index", "index", "frame") for f in frames]
    ts = [get(f, "timestamp", "timestamp_s", "time_s", "t") for f in frames]
    assert idx == sorted(idx) and len(set(idx)) == len(idx)
    assert ts == sorted(ts)


def test_detection_result_is_json_serialisable_and_round_trips(clip):
    cls = find("DetectionResult")
    res = fake().detect(clip, PROMPTS)
    data = res.to_dict()
    text = json.dumps(data)  # masks must be JSON-safe (RLE / polygons / lists)
    back = cls.from_dict(json.loads(text))
    assert canon(back) == canon(res)


def test_verdict_has_all_fields(frame_image):
    v = fake().verify([frame_image])
    assert isinstance(v, find("Verdict"))
    assert isinstance(verdict_field(v, "is_hazard"), bool)
    assert isinstance(verdict_field(v, "hazard_class"), str)
    assert verdict_field(v, "hazard_class")
    assert verdict_field(v, "severity") in (1, 2, 3)
    lane = verdict_field(v, "lane")
    assert lane is None or isinstance(lane, str)
    assert 0.0 <= verdict_field(v, "confidence") <= 1.0
    assert isinstance(verdict_field(v, "reason"), str) and verdict_field(v, "reason")
    desc = verdict_field(v, "report_description")
    assert isinstance(desc, str) and desc


def test_verdict_round_trips_through_dict(frame_image):
    cls = find("Verdict")
    v = fake().verify([frame_image])
    data = json.loads(json.dumps(v.to_dict()))
    assert canon(cls.from_dict(data)) == canon(v)
    for key in ("is_hazard", "severity", "lane", "confidence", "reason", "report_description"):
        assert key in data, f"to_dict() must contain {key!r}"


# ------------------------------------------------------------ fake provider

def test_fake_needs_no_args_no_key_no_real_files(frame_image, clip):
    p = find("FakeProvider")()
    assert p.detect(clip, PROMPTS) is not None
    assert p.verify([frame_image]) is not None


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_fake_is_deterministic(scenario, clip, frame_image):
    a, b = fake(scenario), fake(scenario)
    assert canon(a.detect(clip, PROMPTS)) == canon(b.detect(clip, PROMPTS))
    assert canon(a.detect(clip, PROMPTS)) == canon(a.detect(clip, PROMPTS))
    assert canon(a.verify([frame_image])) == canon(b.verify([frame_image]))
    assert canon(a.verify([frame_image])) == canon(a.verify([frame_image]))


def test_fake_default_is_a_real_multi_frame_track(clip):
    res = fake("default").detect(clip, PROMPTS)
    counts = frames_per_track(res)
    assert counts
    assert max(counts.values()) >= 5, "default hazard track must span at least 5 frames"
    prompts = {get(d, "prompt", "label", "text_prompt", "class_prompt") for d in all_dets(res)}
    assert prompts <= set(PROMPTS)


def test_fake_empty_is_empty_not_error(clip):
    res = fake("empty").detect(clip, PROMPTS)
    assert isinstance(res, find("DetectionResult"))
    assert all_dets(res) == []


def test_fake_short_track_is_short(clip):
    res = fake("short_track").detect(clip, PROMPTS)
    counts = frames_per_track(res)
    assert counts, "short_track must still contain one detection track"
    assert max(counts.values()) <= 2, "short track must be seen in at most 2 frames"


def test_fake_default_vs_short_track_differ(clip):
    d = max(frames_per_track(fake("default").detect(clip, PROMPTS)).values())
    s = max(frames_per_track(fake("short_track").detect(clip, PROMPTS)).values())
    assert d > s


def test_fake_false_positive_is_detected_then_rejected(clip, frame_image):
    p = fake("false_positive")
    res = p.detect(clip, PROMPTS)
    assert all_dets(res), "a false positive is something SAM detects"
    v = p.verify([frame_image])
    assert verdict_field(v, "is_hazard") is False
    assert 0.0 <= verdict_field(v, "confidence") <= 1.0
    assert isinstance(verdict_field(v, "reason"), str) and verdict_field(v, "reason")
    sev = verdict_field(v, "severity")
    assert sev is None or sev in (0, 1, 2, 3)


def test_fake_default_verdict_is_a_hazard(frame_image):
    v = fake("default").verify([frame_image])
    assert verdict_field(v, "is_hazard") is True
    assert verdict_field(v, "confidence") >= 0.5


def test_fake_unknown_scenario_is_rejected():
    with pytest.raises((ValueError, KeyError)):
        fake("no-such-scenario")


def test_fake_detect_does_not_mutate_between_calls(clip):
    p = fake("default")
    first = p.detect(clip, PROMPTS)
    snapshot = canon(first)
    p.detect(clip, PROMPTS)
    assert canon(first) == snapshot
