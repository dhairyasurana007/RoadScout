"""Task 8: end to end with the fake provider: preprocess -> run_detection -> aggregate."""
from __future__ import annotations

from pathlib import Path
from datetime import timedelta

import pytest

from rs8_helpers import (  # noqa: F401
    FFMPEG, FFPROBE, M_PER_DEG, T0, Env, _ffmpeg, image_size, make_track, rs8_cache, rs8_env, norm,
)


@pytest.fixture(scope="session")
def rs8_pre(tmp_path_factory):
    """12 s synthetic 640x360 clip preprocessed to 3 chunks of 4 s at 3 fps (12 frames each)."""
    if not (FFMPEG and FFPROBE):
        pytest.skip("ffmpeg/ffprobe not on PATH")
    base = tmp_path_factory.mktemp("rs8pre")
    clip = base / "clip.mp4"
    _ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-t", 12, "-c:v", "libx264",
             "-preset", "ultrafast", "-pix_fmt", "yuv420p", clip])
    from backend.preprocess import preprocess_video

    res = preprocess_video(clip, base / "out", fps=3, chunk_seconds=4)
    assert len(res.chunks) == 3
    return res


def run_pipeline(e, pre, scenario, **kw):
    from backend.detection import run_detection
    from backend.providers import FakeProvider

    run_detection("drive-a", pre, provider=FakeProvider(scenario), db_path=e.db)
    e.work = Path(str(pre.chunks[0].path)).parent
    return e.run(chunks=pre, **kw)


def test_default_scenario_gives_one_candidate_per_chunk(rs8_env, rs8_pre):
    e = rs8_env
    track = make_track()
    run_pipeline(e, rs8_pre, "default", gpx=track, video_start=T0, offset=0.0)
    cs = sorted(e.cands(), key=lambda c: c["first_ts"])
    assert len(cs) == 3
    assert [c["track_key"] for c in cs] == ["0:1", "1:1", "2:1"]
    for c, chunk in zip(cs, rs8_pre.chunks):
        assert c["frames"] == 12
        assert c["first_ts"] == pytest.approx(chunk.start_time, abs=1e-6)
        assert c["last_ts"] == pytest.approx(chunk.start_time + 11 / 3, abs=1e-5)
        assert c["hazard_id"] is None
        best, crop = e.resolve(c["best_path"]), e.resolve(c["crop_path"])
        assert best.is_file() and crop.is_file()
        assert image_size(best) == (640, 360)
        cw, ch = image_size(crop)
        assert 0 < cw <= 640 and 0 < ch <= 360
        # geotag lies on the northbound leg between the first and last sighting of the object
        lo = track.position_at(c["first_ts"], video_start=T0, clock_offset_seconds=0.0)
        hi = track.position_at(c["last_ts"], video_start=T0, clock_offset_seconds=0.0)
        assert lo.latitude - 1e-7 <= c["lat"] <= hi.latitude + 1e-7
        assert abs(c["heading"]) < 1.0 or abs(c["heading"] - 360) < 1.0
    # the three objects are different places along the road
    assert len({round(c["lat"], 6) for c in cs}) == 3


def test_class_comes_from_the_prompt_config(rs8_env, rs8_pre):
    from backend.hazard_prompts import load_hazard_prompts

    e = rs8_env
    run_pipeline(e, rs8_pre, "default", gpx=make_track(), video_start=T0, offset=0.0)
    prompts = load_hazard_prompts()
    first_class = prompts.class_names[0]
    assert {c["cls"] for c in e.cands()} == {first_class}


@pytest.mark.parametrize("scenario,expected", [("empty", 0), ("short_track", 0), ("false_positive", 3)])
def test_other_fake_scenarios(rs8_env, rs8_pre, scenario, expected):
    e = rs8_env
    run_pipeline(e, rs8_pre, scenario, gpx=make_track(), video_start=T0, offset=0.0)
    cs = e.cands()
    assert len(cs) == expected
    if scenario == "false_positive":
        assert {c["frames"] for c in cs} == {8}


def test_short_track_is_kept_when_the_minimum_is_lowered(rs8_env, rs8_pre):
    e = rs8_env
    run_pipeline(e, rs8_pre, "short_track", gpx=make_track(), video_start=T0, offset=0.0, min_frames=2)
    cs = e.cands()
    assert len(cs) == 3 and {c["frames"] for c in cs} == {2}


def test_pipeline_rerun_is_idempotent(rs8_env, rs8_pre):
    e = rs8_env
    run_pipeline(e, rs8_pre, "default", gpx=make_track(), video_start=T0, offset=0.0)
    first = [(c["track_key"], c["frames"], c["first_ts"], c["last_ts"], c["lat"], c["lon"]) for c in e.cands()]
    run_pipeline(e, rs8_pre, "default", gpx=make_track(), video_start=T0, offset=0.0)
    second = [(c["track_key"], c["frames"], c["first_ts"], c["last_ts"], c["lat"], c["lon"]) for c in e.cands()]
    assert sorted(first) == sorted(second) and len(second) == 3


def test_no_network_and_no_api_key_needed(rs8_env, rs8_pre, monkeypatch):
    """The whole path runs against the fake provider with no META_API_KEY (the env fixture blocks sockets)."""
    monkeypatch.delenv("META_API_KEY", raising=False)
    e = rs8_env
    run_pipeline(e, rs8_pre, "default", gpx=make_track(), video_start=T0, offset=0.0)
    assert len(e.cands()) == 3
