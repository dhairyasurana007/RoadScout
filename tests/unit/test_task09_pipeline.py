"""Task 9: fake scenarios end to end: preprocess -> run_detection -> aggregate_drive -> verify."""
from __future__ import annotations

import io
import math
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rs9_helpers import FFMPEG, FFPROBE, class_names, cls_of, rs9_env, size_of  # noqa: F401

T0 = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
M_PER_DEG = 6371008.8 * math.pi / 180.0


def make_track(seconds=120):
    from backend.gpx_track import GpxTrack, TrackPoint

    pts, lat = [], 38.88
    for s in range(seconds + 1):
        pts.append(TrackPoint(T0 + timedelta(seconds=s), lat, -77.3))
        lat += 10.0 / M_PER_DEG
    return GpxTrack.from_points(pts)


@pytest.fixture(scope="session")
def rs9_pre(tmp_path_factory):
    """12 s synthetic 640x360 clip preprocessed to 3 chunks of 4 s at 3 fps."""
    if not (FFMPEG and FFPROBE):
        pytest.skip("ffmpeg/ffprobe not on PATH")
    base = tmp_path_factory.mktemp("rs9pre")
    clip = base / "clip.mp4"
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        "testsrc2=size=640x360:rate=30", "-t", "12", "-c:v", "libx264", "-preset", "ultrafast",
                        "-pix_fmt", "yuv420p", str(clip)], capture_output=True, timeout=240)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    from backend.preprocess import preprocess_video

    res = preprocess_video(clip, base / "out", fps=3, chunk_seconds=4)
    assert len(res.chunks) == 3
    return res


def make_recorder(scenario):
    from backend.providers import FakeProvider, ModelProvider

    class Wrap(ModelProvider):
        name = "wrap"

        def __init__(self):
            self.inner = FakeProvider(scenario)
            self.calls = []

        def detect(self, video_path, prompts):
            return self.inner.detect(video_path, prompts)

        def verify(self, images, *, class_prompt=None, prompt_version=None):
            self.calls.append({"images": list(images), "class_prompt": class_prompt,
                               "prompt_version": prompt_version})
            return self.inner.verify(images, class_prompt=class_prompt, prompt_version=prompt_version)

    return Wrap()


def detect_and_aggregate(e, pre, scenario, drive="drive-a"):
    from backend.candidates import aggregate_drive
    from backend.detection import run_detection
    from backend.providers import FakeProvider

    run_detection(drive, pre, provider=FakeProvider(scenario), db_path=e.db)
    return aggregate_drive(drive, pre, gpx_track=make_track(), video_start=T0, clock_offset_seconds=0.0,
                           db_path=e.db)


def test_default_scenario_end_to_end(rs9_env, rs9_pre):
    from backend.candidates import load_candidates

    e = rs9_env
    detect_and_aggregate(e, rs9_pre, "default")
    cands = load_candidates("drive-a", db_path=e.db)
    assert len(cands) == 3
    rec = make_recorder("default")
    e.run(provider=rec)
    assert len(rec.calls) == 3
    got = e.by_candidate()
    assert sorted(got) == sorted(c["id"] for c in cands)
    acc = e.accepted()
    for c in cands:
        row = got[c["id"]]
        assert bool(row["is_hazard"]) is True
        assert cls_of(row) == c["hazard_class"] == class_names()[0]
        assert row["model"] == "fake-muse-spark"
        assert row["prompt_version"] and isinstance(row["prompt_version"], str)
        assert row["severity"] in (1, 2, 3)
        assert 0.0 <= row["confidence"] <= 1.0
        assert row["reason"] and row["report_description"]
        assert acc[c["id"]] is True
    assert len({r["prompt_version"] for r in got.values()}) == 1


def test_images_sent_for_real_candidates(rs9_env, rs9_pre):
    from backend.candidates import load_candidates

    e = rs9_env
    detect_and_aggregate(e, rs9_pre, "default")
    cands = load_candidates("drive-a", db_path=e.db)
    rec = make_recorder("default")
    e.run(provider=rec)
    assert len(rec.calls) == len(cands) == 3
    for call in rec.calls:
        imgs = call["images"]
        assert 2 <= len(imgs) <= 4, "best frame, crop and up to two neighbouring frames"
        for i in imgs:
            if not isinstance(i, (bytes, bytearray)):
                assert Path(str(i)).is_file()
        assert size_of(imgs[0]) == (640, 360), "the best frame comes first"
        assert any(size_of(i)[0] < 640 or size_of(i)[1] < 360 for i in imgs[1:]), "the crop must be sent"
        assert call["class_prompt"] == class_names()[0]
    # when the images are paths, the first one of each call is exactly one candidate's best frame
    if all(not isinstance(c["images"][0], (bytes, bytearray)) for c in rec.calls):
        sent = sorted(str(Path(str(c["images"][0])).resolve()) for c in rec.calls)
        assert sent == sorted(str(Path(c["best_frame_path"]).resolve()) for c in cands)


def test_false_positive_scenario_is_rejected_but_stored(rs9_env, rs9_pre):
    e = rs9_env
    detect_and_aggregate(e, rs9_pre, "false_positive")
    e.mp.setenv("ROADSCOUT_FAKE_SCENARIO", "false_positive")
    e.run()  # default provider path
    rows = e.rows()
    assert len(rows) == 3
    assert all(not bool(r["is_hazard"]) for r in rows)
    assert set(e.accepted().values()) == {False}


@pytest.mark.parametrize("scenario", ["empty", "short_track"])
def test_scenarios_with_no_candidates_verify_nothing(rs9_env, rs9_pre, scenario):
    e = rs9_env
    detect_and_aggregate(e, rs9_pre, scenario)
    rec = make_recorder(scenario)
    e.run(provider=rec)
    assert rec.calls == []
    assert e.raw_all() == []


def test_rerun_of_aggregation_then_verification_is_consistent(rs9_env, rs9_pre):
    from backend.candidates import load_candidates

    e = rs9_env
    detect_and_aggregate(e, rs9_pre, "default")
    e.run(provider=make_recorder("default"))
    assert len(e.raw_all()) == 3
    detect_and_aggregate(e, rs9_pre, "default")  # replaces the candidates (new ids); must not raise
    e.run(provider=make_recorder("default"))
    cands = load_candidates("drive-a", db_path=e.db)
    assert len(cands) == 3
    assert sorted(e.by_candidate()) == sorted(c["id"] for c in cands)
    assert len(e.raw_all()) == 3, "no orphan verdicts from the replaced candidates"


def test_whole_path_needs_no_network_and_no_key(rs9_env, rs9_pre, monkeypatch):
    """rs9_env blocks sockets and removes META_API_KEY; the default provider must still work."""
    e = rs9_env
    monkeypatch.delenv("META_API_KEY", raising=False)
    detect_and_aggregate(e, rs9_pre, "default")
    e.run()
    assert len(e.rows()) == 3
