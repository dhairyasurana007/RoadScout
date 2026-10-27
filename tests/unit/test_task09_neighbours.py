"""Task 9: neighbouring frames are added when the chunk videos are supplied (optional chunks= argument)."""
from __future__ import annotations

import inspect

import pytest

from rs9_helpers import find_func, rs9_env, size_of  # noqa: F401
from test_task09_pipeline import detect_and_aggregate, make_recorder, rs9_pre  # noqa: F401


def test_neighbouring_frames_added_with_chunks(rs9_env, rs9_pre):
    if "chunks" not in inspect.signature(find_func()).parameters:
        pytest.skip("no chunks parameter")
    e = rs9_env
    detect_and_aggregate(e, rs9_pre, "default")
    rec = make_recorder("default")
    e.run(provider=rec, chunks=rs9_pre)
    assert len(rec.calls) == 3
    for call in rec.calls:
        imgs = call["images"]
        assert 3 <= len(imgs) <= 4, "best frame, crop and one or two neighbouring frames"
        assert size_of(imgs[0]) == (640, 360)
        assert sum(1 for i in imgs[1:] if size_of(i) == (640, 360)) in (1, 2)
    assert len(e.rows()) == 3


def test_prompt_text_reaches_providers_that_accept_it(rs9_env):
    from backend.providers import ModelProvider
    from rs9_helpers import make_verdict

    e = rs9_env
    e.add_candidates(2)
    seen = []

    class P(ModelProvider):
        name = "p"

        def detect(self, video_path, prompts):
            raise AssertionError

        def verify(self, images, *, class_prompt=None, prompt_version=None, prompt_text=None):
            seen.append(prompt_text)
            return make_verdict(cls=class_prompt, prompt_version=prompt_version or "")

    e.run(provider=P())
    assert len(seen) == 2 and all(isinstance(s, str) and s.strip() for s in seen)
    assert len(e.rows()) == 2
