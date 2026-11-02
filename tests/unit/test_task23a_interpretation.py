"""Task 23a: tests that rest on one reading of the description (kept apart so they are easy to adjust).

Reading: "In demo mode the app never constructs the live provider and never reads META_API_KEY, even if
ROADSCOUT_PROVIDER=live" also covers backend.providers.get_provider() and config.meta_api_key().
"""
from __future__ import annotations

import os

import pytest

from rs23a_fixtures import make_app  # noqa: F401


def test_get_provider_in_demo_mode_never_returns_the_live_provider(make_app, monkeypatch):
    app, seed = make_app(demo="1", extra_env={"ROADSCOUT_PROVIDER": "live", "META_API_KEY": "dummy-key-23a"})
    import backend.providers.live as live
    from backend.providers import get_provider

    def boom(*a, **k):
        raise AssertionError("the live provider must not be constructed in demo mode")

    monkeypatch.setattr(live.LiveProvider, "__init__", boom)
    p = get_provider()
    assert not isinstance(p, live.LiveProvider)
    assert p.name != "live"


def test_meta_api_key_is_none_in_demo_mode_even_when_set(make_app):
    app, seed = make_app(demo="1", extra_env={"META_API_KEY": "dummy-key-23a"})
    from backend import config

    assert config.meta_api_key() is None


def test_meta_api_key_is_read_normally_when_demo_is_off(make_app):
    app, seed = make_app(demo=None, extra_env={"META_API_KEY": "dummy-key-23a"})
    from backend import config

    assert config.meta_api_key() == "dummy-key-23a"
