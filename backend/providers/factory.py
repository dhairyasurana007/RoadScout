"""Choose the provider from config: ROADSCOUT_PROVIDER = fake | fixture | live.

``get_provider(name=None, **kw)``: an explicit ``name`` overrides ROADSCOUT_PROVIDER
(default ``fake``); an unknown name raises ``ValueError``. ``kw`` goes to the provider's
constructor (``scenario`` for fake, ``fixtures_dir`` for fixture, ``api_key``, ``tracker``,
... for live). Without ``kw``, fake reads ROADSCOUT_FAKE_SCENARIO and fixture reads
ROADSCOUT_FIXTURES_DIR. Live raises an error mentioning META_API_KEY if the key is unset.

In demo mode (ROADSCOUT_DEMO) ``live`` is replaced by ``fake``: the live provider is never built and
META_API_KEY is never read.

Live providers share one process-wide ``SpendTracker`` (cap from ROADSCOUT_SPEND_CAP_USD,
log at ROADSCOUT_COST_LOG, default data/cost_log.jsonl).
"""
from __future__ import annotations

from typing import Any

from .. import config
from .base import ModelProvider
from .cost import SpendTracker
from .errors import ProviderConfigError
from .fake import FakeProvider
from .fixture import FixtureProvider
from .live import LiveProvider

PROVIDER_NAMES = ("fake", "fixture", "live")

_tracker: SpendTracker | None = None


def get_spend_tracker() -> SpendTracker:
    """The process-wide spend tracker used by live providers."""
    global _tracker
    if _tracker is None:
        _tracker = SpendTracker(log_path=config.cost_log_path())
    return _tracker


def reset_spend_tracker() -> None:
    """Forget the shared tracker (rebuilt from config on next use). For tests."""
    global _tracker
    _tracker = None


def get_provider(name: str | None = None, **kw: Any) -> ModelProvider:
    name = (name or config.provider_name()).strip().lower()
    if config.demo_mode() and name == "live":
        name = "fake"  # the read-only demo never builds the live provider or reads the key
        kw = {k: v for k, v in kw.items() if k == "scenario"}
    if name == "fake":
        kw.setdefault("scenario", config.fake_scenario())
        return FakeProvider(**kw)
    if name == "fixture":
        return FixtureProvider(**kw)
    if name == "live":
        kw.setdefault("tracker", get_spend_tracker())
        return LiveProvider(**kw)
    raise ProviderConfigError(
        f"unknown ROADSCOUT_PROVIDER {name!r}; choose one of {', '.join(PROVIDER_NAMES)}"
    )
