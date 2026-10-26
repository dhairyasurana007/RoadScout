"""Live provider placeholder (Meta Model API). Implemented in task 26.

It needs META_API_KEY (read from the environment, never logged or echoed) and makes no
network call here: the real calls (``_do_detect``, ``_do_verify``) raise
NotImplementedError until task 26 supplies the endpoints, model IDs and request shapes
from Meta's docs.

The guard layer is already in place around those calls (``_call``): spend-cap check
before the call, per-call timeout, retries with exponential backoff on transient errors
and timeouts, and a charge plus cost-log line after success.
"""
from __future__ import annotations

import time
from typing import Callable, Sequence, TypeVar

from .. import config
from .base import Image, ModelProvider
from .cost import SpendTracker
from .errors import ProviderConfigError, ProviderTimeoutError, TransientProviderError
from .retry import call_with_retries, call_with_timeout
from .types import DetectionResult, Verdict

T = TypeVar("T")


class LiveProvider(ModelProvider):
    name = "live"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        tracker: SpendTracker | None = None,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        backoff: float = 2.0,
        timeout_seconds: float | None = 120.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        key = api_key or config.meta_api_key()
        if not key:
            raise ProviderConfigError(
                "ROADSCOUT_PROVIDER=live needs META_API_KEY (set it in the environment or a "
                "git-ignored .env). Use ROADSCOUT_PROVIDER=fake until task 26."
            )
        self._api_key = key  # never print or log this
        self.tracker = tracker if tracker is not None else SpendTracker(log_path=config.cost_log_path())
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.backoff = backoff
        self.timeout_seconds = timeout_seconds
        self._sleep = sleep

    def __repr__(self) -> str:
        return "LiveProvider(api_key=<hidden>)"

    def estimate_cost(self, operation: str) -> float:
        # Rates come from the environment until task 26 measures real pricing.
        return config.cost_per_call_usd(operation)

    # --- guard layer ---------------------------------------------------------

    def _call(self, operation: str, fn: Callable[[], T]) -> T:
        estimate = self.estimate_cost(operation)
        self.tracker.check(estimate)  # raises SpendCapExceeded before any call is made
        started = time.monotonic()
        attempts = 0

        def attempt() -> T:
            nonlocal attempts
            attempts += 1
            return call_with_timeout(fn, self.timeout_seconds)

        result = call_with_retries(
            attempt,
            max_attempts=self.max_attempts,
            base_delay=self.base_delay,
            backoff=self.backoff,
            sleep=self._sleep,
            retry_on=(TransientProviderError, ProviderTimeoutError),
        )
        self.tracker.charge(
            estimate,
            model=f"live-{operation}",
            operation=operation,
            attempts=attempts,
            duration_s=round(time.monotonic() - started, 3),
        )
        return result

    # --- the real calls (task 26) -------------------------------------------

    def _do_detect(self, video_path: str, prompts: Sequence[str]) -> DetectionResult:
        raise NotImplementedError("live SAM 3.1 detection is a placeholder until task 26")

    def _do_verify(
        self, images: Sequence[Image], class_prompt: str | None, prompt_version: str | None
    ) -> Verdict:
        raise NotImplementedError("live Muse Spark verification is a placeholder until task 26")

    def detect(self, video_path: str, prompts: Sequence[str]) -> DetectionResult:
        return self._call("detect", lambda: self._do_detect(video_path, prompts))

    def verify(
        self,
        images: Sequence[Image],
        *,
        class_prompt: str | None = None,
        prompt_version: str | None = None,
    ) -> Verdict:
        return self._call("verify", lambda: self._do_verify(images, class_prompt, prompt_version))
