"""Retries with exponential backoff and a per-call timeout (both usable on their own)."""
from __future__ import annotations

import threading
import time
from typing import Callable, TypeVar

from .errors import ProviderTimeoutError, RetriesExhausted, TransientProviderError

T = TypeVar("T")

#: Longest wait honoured from a server's Retry-After hint.
MAX_RETRY_AFTER_SECONDS = 30.0


def call_with_timeout(fn: Callable[[], T], timeout_seconds: float | None) -> T:
    """Run ``fn()`` and return its result.

    Raises ``ProviderTimeoutError`` as soon as ``timeout_seconds`` pass, without
    waiting for ``fn`` to finish. ``fn``'s own exceptions pass through unchanged.
    ``fn`` runs in a daemon thread; Python cannot kill a thread, so a timed-out call
    may keep running in the background. ``None`` disables the timeout.
    """
    if timeout_seconds is None:
        return fn()
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # passed through to the caller below
            box["error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise ProviderTimeoutError(f"call did not finish within {timeout_seconds}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def call_with_retries(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    backoff: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    retry_on: tuple[type[BaseException], ...] = (TransientProviderError,),
) -> T:
    """Call ``fn()`` (no arguments), retrying errors in ``retry_on``.

    Waits ``base_delay * backoff**(n-1)`` seconds (via ``sleep``) after failed attempt
    ``n`` (at least the error's ``retry_after`` seconds when it has one, capped at 30);
    never sleeps after the last attempt. Errors outside ``retry_on``
    (including SpendCapExceeded) propagate immediately. When ``max_attempts`` are used
    up it raises ``RetriesExhausted`` (a ``ProviderError``) carrying the last
    error's message.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except retry_on as e:
            if attempt == max_attempts:
                raise RetriesExhausted(
                    f"gave up after {attempt} attempts: {e}", attempts=attempt
                ) from e
            delay = base_delay * backoff ** (attempt - 1)
            hint = getattr(e, "retry_after", None)  # a server's Retry-After, capped
            if isinstance(hint, (int, float)) and hint > 0:
                delay = max(delay, min(float(hint), MAX_RETRY_AFTER_SECONDS))
            sleep(delay)
    raise AssertionError("unreachable")
