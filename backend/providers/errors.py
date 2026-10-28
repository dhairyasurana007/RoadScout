"""Provider errors.

``call_with_retries`` retries only ``TransientProviderError`` by default (the live
provider also retries ``ProviderTimeoutError``). ``SpendCapExceeded`` is never retried.
"""


class ProviderError(Exception):
    """Base class for every provider-layer error."""


class ProviderConfigError(ProviderError, ValueError):
    """Bad or missing configuration (unknown provider or scenario, missing META_API_KEY, bad fixture)."""


class TransientProviderError(ProviderError):
    """A failure worth retrying: rate limit, 5xx, network blip.

    ``retry_after`` (seconds, optional) is the server's Retry-After hint; the retry layer
    waits at least that long (capped at 30 s) before the next attempt."""

    def __init__(self, message: str = "", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    """A call did not finish within the per-call timeout."""


class RetriesExhausted(ProviderError):
    """All retry attempts failed. The message carries the last error's message;
    ``attempts`` is the number of tries and ``__cause__`` the last error."""

    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


class SpendCapExceeded(ProviderError):
    """A charge or call would push total spend over the hard cap. Never retried."""


class FixtureMissing(ProviderError, FileNotFoundError):
    """A fixture file the provider needs does not exist."""
