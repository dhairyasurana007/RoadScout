"""Model providers: the interface, the fake/fixture/live implementations and the guard helpers."""
from .base import ModelProvider
from .cost import SpendTracker
from .errors import (
    FixtureMissing,
    ProviderConfigError,
    ProviderError,
    ProviderTimeoutError,
    RetriesExhausted,
    SpendCapExceeded,
    TransientProviderError,
)
from .factory import get_provider, get_spend_tracker, reset_spend_tracker
from .fake import SCENARIOS, FakeProvider
from .fixture import FixtureProvider
from .live import LiveProvider
from .retry import call_with_retries, call_with_timeout
from .types import (
    Box,
    Detection,
    DetectionResult,
    FrameRecord,
    Verdict,
    rle_area,
    rle_from_bitmap,
    rle_from_box,
    rle_to_bitmap,
    validate_rle,
)

__all__ = [
    "Box", "Detection", "DetectionResult", "FakeProvider", "FixtureMissing",
    "FixtureProvider", "FrameRecord", "LiveProvider", "ModelProvider",
    "ProviderConfigError", "ProviderError", "ProviderTimeoutError", "RetriesExhausted",
    "SCENARIOS", "SpendCapExceeded", "SpendTracker", "TransientProviderError", "Verdict",
    "call_with_retries", "call_with_timeout", "get_provider", "get_spend_tracker",
    "reset_spend_tracker", "rle_area", "rle_from_bitmap", "rle_from_box", "rle_to_bitmap",
    "validate_rle",
]
