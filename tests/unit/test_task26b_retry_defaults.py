"""Task 26b: LiveProvider retries 5 times (2, 4, 8, 16 s) by default; explicit values and Retry-After still win."""
import inspect
import json

import httpx
import pytest

import rs26b_setup  # noqa: F401
import rs26_helpers as H
from rs26_conftest import _safe_env, rec, api, clips, clip  # noqa: F401
from rs26_helpers import err_json, jpeg_bytes, verdict_obj


@pytest.fixture()
def sleeps(api):
    return lambda: [e[1] for e in api.events if e[0] == "sleep"]


def verify(p):
    return p.verify([jpeg_bytes()], class_prompt="pothole", prompt_version="v1", prompt_text="check")


def test_constructor_defaults():
    from backend.providers.live import LiveProvider

    params = inspect.signature(LiveProvider.__init__).parameters
    assert params["max_attempts"].default == 5
    assert params["base_delay"].default == 2.0
    assert params["backoff"].default == 2.0


def test_default_instance_attributes(api):
    p = H.make_provider(api)
    assert getattr(p, "max_attempts", 5) == 5
    assert getattr(p, "base_delay", 2.0) == 2.0
    assert getattr(p, "backoff", 2.0) == 2.0


@pytest.mark.parametrize("exc", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")], ids=["connect", "timeout"])
def test_default_waits_are_2_4_8_16_over_five_attempts(api, sleeps, exc):
    from backend.providers.errors import RetriesExhausted

    api.queue.extend([exc] * 20)
    p = H.make_provider(api)
    with pytest.raises(RetriesExhausted) as ei:
        verify(p)
    assert ei.value.attempts == 5
    assert len(api.responses) == 5
    assert sleeps() == [2.0, 4.0, 8.0, 16.0]


def test_recovers_on_the_fifth_attempt(api, sleeps):
    api.queue.extend([httpx.ConnectError("refused")] * 4 + [json.dumps(verdict_obj())])
    v = verify(H.make_provider(api))
    assert v.hazard_class == "pothole"
    assert len(api.responses) == 5
    assert sleeps() == [2.0, 4.0, 8.0, 16.0]


def test_recovers_after_a_short_outage(api, sleeps):
    api.queue.extend([httpx.ConnectError("refused"), json.dumps(verdict_obj())])
    verify(H.make_provider(api))
    assert sleeps() == [2.0]


@pytest.mark.parametrize("status", [429, 500, 503])
def test_status_failures_use_the_same_schedule(api, sleeps, status):
    from backend.providers.errors import RetriesExhausted

    api.queue.extend([httpx.Response(status, json=err_json("later"))] * 20)
    with pytest.raises(RetriesExhausted) as ei:
        verify(H.make_provider(api))
    assert ei.value.attempts == 5
    assert sleeps() == [2.0, 4.0, 8.0, 16.0]


def test_explicit_values_still_win(api, sleeps):
    from backend.providers.errors import RetriesExhausted

    api.queue.extend([httpx.ConnectError("refused")] * 20)
    p = H.make_provider(api, max_attempts=2, base_delay=0.25, backoff=3.0)
    with pytest.raises(RetriesExhausted) as ei:
        verify(p)
    assert ei.value.attempts == 2
    assert sleeps() == [0.25]


def test_explicit_backoff_and_base_delay_with_default_attempts(api, sleeps):
    from backend.providers.errors import RetriesExhausted

    api.queue.extend([httpx.ConnectError("refused")] * 20)
    p = H.make_provider(api, base_delay=1.0, backoff=3.0)
    with pytest.raises(RetriesExhausted):
        verify(p)
    assert sleeps() == [1.0, 3.0, 9.0, 27.0]


def test_single_attempt_means_no_retry_and_no_sleep(api, sleeps):
    from backend.providers.errors import RetriesExhausted

    api.queue.extend([httpx.ConnectError("refused")] * 3)
    with pytest.raises(RetriesExhausted):
        verify(H.make_provider(api, max_attempts=1))
    assert len(api.responses) == 1 and sleeps() == []


def test_retry_after_still_wins_over_the_default_backoff(api, sleeps):
    api.queue.extend([
        httpx.Response(429, json=err_json("slow"), headers={"Retry-After": "10"}),
        json.dumps(verdict_obj()),
    ])
    verify(H.make_provider(api))
    assert sleeps() == [10.0]


def test_retry_after_below_the_backoff_does_not_shorten_it(api, sleeps):
    api.queue.extend([
        httpx.Response(429, json=err_json("slow"), headers={"Retry-After": "1"}),
        json.dumps(verdict_obj()),
    ])
    verify(H.make_provider(api))
    assert sleeps() == [2.0]


def test_retry_after_is_capped_at_30_seconds_with_the_new_defaults(api, sleeps):
    api.queue.extend([
        httpx.Response(503, json=err_json("slow"), headers={"Retry-After": "900"}),
        json.dumps(verdict_obj()),
    ])
    verify(H.make_provider(api))
    assert sleeps() == [30.0]


def test_permanent_errors_are_still_not_retried(api, sleeps):
    from backend.providers.errors import ProviderError, RetriesExhausted

    api.queue.extend([httpx.Response(401, json=err_json("nope"))] * 5)
    with pytest.raises(ProviderError) as ei:
        verify(H.make_provider(api))
    assert not isinstance(ei.value, RetriesExhausted)
    assert len(api.responses) == 1 and sleeps() == []


def test_detect_uses_the_same_schedule(api, sleeps, clip):
    from backend.providers.errors import RetriesExhausted

    api.queue.extend([httpx.ConnectError("refused")] * 20)
    with pytest.raises(RetriesExhausted) as ei:
        H.make_provider(api).detect(clip, ["x"])
    assert ei.value.attempts == 5
    assert sleeps() == [2.0, 4.0, 8.0, 16.0]
    assert len(api.deletes) == len(api.uploads) > 0
