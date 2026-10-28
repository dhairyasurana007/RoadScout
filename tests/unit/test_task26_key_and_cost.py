"""Task 26: the key never leaks, a live call needs a key, spend cap and cost estimates."""
import json
import logging

import pytest

import rs26_helpers as H
from rs26_conftest import _safe_env, rec, api, clips, clip, provider_factory, provider  # noqa: F401
from rs26_helpers import KEY, frame_line, jpeg_bytes, rec_str, verdict_obj


def good_line(n=1):
    return frame_line(n, rec_str(0, 4, 4, 13, 13, 100, 80, c=0.5))


def verify(p):
    return p.verify([jpeg_bytes()], class_prompt="pothole", prompt_version="v1", prompt_text="check")


# -------------------------------------------------------------------- key

def test_key_not_in_repr_or_str(api, provider):
    for s in (repr(provider), str(provider), format(provider), f"{provider!r}"):
        assert KEY not in s


def test_key_not_in_repr_after_calls(api, provider, clip):
    api.default_text = good_line(1)
    provider.detect(clip, ["x"])
    api.queue.append(json.dumps(verdict_obj()))
    verify(provider)
    assert KEY not in repr(provider) and KEY not in str(provider)


def test_key_not_in_logs(api, provider, clip, caplog, rec):
    caplog.set_level(logging.DEBUG)
    api.default_text = good_line(1)
    provider.detect(clip, ["x"])
    api.queue.append(json.dumps(verdict_obj()))
    verify(provider)
    api.queue.extend([H.httpx.Response(500, json=H.err_json("boom"))] * 5)
    with pytest.raises(Exception):
        verify(provider)
    assert KEY not in caplog.text
    for r in caplog.records:
        assert KEY not in r.getMessage()
        assert KEY not in repr(r.args)


def test_key_not_printed(api, provider, clip, capsys, rec):
    api.default_text = good_line(1)
    provider.detect(clip, ["x"])
    api.queue.append(json.dumps(verdict_obj()))
    verify(provider)
    out = capsys.readouterr()
    assert KEY not in out.out and KEY not in out.err


def test_key_not_in_the_request_body_or_url(api, provider, clip):
    api.default_text = good_line(1)
    provider.detect(clip, ["x"])
    for r in api.requests:
        assert KEY not in r.url
        assert KEY.encode() not in r.body


def test_key_from_environment_is_used(api, clip, monkeypatch):
    from backend.providers.cost import SpendTracker
    from backend.providers.live import LiveProvider

    monkeypatch.setenv("META_API_KEY", KEY)
    p = LiveProvider(client=api.client(), tracker=SpendTracker(cap_usd=10, log_path=None))
    api.default_text = good_line(1)
    p.detect(clip, ["x"])
    assert all(r.headers["authorization"] == f"Bearer {KEY}" for r in api.requests)
    assert KEY not in repr(p)


@pytest.mark.parametrize("op", ["detect", "verify"])
def test_no_key_means_no_call(api, clip, op):
    from backend.providers.cost import SpendTracker
    from backend.providers.errors import ProviderConfigError
    from backend.providers.live import LiveProvider

    with pytest.raises(ProviderConfigError) as ei:
        p = LiveProvider(client=api.client(), tracker=SpendTracker(cap_usd=10, log_path=None))
        if op == "detect":
            p.detect(clip, ["x"])
        else:
            verify(p)
    assert "META_API_KEY" in str(ei.value)
    assert api.requests == []


def test_blank_key_means_no_call(api, clip, monkeypatch):
    from backend.providers.cost import SpendTracker
    from backend.providers.errors import ProviderConfigError
    from backend.providers.live import LiveProvider

    monkeypatch.setenv("META_API_KEY", "")
    with pytest.raises(ProviderConfigError):
        LiveProvider(client=api.client(), tracker=SpendTracker(cap_usd=10, log_path=None)).detect(clip, ["x"])
    assert api.requests == []


def test_get_provider_live_builds_a_live_provider_with_kwargs(api):
    from backend.providers import get_provider
    from backend.providers.cost import SpendTracker
    from backend.providers.live import LiveProvider

    p = get_provider("live", api_key=KEY, client=api.client(), tracker=SpendTracker(cap_usd=1, log_path=None))
    assert isinstance(p, LiveProvider)
    assert p.name == "live"
    assert KEY not in repr(p)


def test_default_client_is_built_without_network(monkeypatch):
    from backend.providers.cost import SpendTracker
    from backend.providers.live import LiveProvider

    p = LiveProvider(api_key=KEY, tracker=SpendTracker(cap_usd=1, log_path=None))
    assert KEY not in repr(p)


# -------------------------------------------------------------------- cost

def test_estimates_default(api, provider):
    assert provider.estimate_cost("detect") == pytest.approx(0.012)
    assert provider.estimate_cost("verify") == pytest.approx(0.007)


def test_estimates_env_override(api, monkeypatch):
    monkeypatch.setenv("ROADSCOUT_COST_PER_DETECT_USD", "0.5")
    monkeypatch.setenv("ROADSCOUT_COST_PER_VERIFY_USD", "0.25")
    p = H.make_provider(api)
    assert p.estimate_cost("detect") == pytest.approx(0.5)
    assert p.estimate_cost("verify") == pytest.approx(0.25)


def test_detect_charges_per_prompt(api, clip):
    from backend.providers.cost import SpendTracker

    tr = SpendTracker(cap_usd=10, log_path=None)
    p = H.make_provider(api, tracker=tr)
    api.default_text = good_line(1)
    p.detect(clip, ["a", "b", "c"])
    assert tr.spent == pytest.approx(0.036)
    assert all(e["cost_usd"] > 0 for e in tr.entries)


def test_detect_one_prompt_charges_one_estimate(api, clip):
    from backend.providers.cost import SpendTracker

    tr = SpendTracker(cap_usd=10, log_path=None)
    H.make_provider(api, tracker=tr).detect(clip, ["a"])
    assert tr.spent == pytest.approx(0.012)


def test_verify_charges_the_estimate(api):
    from backend.providers.cost import SpendTracker

    tr = SpendTracker(cap_usd=10, log_path=None)
    p = H.make_provider(api, tracker=tr)
    api.queue.append(json.dumps(verdict_obj()))
    verify(p)
    assert tr.spent == pytest.approx(0.007)
    assert [e["operation"] for e in tr.entries] == ["verify"]
    assert tr.entries[0]["model"] == "muse-spark-1.3" or tr.entries[0]["model"]


def test_env_detect_cost_is_per_prompt(api, clip, monkeypatch):
    from backend.providers.cost import SpendTracker

    monkeypatch.setenv("ROADSCOUT_COST_PER_DETECT_USD", "0.05")
    tr = SpendTracker(cap_usd=10, log_path=None)
    H.make_provider(api, tracker=tr).detect(clip, ["a", "b"])
    assert tr.spent == pytest.approx(0.10)


def test_env_verify_cost_is_charged(api, monkeypatch):
    from backend.providers.cost import SpendTracker

    monkeypatch.setenv("ROADSCOUT_COST_PER_VERIFY_USD", "0.3")
    tr = SpendTracker(cap_usd=10, log_path=None)
    p = H.make_provider(api, tracker=tr)
    api.queue.append(json.dumps(verdict_obj()))
    verify(p)
    assert tr.spent == pytest.approx(0.3)


def test_cost_log_entries_hold_no_key(api, clip, tmp_path):
    from backend.providers.cost import SpendTracker

    log = tmp_path / "cost.jsonl"
    tr = SpendTracker(cap_usd=10, log_path=log)
    api.default_text = good_line(1)
    H.make_provider(api, tracker=tr).detect(clip, ["a"])
    assert log.exists() and KEY not in log.read_text(encoding="utf-8")
    assert str(clip) not in log.read_text(encoding="utf-8")


# -------------------------------------------------------------------- cap

def test_cap_blocks_detect_before_any_http_traffic(api, clip):
    from backend.providers.cost import SpendTracker
    from backend.providers.errors import SpendCapExceeded

    tr = SpendTracker(cap_usd=0.005, log_path=None)
    p = H.make_provider(api, tracker=tr)
    with pytest.raises(SpendCapExceeded):
        p.detect(clip, ["a"])
    assert api.requests == []
    assert tr.spent == 0


def test_cap_blocks_verify_before_any_http_traffic(api):
    from backend.providers.cost import SpendTracker
    from backend.providers.errors import SpendCapExceeded

    tr = SpendTracker(cap_usd=0.005, log_path=None)
    p = H.make_provider(api, tracker=tr)
    with pytest.raises(SpendCapExceeded):
        verify(p)
    assert api.requests == []
    assert tr.spent == 0


def test_cap_already_spent_blocks_everything(api, clip):
    from backend.providers.cost import SpendTracker
    from backend.providers.errors import SpendCapExceeded

    tr = SpendTracker(cap_usd=1.0, log_path=None)
    tr.charge(1.0, model="m", operation="detect")
    p = H.make_provider(api, tracker=tr)
    with pytest.raises(SpendCapExceeded):
        p.detect(clip, ["a"])
    with pytest.raises(SpendCapExceeded):
        verify(p)
    assert api.requests == []


def test_cap_is_checked_against_the_env_estimate(api, clip, monkeypatch):
    from backend.providers.cost import SpendTracker
    from backend.providers.errors import SpendCapExceeded

    monkeypatch.setenv("ROADSCOUT_COST_PER_DETECT_USD", "0.5")
    tr = SpendTracker(cap_usd=0.4, log_path=None)
    with pytest.raises(SpendCapExceeded):
        H.make_provider(api, tracker=tr).detect(clip, ["a"])
    assert api.requests == []


def test_cap_exactly_fits_one_detect_prompt(api, clip):
    from backend.providers.cost import SpendTracker

    tr = SpendTracker(cap_usd=0.012, log_path=None)
    p = H.make_provider(api, tracker=tr)
    p.detect(clip, ["a"])
    assert tr.spent == pytest.approx(0.012)


def test_cap_stops_a_multi_prompt_detect_that_no_longer_fits(api, clip):
    from backend.providers.cost import SpendTracker
    from backend.providers.errors import SpendCapExceeded

    tr = SpendTracker(cap_usd=0.015, log_path=None)
    p = H.make_provider(api, tracker=tr)
    with pytest.raises(SpendCapExceeded):
        p.detect(clip, ["a", "b", "c"])
    assert len(api.responses) <= 1
    assert tr.spent <= 0.015 + 1e-9


def test_cap_error_is_not_retried(api, clip, rec):
    from backend.providers.cost import SpendTracker
    from backend.providers.errors import SpendCapExceeded

    tr = SpendTracker(cap_usd=0.001, log_path=None)
    with pytest.raises(SpendCapExceeded):
        H.make_provider(api, tracker=tr).detect(clip, ["a"])
    assert rec.sleeps == []


def test_failed_permanent_call_does_not_over_charge(api, clip):
    """Whatever is charged on a failure, it never exceeds the estimate for the prompts attempted."""
    from backend.providers.cost import SpendTracker
    from backend.providers.errors import ProviderError

    tr = SpendTracker(cap_usd=10, log_path=None)
    p = H.make_provider(api, tracker=tr)
    api.queue.append(H.httpx.Response(400, json=H.err_json("bad")))
    with pytest.raises(ProviderError):
        p.detect(clip, ["a", "b", "c"])
    assert tr.spent <= 0.036 + 1e-9
