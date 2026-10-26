"""Task 3: retries with backoff, timeouts, cost log, spend counter and hard cap."""
import json
import threading
import time

import pytest

import backend.config as config
from rs3_helpers import clean_env, find, no_network  # noqa: F401


# ------------------------------------------------------------------ retries

class Flaky:
    """Fails `fails` times with `exc`, then returns 'ok'."""

    def __init__(self, fails, exc):
        self.fails, self.exc, self.calls = fails, exc, 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.fails:
            raise self.exc
        return "ok"


def retry(fn, sleeps, **kw):
    return find("call_with_retries")(fn, sleep=sleeps.append, **kw)


def test_retry_returns_value_without_sleeping_on_first_success():
    sleeps = []
    f = Flaky(0, None)
    assert retry(f, sleeps) == "ok"
    assert f.calls == 1 and sleeps == []


def test_retry_recovers_from_transient_errors():
    Transient = find("TransientProviderError")
    sleeps = []
    f = Flaky(2, Transient("503"))
    assert retry(f, sleeps, max_attempts=4, base_delay=0.5, backoff=2.0) == "ok"
    assert f.calls == 3
    assert len(sleeps) == 2, "one sleep between each failed attempt"
    assert all(s > 0 for s in sleeps)


def test_retry_backoff_grows():
    Transient = find("TransientProviderError")
    sleeps = []
    f = Flaky(10, Transient("429"))
    with pytest.raises(Exception):
        retry(f, sleeps, max_attempts=4, base_delay=1.0, backoff=2.0)
    assert len(sleeps) == 3
    assert sleeps[-1] > sleeps[0], "delays must grow (exponential backoff)"
    assert sleeps[0] >= 0.5, "first delay is about base_delay"


def test_retry_gives_up_after_max_attempts_and_raises_provider_error():
    Transient = find("TransientProviderError")
    sleeps = []
    f = Flaky(100, Transient("still down"))
    with pytest.raises(find("ProviderError")) as ei:
        retry(f, sleeps, max_attempts=3, base_delay=0.1)
    assert f.calls == 3
    assert len(sleeps) == 2, "no sleep after the final failed attempt"
    assert "still down" in str(ei.value) or "still down" in str(ei.value.__cause__ or "")


def test_retry_max_attempts_one_means_no_retry():
    Transient = find("TransientProviderError")
    sleeps = []
    f = Flaky(5, Transient("x"))
    with pytest.raises(find("ProviderError")):
        retry(f, sleeps, max_attempts=1)
    assert f.calls == 1 and sleeps == []


def test_retry_does_not_retry_permanent_provider_errors():
    sleeps = []
    f = Flaky(5, find("ProviderError")("400 bad request"))
    with pytest.raises(find("ProviderError")):
        retry(f, sleeps, max_attempts=5)
    assert f.calls == 1 and sleeps == []


def test_retry_does_not_retry_unrelated_exceptions():
    sleeps = []
    f = Flaky(5, ValueError("programming error"))
    with pytest.raises(ValueError):
        retry(f, sleeps, max_attempts=5)
    assert f.calls == 1 and sleeps == []


def test_retry_never_retries_the_spend_cap():
    sleeps = []
    f = Flaky(5, find("SpendCapExceeded")("cap"))
    with pytest.raises(find("SpendCapExceeded")):
        retry(f, sleeps, max_attempts=5)
    assert f.calls == 1 and sleeps == []


def test_retry_defaults_are_sane_and_retry_at_least_once():
    Transient = find("TransientProviderError")
    sleeps = []
    f = Flaky(1000, Transient("down"))
    with pytest.raises(find("ProviderError")):
        find("call_with_retries")(f, sleep=sleeps.append)
    assert 2 <= f.calls <= 10, "default attempts should be small but retry at least once"
    assert len(sleeps) == f.calls - 1


def test_retry_retries_timeouts():
    Timeout = find("ProviderTimeoutError")
    if not issubclass(Timeout, find("TransientProviderError")):
        pytest.skip("timeouts are not classed as transient in this design")
    sleeps = []
    f = Flaky(1, Timeout("slow"))
    assert retry(f, sleeps, max_attempts=3) == "ok"


# ------------------------------------------------------------------ timeout

def test_timeout_returns_fast_result():
    assert find("call_with_timeout")(lambda: 42, 5) == 42


def test_timeout_raises_when_call_is_too_slow():
    def slow():
        time.sleep(2.0)
        return "late"

    start = time.monotonic()
    with pytest.raises(find("ProviderTimeoutError")):
        find("call_with_timeout")(slow, 0.2)
    assert time.monotonic() - start < 1.5, "must not wait for the slow call to finish"


def test_timeout_propagates_the_calls_own_exception():
    def boom():
        raise KeyError("inner")

    with pytest.raises(KeyError):
        find("call_with_timeout")(boom, 5)


def test_timeout_error_is_a_provider_error():
    assert issubclass(find("ProviderTimeoutError"), find("ProviderError"))


# ------------------------------------------------------------- spend tracker

def tracker(tmp_path, cap, name="cost.jsonl"):
    return find("SpendTracker")(cap_usd=cap, log_path=tmp_path / name)


def cost_of(entry):
    for k in ("cost_usd", "cost", "usd"):
        if k in entry:
            return entry[k]
    raise AssertionError(f"cost log entry has no cost field: {entry}")


def test_tracker_starts_at_zero(tmp_path):
    t = tracker(tmp_path, 5.0)
    assert t.spent == 0
    assert t.remaining == pytest.approx(5.0)


def test_tracker_running_total(tmp_path):
    t = tracker(tmp_path, 5.0)
    t.charge(0.25, model="sam-3.1", operation="detect")
    t.charge(0.50, model="muse-spark", operation="verify")
    assert t.spent == pytest.approx(0.75)
    assert t.remaining == pytest.approx(4.25)


def test_tracker_allows_spending_exactly_up_to_the_cap(tmp_path):
    t = tracker(tmp_path, 1.0)
    t.charge(0.5)
    t.charge(0.5)
    assert t.spent == pytest.approx(1.0)


def test_tracker_raises_when_cap_would_be_exceeded_and_does_not_count_it(tmp_path):
    t = tracker(tmp_path, 1.0)
    t.charge(0.9)
    with pytest.raises(find("SpendCapExceeded")):
        t.charge(0.2)
    assert t.spent == pytest.approx(0.9), "a refused charge must not be added"
    t.charge(0.1)  # the remaining budget is still usable
    assert t.spent == pytest.approx(1.0)


def test_tracker_single_call_larger_than_cap_is_refused(tmp_path):
    t = tracker(tmp_path, 0.5)
    with pytest.raises(find("SpendCapExceeded")):
        t.charge(0.51)
    assert t.spent == 0


def test_tracker_zero_cap_blocks_every_paid_call(tmp_path):
    t = tracker(tmp_path, 0.0)
    with pytest.raises(find("SpendCapExceeded")):
        t.charge(0.01)
    t.charge(0.0)  # free calls (fake / fixture) are still fine


def test_tracker_rejects_negative_charges(tmp_path):
    t = tracker(tmp_path, 1.0)
    t.charge(0.5)
    with pytest.raises(ValueError):
        t.charge(-0.4)  # would otherwise be a way to dodge the cap
    assert t.spent == pytest.approx(0.5)


def test_tracker_check_precheck_does_not_spend(tmp_path):
    t = tracker(tmp_path, 1.0)
    t.charge(0.6)
    t.check(0.4)  # fits exactly, no raise
    with pytest.raises(find("SpendCapExceeded")):
        t.check(0.5)
    assert t.spent == pytest.approx(0.6)


def test_tracker_cap_exceeded_message_is_informative(tmp_path):
    t = tracker(tmp_path, 1.0)
    t.charge(0.9)
    with pytest.raises(find("SpendCapExceeded")) as ei:
        t.charge(0.5)
    msg = str(ei.value)
    assert "cap" in msg.lower() or "limit" in msg.lower()
    assert "1.0" in msg or "1.00" in msg


def test_tracker_is_thread_safe_and_cap_is_hard(tmp_path):
    t = tracker(tmp_path, 1.0)
    ok, refused = [], []

    def worker():
        try:
            t.charge(0.01)
            ok.append(1)
        except find("SpendCapExceeded"):
            refused.append(1)

    threads = [threading.Thread(target=worker) for _ in range(200)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(ok) == 100 and len(refused) == 100
    assert t.spent == pytest.approx(1.0)


# ------------------------------------------------------------------ cost log

def read_log(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_cost_log_has_one_json_line_per_charge(tmp_path):
    t = tracker(tmp_path, 10.0)
    t.charge(0.25, model="sam-3.1", operation="detect")
    t.charge(0.125, model="muse-spark", operation="verify")
    entries = read_log(tmp_path / "cost.jsonl")
    assert len(entries) == 2
    assert [cost_of(e) for e in entries] == [pytest.approx(0.25), pytest.approx(0.125)]
    assert "sam-3.1" in entries[0].values()
    assert "muse-spark" in entries[1].values()
    assert "detect" in entries[0].values()
    for e in entries:
        assert any(("time" in k or k in ("ts", "at")) for k in e), f"no timestamp in {e}"


def test_cost_log_records_running_total_if_present(tmp_path):
    t = tracker(tmp_path, 10.0)
    t.charge(0.25)
    t.charge(0.5)
    entries = read_log(tmp_path / "cost.jsonl")
    for key in ("cumulative_usd", "total_usd", "spent_usd", "running_total_usd"):
        if key in entries[-1]:
            assert entries[-1][key] == pytest.approx(0.75)
            return  # field name is optional; value is checked when present


def test_cost_log_is_appended_across_charges_not_rewritten(tmp_path):
    t = tracker(tmp_path, 10.0)
    for _ in range(5):
        t.charge(0.01)
    assert len(read_log(tmp_path / "cost.jsonl")) == 5


def test_refused_charge_is_not_logged_as_spend(tmp_path):
    t = tracker(tmp_path, 0.5)
    t.charge(0.4, model="sam-3.1")
    with pytest.raises(find("SpendCapExceeded")):
        t.charge(0.4, model="sam-3.1")
    total_logged = sum(cost_of(e) for e in read_log(tmp_path / "cost.jsonl")
                       if not any(str(v).lower() in ("refused", "rejected", "blocked")
                                  for v in e.values()))
    assert total_logged == pytest.approx(0.4)


def test_cost_log_creates_missing_parent_directories(tmp_path):
    p = tmp_path / "nested" / "dir" / "cost.jsonl"
    t = find("SpendTracker")(cap_usd=1.0, log_path=p)
    t.charge(0.1)
    assert p.exists()


def test_cost_log_never_contains_api_keys_or_frames(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "dummy-key-for-tests-0000")
    t = tracker(tmp_path, 1.0)
    t.charge(0.1, model="sam-3.1", operation="detect")
    assert "dummy-key-for-tests-0000" not in (tmp_path / "cost.jsonl").read_text()


# ------------------------------------------------------------ cap configuration

def test_default_cap_is_well_under_the_credit_budget():
    cap = config.spend_cap_usd()
    assert isinstance(cap, (int, float))
    assert 0 < cap <= 100, "default hard cap must be positive and well under the $150 credit budget"


def test_cap_is_configurable_via_env(monkeypatch):
    monkeypatch.setenv("ROADSCOUT_SPEND_CAP_USD", "12.5")
    assert config.spend_cap_usd() == pytest.approx(12.5)


def test_tracker_defaults_to_the_configured_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("ROADSCOUT_SPEND_CAP_USD", "2.0")
    t = find("SpendTracker")(log_path=tmp_path / "c.jsonl")
    t.charge(1.5)
    with pytest.raises(find("SpendCapExceeded")):
        t.charge(1.0)


def test_tracker_default_cap_without_env_is_the_safe_default(tmp_path):
    t = find("SpendTracker")(log_path=tmp_path / "c.jsonl")
    assert t.remaining == pytest.approx(config.spend_cap_usd())
    assert t.remaining <= 100
