"""Task 26: error classification, retries with Retry-After, cleanup of the upload, error text hygiene."""
import json
from pathlib import Path

import httpx
import pytest

import rs26_helpers as H
from rs26_conftest import _safe_env, rec, api, clips, clip, provider_factory, provider  # noqa: F401
from rs26_helpers import FILE_ID, KEY, err_json, frame_line, jpeg_bytes, out_json, rec_str, verdict_obj

TRANSIENT = [429, 500, 503]
PERMANENT = [400, 401, 402, 403, 404, 413, 504]
PHRASE = "unique-phrase-for-error-tests-5521"


def resp(status, message="server says no", headers=None, body=None):
    if body is not None:
        return httpx.Response(status, content=body, headers=headers or {})
    return httpx.Response(status, json=err_json(message, "t", "c"), headers=headers or {})


def good_line(n=1):
    return frame_line(n, rec_str(0, 4, 4, 13, 13, 100, 80, c=0.5))


def verify(provider):
    return provider.verify([jpeg_bytes()], class_prompt="pothole", prompt_version="v1", prompt_text="check it")


def n_responses(api):
    return len(api.responses)


# ------------------------------------------------------------- classification

@pytest.mark.parametrize("status", PERMANENT)
def test_permanent_status_is_not_retried_verify(api, provider, status):
    from backend.providers.errors import ProviderError, RetriesExhausted, TransientProviderError

    api.queue.extend([resp(status, "permanent failure text")] * 5)
    with pytest.raises(ProviderError) as ei:
        verify(provider)
    assert not isinstance(ei.value, (TransientProviderError, RetriesExhausted))
    assert n_responses(api) == 1
    assert "permanent failure text" in str(ei.value)


@pytest.mark.parametrize("status", PERMANENT)
def test_permanent_status_is_not_retried_detect(api, provider, clip, status):
    from backend.providers.errors import ProviderError, RetriesExhausted, TransientProviderError

    api.queue.extend([resp(status, "permanent failure text")] * 5)
    with pytest.raises(ProviderError) as ei:
        provider.detect(clip, [PHRASE, "second"])
    assert not isinstance(ei.value, (TransientProviderError, RetriesExhausted))
    assert n_responses(api) == 1, "a permanent error must stop the whole detect call"
    assert api.kinds()[-1] == "delete"


@pytest.mark.parametrize("status", TRANSIENT)
def test_transient_status_is_retried_then_succeeds_verify(api, provider, status, rec):
    api.queue.extend([resp(status), json.dumps(verdict_obj())])
    v = verify(provider)
    assert v.hazard_class == "pothole"
    assert n_responses(api) == 2


@pytest.mark.parametrize("status", TRANSIENT)
def test_transient_status_is_retried_then_succeeds_detect(api, provider, clip, status, rec):
    api.queue.extend([resp(status), good_line(1)])
    res = provider.detect(clip, ["x"])
    assert [f.frame_index for f in res.frames] == [1]
    assert n_responses(api) == 2
    assert {d.path.rsplit("/", 1)[1] for d in api.deletes} == set(api.file_ids)


@pytest.mark.parametrize("status", TRANSIENT)
def test_transient_status_exhausts_retries(api, provider, status, rec):
    from backend.providers.errors import ProviderError, RetriesExhausted

    api.queue.extend([resp(status, "try again later")] * 20)
    with pytest.raises(ProviderError) as ei:
        verify(provider)
    assert isinstance(ei.value, RetriesExhausted)
    assert ei.value.attempts >= 2
    assert n_responses(api) == ei.value.attempts
    assert "try again later" in str(ei.value)
    assert len(rec.sleeps) >= 1


@pytest.mark.parametrize("exc", [
    httpx.ConnectError("connection refused"),
    httpx.ReadTimeout("read timed out"),
    httpx.ConnectTimeout("connect timed out"),
    httpx.RemoteProtocolError("server dropped"),
])
def test_transport_errors_are_transient_and_retried(api, provider, exc, rec):
    api.queue.extend([exc, json.dumps(verdict_obj())])
    assert verify(provider).hazard_class == "pothole"
    assert n_responses(api) == 2


@pytest.mark.parametrize("exc", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
def test_transport_errors_exhaust_as_provider_error(api, provider, exc, rec):
    from backend.providers.errors import ProviderError

    api.queue.extend([exc] * 20)
    with pytest.raises(ProviderError) as ei:
        verify(provider)
    assert n_responses(api) >= 2
    assert not isinstance(ei.value, httpx.HTTPError), "raw httpx errors must not leak"


def test_transport_error_during_detect_cleans_up(api, provider, clip, rec):
    from backend.providers.errors import ProviderError

    api.queue.extend([httpx.ConnectError("refused")] * 20)
    with pytest.raises(ProviderError):
        provider.detect(clip, ["x"])
    assert {d.path.rsplit("/", 1)[1] for d in api.deletes} == set(api.file_ids)
    assert api.file_ids


def test_non_json_error_body_does_not_crash(api, provider):
    from backend.providers.errors import ProviderError

    api.queue.extend([resp(400, body="<html><body>Bad request</body></html>", headers={"content-type": "text/html"})] * 3)
    with pytest.raises(ProviderError) as ei:
        verify(provider)
    assert str(ei.value).strip()


def test_empty_error_body_does_not_crash(api, provider):
    from backend.providers.errors import ProviderError

    api.queue.extend([httpx.Response(403)] * 3)
    with pytest.raises(ProviderError) as ei:
        verify(provider)
    assert str(ei.value).strip()


def test_error_message_comes_from_the_json_body(api, provider):
    from backend.providers.errors import ProviderError

    api.queue.append(httpx.Response(402, json=err_json("Insufficient credits for this request", "billing", "no_credit")))
    with pytest.raises(ProviderError) as ei:
        verify(provider)
    assert "Insufficient credits for this request" in str(ei.value)


def test_upload_413_is_permanent(api, provider, clip):
    from backend.providers.errors import ProviderError, RetriesExhausted

    api.upload_fn = lambda req: resp(413, "file too large")
    with pytest.raises(ProviderError) as ei:
        provider.detect(clip, ["x"])
    assert not isinstance(ei.value, RetriesExhausted)
    assert len(api.uploads) == 1
    assert api.responses == []
    assert "file too large" in str(ei.value)


def test_upload_transient_failure_is_retried(api, provider, clip, rec):
    state = {"n": 0}

    def upload(req):
        state["n"] += 1
        if state["n"] == 1:
            return resp(503, "busy")
        return None

    api.upload_fn = upload
    api.default_text = good_line(2)
    res = provider.detect(clip, ["x"])
    assert [f.frame_index for f in res.frames] == [2]
    assert len(api.uploads) >= 2


# ------------------------------------------------------------- Retry-After

def _total_sleep_between_requests(rec, first_idx=0):
    """Total seconds slept between the first two /responses requests."""
    seen = 0
    total = 0.0
    counting = False
    for e in rec.events:
        if e[0] == "req" and e[1] == "POST" and e[2].endswith("/responses"):
            seen += 1
            if seen == 1:
                counting = True
                continue
            if seen == 2:
                break
        elif e[0] == "sleep" and counting:
            total += e[1]
    return total


@pytest.mark.parametrize("retry_after", ["7", "2", "15"])
def test_retry_after_is_honoured(api, provider, rec, retry_after):
    api.queue.extend([resp(429, "slow down", headers={"Retry-After": retry_after}), json.dumps(verdict_obj())])
    verify(provider)
    waited = _total_sleep_between_requests(rec)
    assert waited >= float(retry_after) - 1e-6, f"waited {waited}s, Retry-After was {retry_after}"
    assert waited <= 35


def test_retry_after_is_capped_at_30_seconds(api, provider, rec):
    api.queue.extend([resp(503, "later", headers={"Retry-After": "600"}), json.dumps(verdict_obj())])
    verify(provider)
    waited = _total_sleep_between_requests(rec)
    assert waited >= 30 - 1e-6, f"waited only {waited}s"
    assert waited <= 35, f"waited {waited}s: the cap is 30 s"
    assert max(rec.sleeps) <= 35


def test_retry_after_on_500_is_honoured_too(api, provider, rec):
    api.queue.extend([resp(500, "oops", headers={"Retry-After": "9"}), json.dumps(verdict_obj())])
    verify(provider)
    assert _total_sleep_between_requests(rec) >= 9 - 1e-6


def test_retry_after_zero_does_not_block(api, provider, rec):
    api.queue.extend([resp(429, "x", headers={"Retry-After": "0"}), json.dumps(verdict_obj())])
    assert verify(provider).hazard_class == "pothole"


@pytest.mark.parametrize("value", ["soon", "", "-5", "abc123"])
def test_unusable_retry_after_falls_back_to_normal_backoff(api, provider, rec, value):
    api.queue.extend([resp(429, "x", headers={"Retry-After": value}), json.dumps(verdict_obj())])
    assert verify(provider).hazard_class == "pothole"
    assert n_responses(api) == 2
    assert all(s >= 0 for s in rec.sleeps)
    assert max(rec.sleeps, default=0) <= 35


def test_no_retry_after_header_still_backs_off_a_little(api, provider, rec):
    api.queue.extend([resp(503, "x"), json.dumps(verdict_obj())])
    verify(provider)
    assert all(0 <= s <= 35 for s in rec.sleeps)


def test_retry_after_on_detect_responses_call(api, provider, clip, rec):
    api.queue.extend([resp(429, "slow", headers={"Retry-After": "11"}), good_line(1)])
    provider.detect(clip, ["x"])
    assert _total_sleep_between_requests(rec) >= 11 - 1e-6


# ------------------------------------------------------------- cleanup

def test_upload_is_deleted_after_success(api, provider, clip):
    api.default_text = good_line(1)
    provider.detect(clip, ["x"])
    assert [d.path.rsplit("/", 1)[1] for d in api.deletes] == [FILE_ID]


@pytest.mark.parametrize("status", [400, 401, 404, 413, 504])
def test_upload_is_deleted_after_a_permanent_error_on_the_first_prompt(api, provider, clip, status):
    from backend.providers.errors import ProviderError

    api.queue.append(resp(status, "nope"))
    with pytest.raises(ProviderError):
        provider.detect(clip, ["x", "y"])
    assert [d.path.rsplit("/", 1)[1] for d in api.deletes] == [FILE_ID]


def test_upload_is_deleted_after_an_error_on_a_later_prompt(api, provider, clip):
    from backend.providers.errors import ProviderError

    api.queue.extend([good_line(1), resp(400, "bad second")])
    with pytest.raises(ProviderError) as ei:
        provider.detect(clip, ["a", "b", "c"])
    assert "bad second" in str(ei.value)
    assert n_responses(api) == 2
    assert [d.path.rsplit("/", 1)[1] for d in api.deletes] == [FILE_ID]


def test_delete_failure_is_ignored_on_success(api, provider, clip):
    api.default_text = good_line(3)
    api.delete_fn = lambda req: resp(500, "cannot delete")
    res = provider.detect(clip, ["x"])
    assert [f.frame_index for f in res.frames] == [3]


@pytest.mark.parametrize("exc", [httpx.ConnectError("no route"), httpx.ReadTimeout("slow")])
def test_delete_transport_failure_is_ignored(api, provider, clip, exc, rec):
    api.default_text = good_line(3)
    api.delete_fn = lambda req: exc
    res = provider.detect(clip, ["x"])
    assert [f.frame_index for f in res.frames] == [3]


def test_delete_404_is_ignored(api, provider, clip):
    api.default_text = good_line(3)
    api.delete_fn = lambda req: resp(404, "gone already")
    assert provider.detect(clip, ["x"]).frames


def test_delete_failure_never_hides_the_real_error(api, provider, clip, rec):
    from backend.providers.errors import ProviderError

    api.queue.append(resp(400, "the real problem"))
    api.delete_fn = lambda req: resp(500, "delete exploded")
    with pytest.raises(ProviderError) as ei:
        provider.detect(clip, ["x"])
    assert "the real problem" in str(ei.value)
    assert "delete exploded" not in str(ei.value)


def test_delete_transport_failure_never_hides_the_real_error(api, provider, clip, rec):
    from backend.providers.errors import ProviderError

    api.queue.append(resp(401, "the real problem"))
    api.delete_fn = lambda req: httpx.ConnectError("delete net down")
    with pytest.raises(ProviderError) as ei:
        provider.detect(clip, ["x"])
    assert "the real problem" in str(ei.value)


def test_unparseable_output_does_not_leave_the_upload_behind(api, provider, clip):
    api.default_text = "complete garbage \x00 <3f> <|box"
    provider.detect(clip, ["x"])
    assert len(api.deletes) == 1


def test_upload_deleted_when_the_spend_cap_trips_mid_call(api, clip):
    from backend.providers.cost import SpendTracker
    from backend.providers.errors import SpendCapExceeded

    tr = SpendTracker(cap_usd=0.015, log_path=None)
    p = H.make_provider(api, tracker=tr)
    api.default_text = good_line(1)
    with pytest.raises(SpendCapExceeded):
        p.detect(clip, ["a", "b"])
    assert n_responses(api) <= 1
    assert {d.path.rsplit("/", 1)[1] for d in api.deletes} == set(api.file_ids)
    assert tr.spent <= 0.015 + 1e-9


# ------------------------------------------------------------- error text hygiene

def _msgs(exc):
    out = [str(exc), repr(exc)]
    cause = exc.__cause__
    while cause is not None:
        out += [str(cause), repr(cause)]
        cause = cause.__cause__
    return out


@pytest.mark.parametrize("status", PERMANENT + TRANSIENT)
def test_errors_never_contain_key_prompt_or_paths_detect(api, provider, clip, rec, status):
    from backend.providers.errors import ProviderError

    api.queue.extend([resp(status, "generic failure")] * 20)
    with pytest.raises(ProviderError) as ei:
        provider.detect(clip, [PHRASE])
    for m in _msgs(ei.value):
        assert KEY not in m
        assert PHRASE not in m, "the request body must not be echoed"
        assert str(clip) not in m
        assert Path(clip).name not in m
        assert str(Path(clip).parent) not in m


@pytest.mark.parametrize("status", PERMANENT + TRANSIENT)
def test_errors_never_contain_key_prompt_or_paths_verify(api, provider, tmp_path, rec, status):
    from backend.providers.errors import ProviderError

    img = tmp_path / "uniqueimagename_6619.jpg"
    img.write_bytes(jpeg_bytes())
    api.queue.extend([resp(status, "generic failure")] * 20)
    with pytest.raises(ProviderError) as ei:
        provider.verify([img], class_prompt="pothole", prompt_version="v1", prompt_text="UNIQUE-VERIFY-PROMPT-7710")
    for m in _msgs(ei.value):
        assert KEY not in m
        assert "UNIQUE-VERIFY-PROMPT-7710" not in m
        assert "uniqueimagename_6619" not in m
        assert str(tmp_path) not in m
        assert "/9j/" not in m  # base64 of the image


def test_transport_error_text_has_no_key_or_path(api, provider, clip, rec):
    from backend.providers.errors import ProviderError

    api.queue.extend([httpx.ConnectError("connection refused")] * 20)
    with pytest.raises(ProviderError) as ei:
        provider.detect(clip, [PHRASE])
    for m in _msgs(ei.value):
        assert KEY not in m
        assert Path(clip).name not in m


def test_server_message_echoing_the_key_is_not_passed_on(api, provider):
    """Strict hygiene: even if a server echoes the credential in its error text, it must not reach us."""
    from backend.providers.errors import ProviderError

    api.queue.append(httpx.Response(401, json=err_json(f"Incorrect API key provided: {KEY}", "auth", "bad_key")))
    with pytest.raises(ProviderError) as ei:
        verify(provider)
    for m in _msgs(ei.value):
        assert KEY not in m
