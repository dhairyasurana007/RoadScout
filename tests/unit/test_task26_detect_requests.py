"""Task 26: the HTTP requests detect() sends (upload, one /responses call per prompt, delete)."""
import json

import pytest

import rs26_helpers as H
from rs26_conftest import _safe_env, rec, api, clips, clip, provider_factory, provider  # noqa: F401
from rs26_helpers import BASE, FILE_ID, KEY


def expected_body(phrase, file_id=FILE_ID, model="sam-3.1"):
    return {
        "model": model,
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": phrase},
            {"type": "input_video", "file_id": file_id},
        ]}],
        "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
    }


def strip_stream(body):
    body = dict(body)
    assert body.pop("stream", False) in (False, None), "detect requests must be non-streaming"
    return body


def test_upload_request_shape(api, provider, clip):
    provider.detect(clip, ["pothole"])
    up = api.uploads
    assert len(up) == 1
    r = up[0]
    assert r.url == f"{BASE}/files"
    assert r.headers["authorization"] == f"Bearer {KEY}"
    assert r.headers["content-type"].lower().startswith("multipart/form-data")
    body = r.body
    assert b'name="file"' in body
    assert b'name="purpose"' in body
    # the purpose field value is exactly user_data
    import re

    assert re.search(rb'name="purpose"[^\r\n]*\r\n(?:[^\r\n]+\r\n)*\r\nuser_data\r\n', body)
    # the real chunk bytes are in the file part
    with open(clip, "rb") as f:
        assert f.read() in body


def test_upload_happens_once_for_many_prompts(api, provider, clip):
    provider.detect(clip, ["pothole", "debris", "guardrail"])
    assert len(api.uploads) == 1
    assert len(api.responses) == 3


def test_request_order_is_upload_then_responses_then_delete(api, provider, clip):
    provider.detect(clip, ["a", "b"])
    assert api.kinds() == ["upload", "responses", "responses", "delete"]


def test_responses_body_is_exact_and_in_prompt_order(api, provider, clip):
    prompts = ["pothole in the road", "debris in lane", "bent sign"]
    provider.detect(clip, prompts)
    assert len(api.responses) == 3
    for r, phrase in zip(api.responses, prompts):
        assert r.url == f"{BASE}/responses"
        assert strip_stream(r.json) == expected_body(phrase)


def test_include_confidence_is_the_string_true(api, provider, clip):
    provider.detect(clip, ["x"])
    md = api.responses[0].json["metadata"]
    assert md["include_confidence"] == "true" and isinstance(md["include_confidence"], str)
    assert md["mask_encoding"] == "one_bit"


def test_auth_header_on_every_request(api, provider, clip):
    provider.detect(clip, ["x", "y"])
    assert len(api.requests) == 4
    for r in api.requests:
        assert r.headers["authorization"] == f"Bearer {KEY}"


def test_delete_uses_the_uploaded_id_and_responses_reference_it(api, provider, clip):
    provider.detect(clip, ["x", "y"])
    assert len(api.deletes) == 1
    d = api.deletes[0]
    assert d.url == f"{BASE}/files/{FILE_ID}"
    for r in api.responses:
        assert r.json["input"][0]["content"][1]["file_id"] == FILE_ID


def test_non_streaming(api, provider, clip):
    provider.detect(clip, ["x"])
    assert api.responses[0].json.get("stream") in (None, False)


def test_default_model_and_env_override(api, clip, monkeypatch):
    monkeypatch.setenv("ROADSCOUT_SAM_MODEL", "sam-test-9")
    H.make_provider(api).detect(clip, ["x"])
    assert api.responses[0].json["model"] == "sam-test-9"


def test_default_base_url_is_api_meta_ai_v1(api, clip):
    H.make_provider(api).detect(clip, ["x"])
    assert api.uploads[0].url == "https://api.meta.ai/v1/files"
    assert api.responses[0].url == "https://api.meta.ai/v1/responses"


def test_base_url_from_env(api, clip, monkeypatch):
    monkeypatch.setenv("ROADSCOUT_META_BASE_URL", "https://example.test/alt/v9")
    H.make_provider(api).detect(clip, ["x"])
    assert {r.url for r in api.requests if r.method == "POST"} == {
        "https://example.test/alt/v9/files", "https://example.test/alt/v9/responses"}
    assert api.deletes[0].url == f"https://example.test/alt/v9/files/{FILE_ID}"


def test_base_url_argument_beats_env(api, clip, monkeypatch):
    monkeypatch.setenv("ROADSCOUT_META_BASE_URL", "https://env.test/v1")
    H.make_provider(api, base_url="https://arg.test/api").detect(clip, ["x"])
    assert api.uploads[0].url == "https://arg.test/api/files"
    assert api.responses[0].url == "https://arg.test/api/responses"


def test_phrases_are_sent_verbatim(api, provider, clip):
    phrase = "pothole in the road surface (large) / 2"
    provider.detect(clip, [phrase])
    assert api.responses[0].json["input"][0]["content"][0]["text"] == phrase


def test_empty_prompt_list_is_an_empty_result(api, provider, clip):
    res = provider.detect(clip, [])
    assert res.is_empty and res.frames == ()
    assert api.responses == []
    # whatever was uploaded (if anything) is cleaned up
    assert len(api.deletes) == len(api.uploads)


def test_two_detect_calls_each_upload_and_delete(api, provider, clip):
    provider.detect(clip, ["x"])
    provider.detect(clip, ["x"])
    assert len(api.uploads) == 2 and len(api.deletes) == 2
    assert {d.path.rsplit("/", 1)[1] for d in api.deletes} == set(api.file_ids)


def test_body_is_valid_json_content_type(api, provider, clip):
    provider.detect(clip, ["x"])
    r = api.responses[0]
    assert "json" in r.headers["content-type"].lower()
    json.loads(r.body)


def test_upload_response_without_id_is_a_clear_provider_error(api, provider, clip):
    from backend.providers.errors import ProviderError

    api.upload_fn = lambda req: {"object": "file"}
    with pytest.raises(ProviderError) as ei:
        provider.detect(clip, ["x"])
    assert api.responses == []
    assert KEY not in str(ei.value)
