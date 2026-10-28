"""Task 26: verify() request shape, image handling and verdict mapping."""
import base64
import json
from pathlib import Path

import pytest

import rs26_helpers as H
from rs26_conftest import _safe_env, rec, api, clips, clip, provider_factory, provider  # noqa: F401
from rs26_helpers import BASE, KEY, data_url, jpeg_bytes, png_bytes, verdict_obj

PROMPT = "Check this hazard. UNIQUE-PROMPT-TEXT-4471"


def verify(provider, images, **kw):
    kw.setdefault("class_prompt", "pothole")
    kw.setdefault("prompt_version", "v1")
    kw.setdefault("prompt_text", PROMPT)
    return provider.verify(images, **kw)


def reply(api, obj=None, text=None):
    api.queue.append(text if text is not None else json.dumps(obj if obj is not None else verdict_obj()))


def strip_stream(body):
    body = dict(body)
    assert body.pop("stream", False) in (False, None)
    return body


# ----------------------------------------------------------------- request

def test_request_body_is_exact(api, provider):
    a, b = jpeg_bytes(), png_bytes()
    reply(api)
    verify(provider, [a, b])
    assert len(api.requests) == 1
    r = api.requests[0]
    assert r.method == "POST" and r.url == f"{BASE}/responses"
    assert r.headers["authorization"] == f"Bearer {KEY}"
    assert strip_stream(r.json) == {
        "model": "muse-spark-1.3",
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": PROMPT},
            {"type": "input_image", "image_url": data_url("image/jpeg", a)},
            {"type": "input_image", "image_url": data_url("image/png", b)},
        ]}],
        "text": {"format": {"type": "json_schema", "name": "verdict", "schema": H.verification_schema()}},
    }


def test_no_files_endpoint_is_used_for_verification(api, provider):
    reply(api)
    verify(provider, [jpeg_bytes()])
    assert api.kinds() == ["responses"]


def test_images_keep_their_order(api, provider):
    imgs = [jpeg_bytes((i * 20, 0, 0), (8 + i, 8)) for i in range(5)]
    reply(api)
    verify(provider, imgs)
    parts = api.requests[0].json["input"][0]["content"][1:]
    assert [p["image_url"] for p in parts] == [data_url("image/jpeg", b) for b in imgs]


def test_paths_str_and_pathlib_and_bytes(api, provider, tmp_path):
    a, b, c = jpeg_bytes((1, 2, 3)), png_bytes((4, 5, 6)), jpeg_bytes((7, 8, 9), (9, 9))
    pa, pb = tmp_path / "a.jpg", tmp_path / "b.png"
    pa.write_bytes(a)
    pb.write_bytes(b)
    reply(api)
    verify(provider, [str(pa), pb, c])
    urls = [p["image_url"] for p in api.requests[0].json["input"][0]["content"][1:]]
    assert urls == [data_url("image/jpeg", a), data_url("image/png", b), data_url("image/jpeg", c)]


def test_mime_is_sniffed_from_magic_bytes_not_the_extension(api, provider, tmp_path):
    png_named_jpg = tmp_path / "really_png.jpg"
    jpg_named_png = tmp_path / "really_jpeg.png"
    png_named_jpg.write_bytes(png_bytes())
    jpg_named_png.write_bytes(jpeg_bytes())
    reply(api)
    verify(provider, [png_named_jpg, jpg_named_png])
    urls = [p["image_url"] for p in api.requests[0].json["input"][0]["content"][1:]]
    assert urls[0].startswith("data:image/png;base64,")
    assert urls[1].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize("raw", [b"GIF89a-not-supported", b"RIFF\x00\x00\x00\x00WEBPVP8 ", b"\x00\x01\x02\x03", b"x"])
def test_unknown_magic_defaults_to_jpeg(api, provider, raw):
    reply(api)
    verify(provider, [raw])
    url = api.requests[0].json["input"][0]["content"][1]["image_url"]
    assert url == data_url("image/jpeg", raw)


def test_png_signature_only_is_png(api, provider):
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
    reply(api)
    verify(provider, [raw])
    assert api.requests[0].json["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")


def test_jpeg_signature_only_is_jpeg(api, provider):
    raw = b"\xff\xd8\xff\xe0" + b"\x00" * 10
    reply(api)
    verify(provider, [raw])
    assert api.requests[0].json["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")


def test_base64_roundtrips_the_exact_bytes(api, provider):
    raw = jpeg_bytes((12, 200, 90), (31, 17))
    reply(api)
    verify(provider, [raw])
    url = api.requests[0].json["input"][0]["content"][1]["image_url"]
    assert base64.b64decode(url.split(",", 1)[1]) == raw


def test_fifty_images_are_allowed(api, provider):
    reply(api)
    verify(provider, [jpeg_bytes()] * 50)
    assert len(api.requests[0].json["input"][0]["content"]) == 51


def test_more_than_fifty_images_raise_before_any_call(api, provider):
    from backend.providers.errors import ProviderError, TransientProviderError

    with pytest.raises(ProviderError) as ei:
        verify(provider, [jpeg_bytes()] * 51)
    assert not isinstance(ei.value, TransientProviderError)
    assert api.requests == []
    assert "50" in str(ei.value)


def test_too_many_images_does_not_spend(api):
    from backend.providers.cost import SpendTracker
    from backend.providers.errors import ProviderError

    tr = SpendTracker(cap_usd=100, log_path=None)
    p = H.make_provider(api, tracker=tr)
    with pytest.raises(ProviderError):
        verify(p, [jpeg_bytes()] * 60)
    assert tr.spent == 0


def test_model_env_override(api, monkeypatch):
    monkeypatch.setenv("ROADSCOUT_VERIFY_MODEL", "muse-spark-test-2")
    p = H.make_provider(api)
    reply(api)
    v = verify(p, [jpeg_bytes()])
    assert api.requests[0].json["model"] == "muse-spark-test-2"
    assert v.model == "muse-spark-test-2"


def test_prompt_text_is_sent_verbatim(api, provider):
    reply(api)
    verify(provider, [jpeg_bytes()], prompt_text="line1\nline2 {braces} \"quotes\"")
    assert api.requests[0].json["input"][0]["content"][0] == {
        "type": "input_text", "text": "line1\nline2 {braces} \"quotes\""}


# ----------------------------------------------------------------- verdict

def test_verdict_mapping(api, provider):
    reply(api, verdict_obj())
    v = verify(provider, [jpeg_bytes()], prompt_version="v7")
    from backend.providers.types import Verdict

    assert isinstance(v, Verdict)
    assert v.is_hazard is True
    assert v.hazard_class == "pothole"
    assert v.severity == 2
    assert v.lane == "left"
    assert v.confidence == pytest.approx(0.83)
    assert v.reason == "Dark crater with broken edges."
    assert v.report_description == "A pothole in the left lane."
    assert v.model == "muse-spark-1.3"
    assert v.prompt_version == "v7"


def test_prompt_version_defaults_to_empty_string(api, provider):
    reply(api)
    v = provider.verify([jpeg_bytes()], prompt_text=PROMPT)
    assert v.prompt_version == ""


def test_lane_null_maps_to_none(api, provider):
    reply(api, verdict_obj(lane=None))
    assert verify(provider, [jpeg_bytes()]).lane is None


def test_not_a_hazard_with_severity_zero(api, provider):
    reply(api, verdict_obj(is_hazard=False, severity=0, confidence=0.1, lane=None, **{"class": "none"}))
    v = verify(provider, [jpeg_bytes()])
    assert v.is_hazard is False and v.severity == 0 and v.hazard_class == "none"


@pytest.mark.parametrize("sev", [1, 2, 3])
def test_severity_range(api, provider, sev):
    reply(api, verdict_obj(severity=sev))
    assert verify(provider, [jpeg_bytes()]).severity == sev


def test_confidence_bounds(api, provider):
    reply(api, verdict_obj(confidence=0))
    assert verify(provider, [jpeg_bytes()]).confidence == 0.0
    reply(api, verdict_obj(confidence=1))
    assert verify(provider, [jpeg_bytes()]).confidence == 1.0


@pytest.mark.parametrize("fence", ["```json\n{}\n```", "```\n{}\n```", "  ```JSON\n{}\n```  \n", "```json\r\n{}\r\n```"])
def test_markdown_fences_are_tolerated(api, provider, fence):
    reply(api, text=fence.replace("{}", json.dumps(verdict_obj(reason="fenced ok"))))
    assert verify(provider, [jpeg_bytes()]).reason == "fenced ok"


def test_surrounding_whitespace_is_tolerated(api, provider):
    reply(api, text="\n\n  " + json.dumps(verdict_obj()) + "  \n")
    assert verify(provider, [jpeg_bytes()]).hazard_class == "pothole"


def test_reasoning_item_before_the_message_is_tolerated(api, provider):
    api.queue.append(H.out_json(json.dumps(verdict_obj()), extra_items=[
        {"type": "reasoning", "id": "rs_1", "summary": []}]))
    assert verify(provider, [jpeg_bytes()]).is_hazard is True


def test_reply_text_split_over_parts_is_concatenated(api, provider):
    t = json.dumps(verdict_obj())
    api.queue.append({"output": [{"type": "message", "content": [
        {"type": "output_text", "text": t[:20]}, {"type": "output_text", "text": t[20:]}]}]})
    assert verify(provider, [jpeg_bytes()]).hazard_class == "pothole"


def test_extra_keys_are_ignored(api, provider):
    reply(api, verdict_obj(extra_thing="x"))
    assert verify(provider, [jpeg_bytes()]).hazard_class == "pothole"


BAD_REPLIES = {
    "not_json": "I think it is a pothole.",
    "empty": "",
    "json_array": "[1, 2, 3]",
    "json_string": '"pothole"',
    "truncated": '{"is_hazard": true, "class": "pothole"',
    "missing_class": {k: v for k, v in verdict_obj().items() if k != "class"},
    "missing_lane": {k: v for k, v in verdict_obj().items() if k != "lane"},
    "missing_reason": {k: v for k, v in verdict_obj().items() if k != "reason"},
    "missing_description": {k: v for k, v in verdict_obj().items() if k != "report_description"},
    "missing_confidence": {k: v for k, v in verdict_obj().items() if k != "confidence"},
    "missing_severity": {k: v for k, v in verdict_obj().items() if k != "severity"},
    "missing_is_hazard": {k: v for k, v in verdict_obj().items() if k != "is_hazard"},
    "severity_too_high": verdict_obj(severity=4),
    "severity_negative": verdict_obj(severity=-1),
    "severity_string": verdict_obj(severity="high"),
    "severity_float": verdict_obj(severity=1.5),
    "confidence_too_high": verdict_obj(confidence=1.5),
    "confidence_negative": verdict_obj(confidence=-0.1),
    "confidence_string": verdict_obj(confidence="high"),
    "is_hazard_string": verdict_obj(is_hazard="yes"),
    "class_number": verdict_obj(**{"class": 5}),
    "lane_number": verdict_obj(lane=3),
    "hazard_with_severity_zero": verdict_obj(is_hazard=True, severity=0),
}


@pytest.mark.parametrize("name", sorted(BAD_REPLIES))
def test_bad_replies_raise_a_clear_provider_error(api, provider, name):
    from backend.providers.errors import ProviderConfigError, ProviderError

    bad = BAD_REPLIES[name]
    api.default_text = bad if isinstance(bad, str) else json.dumps(bad)
    api.queue.extend([api.default_text] * 5)
    with pytest.raises(ProviderError) as ei:
        verify(provider, [jpeg_bytes()])
    assert not isinstance(ei.value, ProviderConfigError)
    assert str(ei.value).strip()
    assert KEY not in str(ei.value)
    assert "UNIQUE-PROMPT-TEXT-4471" not in str(ei.value)


def test_no_output_at_all_raises(api, provider):
    from backend.providers.errors import ProviderError

    api.queue.extend([{"output": []}] * 5)
    with pytest.raises(ProviderError):
        verify(provider, [jpeg_bytes()])


def test_verdict_is_not_invented_on_failure(api, provider):
    """A bad reply must raise, never return something."""
    from backend.providers.errors import ProviderError

    api.queue.extend(["nonsense"] * 5)
    got = None
    try:
        got = verify(provider, [jpeg_bytes()])
    except ProviderError:
        pass
    assert got is None
