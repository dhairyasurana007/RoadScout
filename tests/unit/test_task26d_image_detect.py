"""Task 26d: LiveProvider.detect on an image file sends it as input_image (no upload), parses like video."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest

import rs26d_setup  # noqa: F401
import rs26_helpers as H
from rs26_conftest import _safe_env, rec, api, clips, clip, provider_factory, provider  # noqa: F401
from rs26_helpers import BASE, KEY, box_tok, err_json, frame_line, jpeg_bytes, make_raster, mask_tok, payload_for, png_bytes, rec_str

W, HH = 80, 60
PHRASE_ERR = "unique-image-phrase-3391"


@pytest.fixture()
def jpg(tmp_path):
    p = tmp_path / "uniqueimage_4417.jpg"
    p.write_bytes(jpeg_bytes(size=(W, HH)))
    return p


@pytest.fixture()
def png(tmp_path):
    p = tmp_path / "uniquepng_5528.png"
    p.write_bytes(png_bytes(size=(W, HH)))
    return p


def line(obj=0, c=0.8, x1=10, y1=12, x2=29, y2=31, frame=0, **kw):
    return frame_line(frame, rec_str(obj, x1, y1, x2, y2, W, HH, c=c, **kw))


def data_url(mime, raw):
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def sleeps(api):
    return [e[1] for e in api.events if e[0] == "sleep"]


def strip_stream(body):
    body = dict(body)
    assert body.pop("stream", False) in (False, None)
    return body


# ------------------------------------------------------------------ request shape

def test_jpeg_request_body_is_exact(api, provider, jpg):
    provider.detect(str(jpg), ["face"])
    assert len(api.requests) == 1
    r = api.requests[0]
    assert r.method == "POST" and r.url == f"{BASE}/responses"
    assert r.headers["authorization"] == f"Bearer {KEY}"
    assert strip_stream(r.json) == {
        "model": "sam-3.1",
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "face"},
            {"type": "input_image", "image_url": data_url("image/jpeg", jpg.read_bytes())},
        ]}],
        "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
    }


def test_png_uses_the_png_mime(api, provider, png):
    provider.detect(str(png), ["face"])
    part = api.requests[0].json["input"][0]["content"][1]
    assert part == {"type": "input_image", "image_url": data_url("image/png", png.read_bytes())}


def test_no_upload_and_no_delete_for_images(api, provider, jpg):
    provider.detect(str(jpg), ["face", "license plate"])
    assert api.kinds() == ["responses", "responses"]
    assert api.uploads == [] and api.deletes == []
    assert all("file_id" not in json.dumps(r.json) and "input_video" not in json.dumps(r.json)
               for r in api.responses)


def test_one_request_per_prompt_in_order_with_the_same_image(api, provider, jpg):
    provider.detect(str(jpg), ["face", "license plate", "third thing"])
    texts = [r.json["input"][0]["content"][0]["text"] for r in api.responses]
    assert texts == ["face", "license plate", "third thing"]
    urls = {r.json["input"][0]["content"][1]["image_url"] for r in api.responses}
    assert len(urls) == 1


def test_model_env_override_applies_to_images(api, jpg, monkeypatch):
    monkeypatch.setenv("ROADSCOUT_SAM_MODEL", "sam-test-77")
    H.make_provider(api).detect(str(jpg), ["face"])
    assert api.requests[0].json["model"] == "sam-test-77"


def test_base_url_is_respected(api, jpg):
    H.make_provider(api, base_url="https://example.test/alt").detect(str(jpg), ["face"])
    assert api.requests[0].url == "https://example.test/alt/responses"


def test_magic_bytes_beat_a_wrong_extension_jpeg_named_png(api, provider, tmp_path):
    p = tmp_path / "really_a_jpeg.png"
    p.write_bytes(jpeg_bytes())
    provider.detect(str(p), ["face"])
    assert api.requests[0].json["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")
    assert api.uploads == []


def test_magic_bytes_beat_a_wrong_extension_png_named_jpg(api, provider, tmp_path):
    p = tmp_path / "really_a_png.jpg"
    p.write_bytes(png_bytes())
    provider.detect(str(p), ["face"])
    assert api.requests[0].json["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")


def test_magic_bytes_work_without_any_extension(api, provider, tmp_path):
    p = tmp_path / "frame_without_extension"
    p.write_bytes(png_bytes())
    provider.detect(str(p), ["face"])
    assert api.requests[0].json["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert api.uploads == []


def test_jpeg_signature_only_is_an_image(api, provider, tmp_path):
    p = tmp_path / "sig.bin"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
    provider.detect(str(p), ["face"])
    assert api.requests[0].json["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg;base64,")


def test_png_signature_only_is_an_image(api, provider, tmp_path):
    p = tmp_path / "sig.bin"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    provider.detect(str(p), ["face"])
    assert api.requests[0].json["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize("name,mime", [("x.jpg", "image/jpeg"), ("x.jpeg", "image/jpeg"), ("x.png", "image/png")])
def test_extension_is_the_fallback_when_the_magic_is_unknown(api, provider, tmp_path, name, mime):
    p = tmp_path / name
    p.write_bytes(b"not-a-known-signature-but-named-as-image" * 4)
    provider.detect(str(p), ["face"])
    assert api.uploads == []
    assert api.requests[0].json["input"][0]["content"][1]["image_url"].startswith(f"data:{mime};base64,")


def test_image_file_is_untouched(api, provider, jpg):
    before = jpg.read_bytes()
    provider.detect(str(jpg), ["face"])
    assert jpg.read_bytes() == before and jpg.is_file()


def test_pathlib_path_is_accepted(api, provider, jpg):
    provider.detect(jpg, ["face"])
    assert api.uploads == [] and len(api.responses) == 1


# ------------------------------------------------------------------ parsing

def test_normal_answer_is_mapped_like_video(api, provider, jpg):
    api.default_text = line(obj=0, c=0.64, x1=10, y1=12, x2=29, y2=31)
    res = provider.detect(str(jpg), ["face"])
    assert [f.frame_index for f in res.frames] == [0]
    f = res.frames[0]
    assert f.timestamp == 0.0
    (d,) = f.detections
    assert tuple(d.box) == (10, 12, 30, 32)
    assert d.score == pytest.approx(0.64)
    assert d.prompt == "face"
    assert d.track_id == 0
    assert d.mask["size"] == [HH, W]
    expected = H.expected_bitmap(W, HH, 10, 12, make_raster(20, 20), 20, 20)
    assert H.bitmap_of(d) == expected


def test_no_ffprobe_is_needed_for_an_image(api, provider, tmp_path, monkeypatch):
    import backend.providers.live as live

    def boom(*a, **k):
        raise AssertionError("probe_video must not be called for an image")

    monkeypatch.setattr(live, "probe_video", boom, raising=False)
    p = tmp_path / "i.jpg"
    p.write_bytes(jpeg_bytes())
    api.default_text = line()
    res = provider.detect(str(p), ["face"])
    assert res.frames[0].timestamp == 0.0


def test_two_prompts_are_namespaced_and_combined_in_one_frame_record(api, provider, jpg):
    api.by_prompt = {
        "face": frame_line(0, rec_str(0, 5, 5, 14, 14, W, HH, c=0.5), rec_str(2, 30, 30, 39, 39, W, HH, c=0.6)),
        "license plate": frame_line(0, rec_str(0, 50, 20, 69, 29, W, HH, c=0.7)),
    }
    res = provider.detect(str(jpg), ["face", "license plate"])
    assert len(res.frames) == 1 and res.frames[0].frame_index == 0
    dets = {d.track_id: d for d in res.frames[0].detections}
    assert sorted(dets) == [0, 2, 1000]
    assert dets[0].prompt == dets[2].prompt == "face"
    assert dets[1000].prompt == "license plate"
    assert tuple(dets[1000].box) == (50, 20, 70, 30)


def test_only_the_second_prompt_matching(api, provider, jpg):
    api.by_prompt = {"license plate": line(obj=3)}
    res = provider.detect(str(jpg), ["face", "license plate"])
    assert [d.track_id for d in H.dets_of(res)] == [1003]


@pytest.mark.parametrize("text", ["", "\n", "  \n \n"])
def test_empty_answer_is_an_empty_result(api, provider, jpg, text):
    api.default_text = text
    res = provider.detect(str(jpg), ["face", "license plate"])
    assert res.is_empty and res.frames == ()


def test_empty_output_list_is_an_empty_result(api, provider, jpg):
    api.queue.append({"output": []})
    assert provider.detect(str(jpg), ["face"]).is_empty


def test_missing_confidence_scores_half(api, provider, jpg):
    api.default_text = line(c=None)
    assert H.dets_of(provider.detect(str(jpg), ["face"]))[0].score == 0.5


def test_malformed_mask_falls_back_to_the_filled_box(api, provider, jpg):
    rec = "0" + box_tok(10, 12, 29, 31, W, HH, 0.8) + mask_tok(20, 20, "!zzzz", 0.8)
    api.default_text = frame_line(0, rec)
    (d,) = H.dets_of(provider.detect(str(jpg), ["face"]))
    bm = H.bitmap_of(d)
    assert sum(map(sum, bm)) == 400
    assert bm[12][10] == 1 and bm[31][29] == 1 and bm[32][29] == 0


def test_missing_mask_falls_back_to_the_filled_box(api, provider, jpg):
    api.default_text = line(mask=False)
    (d,) = H.dets_of(provider.detect(str(jpg), ["face"]))
    assert sum(map(sum, H.bitmap_of(d))) == 400


def test_mask_clipped_at_the_image_edge(api, provider, jpg):
    api.default_text = frame_line(0, rec_str(0, 70, 50, 89, 69, W, HH, c=0.5, raster="ones"))
    (d,) = H.dets_of(provider.detect(str(jpg), ["face"]))
    assert sum(map(sum, H.bitmap_of(d))) == 10 * 10


def test_garbage_lines_are_skipped(api, provider, jpg):
    api.default_text = "nonsense\n" + line(obj=1) + "\n<<>>\n"
    assert [d.track_id for d in H.dets_of(provider.detect(str(jpg), ["face"]))] == [1]


# ------------------------------------------------------------------ size and unreadable files

def test_oversize_image_is_refused_before_any_request(api, provider, tmp_path):
    p = tmp_path / "huge_unique_9921.jpg"
    with open(p, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0")
        f.truncate(50_000_001)
    from backend.providers.errors import ProviderError, TransientProviderError

    with pytest.raises(ProviderError) as ei:
        provider.detect(str(p), ["face"])
    assert not isinstance(ei.value, TransientProviderError)
    assert api.requests == []
    assert "huge_unique_9921" not in str(ei.value) and str(tmp_path) not in str(ei.value)
    assert str(ei.value).strip()


def test_oversize_check_happens_before_spend(api, tmp_path):
    from backend.providers.cost import SpendTracker
    from backend.providers.errors import ProviderError

    p = tmp_path / "huge.png"
    with open(p, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.truncate(50_000_001)
    tr = SpendTracker(cap_usd=10, log_path=None)
    with pytest.raises(ProviderError):
        H.make_provider(api, tracker=tr).detect(str(p), ["face"])
    assert tr.spent == 0 and api.requests == []


@pytest.mark.parametrize("name", ["empty_unique_1.jpg", "empty_unique_2.png"])
def test_empty_file_is_a_clear_error(api, provider, tmp_path, name):
    from backend.providers.errors import ProviderError

    p = tmp_path / name
    p.write_bytes(b"")
    with pytest.raises(ProviderError) as ei:
        provider.detect(str(p), ["face"])
    assert api.requests == []
    assert str(ei.value).strip() and "empty_unique" not in str(ei.value) and str(tmp_path) not in str(ei.value)


def test_missing_image_is_a_clear_error(api, provider, tmp_path):
    from backend.providers.errors import ProviderError

    p = tmp_path / "missing_unique_3.jpg"
    with pytest.raises(ProviderError) as ei:
        provider.detect(str(p), ["face"])
    assert api.requests == []
    assert "missing_unique" not in str(ei.value) and str(tmp_path) not in str(ei.value)


def test_directory_named_like_an_image_is_a_clear_error(api, provider, tmp_path):
    from backend.providers.errors import ProviderError

    d = tmp_path / "dir_unique_4.jpg"
    d.mkdir()
    with pytest.raises(ProviderError) as ei:
        provider.detect(str(d), ["face"])
    assert api.requests == []
    assert "dir_unique" not in str(ei.value)


def test_non_image_non_video_file_keeps_the_video_error_handling(api, provider, tmp_path):
    """Unknown bytes with an unknown extension: today's video path (ffprobe fails -> ProviderError, no upload)."""
    from backend.providers.errors import ProviderError

    p = tmp_path / "junk_unique_5.bin"
    p.write_bytes(b"this is neither an image nor a video" * 10)
    with pytest.raises(ProviderError) as ei:
        provider.detect(str(p), ["face"])
    assert api.uploads == []
    assert "junk_unique" not in str(ei.value)


# ------------------------------------------------------------------ videos are unchanged

def test_video_still_uploads_and_sends_input_video(api, provider, clip):
    api.default_text = ""
    provider.detect(clip, ["pothole"])
    assert api.kinds() == ["upload", "responses", "delete"]
    part = api.responses[0].json["input"][0]["content"][1]
    assert part == {"type": "input_video", "file_id": H.FILE_ID}


# ------------------------------------------------------------------ cost

def test_image_costs_a_quarter_cent_per_prompt(api, jpg):
    from backend.providers.cost import SpendTracker

    tr = SpendTracker(cap_usd=10, log_path=None)
    H.make_provider(api, tracker=tr).detect(str(jpg), ["a", "b", "c"])
    assert tr.spent == pytest.approx(0.0075)


def test_image_single_prompt_cost(api, jpg):
    from backend.providers.cost import SpendTracker

    tr = SpendTracker(cap_usd=10, log_path=None)
    H.make_provider(api, tracker=tr).detect(str(jpg), ["a"])
    assert tr.spent == pytest.approx(0.0025)
    assert [e["operation"] for e in tr.entries] == ["detect"]


def test_video_cost_is_unchanged(api, clip):
    from backend.providers.cost import SpendTracker

    tr = SpendTracker(cap_usd=10, log_path=None)
    H.make_provider(api, tracker=tr).detect(clip, ["a", "b"])
    assert tr.spent == pytest.approx(0.024)


def test_env_override_wins_for_images_and_stays_per_prompt(api, jpg, monkeypatch):
    from backend.providers.cost import SpendTracker

    monkeypatch.setenv("ROADSCOUT_COST_PER_DETECT_USD", "0.05")
    tr = SpendTracker(cap_usd=10, log_path=None)
    H.make_provider(api, tracker=tr).detect(str(jpg), ["a", "b"])
    assert tr.spent == pytest.approx(0.10)


def test_cap_that_fits_the_image_price_but_not_the_video_price_passes(api, jpg):
    from backend.providers.cost import SpendTracker

    tr = SpendTracker(cap_usd=0.005, log_path=None)       # two image prompts exactly; a video price would not fit
    H.make_provider(api, tracker=tr).detect(str(jpg), ["a", "b"])
    assert tr.spent == pytest.approx(0.005)


def test_spend_cap_blocks_an_image_before_any_request(api, jpg):
    from backend.providers.cost import SpendTracker
    from backend.providers.errors import SpendCapExceeded

    tr = SpendTracker(cap_usd=0.002, log_path=None)
    with pytest.raises(SpendCapExceeded):
        H.make_provider(api, tracker=tr).detect(str(jpg), ["a"])
    assert api.requests == [] and tr.spent == 0


def test_spend_cap_stops_a_multi_prompt_image_call(api, jpg):
    from backend.providers.cost import SpendTracker
    from backend.providers.errors import SpendCapExceeded

    tr = SpendTracker(cap_usd=0.004, log_path=None)
    with pytest.raises(SpendCapExceeded):
        H.make_provider(api, tracker=tr).detect(str(jpg), ["a", "b"])
    assert len(api.responses) <= 1
    assert api.uploads == [] and api.deletes == []
    assert tr.spent <= 0.004 + 1e-9


def test_env_override_cap_check_for_images(api, jpg, monkeypatch):
    from backend.providers.cost import SpendTracker
    from backend.providers.errors import SpendCapExceeded

    monkeypatch.setenv("ROADSCOUT_COST_PER_DETECT_USD", "0.5")
    tr = SpendTracker(cap_usd=0.4, log_path=None)
    with pytest.raises(SpendCapExceeded):
        H.make_provider(api, tracker=tr).detect(str(jpg), ["a"])
    assert api.requests == []


# ------------------------------------------------------------------ errors, retries, hygiene

@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 413, 504])
def test_permanent_errors_are_not_retried_for_images(api, provider, jpg, status):
    from backend.providers.errors import ProviderError, RetriesExhausted, TransientProviderError

    api.queue.extend([httpx.Response(status, json=err_json("permanent image failure"))] * 5)
    with pytest.raises(ProviderError) as ei:
        provider.detect(str(jpg), ["face", "license plate"])
    assert not isinstance(ei.value, (RetriesExhausted, TransientProviderError))
    assert len(api.responses) == 1
    assert "permanent image failure" in str(ei.value)
    assert api.uploads == [] and api.deletes == []


def test_the_original_meta_400_message_reaches_the_error(api, provider, jpg):
    from backend.providers.errors import ProviderError

    msg = "unsupported media type at input[0].content[1]: image/jpeg. Supported: MP4"
    api.queue.append(httpx.Response(400, json=err_json(msg)))
    with pytest.raises(ProviderError) as ei:
        provider.detect(str(jpg), ["face"])
    assert "400" in str(ei.value)


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_errors_retry_the_whole_call_for_images(api, provider, jpg, status):
    api.queue.extend([httpx.Response(status, json=err_json("busy")), line(obj=1)])
    res = provider.detect(str(jpg), ["face"])
    assert [d.track_id for d in H.dets_of(res)] == [1]
    assert len(api.responses) == 2
    assert api.uploads == [] and api.deletes == []
    assert sleeps(api) == [2.0]


def test_a_transient_failure_on_the_second_prompt_repeats_the_call(api, provider, jpg):
    api.queue.extend([line(obj=0), httpx.Response(503, json=err_json("busy"))])
    api.by_prompt = {"face": line(obj=0), "license plate": line(obj=4)}
    res = provider.detect(str(jpg), ["face", "license plate"])
    ids = sorted(d.track_id for d in H.dets_of(res))
    assert ids == [0, 1004]
    assert len(api.responses) == 4
    assert api.uploads == [] and api.deletes == []


def test_retry_after_is_honoured_for_images(api, provider, jpg):
    api.queue.extend([httpx.Response(429, json=err_json("slow"), headers={"Retry-After": "9"}), ""])
    provider.detect(str(jpg), ["face"])
    assert sleeps(api) == [9.0]


def test_connection_errors_exhaust_after_five_attempts(api, provider, jpg):
    from backend.providers.errors import RetriesExhausted

    api.queue.extend([httpx.ConnectError("refused")] * 20)
    with pytest.raises(RetriesExhausted) as ei:
        provider.detect(str(jpg), ["face"])
    assert ei.value.attempts == 5
    assert sleeps(api) == [2.0, 4.0, 8.0, 16.0]


@pytest.mark.parametrize("status", [400, 429, 503, 413])
def test_errors_never_contain_key_prompt_path_or_image_data(api, provider, jpg, rec, status):
    from backend.providers.errors import ProviderError

    api.queue.extend([httpx.Response(status, json=err_json("generic failure"))] * 20)
    with pytest.raises(ProviderError) as ei:
        provider.detect(str(jpg), [PHRASE_ERR])
    texts = [str(ei.value), repr(ei.value)]
    cause = ei.value.__cause__
    while cause is not None:
        texts += [str(cause), repr(cause)]
        cause = cause.__cause__
    for m in texts:
        assert KEY not in m
        assert PHRASE_ERR not in m
        assert "uniqueimage_4417" not in m
        assert str(jpg.parent) not in m
        assert "/9j/" not in m and "base64" not in m


def test_server_echoing_the_key_is_redacted_for_images(api, provider, jpg):
    from backend.providers.errors import ProviderError

    api.queue.append(httpx.Response(401, json=err_json(f"Incorrect API key provided: {KEY}")))
    with pytest.raises(ProviderError) as ei:
        provider.detect(str(jpg), ["face"])
    assert KEY not in str(ei.value)


def test_key_not_in_repr_or_logs_for_images(api, provider, jpg, caplog):
    import logging

    caplog.set_level(logging.DEBUG)
    provider.detect(str(jpg), ["face"])
    assert KEY not in repr(provider) and KEY not in caplog.text
    assert KEY.encode() not in api.requests[0].body
