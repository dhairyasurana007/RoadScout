"""Task 26: parsing SAM 3.1 response text into DetectionResult."""
import pytest

import rs26_helpers as H
from rs26_conftest import _safe_env, rec, api, clips, clip, provider_factory, provider  # noqa: F401
from rs26_helpers import (box_tok, by_track, dets_of, frame_line, frame_of, mask_tok, out_json, rec_str)

W, HH = 406, 720


def detect(api, provider, clip, text, prompts=("pothole",)):
    api.default_text = text
    return provider.detect(clip, list(prompts))


def test_spec_example_shape_maps_to_internal_types(api, provider, clip):
    r1 = rec_str(0, 72, 382, 95, 396, W, HH, c=0.64)
    r2 = rec_str(2, 10, 20, 29, 49, W, HH, c=0.9)
    res = detect(api, provider, clip, frame_line(3, r1, r2))
    assert len(res.frames) == 1
    f = res.frames[0]
    assert f.frame_index == 3
    assert f.timestamp == pytest.approx(3 / 3.0)
    t = by_track(f)
    assert set(t) == {0, 2}
    d0 = t[0]
    assert tuple(d0.box) == (72, 382, 96, 397)
    assert d0.score == pytest.approx(0.64)
    assert d0.prompt == "pothole"
    assert d0.mask["size"] == [HH, W]
    assert tuple(t[2].box) == (10, 20, 30, 50)
    assert t[2].score == pytest.approx(0.9)


def test_frames_without_matches_are_absent(api, provider, clip):
    res = detect(api, provider, clip, frame_line(5, rec_str(1, 1, 1, 8, 8, W, HH, c=0.5)))
    assert [f.frame_index for f in res.frames] == [5]


@pytest.mark.parametrize("text", ["", "\n", "   \n  \n", "\n\n"])
def test_empty_text_is_an_empty_result_not_an_error(api, provider, clip, text):
    res = detect(api, provider, clip, text)
    assert res.is_empty
    assert res.frames == ()


def test_empty_output_list_is_an_empty_result(api, provider, clip):
    api.queue.append({"output": []})
    res = provider.detect(clip, ["x"])
    assert res.is_empty


def test_output_item_without_content_is_tolerated(api, provider, clip):
    line = frame_line(1, rec_str(0, 1, 1, 6, 6, W, HH, c=0.7))
    api.queue.append(out_json(line, extra_items=[{"type": "reasoning", "id": "rs_1", "summary": []}]))
    res = provider.detect(clip, ["x"])
    assert [f.frame_index for f in res.frames] == [1]


def test_out_of_order_frames_are_sorted(api, provider, clip):
    text = "\n".join([
        frame_line(7, rec_str(0, 1, 1, 4, 4, W, HH, c=0.5)),
        frame_line(2, rec_str(0, 1, 1, 4, 4, W, HH, c=0.5)),
        frame_line(5, rec_str(0, 1, 1, 4, 4, W, HH, c=0.5)),
    ])
    res = detect(api, provider, clip, text)
    assert [f.frame_index for f in res.frames] == [2, 5, 7]
    ts = [f.timestamp for f in res.frames]
    assert ts == sorted(ts)


def test_multiple_text_parts_are_concatenated(api, provider, clip):
    a = frame_line(1, rec_str(0, 1, 1, 4, 4, W, HH, c=0.5)) + "\n"
    b = frame_line(2, rec_str(0, 1, 1, 4, 4, W, HH, c=0.5))
    api.queue.append({"output": [{"type": "message", "content": [
        {"type": "output_text", "text": a}, {"type": "output_text", "text": b}]}]})
    res = provider.detect(clip, ["x"])
    assert [f.frame_index for f in res.frames] == [1, 2]


def test_text_split_in_the_middle_of_a_record_is_rejoined(api, provider, clip):
    full = frame_line(4, rec_str(0, 5, 6, 14, 15, W, HH, c=0.5))
    cut = len(full) // 2
    api.queue.append({"output": [
        {"type": "message", "content": [{"type": "output_text", "text": full[:cut]}]},
        {"type": "message", "content": [{"type": "output_text", "text": full[cut:]}]},
    ]})
    res = provider.detect(clip, ["x"])
    assert [f.frame_index for f in res.frames] == [4]
    assert tuple(res.frames[0].detections[0].box) == (5, 6, 15, 16)


def test_non_text_content_parts_are_ignored(api, provider, clip):
    line = frame_line(1, rec_str(0, 1, 1, 6, 6, W, HH, c=0.7))
    api.queue.append({"output": [{"type": "message", "content": [
        {"type": "refusal", "refusal": "no"}, {"type": "output_text", "text": line}]}]})
    res = provider.detect(clip, ["x"])
    assert [f.frame_index for f in res.frames] == [1]


def test_missing_confidence_scores_half(api, provider, clip):
    res = detect(api, provider, clip, frame_line(0, rec_str(0, 3, 3, 12, 12, W, HH, c=None)))
    assert res.frames[0].detections[0].score == 0.5


def test_zero_confidence_is_zero_not_the_default(api, provider, clip):
    res = detect(api, provider, clip, frame_line(0, rec_str(0, 3, 3, 12, 12, W, HH, c=0)))
    assert res.frames[0].detections[0].score == 0.0


def test_full_confidence(api, provider, clip):
    res = detect(api, provider, clip, frame_line(0, rec_str(0, 3, 3, 12, 12, W, HH, c=1.0)))
    assert res.frames[0].detections[0].score == 1.0


def test_one_pixel_object_has_a_valid_half_open_box_and_mask(api, provider, clip):
    res = detect(api, provider, clip, frame_line(0, rec_str(0, 50, 60, 50, 60, W, HH, c=0.4, raster="ones")))
    d = res.frames[0].detections[0]
    assert tuple(d.box) == (50, 60, 51, 61)
    bm = H.bitmap_of(d)
    assert sum(map(sum, bm)) == 1 and bm[60][50] == 1


def test_track_ids_are_namespaced_per_prompt(api, provider, clip):
    api.by_prompt = {
        "a": frame_line(3, rec_str(0, 1, 1, 6, 6, W, HH, c=0.5), rec_str(2, 20, 20, 30, 30, W, HH, c=0.5)),
        "b": frame_line(3, rec_str(0, 1, 1, 6, 6, W, HH, c=0.5), rec_str(2, 20, 20, 30, 30, W, HH, c=0.5)),
        "c": frame_line(3, rec_str(16, 1, 1, 6, 6, W, HH, c=0.5)),
    }
    res = provider.detect(clip, ["a", "b", "c"])
    f = frame_of(res, 3)
    assert sorted(d.track_id for d in f.detections) == [0, 2, 1000, 1002, 2016]
    prompts = {d.track_id: d.prompt for d in f.detections}
    assert prompts == {0: "a", 2: "a", 1000: "b", 1002: "b", 2016: "c"}


def test_all_prompts_combine_into_one_record_per_frame(api, provider, clip):
    api.by_prompt = {
        "a": frame_line(3, rec_str(0, 1, 1, 6, 6, W, HH, c=0.5)) + "\n" + frame_line(1, rec_str(0, 1, 1, 6, 6, W, HH, c=0.5)),
        "b": frame_line(3, rec_str(0, 9, 9, 16, 16, W, HH, c=0.6)) + "\n" + frame_line(8, rec_str(1, 9, 9, 16, 16, W, HH, c=0.6)),
    }
    res = provider.detect(clip, ["a", "b"])
    assert [f.frame_index for f in res.frames] == [1, 3, 8]
    assert len(frame_of(res, 3).detections) == 2
    assert len(frame_of(res, 1).detections) == 1
    assert [d.track_id for d in frame_of(res, 8).detections] == [1001]
    assert len({f.frame_index for f in res.frames}) == len(res.frames)


def test_prompt_index_follows_the_given_order_even_if_some_prompts_find_nothing(api, provider, clip):
    api.by_prompt = {"third": frame_line(0, rec_str(4, 1, 1, 6, 6, W, HH, c=0.5))}
    res = provider.detect(clip, ["first", "second", "third"])
    d = dets_of(res)[0]
    assert d.track_id == 2004 and d.prompt == "third"


def test_each_prompt_only_sees_its_own_answer(api, provider, clip):
    api.by_prompt = {"a": frame_line(0, rec_str(1, 1, 1, 6, 6, W, HH, c=0.5))}
    res = provider.detect(clip, ["a", "b"])
    assert [d.prompt for d in dets_of(res)] == ["a"]


def test_result_is_a_valid_detection_result(api, provider, clip):
    from backend.providers.types import DetectionResult

    text = "\n".join(frame_line(n, rec_str(0, 1, 1, 6, 6, W, HH, c=0.5)) for n in (9, 0, 4))
    res = detect(api, provider, clip, text)
    assert isinstance(res, DetectionResult)
    assert DetectionResult.from_dict(res.to_dict()).to_dict() == res.to_dict()


# ------------------------------------------------------------------ timestamps

def test_timestamp_uses_the_chunk_fps_3(api, provider, clips):
    api.default_text = "\n".join(frame_line(n, rec_str(0, 1, 1, 6, 6, W, HH, c=0.5)) for n in (0, 1, 4, 8))
    res = provider.detect(clips["fps3"], ["x"])
    got = {f.frame_index: f.timestamp for f in res.frames}
    assert got == {n: pytest.approx(n / 3.0, abs=1e-6) for n in (0, 1, 4, 8)}


def test_timestamp_uses_the_chunk_fps_ntsc(api, provider, clips):
    api.default_text = "\n".join(frame_line(n, rec_str(0, 1, 1, 6, 6, W, HH, c=0.5)) for n in (0, 30, 59))
    res = provider.detect(clips["ntsc"], ["x"])
    got = {f.frame_index: f.timestamp for f in res.frames}
    for n in (0, 30, 59):
        assert got[n] == pytest.approx(n * 1001 / 30000, abs=1e-6)


def test_timestamp_uses_the_chunk_fps_5(api, provider, clips):
    api.default_text = frame_line(7, rec_str(0, 1, 1, 6, 6, W, HH, c=0.5))
    res = provider.detect(clips["fps5"], ["x"])
    assert res.frames[0].timestamp == pytest.approx(7 / 5.0, abs=1e-6)


def test_timestamps_independent_of_which_prompt(api, provider, clip):
    api.by_prompt = {
        "a": frame_line(6, rec_str(0, 1, 1, 6, 6, W, HH, c=0.5)),
        "b": frame_line(6, rec_str(0, 1, 1, 6, 6, W, HH, c=0.5)),
    }
    res = provider.detect(clip, ["a", "b"])
    assert len(res.frames) == 1 and res.frames[0].timestamp == pytest.approx(2.0)


# ------------------------------------------------------------ malformed input

GOOD = lambda n=2, obj=0: frame_line(n, rec_str(obj, 4, 4, 13, 13, W, HH, c=0.5))  # noqa: E731


def test_garbage_lines_are_skipped_without_crashing(api, provider, clip):
    text = "hello world\n" + GOOD(2) + "\n<<<>>>\n<xf>0<|box;nonsense|>\n"
    res = detect(api, provider, clip, text)
    assert [f.frame_index for f in res.frames] == [2]


def test_bad_record_is_skipped_but_the_rest_of_the_line_survives(api, provider, clip):
    good1 = rec_str(0, 4, 4, 13, 13, W, HH, c=0.5)
    good2 = rec_str(5, 20, 20, 29, 29, W, HH, c=0.6)
    line = "<2f>" + good1 + ",7<|box;x1=abc;y1=;x2=;y2=|>," + good2
    res = detect(api, provider, clip, line)
    t = by_track(frame_of(res, 2))
    assert 0 in t and 5 in t
    assert 7 not in t


def test_inverted_box_is_skipped_not_fatal(api, provider, clip):
    bad = "9" + box_tok(50, 50, 40, 40, W, HH, 0.5)
    line = "<2f>" + rec_str(0, 4, 4, 13, 13, W, HH, c=0.5) + "," + bad
    res = detect(api, provider, clip, line)
    assert [d.track_id for d in frame_of(res, 2).detections] == [0]


def test_a_frame_with_only_a_bad_record_has_no_detections(api, provider, clip):
    text = "<4f>3<|box;broken|>\n" + GOOD(6)
    res = detect(api, provider, clip, text)
    assert [f.frame_index for f in res.frames if f.detections] == [6]


def test_blank_lines_between_records_are_fine(api, provider, clip):
    text = "\n\n" + GOOD(1) + "\n\n\n" + GOOD(3) + "\n"
    res = detect(api, provider, clip, text)
    assert [f.frame_index for f in res.frames] == [1, 3]


def test_crlf_line_endings(api, provider, clip):
    text = GOOD(1) + "\r\n" + GOOD(3) + "\r\n"
    res = detect(api, provider, clip, text)
    assert [f.frame_index for f in res.frames] == [1, 3]
    assert tuple(res.frames[0].detections[0].box) == (4, 4, 14, 14)


def test_many_objects_on_one_line(api, provider, clip):
    recs = [rec_str(i, 2 * i, 2 * i, 2 * i + 5, 2 * i + 5, W, HH, c=0.5) for i in range(18)]
    res = detect(api, provider, clip, frame_line(2, *recs))
    assert sorted(by_track(frame_of(res, 2))) == list(range(18))
