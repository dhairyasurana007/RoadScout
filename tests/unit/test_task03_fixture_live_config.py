"""Task 3: recorded-fixture provider, placeholder live provider, config switch."""
import json
import re
from pathlib import Path

import pytest

import backend.config as config
from rs3_helpers import (  # noqa: F401
    PROMPTS, all_dets, canon, clean_env, clip, find, frame_image, no_network,
    verdict_field,
)

ROOT = Path(config.__file__).resolve().parent.parent


def write_fixtures(dirpath, scenario="default", clip_p=None, img_p=None):
    fake = find("FakeProvider")(scenario=scenario)
    det = fake.detect(clip_p, PROMPTS)
    ver = fake.verify([img_p])
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "detect.json").write_text(json.dumps(det.to_dict()), encoding="utf-8")
    (dirpath / "verify.json").write_text(json.dumps(ver.to_dict()), encoding="utf-8")
    return det, ver


# ------------------------------------------------------- fixture provider

def test_fixture_provider_replays_saved_json(tmp_path, clip, frame_image):
    det, ver = write_fixtures(tmp_path / "fx", "default", clip, frame_image)
    p = find("FixtureProvider")(tmp_path / "fx")
    assert isinstance(p, find("ModelProvider"))
    got = p.detect(clip, PROMPTS)
    assert isinstance(got, find("DetectionResult"))
    assert canon(got) == canon(det)
    gv = p.verify([frame_image])
    assert isinstance(gv, find("Verdict"))
    assert canon(gv) == canon(ver)


def test_fixture_provider_is_deterministic_and_input_independent(tmp_path, clip, frame_image):
    write_fixtures(tmp_path / "fx", "default", clip, frame_image)
    p = find("FixtureProvider")(tmp_path / "fx")
    a = canon(p.detect(clip, PROMPTS))
    assert a == canon(p.detect(clip, PROMPTS))
    assert a == canon(p.detect(clip, ["something else"]))


def test_fixture_provider_replays_hand_edited_json(tmp_path, clip, frame_image):
    """The fixture is plain JSON: editing a field changes what is replayed."""
    write_fixtures(tmp_path / "fx", "default", clip, frame_image)
    vp = tmp_path / "fx" / "verify.json"
    data = json.loads(vp.read_text())
    data["reason"] = "edited reason text"
    data["confidence"] = 0.123
    vp.write_text(json.dumps(data))
    v = find("FixtureProvider")(tmp_path / "fx").verify([frame_image])
    assert verdict_field(v, "reason") == "edited reason text"
    assert verdict_field(v, "confidence") == pytest.approx(0.123)


def test_fixture_provider_replays_empty_detection_as_empty(tmp_path, clip, frame_image):
    write_fixtures(tmp_path / "fx", "empty", clip, frame_image)
    res = find("FixtureProvider")(tmp_path / "fx").detect(clip, PROMPTS)
    assert all_dets(res) == []


def test_fixture_provider_replays_false_positive(tmp_path, clip, frame_image):
    write_fixtures(tmp_path / "fx", "false_positive", clip, frame_image)
    v = find("FixtureProvider")(tmp_path / "fx").verify([frame_image])
    assert verdict_field(v, "is_hazard") is False


def test_fixture_provider_missing_fixture_fails_clearly(tmp_path, clip):
    (tmp_path / "empty").mkdir()
    try:
        p = find("FixtureProvider")(tmp_path / "empty")
        p.detect(clip, PROMPTS)
    except (FileNotFoundError, find("ProviderError"), LookupError) as e:
        assert str(e)
    else:
        pytest.fail("missing fixture must raise, not return a made-up result")


def test_fixture_provider_malformed_json_fails_clearly(tmp_path, clip):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "detect.json").write_text("{not json", encoding="utf-8")
    with pytest.raises((ValueError, find("ProviderError"))):
        find("FixtureProvider")(d).detect(clip, PROMPTS)


# ----------------------------------------------------------- live provider

def test_live_without_key_fails_clearly_naming_the_variable(clip, frame_image):
    for call in ("detect", "verify"):
        try:
            p = find("LiveProvider")()
            if call == "detect":
                p.detect(clip, PROMPTS)
            else:
                p.verify([frame_image])
        except Exception as e:  # noqa: BLE001
            assert "META_API_KEY" in str(e), "error must name the missing variable"
        else:
            pytest.fail(f"live {call} without a key must fail")


def test_live_is_a_placeholder_even_with_a_key_and_makes_no_network_call(
        monkeypatch, clip, frame_image, no_network):
    fake_key = "dummy-key-for-tests-0000"
    monkeypatch.setenv("META_API_KEY", fake_key)
    monkeypatch.setattr("time.sleep", lambda s: None)
    for call in ("detect", "verify"):
        try:
            p = find("LiveProvider")()
            if call == "detect":
                p.detect(clip, PROMPTS)
            else:
                p.verify([frame_image])
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            assert fake_key not in msg, "the key must never appear in an error message"
            assert re.search(r"placeholder|not implemented|task 26", msg, re.I), (
                "placeholder error should say it is not implemented until task 26")
        else:
            pytest.fail("placeholder live provider must raise")
    assert no_network == []


def test_live_provider_is_not_selected_by_default():
    p = find("get_provider")()
    assert not isinstance(p, find("LiveProvider"))


# ------------------------------------------------------------ config switch

def test_config_provider_name_defaults_to_fake():
    assert config.provider_name() == "fake"


@pytest.mark.parametrize("name", ["fake", "fixture", "live"])
def test_config_provider_name_reads_env_at_call_time(monkeypatch, name):
    monkeypatch.setenv("ROADSCOUT_PROVIDER", name)
    assert config.provider_name() == name


def test_get_provider_env_switch_fake(monkeypatch):
    monkeypatch.setenv("ROADSCOUT_PROVIDER", "fake")
    assert isinstance(find("get_provider")(), find("FakeProvider"))


def test_get_provider_env_switch_fixture(monkeypatch, tmp_path, clip, frame_image):
    write_fixtures(tmp_path / "fx", "default", clip, frame_image)
    monkeypatch.setenv("ROADSCOUT_PROVIDER", "fixture")
    monkeypatch.setenv("ROADSCOUT_FIXTURES_DIR", str(tmp_path / "fx"))
    p = find("get_provider")()
    assert isinstance(p, find("FixtureProvider"))
    assert all_dets(p.detect(clip, PROMPTS))


def test_get_provider_explicit_name_overrides_env(monkeypatch):
    monkeypatch.setenv("ROADSCOUT_PROVIDER", "live")
    assert isinstance(find("get_provider")("fake"), find("FakeProvider"))


def test_get_provider_live_selected_but_unusable_without_key(monkeypatch, clip):
    monkeypatch.setenv("ROADSCOUT_PROVIDER", "live")
    try:
        p = find("get_provider")()
        assert isinstance(p, find("LiveProvider"))
        p.detect(clip, PROMPTS)
    except Exception as e:  # noqa: BLE001
        assert "META_API_KEY" in str(e)
    else:
        pytest.fail("live provider without a key must fail")


def test_get_provider_unknown_name_is_rejected(monkeypatch):
    monkeypatch.setenv("ROADSCOUT_PROVIDER", "banana")
    with pytest.raises(ValueError):
        find("get_provider")()


def test_env_example_documents_the_new_switches():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "ROADSCOUT_PROVIDER" in text
    assert "ROADSCOUT_SPEND_CAP_USD" in text
    assert "META_API_KEY" in text
    # the example file must hold only a placeholder, never a key-looking value
    m = re.search(r"^META_API_KEY=(.*)$", text, re.M)
    assert m and ("your" in m.group(1).lower() or m.group(1).strip() == "")
