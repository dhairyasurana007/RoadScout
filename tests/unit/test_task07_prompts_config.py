"""Task 7: hazard class prompts come from a swappable config file and are what the provider is sent."""
import json

import pytest

from rs7_helpers import (  # noqa: F401  (fixtures are imported for pytest)
    CLASS_KEYWORDS,
    PROMPTS_ENV,
    RecordingProvider,
    class_groups,
    find_default_config,
    parse_config,
    required_groups,
    rs7_env,
    rs7_work,
    strings,
    swapped_copy,
)


# ------------------------------------------------------------------ the shipped default config


def test_default_config_file_exists_under_repo():
    assert find_default_config() is not None, "expected a prompt config such as config/prompts.json"


def test_default_config_has_all_five_hazard_classes_with_phrasings():
    cfg = parse_config(find_default_config())
    groups = class_groups(cfg)
    found = required_groups(groups)
    missing = [k for k in CLASS_KEYWORDS if k not in found]
    assert not missing, "classes missing from the config (by keyword): %s; groups=%s" % (missing, list(groups))
    for key, name in found.items():
        phr = groups[name]
        assert len(phr) >= 1, "class %r has no phrasings" % name
        assert all(isinstance(p, str) and p.strip() for p in phr), "blank phrasing in class %r" % name


def test_default_config_phrasings_are_plain_text_not_empty_and_unique_per_class():
    groups = class_groups(parse_config(find_default_config()))
    for name, phr in groups.items():
        assert len(set(p.strip().lower() for p in phr)) == len(phr), "duplicate phrasing in %r" % name


# ------------------------------------------------------------------ prompts reach the provider


def test_every_chunk_is_sent_to_detect_with_all_classes_from_config(rs7_env):
    rec = RecordingProvider()
    rs7_env.run(provider=rec)
    cfg_groups = class_groups(parse_config(find_default_config()))
    required = required_groups(cfg_groups)
    all_cfg = set(strings(parse_config(find_default_config())))
    assert rec.calls, "provider.detect was never called"
    chunks_called = sorted({c["chunk"] for c in rec.calls})
    assert chunks_called == [0, 1, 2]
    for c in rec.calls:
        assert c["exists"], "detect() was given a path that does not exist: %r" % (c["path"],)
        assert c["prompts"], "detect() was called with no prompts"
        assert all(isinstance(p, str) for p in c["prompts"])
        assert set(c["prompts"]) <= all_cfg, "prompts not from the config: %s" % (set(c["prompts"]) - all_cfg)
    for chunk in chunks_called:
        sent = set(rec.prompts_for_chunk(chunk))
        for key, name in required.items():
            assert sent & set(cfg_groups[name]), "chunk %d: no phrasing of class %r was sent" % (chunk, name)


def test_chunk_paths_given_to_provider_are_the_preprocessed_chunk_files(rs7_env):
    rec = RecordingProvider()
    rs7_env.run(provider=rec)
    names = {str(c["path"]).replace("\\", "/").rsplit("/", 1)[-1] for c in rec.calls}
    assert names == {"chunk_0000.mp4", "chunk_0001.mp4", "chunk_0002.mp4"}
    work = str(rs7_env.work_dir).replace("\\", "/")
    for c in rec.calls:
        assert str(c["path"]).replace("\\", "/").startswith(work), "chunk path outside the work dir"


def test_each_chunk_gets_the_same_prompt_set(rs7_env):
    rec = RecordingProvider()
    rs7_env.run(provider=rec)
    sets = {c: frozenset(rec.prompts_for_chunk(c)) for c in (0, 1, 2)}
    assert len(set(sets.values())) == 1, "different chunks were sent different prompts: %s" % sets


# ------------------------------------------------------------------ swapping the config


def test_env_var_swaps_the_prompts_used(rs7_env, tmp_path):
    path, prefix = swapped_copy(tmp_path)
    rs7_env.mp.setenv(PROMPTS_ENV, str(path))
    rec = RecordingProvider()
    rs7_env.run(provider=rec)
    assert rec.calls
    for c in rec.calls:
        assert c["prompts"], "no prompts sent"
        assert all(p.startswith(prefix) for p in c["prompts"]), \
            "prompts did not come from the file named by %s: %s" % (PROMPTS_ENV, c["prompts"])
    # The stored class prompt is the swapped phrasing too (the fake provider echoes prompts[0]).
    rows = rs7_env.rows()
    assert rows and all(r["prompt"].startswith(prefix) for r in rows)


def test_config_is_read_per_run_not_cached_at_import(rs7_env, tmp_path):
    rec1 = RecordingProvider()
    rs7_env.run("drive-a", provider=rec1)
    assert rec1.calls and not any(p.startswith("swapped: ") for c in rec1.calls for p in c["prompts"])
    path, prefix = swapped_copy(tmp_path)
    rs7_env.mp.setenv(PROMPTS_ENV, str(path))
    rs7_env.add_drive("drive-b")
    rec2 = RecordingProvider()
    rs7_env.run("drive-b", provider=rec2)
    assert rec2.calls and all(p.startswith(prefix) for c in rec2.calls for p in c["prompts"])
    rs7_env.mp.delenv(PROMPTS_ENV)
    rs7_env.add_drive("drive-c")
    rec3 = RecordingProvider()
    rs7_env.run("drive-c", provider=rec3)
    assert not any(p.startswith(prefix) for c in rec3.calls for p in c["prompts"])


def test_config_path_parameter_swaps_the_prompts_used(rs7_env, tmp_path):
    if not rs7_env.has_prompts_param():
        pytest.skip("no prompt-config path parameter (the env var route is tested separately)")
    path, prefix = swapped_copy(tmp_path)
    rec = RecordingProvider()
    rs7_env.run(provider=rec, prompts_path=path)
    assert rec.calls and all(p.startswith(prefix) for c in rec.calls for p in c["prompts"])


def test_custom_json_config_with_documented_shape(rs7_env, tmp_path):
    path = tmp_path / "mine.json"
    path.write_text(json.dumps({"classes": {"alpha thing": ["alpha one", "alpha two"], "beta thing": ["beta only"]}}),
                    encoding="utf-8")
    rs7_env.mp.setenv(PROMPTS_ENV, str(path))
    rec = RecordingProvider()
    rs7_env.run(provider=rec)
    allowed = {"alpha one", "alpha two", "beta only"}
    for chunk in (0, 1, 2):
        sent = set(rec.prompts_for_chunk(chunk))
        assert sent <= allowed, "unexpected prompts %s" % (sent - allowed)
        assert sent & {"alpha one", "alpha two"}, "class alpha thing not sent for chunk %d" % chunk
        assert "beta only" in sent, "class beta thing not sent for chunk %d" % chunk


# ------------------------------------------------------------------ bad config


def test_missing_config_file_is_a_clear_error_with_no_calls_and_no_rows(rs7_env, tmp_path):
    rs7_env.mp.setenv(PROMPTS_ENV, str(tmp_path / "does-not-exist.json"))
    rec = RecordingProvider()
    with pytest.raises(Exception) as ei:
        rs7_env.run(provider=rec)
    assert not rec.calls
    assert rs7_env.all_count() == 0
    assert "does-not-exist" in str(ei.value) or isinstance(ei.value, (FileNotFoundError, ValueError))


def test_malformed_config_is_an_error_with_no_calls_and_no_rows(rs7_env, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    rs7_env.mp.setenv(PROMPTS_ENV, str(bad))
    rec = RecordingProvider()
    with pytest.raises(Exception):
        rs7_env.run(provider=rec)
    assert not rec.calls
    assert rs7_env.all_count() == 0


def test_config_without_any_prompts_never_sends_an_empty_prompt_list(rs7_env, tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"classes": {}}), encoding="utf-8")
    rs7_env.mp.setenv(PROMPTS_ENV, str(empty))
    rec = RecordingProvider()
    try:
        rs7_env.run(provider=rec)
    except Exception:
        pass  # raising is fine
    assert all(c["prompts"] for c in rec.calls), "detect() called with an empty prompt list"
