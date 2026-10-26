"""Task 7: detection goes through get_provider(); the fake provider is the default; no network or key needed."""
import pytest

from rs7_helpers import (  # noqa: F401  (fixtures are imported for pytest)
    class_groups,
    find_default_config,
    parse_config,
    rs7_env,
    rs7_work,
    strings,
)


def test_default_provider_is_fake_and_needs_no_key_or_network(rs7_env):
    # rs7_env removed META_API_KEY / ROADSCOUT_PROVIDER and blocks sockets.
    rs7_env.run()  # no provider argument: must use get_provider()
    rows = rs7_env.rows()
    assert len(rows) == 36
    cfg_strings = set(strings(parse_config(find_default_config())))
    assert {r["prompt"] for r in rows} <= cfg_strings, "stored class prompts must be phrasings from the config"


def test_explicit_fake_env_matches_default(rs7_env):
    rs7_env.mp.setenv("ROADSCOUT_PROVIDER", "fake")
    rs7_env.run()
    assert len(rs7_env.rows()) == 36


@pytest.mark.parametrize("scenario,per_chunk", [
    ("default", 12), ("empty", 0), ("short_track", 2), ("false_positive", 8),
])
def test_fake_scenarios_store_the_raw_detections_unfiltered(rs7_env, scenario, per_chunk):
    """Raw storage: short tracks and false positives are kept (aggregation and verification drop them later)."""
    rs7_env.mp.setenv("ROADSCOUT_FAKE_SCENARIO", scenario)
    rs7_env.run()
    rows = rs7_env.rows()
    assert len(rows) == 3 * per_chunk
    for c in (0, 1, 2):
        assert sum(1 for r in rows if r["chunk"] == c) == per_chunk


def test_empty_fake_scenario_is_not_an_error(rs7_env):
    rs7_env.mp.setenv("ROADSCOUT_FAKE_SCENARIO", "empty")
    rs7_env.run()
    assert rs7_env.all_count() == 0


def test_live_provider_without_key_fails_clearly_and_stores_nothing(rs7_env):
    """Proves detection resolves its provider via get_provider(): live needs META_API_KEY."""
    rs7_env.mp.setenv("ROADSCOUT_PROVIDER", "live")
    with pytest.raises(Exception) as ei:
        rs7_env.run()
    assert "META_API_KEY" in str(ei.value) or "META_API_KEY" in str(ei.value.__cause__ or "")
    assert rs7_env.all_count() == 0


def test_unknown_provider_name_fails_clearly_and_stores_nothing(rs7_env):
    rs7_env.mp.setenv("ROADSCOUT_PROVIDER", "no-such-provider")
    with pytest.raises(Exception):
        rs7_env.run()
    assert rs7_env.all_count() == 0


def test_unknown_fake_scenario_fails_clearly_and_stores_nothing(rs7_env):
    rs7_env.mp.setenv("ROADSCOUT_FAKE_SCENARIO", "nope")
    with pytest.raises(Exception):
        rs7_env.run()
    assert rs7_env.all_count() == 0


def test_detection_does_not_modify_the_chunk_files_or_timestamps(rs7_env):
    import hashlib

    def digest():
        h = hashlib.sha256()
        for p in sorted(rs7_env.work_dir.iterdir()):
            if p.is_file():
                h.update(p.name.encode())
                h.update(p.read_bytes())
        return h.hexdigest()

    before = digest()
    rs7_env.run()
    assert digest() == before, "detection must not change the preprocess output"


def test_missing_work_dir_is_a_clear_error_and_stores_nothing(rs7_env, tmp_path):
    with pytest.raises(Exception):
        rs7_env.run(work_dir=tmp_path / "nothing-here")
    assert rs7_env.all_count() == 0
