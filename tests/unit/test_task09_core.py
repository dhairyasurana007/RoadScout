"""Task 9: every candidate is verified through the provider, one verdict stored per candidate."""
from __future__ import annotations

import pytest

from rs9_helpers import (  # noqa: F401
    BEST_SIZE, CROP_SIZE, PV, PROMPT_ENV, THRESH_ENV, Env, class_names, close, cls_of, make_verdict,
    rgb_of, rs9_env, size_of, stub, write_prompt_config,
)

REQUIRED_COLUMNS = ["candidate_id", "is_hazard", "severity", "lane", "confidence", "reason",
                    "report_description", "model", "prompt_version"]


def test_verdicts_table_exists_after_init_db_with_required_columns(rs9_env):
    cols = rs9_env.columns()
    assert cols, "table verdicts does not exist after init_db()"
    for c in REQUIRED_COLUMNS:
        assert c in cols, "verdicts has no column %r (has %s)" % (c, cols)
    assert any(c in cols for c in ("class", "hazard_class")), "verdicts needs a class column: %s" % cols


def test_every_candidate_is_verified_once_through_the_provider(rs9_env):
    e = rs9_env
    cands = e.add_candidates(3)
    p = stub(e)
    e.run(provider=p)
    assert len(p.calls) == 3
    assert sorted(c["cand"]["id"] for c in p.calls) == [c["id"] for c in cands]
    got = e.by_candidate()
    assert sorted(got) == [c["id"] for c in cands]


def test_provider_receives_class_prompt_and_prompt_version(rs9_env):
    e = rs9_env
    cands = e.add_candidates(3)
    p = stub(e)
    e.run(provider=p)
    for call in p.calls:
        assert call["class_prompt"] == call["cand"]["cls"]
        assert call["prompt_version"] == PV


def test_images_sent_best_frame_first_then_crop(rs9_env):
    e = rs9_env
    e.add_candidates(3)
    p = stub(e)
    e.run(provider=p)
    for call in p.calls:
        imgs = call["images"]
        assert 2 <= len(imgs) <= 4, "best frame + crop + up to two neighbours, got %d images" % len(imgs)
        assert all(i is not None for i in imgs)
        c = call["cand"]
        assert close(rgb_of(imgs[0]), c["best_rgb"]), "first image must be the best frame"
        assert any(close(rgb_of(i), c["crop_rgb"]) for i in imgs[1:]), "the crop must be sent"


def test_candidate_without_crop_sends_best_frame_only(rs9_env):
    e = rs9_env
    e.add_candidates(2, crop=False)
    p = stub(e)
    e.run(provider=p)
    assert len(p.calls) == 2
    for call in p.calls:
        imgs = call["images"]
        assert 1 <= len(imgs) <= 3
        assert all(i is not None for i in imgs)
        assert close(rgb_of(imgs[0]), call["cand"]["best_rgb"])
    assert len(e.by_candidate()) == 2


def test_verdict_fields_round_trip_unchanged(rs9_env):
    e = rs9_env
    names = class_names()
    cands = e.add_candidates(5)
    script = {
        0: make_verdict(True, names[0], 3, "left", 0.97, 'Deep "pothole" with a sharp edge, café; it\'s bad.',
                        "Large pothole in the left lane.", model="stub-muse-1"),
        1: make_verdict(False, names[1], 0, None, 0.88, "Just a shadow.", "No hazard.", model="stub-muse-1"),
        2: make_verdict(True, names[2], 1, "right", 0.55, "Bent rail.", "Guardrail bent.", model="stub-muse-1"),
        3: make_verdict(True, names[3], 2, None, 0.0, "Unsure.", "Maybe a sign.", model="stub-muse-1"),
        4: make_verdict(True, names[4], 2, "center", 1.0, "Lines gone.", "Faded markings.", model="stub-muse-1"),
    }
    e.run(provider=stub(e, script=script))
    got = e.by_candidate()
    assert len(got) == 5
    for c in cands:
        want, row = script[c["k"]], got[c["id"]]
        assert bool(row["is_hazard"]) == want.is_hazard and row["is_hazard"] in (0, 1, True, False)
        assert cls_of(row) == want.hazard_class
        assert row["severity"] == want.severity and isinstance(row["severity"], int)
        assert row["lane"] == want.lane
        assert row["confidence"] == want.confidence
        assert row["reason"] == want.reason
        assert row["report_description"] == want.report_description
        assert row["model"] == "stub-muse-1"
        assert row["prompt_version"] == PV


def test_verdict_class_may_differ_from_the_detector_class(rs9_env):
    """Muse Spark judges the class; a known class it returns is stored as returned."""
    e = rs9_env
    names = class_names()
    cands = e.add_candidates(1, classes=[names[0]])
    e.run(provider=stub(e, script={0: make_verdict(True, names[2], 2, "left", 0.9)}))
    row = e.by_candidate()[cands[0]["id"]]
    assert cls_of(row) == names[2]


def test_model_and_prompt_version_are_stored_on_every_verdict(rs9_env):
    e = rs9_env
    e.add_candidates(4)
    p = stub(e, default=lambda c, cp, pv: make_verdict(cls=cp, model="stub-muse-9", prompt_version=""),
             echo_pv=False)
    e.run(provider=p)
    rows = e.rows()
    assert len(rows) == 4
    for r in rows:
        assert r["model"] == "stub-muse-9"
        assert r["prompt_version"] == PV, "prompt_version must come from the verification prompt config"


def test_changing_the_prompt_config_changes_the_stored_prompt_version(rs9_env):
    e = rs9_env
    e.add_candidates(3)
    e.set_version("t9-alpha")
    e.run(provider=stub(e))
    assert {r["prompt_version"] for r in e.rows()} == {"t9-alpha"}
    e.set_version("t9-beta")
    p = stub(e)
    e.run(provider=p)
    assert {r["prompt_version"] for r in e.rows()} == {"t9-beta"}
    assert {c["prompt_version"] for c in p.calls} == {"t9-beta"}
    assert len(e.rows()) == 3


def test_default_prompt_config_exists_and_gives_a_version(rs9_env, monkeypatch):
    e = rs9_env
    monkeypatch.delenv(PROMPT_ENV, raising=False)
    e.add_candidates(2)
    p = stub(e)
    e.run(provider=p)
    versions = {r["prompt_version"] for r in e.rows()}
    assert len(versions) == 1
    v = versions.pop()
    assert isinstance(v, str) and v.strip()
    assert {c["prompt_version"] for c in p.calls} == {v}


@pytest.mark.parametrize("bad", ["missing", "not json {", '{"prompt": "no version here"}',
                                 '{"version": "   ", "prompt": "x"}', '[1, 2]'])
def test_bad_prompt_config_is_a_clear_error_before_any_call(rs9_env, bad):
    e = rs9_env
    e.add_candidates(2)
    if bad == "missing":
        e.prompt_cfg.unlink()
    else:
        e.prompt_cfg.write_text(bad, encoding="utf-8")
    p = stub(e)
    with pytest.raises(Exception) as ei:
        e.run(provider=p)
    assert str(e.prompt_cfg.name) in str(ei.value) or "prompt" in str(ei.value).lower()
    assert p.calls == []
    assert e.rows() == []


def test_defaults_to_get_provider_which_is_the_fake_provider(rs9_env):
    e = rs9_env
    e.add_candidates(3)
    e.run()  # no provider: get_provider() with no env -> fake, default scenario
    rows = e.rows()
    assert len(rows) == 3
    assert {r["model"] for r in rows} == {"fake-muse-spark"}
    assert all(bool(r["is_hazard"]) for r in rows)
    assert {r["prompt_version"] for r in rows} == {PV}


def test_default_provider_follows_the_fake_scenario_env(rs9_env, monkeypatch):
    e = rs9_env
    e.add_candidates(3)
    monkeypatch.setenv("ROADSCOUT_FAKE_SCENARIO", "false_positive")
    e.run()
    rows = e.rows()
    assert len(rows) == 3
    assert all(not bool(r["is_hazard"]) for r in rows)


def test_default_provider_comes_from_get_provider_config(rs9_env, monkeypatch):
    e = rs9_env
    e.add_candidates(2)
    monkeypatch.setenv("ROADSCOUT_PROVIDER", "no-such-provider")
    with pytest.raises(ValueError):
        e.run()
    assert e.rows() == []


def test_idempotent_rerun_replaces_instead_of_duplicating(rs9_env):
    e = rs9_env
    e.add_candidates(3)
    e.run(provider=stub(e))
    first = e.rows()
    e.run(provider=stub(e))
    second = e.rows()
    assert len(first) == len(second) == 3
    assert len({r["candidate_id"] for r in second}) == 3
    assert len(e.raw_all()) == 3


def test_rerun_replaces_old_verdicts_with_the_new_ones(rs9_env):
    e = rs9_env
    names = class_names()
    cands = e.add_candidates(3)
    e.run(provider=stub(e, default=lambda c, cp, pv: make_verdict(True, cp, 3, "left", 0.9, model="m1")))
    e.run(provider=stub(e, default=lambda c, cp, pv: make_verdict(False, cp, 0, None, 0.4, "no", "no", model="m2")))
    rows = e.rows()
    assert len(rows) == 3
    assert all(not bool(r["is_hazard"]) and r["model"] == "m2" and r["severity"] == 0 for r in rows)


def test_other_drives_are_untouched_and_not_verified(rs9_env):
    e = rs9_env
    a = e.add_candidates(2, drive="drive-a", base_id=9101)
    b = e.add_candidates(2, drive="drive-b", base_id=9201)
    pb = stub(e, default=lambda c, cp, pv: make_verdict(True, cp, 1, "right", 0.8, model="b-model"))
    e.run("drive-b", provider=pb)
    b_before = e.rows("drive-b")
    pa = stub(e)
    e.run("drive-a", provider=pa)
    assert {c["cand"]["drive"] for c in pa.calls} == {"drive-a"}
    assert len(pa.calls) == 2
    assert e.rows("drive-b") == b_before
    assert len(e.rows("drive-a")) == 2
    # verifying drive-a again must not touch drive-b either
    e.run("drive-a", provider=stub(e))
    assert e.rows("drive-b") == b_before


def test_drive_without_candidates_verifies_nothing(rs9_env):
    e = rs9_env
    p = stub(e)
    e.run("drive-a", provider=p)  # no candidates at all
    e.run("never-heard-of-it", provider=p)
    assert p.calls == []
    assert e.raw_all() == []


def test_drive_with_candidates_elsewhere_but_none_here(rs9_env):
    e = rs9_env
    e.add_candidates(2, drive="drive-b", base_id=9201)
    p = stub(e)
    e.run("drive-a", provider=p)
    assert p.calls == [] and e.raw_all() == []


def test_blank_drive_id_is_rejected(rs9_env):
    e = rs9_env
    p = stub(e)
    for bad in ("", "   "):
        with pytest.raises(ValueError):
            e.run(bad, provider=p)
    assert p.calls == []


def test_replacing_candidates_does_not_break_and_leaves_no_orphan_verdicts(rs9_env):
    """aggregate_drive deletes and re-inserts a drive's candidates (foreign keys are on)."""
    e = rs9_env
    e.add_candidates(3, base_id=9101)
    e.run(provider=stub(e))
    assert len(e.raw_all()) == 3
    e.delete_candidates()  # must not raise an IntegrityError because verdicts reference candidates
    e.add_candidates(3, base_id=9301)
    e.run(provider=stub(e))
    assert len(e.rows()) == 3
    assert len(e.raw_all()) == 3, "verdicts of the replaced candidates must not linger"
