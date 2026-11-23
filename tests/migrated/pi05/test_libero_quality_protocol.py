import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "quality_protocol", Path(__file__).parents[1] / "scripts/libero_quality_protocol.py"
)
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)


def row(success, seed=0):
    return dict(
        success=success,
        termination="success" if success else "horizon",
        environment_seed=seed,
        initial_state_sha256="state",
        initial_observation_sha256="image",
    )


def test_complete_fixed_protocol_and_request_seeds():
    for model in ["pi05", "oft"]:
        spec = quality.protocol(model)
        episodes = quality.episode_plan(spec)
        assert len(episodes) == len({r["episode_id"] for r in episodes}) == 40
        assert spec["allowed_success_drop_pp"] is None
        assert spec["numerical_difference_policy"] == "diagnostic_only"
        seeds = [quality.request_seed(r["episode_id"], i) for r in episodes for i in range(3)]
        assert len(set(seeds)) == 120
        assert all(0 <= s < 2**63 for s in seeds)
        assert quality.request_seed("libero_10/0/0", 2) == quality.request_seed("libero_10/0/0", 2)


def test_zero_seed_is_valid_and_all_four_paired_outcomes_count():
    left = dict(a=row(True), b=row(True), c=row(False), d=row(False))
    right = dict(a=row(True), b=row(False), c=row(True), d=row(False))
    result = quality.paired(left, right)
    assert result["both_success"] == result["both_failure"] == 1
    assert result["official_only_success"] == ["b"]
    assert result["candidate_only_success"] == ["c"]
    assert result["delta_percentage_points"] == 0


@pytest.mark.parametrize("field", ["environment_seed", "initial_state_sha256", "initial_observation_sha256"])
def test_unpaired_inputs_rejected(field):
    bad = row(True)
    bad[field] = "different"
    with pytest.raises(ValueError, match=field):
        quality.paired({"a": row(True)}, {"a": bad})


def test_technical_errors_remain_in_denominator():
    error = row(False)
    error["termination"] = "infrastructure_error"
    result = quality.rate([row(True), row(False), error], 40)
    assert result["success_rate"] == 1 / 3
    assert result["technical_errors"] == 1 and result["pending"] == 37
    assert result["wilson95"][0] < 1 / 3 < result["wilson95"][1]


def test_incomplete_campaign_is_not_complete(tmp_path):
    spec = quality.protocol("oft")
    digest = quality.digest(spec)
    (tmp_path / "PROTOCOL.json").write_text(json.dumps(dict(protocol=spec, sha256=digest)))
    result = quality.summarize(tmp_path)
    assert not result["complete"] and result["modes"]["official"]["pending"] == 40
    assert result["paired"]["fp8"]["paired_episodes"] == 0
    folder = tmp_path / "official/episodes"
    folder.mkdir(parents=True)
    entry = dict(row(True), episode_id="libero_spatial/0/0", mode="official", protocol_sha256=digest)
    (folder / "one.json").write_text(json.dumps(entry))
    assert quality.summarize(tmp_path)["modes"]["official"]["successes"] == 1
    (folder / "duplicate.json").write_text(json.dumps(entry))
    with pytest.raises(ValueError, match="duplicate"):
        quality.summarize(tmp_path)
