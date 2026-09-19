import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "realistic_niah_v3_2_empirical_law_analysis.json"
DOC = ROOT / "docs" / "realistic_niah_v3_2_empirical_law_analysis_spec.md"
FREEZE = (
    ROOT
    / "configs"
    / "realistic_niah_v3_2_empirical_law_analysis.freeze.json"
)


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_v3_2_freezes_completed_v3_1_inference_grid() -> None:
    config = load_config()
    immutable = config["immutable_input"]
    assert config["analysis_version"] == "V3.2"
    assert config["inference_protocol"] == "realistic_niah_v3_1"
    assert immutable["requests"] == 161_280
    assert immutable["physical_model_revisions"] == 14
    assert immutable["comparison_slots"] == 12
    assert immutable["model_mode_slots"] == 48
    assert immutable["seeds_per_cell"] == 30
    assert len(immutable["N_levels"]) == 14
    assert len(immutable["L_levels"]) == 8
    assert len(immutable["prompt_modes"]) == 4


def test_v3_2_uses_only_registered_structures() -> None:
    config = load_config()
    candidates = config["candidate_registry"]
    assert len(candidates) == 13
    assert candidates[0] == {"id": "intercept", "terms": []}
    assert sum("interaction" in candidate for candidate in candidates) == 4
    for candidate in candidates:
        if "interaction" in candidate:
            assert candidate["parent"]
            assert candidate["interaction"] in candidate["terms"]


def test_v3_2_formal_estimands_and_accuracy_families_are_fixed() -> None:
    config = load_config()
    estimands = config["estimands"]
    families = config["accuracy_families"]
    assert estimands["accuracy"]["name"] == "parsed_exact_accuracy"
    assert estimands["bias"]["name"] == "trimmed_signed_bias_10"
    assert estimands["bias"]["minimum_parseable"] == 20
    assert "mean_signed_deviation" in estimands["discarded_law_estimands"]
    assert families["headline"]["id"] == "bernoulli_logit"
    assert [item["id"] for item in families["link_robustness"]] == [
        "bernoulli_probit",
        "bernoulli_cloglog",
    ]
    assert families["overdispersion_robustness"]["id"] == (
        "beta_binomial_logit"
    )


def test_v3_2_has_no_bootstrap_or_nested_axis_selection() -> None:
    config = load_config()
    selection = config["selection"]
    assert config["uncertainty"]["bootstrap_repetitions"] == 0
    assert selection["condition_folds"] == 5
    assert not selection["held_seed_nested_selection"]
    assert not selection["held_N_nested_selection"]
    assert not selection["held_L_nested_selection"]
    assert selection["headline_lomo"]
    assert selection["trimmed_bias_lomo"]


def test_v3_2_document_matches_the_machine_readable_contract() -> None:
    text = DOC.read_text(encoding="utf-8")
    for required in (
        "161,280",
        "10%",
        "m >= 20",
        "Bernoulli-logit",
        "probit",
        "cloglog",
        "Beta-Binomial",
        "HC3",
        "Benjamini-Hochberg",
        "LOMO",
        "No bootstrap",
    ):
        assert required in text


def test_v3_2_freeze_hashes_match() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["schema_version"] == (
        "realistic_niah_v3_2_empirical_law_analysis_freeze_v2"
    )
    assert freeze["amendment"]["scope"] == "human_readable_spec_only"
    assert freeze["amendment"]["inference_requests_changed"] is False
    assert freeze["amendment"]["base_machine_readable_contract_changed"] is False
    for relative, expected in freeze["files"].items():
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected
