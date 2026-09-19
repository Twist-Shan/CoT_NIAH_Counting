from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.causal import (
    bootstrap_seed_mean_ci,
    holm_adjust,
    sign_flip_pvalue,
)


PRIMARY_GRAMMARS = (
    "adjacent_rank_after_city",
    "adjacent_rank_before_city",
    "same_unit_rank_before_city",
    "structural_unmarked",
)
EXPECTED_RANDOM_REPEATS = (1, 2, 3)
ANCHOR_KEY = ("model_label", "seed", "request_id", "from_occurrence", "to_occurrence")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_shards(run_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    shards = sorted((run_dir / "shards").glob("*.jsonl"))
    for shard in shards:
        with shard.open("r", encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        if len(lines) != 1:
            raise ValueError(f"Expected one row in atomic shard {shard}; found {len(lines)}")
        records.extend(lines)
    if not records:
        raise ValueError(f"No completed trial shards found in {run_dir}")
    frame = pd.DataFrame(records)
    required = {
        *ANCHOR_KEY,
        "condition",
        "repeat",
        "correct_next_needle",
        "behavior_outcome",
        "split",
        "gold_count",
        "routed_target_grammar_class",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Trial shards in {run_dir} lack columns: {missing}")
    if "status" in frame and not frame["status"].astype(str).eq("ok").all():
        bad = frame.loc[~frame["status"].astype(str).eq("ok"), "status"].value_counts()
        raise ValueError(f"Non-ok completed shards in {run_dir}: {bad.to_dict()}")
    if "trial_complete" in frame and not frame["trial_complete"].astype(bool).all():
        raise ValueError(f"Incomplete atomic trial shard found in {run_dir}")
    return frame


def _read_anchor_registry(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "selected_anchor_registry.jsonl").open(
        "r", encoding="utf-8"
    ) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _registry_map(records: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], str]:
    result: dict[tuple[Any, ...], str] = {}
    for row in records:
        key = (
            str(row["request_id"]),
            int(row["from_occurrence"]),
            int(row["to_occurrence"]),
        )
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
        if key in result:
            raise ValueError(f"Duplicate anchor-registry key: {key}")
        result[key] = canonical
    return result


def _registry_relation(candidate_dir: Path, reference_dir: Path) -> str:
    candidate = _registry_map(_read_anchor_registry(candidate_dir))
    reference = _registry_map(_read_anchor_registry(reference_dir))
    mismatched = [
        key
        for key, value in candidate.items()
        if key not in reference or reference[key] != value
    ]
    if mismatched:
        raise ValueError(
            f"Run registry {candidate_dir} is not an exact row-wise subset of "
            f"the clean reference registry; examples={mismatched[:5]}"
        )
    return "exact_match" if len(candidate) == len(reference) else "exact_subset"


def _load_run(run_dir: Path, bank_size: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    registry_path = run_dir / "selected_anchor_registry.jsonl"
    if not manifest_path.exists() or not registry_path.exists():
        raise FileNotFoundError(f"Run lacks manifest/anchor registry: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry_sha = _sha256(registry_path)
    expected_registry_sha = str(manifest.get("selected_anchor_registry_sha256", ""))
    if registry_sha != expected_registry_sha:
        raise ValueError(
            f"Anchor registry hash mismatch in {run_dir}: {registry_sha} != "
            f"{expected_registry_sha}"
        )
    frame = _read_shards(run_dir)
    planned = pd.to_numeric(frame.get("planned_bank_size"), errors="coerce")
    if planned.isna().any() or set(planned.astype(int)) != {int(bank_size)}:
        raise ValueError(
            f"Run {run_dir} does not contain exactly planned K={bank_size}"
        )
    frame = frame.copy()
    frame["registered_bank_size"] = int(bank_size)
    return frame, manifest


def _assert_metadata_constant(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    grouped = frame.groupby(list(ANCHOR_KEY), dropna=False, sort=False)
    for column in columns:
        bad = grouped[column].nunique(dropna=False)
        if bool((bad != 1).any()):
            example = bad[bad != 1].head(5).to_dict()
            raise ValueError(f"Anchor metadata {column} is inconsistent: {example}")


def _anchor_table(
    trials: pd.DataFrame,
    *,
    bank_size: int,
    clean_reference: pd.DataFrame | None,
    require_complete: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = trials.loc[trials["condition"].astype(str).eq("selected_bank")].copy()
    random = trials.loc[
        trials["condition"].astype(str).eq("layer_matched_random")
    ].copy()
    clean = trials.loc[trials["condition"].astype(str).eq("clean")].copy()
    if clean.empty and clean_reference is not None:
        clean = clean_reference.copy()

    for name, arm in (("clean", clean), ("selected", selected)):
        duplicated = arm.duplicated(list(ANCHOR_KEY), keep=False)
        if bool(duplicated.any()):
            raise ValueError(f"Duplicate {name} rows for an anchor at K={bank_size}")
    random_duplicates = random.duplicated([*ANCHOR_KEY, "repeat"], keep=False)
    if bool(random_duplicates.any()):
        raise ValueError(f"Duplicate random repeat rows for an anchor at K={bank_size}")

    metadata_columns = (
        "split",
        "gold_count",
        "routed_target_grammar_class",
        "target_retrieval_surface_variant",
        "query_site_id",
    )
    available_metadata = [column for column in metadata_columns if column in trials]
    _assert_metadata_constant(trials, available_metadata)
    metadata = (
        trials[[*ANCHOR_KEY, "condition", "repeat", *available_metadata]]
        .copy()
        .sort_values([*ANCHOR_KEY, "condition", "repeat"])
        .groupby(list(ANCHOR_KEY), as_index=False, dropna=False)
        .first()[[*ANCHOR_KEY, *available_metadata]]
    )

    def correctness(arm: pd.DataFrame, name: str) -> pd.DataFrame:
        result = arm[[*ANCHOR_KEY, "correct_next_needle", "behavior_outcome"]].copy()
        result[f"{name}_correct"] = result.pop("correct_next_needle").astype(bool).astype(float)
        result = result.rename(columns={"behavior_outcome": f"{name}_outcome"})
        return result

    table = metadata.merge(correctness(clean, "clean"), on=list(ANCHOR_KEY), how="left")
    table = table.merge(
        correctness(selected, "selected"), on=list(ANCHOR_KEY), how="left"
    )
    random_work = random[[*ANCHOR_KEY, "repeat", "correct_next_needle"]].copy()
    random_work["repeat"] = pd.to_numeric(random_work["repeat"], errors="raise").astype(int)
    random_work["correct_next_needle"] = (
        random_work["correct_next_needle"].astype(bool).astype(float)
    )
    unexpected_repeats = sorted(
        set(random_work["repeat"].unique()) - set(EXPECTED_RANDOM_REPEATS)
    )
    if unexpected_repeats:
        raise ValueError(
            f"Unexpected random repeats at K={bank_size}: {unexpected_repeats}"
        )
    random_wide = random_work.pivot(
        index=list(ANCHOR_KEY), columns="repeat", values="correct_next_needle"
    ).reset_index()
    random_wide = random_wide.rename(
        columns={repeat: f"random_repeat_{repeat}_correct" for repeat in EXPECTED_RANDOM_REPEATS}
    )
    table = table.merge(random_wide, on=list(ANCHOR_KEY), how="left")
    random_columns = [f"random_repeat_{repeat}_correct" for repeat in EXPECTED_RANDOM_REPEATS]
    for column in random_columns:
        if column not in table:
            table[column] = np.nan
    table["random_mean_correct"] = table[random_columns].mean(axis=1)
    table["clean_failure"] = 1.0 - table["clean_correct"]
    table["selected_failure"] = 1.0 - table["selected_correct"]
    table["random_mean_failure"] = 1.0 - table["random_mean_correct"]
    for repeat in EXPECTED_RANDOM_REPEATS:
        table[f"random_repeat_{repeat}_failure"] = (
            1.0 - table[f"random_repeat_{repeat}_correct"]
        )
    table["selected_minus_random_failure"] = (
        table["selected_failure"] - table["random_mean_failure"]
    )
    table["registered_bank_size"] = int(bank_size)
    complete_columns = ["clean_correct", "selected_correct", *random_columns]
    table["complete_five_arm_pair"] = table[complete_columns].notna().all(axis=1)
    incomplete = table.loc[~table["complete_five_arm_pair"]]
    if require_complete and not incomplete.empty:
        raise ValueError(
            f"K={bank_size} has {len(incomplete)} incomplete five-arm anchors; "
            f"examples={incomplete['request_id'].head(5).tolist()}"
        )
    audit = {
        "registered_bank_size": int(bank_size),
        "observed_trial_rows": int(len(trials)),
        "observed_anchor_units": int(len(table)),
        "complete_five_arm_anchor_units": int(table["complete_five_arm_pair"].sum()),
        "incomplete_anchor_units": int((~table["complete_five_arm_pair"]).sum()),
        "condition_rows": {
            str(key): int(value)
            for key, value in trials["condition"].astype(str).value_counts().items()
        },
    }
    return table, audit


def _scope(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "full_panel":
        return frame
    return frame.loc[frame["split"].astype(str).eq(name)]


def _population(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "all_examples":
        return frame
    if name == "clean_correct_only":
        return frame.loc[frame["clean_correct"].eq(1.0)]
    raise ValueError(name)


def _grammar(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "pooled":
        return frame
    if name == "macro_primary_grammars":
        return frame.loc[
            frame["routed_target_grammar_class"].astype(str).isin(PRIMARY_GRAMMARS)
        ]
    return frame.loc[frame["routed_target_grammar_class"].astype(str).eq(name)]


def _seed_means(
    frame: pd.DataFrame,
    metric: str,
    *,
    equalize_by: str | None = None,
) -> pd.DataFrame:
    work = frame.loc[np.isfinite(pd.to_numeric(frame[metric], errors="coerce"))].copy()
    work[metric] = pd.to_numeric(work[metric], errors="raise")
    if equalize_by is not None:
        strata = (
            work.groupby(
                ["model_label", "seed", equalize_by],
                as_index=False,
                dropna=False,
            )
            .agg(
                stratum_value=(metric, "mean"),
                stratum_anchor_units=(metric, "size"),
            )
        )
        return (
            strata.groupby(["model_label", "seed"], as_index=False, dropna=False)
            .agg(
                value=("stratum_value", "mean"),
                n_anchor_units=("stratum_anchor_units", "sum"),
                n_equalized_strata=(equalize_by, "nunique"),
            )
        )
    return (
        work.groupby(["model_label", "seed"], as_index=False, dropna=False)
        .agg(value=(metric, "mean"), n_anchor_units=(metric, "size"))
        .assign(n_equalized_strata=1)
    )


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _summary(
    frame: pd.DataFrame,
    *,
    metric: str,
    bootstrap_samples: int,
    seed_parts: Iterable[object],
    equalize_by: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    seed_frame = _seed_means(frame, metric, equalize_by=equalize_by)
    if seed_frame.empty:
        return {
            "mean": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n_seeds": 0,
            "n_anchor_units": 0,
            "anchor_weighted_mean": np.nan,
            "mean_equalized_strata_per_seed": np.nan,
        }, seed_frame
    statistics = bootstrap_seed_mean_ci(
        seed_frame["value"],
        samples=int(bootstrap_samples),
        seed=_stable_seed(*seed_parts),
    )
    result = {
        "mean": float(statistics["mean_effect"]),
        "ci95_low": float(statistics["ci_low"]),
        "ci95_high": float(statistics["ci_high"]),
        "n_seeds": int(statistics["n_seeds"]),
        "n_anchor_units": int(len(frame)),
        "anchor_weighted_mean": float(pd.to_numeric(frame[metric]).mean()),
        "mean_equalized_strata_per_seed": float(
            seed_frame["n_equalized_strata"].mean()
        ),
    }
    return result, seed_frame


def _analyze(
    anchors: pd.DataFrame,
    trials: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    estimand_rows: list[dict[str, Any]] = []
    arm_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    flow_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    populations = ("all_examples", "clean_correct_only")

    for bank_size, bank_frame in anchors.groupby("registered_bank_size", sort=True):
        available_splits = set(bank_frame["split"].astype(str).unique())
        scopes = tuple(
            scope
            for scope in ("confirmation", "full_panel", "discovery")
            if (
                scope in available_splits
                or (scope == "full_panel" and len(available_splits) > 1)
            )
        )
        grammar_names = (
            "pooled",
            "macro_primary_grammars",
            *sorted(bank_frame["routed_target_grammar_class"].astype(str).unique()),
        )
        for scope_name in scopes:
            scoped = _scope(bank_frame, scope_name)
            for population_name in populations:
                populated = _population(scoped, population_name)
                for grammar_name in grammar_names:
                    cohort = _grammar(populated, grammar_name)
                    if cohort.empty:
                        continue
                    context = {
                        "registered_bank_size": int(bank_size),
                        "evaluation_scope": scope_name,
                        "analysis_population": population_name,
                        "grammar_class": grammar_name,
                    }
                    effect_summary, effect_seeds = _summary(
                        cohort,
                        metric="selected_minus_random_failure",
                        bootstrap_samples=bootstrap_samples,
                        seed_parts=("effect", *context.values()),
                        equalize_by=(
                            "routed_target_grammar_class"
                            if grammar_name == "macro_primary_grammars"
                            else None
                        ),
                    )
                    pvalue = (
                        sign_flip_pvalue(effect_seeds["value"])
                        if len(effect_seeds) >= 2
                        else np.nan
                    )
                    estimand_rows.append(
                        {
                            **context,
                            "estimand": "selected_failure_minus_mean_random_failure",
                            **effect_summary,
                            "sign_flip_p": pvalue,
                            "sign_flip_method": (
                                "exact"
                                if 2 <= len(effect_seeds) <= 20
                                else (
                                    "deterministic_monte_carlo_200000"
                                    if len(effect_seeds) > 20
                                    else "not_run"
                                )
                            ),
                            "inferential_status": (
                                "registered_confirmation"
                                if scope_name == "confirmation" and len(effect_seeds) >= 10
                                else "descriptive_or_secondary"
                            ),
                        }
                    )
                    for _, seed_row in effect_seeds.iterrows():
                        seed_rows.append(
                            {
                                **context,
                                "seed": int(seed_row["seed"]),
                                "seed_effect": float(seed_row["value"]),
                                "n_anchor_units": int(seed_row["n_anchor_units"]),
                            }
                        )
                    arm_metrics = {
                        "clean": "clean_failure",
                        "selected_bank": "selected_failure",
                        "layer_matched_random_mean": "random_mean_failure",
                        **{
                            f"layer_matched_random_repeat_{repeat}": f"random_repeat_{repeat}_failure"
                            for repeat in EXPECTED_RANDOM_REPEATS
                        },
                    }
                    for arm_name, metric in arm_metrics.items():
                        arm_summary, _ = _summary(
                            cohort,
                            metric=metric,
                            bootstrap_samples=bootstrap_samples,
                            seed_parts=("arm", arm_name, *context.values()),
                            equalize_by=(
                                "routed_target_grammar_class"
                                if grammar_name == "macro_primary_grammars"
                                else None
                            ),
                        )
                        arm_rows.append(
                            {
                                **context,
                                "arm": arm_name,
                                "metric": "failure_rate",
                                **arm_summary,
                            }
                        )

        for scope_name in scopes:
            scoped = _scope(bank_frame, scope_name)
            flow_rows.append(
                {
                    "registered_bank_size": int(bank_size),
                    "evaluation_scope": scope_name,
                    "stage": "completed_five_arm_pairs",
                    "gold_count": "all",
                    "n_prompt_units": np.nan,
                    "n_eligible_anchor_units": int(len(scoped)),
                    "n_ineligible_prompt_units": np.nan,
                    "n_seeds": int(scoped["seed"].nunique()),
                    "n_clean_correct": int(scoped["clean_correct"].eq(1.0).sum()),
                    "n_complete_five_arm_pairs": int(
                        scoped["complete_five_arm_pair"].sum()
                    ),
                }
            )

        bank_trials = trials.loc[trials["registered_bank_size"].eq(bank_size)].copy()
        bank_trials["arm"] = np.where(
            bank_trials["condition"].astype(str).eq("layer_matched_random"),
            "layer_matched_random_pooled_repeats",
            bank_trials["condition"].astype(str),
        )
        bank_trials["analysis_scope"] = bank_trials["split"].astype(str)
        failure_source = bank_trials
        if bank_trials["split"].astype(str).nunique() > 1:
            pooled_trials = bank_trials.copy()
            pooled_trials["analysis_scope"] = "full_panel"
            failure_source = pd.concat([bank_trials, pooled_trials], ignore_index=True)
        grouped_failures = failure_source.groupby(
            [
                "analysis_scope",
                "arm",
                "routed_target_grammar_class",
                "behavior_outcome",
            ],
            dropna=False,
            sort=True,
        )
        denominators = failure_source.groupby(
            ["analysis_scope", "arm", "routed_target_grammar_class"],
            dropna=False,
        ).size()
        for (scope_name, arm_name, grammar_name, outcome), block in grouped_failures:
            denominator = int(denominators.loc[(scope_name, arm_name, grammar_name)])
            failure_rows.append(
                {
                    "registered_bank_size": int(bank_size),
                    "evaluation_scope": str(scope_name),
                    "arm": str(arm_name),
                    "grammar_class": str(grammar_name),
                    "behavior_outcome": str(outcome),
                    "n_trials": int(len(block)),
                    "fraction_within_arm_grammar": float(len(block) / denominator),
                }
            )

    estimands = pd.DataFrame(estimand_rows)
    if not estimands.empty:
        estimands["holm_family"] = ""
        estimands["holm_p"] = np.nan
        pooled_mask = estimands["grammar_class"].eq("pooled")
        for keys, indices in estimands.loc[pooled_mask].groupby(
            ["evaluation_scope", "analysis_population"], sort=True
        ).groups.items():
            finite = [index for index in indices if np.isfinite(estimands.at[index, "sign_flip_p"])]
            if finite:
                estimands.loc[finite, "holm_p"] = holm_adjust(
                    estimands.loc[finite, "sign_flip_p"].to_numpy(dtype=float)
                )
                estimands.loc[finite, "holm_family"] = (
                    f"pooled_K_grid:{keys[0]}:{keys[1]}"
                )
        primary_grammar_mask = (
            estimands["registered_bank_size"].eq(125)
            & estimands["grammar_class"].isin(PRIMARY_GRAMMARS)
        )
        for keys, indices in estimands.loc[primary_grammar_mask].groupby(
            ["evaluation_scope", "analysis_population"], sort=True
        ).groups.items():
            finite = [index for index in indices if np.isfinite(estimands.at[index, "sign_flip_p"])]
            if finite:
                estimands.loc[finite, "holm_p"] = holm_adjust(
                    estimands.loc[finite, "sign_flip_p"].to_numpy(dtype=float)
                )
                estimands.loc[finite, "holm_family"] = (
                    f"K125_primary_grammars:{keys[0]}:{keys[1]}"
                )

    return (
        estimands,
        pd.DataFrame(arm_rows),
        pd.DataFrame(seed_rows),
        pd.DataFrame(flow_rows),
        pd.DataFrame(failure_rows),
    )


def _analyze_count_strata(
    anchors: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimand_rows: list[dict[str, Any]] = []
    arm_rows: list[dict[str, Any]] = []
    populations = ("all_examples", "clean_correct_only")
    arm_metrics = {
        "clean": "clean_failure",
        "selected_bank": "selected_failure",
        "layer_matched_random_mean": "random_mean_failure",
        **{
            f"layer_matched_random_repeat_{repeat}": f"random_repeat_{repeat}_failure"
            for repeat in EXPECTED_RANDOM_REPEATS
        },
    }
    for bank_size, bank_frame in anchors.groupby("registered_bank_size", sort=True):
        available_splits = set(bank_frame["split"].astype(str).unique())
        scopes = tuple(
            scope
            for scope in ("confirmation", "full_panel", "discovery")
            if (
                scope in available_splits
                or (scope == "full_panel" and len(available_splits) > 1)
            )
        )
        for scope_name in scopes:
            scoped = _scope(bank_frame, scope_name)
            for population_name in populations:
                populated = _population(scoped, population_name)
                for count in sorted(
                    pd.to_numeric(populated["gold_count"], errors="raise")
                    .astype(int)
                    .unique()
                ):
                    cohort = populated.loc[
                        pd.to_numeric(populated["gold_count"], errors="raise")
                        .astype(int)
                        .eq(int(count))
                    ]
                    if cohort.empty:
                        continue
                    context = {
                        "registered_bank_size": int(bank_size),
                        "evaluation_scope": scope_name,
                        "analysis_population": population_name,
                        "gold_count": int(count),
                    }
                    effect_summary, effect_seeds = _summary(
                        cohort,
                        metric="selected_minus_random_failure",
                        bootstrap_samples=bootstrap_samples,
                        seed_parts=("count_effect", *context.values()),
                    )
                    estimand_rows.append(
                        {
                            **context,
                            "estimand": "selected_failure_minus_mean_random_failure",
                            **effect_summary,
                            "sign_flip_p": (
                                sign_flip_pvalue(effect_seeds["value"])
                                if len(effect_seeds) >= 2
                                else np.nan
                            ),
                            "sign_flip_method": (
                                "exact"
                                if 2 <= len(effect_seeds) <= 20
                                else (
                                    "deterministic_monte_carlo_200000"
                                    if len(effect_seeds) > 20
                                    else "not_run"
                                )
                            ),
                            "inferential_status": "descriptive_count_stratum",
                        }
                    )
                    for arm_name, metric in arm_metrics.items():
                        arm_summary, _ = _summary(
                            cohort,
                            metric=metric,
                            bootstrap_samples=bootstrap_samples,
                            seed_parts=("count_arm", arm_name, *context.values()),
                        )
                        arm_rows.append(
                            {
                                **context,
                                "arm": arm_name,
                                "metric": "failure_rate",
                                **arm_summary,
                            }
                        )
    estimands = pd.DataFrame(estimand_rows)
    if not estimands.empty:
        estimands["holm_family"] = ""
        estimands["holm_p"] = np.nan
        for keys, indices in estimands.groupby(
            [
                "registered_bank_size",
                "evaluation_scope",
                "analysis_population",
            ],
            sort=True,
        ).groups.items():
            finite = [
                index
                for index in indices
                if np.isfinite(estimands.at[index, "sign_flip_p"])
            ]
            if finite:
                estimands.loc[finite, "holm_p"] = holm_adjust(
                    estimands.loc[finite, "sign_flip_p"].to_numpy(dtype=float)
                )
                estimands.loc[finite, "holm_family"] = (
                    f"count_strata:K{keys[0]}:{keys[1]}:{keys[2]}"
                )
    return estimands, pd.DataFrame(arm_rows)


def _parse_run(value: str) -> tuple[int, Path]:
    try:
        raw_k, raw_path = value.split("=", 1)
        bank_size = int(raw_k)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("--run must be K=/absolute/or/relative/path") from error
    return bank_size, Path(raw_path)


def _registered_panel_flow(
    run_dir: Path,
    selection: dict[str, Any],
) -> pd.DataFrame:
    with (run_dir / "selected_anchor_registry.jsonl").open(
        "r", encoding="utf-8"
    ) as handle:
        registry = pd.DataFrame(json.loads(line) for line in handle if line.strip())
    panel = selection["sample_panel"]
    discovery = {int(value) for value in panel["discovery_seeds"]}
    confirmation = {int(value) for value in panel["registered_confirmation_seeds"]}
    counts = [int(value) for value in panel["counts"]]
    registry = registry.copy()
    registry["split"] = np.where(
        pd.to_numeric(registry["seed"], errors="raise").astype(int).isin(discovery),
        "discovery",
        "confirmation",
    )
    rows: list[dict[str, Any]] = []
    scopes = {
        "full_panel": discovery | confirmation,
        "discovery": discovery,
        "confirmation": confirmation,
    }
    for scope_name, seeds in scopes.items():
        scoped = registry.loc[
            pd.to_numeric(registry["seed"], errors="raise").astype(int).isin(seeds)
        ]
        for count_label in ("all", *counts):
            count_registry = (
                scoped
                if count_label == "all"
                else scoped.loc[
                    pd.to_numeric(scoped["gold_count"], errors="raise")
                    .astype(int)
                    .eq(int(count_label))
                ]
            )
            prompt_units = len(seeds) * (len(counts) if count_label == "all" else 1)
            rows.append(
                {
                    "registered_bank_size": "shared_registry",
                    "evaluation_scope": scope_name,
                    "stage": "registered_prompt_panel",
                    "gold_count": count_label,
                    "n_prompt_units": int(prompt_units),
                    "n_eligible_anchor_units": int(len(count_registry)),
                    "n_ineligible_prompt_units": int(prompt_units - len(count_registry)),
                    "n_seeds": int(len(seeds)),
                    "n_clean_correct": np.nan,
                    "n_complete_five_arm_pairs": np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze grammar-routed native-thinking targeted-retrieval ablations "
            "with seed-equal paired selected-minus-random estimands."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=_parse_run,
        metavar="K=DIR",
        help="One frozen bank-size behavior output directory; repeat for a dose grid.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection-config",
        type=Path,
        help=(
            "Frozen full-panel selection JSON used to add the 300-prompt "
            "denominator flow and validate the shared anchor-registry hash."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write an explicitly provisional analysis from currently complete five-arm pairs.",
    )
    args = parser.parse_args()

    runs = dict(args.run)
    if len(runs) != len(args.run):
        raise ValueError("Each --run bank size must be unique")
    loaded: dict[int, tuple[pd.DataFrame, dict[str, Any]]] = {
        bank_size: _load_run(path, bank_size)
        for bank_size, path in sorted(runs.items())
    }
    clean_reference: pd.DataFrame | None = None
    clean_reference_dir: Path | None = None
    for bank_size, (frame, _) in loaded.items():
        candidate = frame.loc[frame["condition"].astype(str).eq("clean")]
        if not candidate.empty:
            clean_reference = candidate.copy()
            clean_reference_dir = runs[bank_size]
            break
    if clean_reference is None or clean_reference_dir is None:
        raise ValueError("At least one supplied run must contain the shared clean arm")
    reference_registry_sha256 = _sha256(
        clean_reference_dir / "selected_anchor_registry.jsonl"
    )
    registry_relations = {
        bank_size: _registry_relation(run_dir, clean_reference_dir)
        for bank_size, run_dir in runs.items()
    }

    anchor_frames: list[pd.DataFrame] = []
    trial_frames: list[pd.DataFrame] = []
    run_audits: list[dict[str, Any]] = []
    for bank_size, (frame, manifest) in loaded.items():
        anchors, audit = _anchor_table(
            frame,
            bank_size=bank_size,
            clean_reference=clean_reference,
            require_complete=not args.allow_incomplete,
        )
        expected_anchors = int(
            manifest.get(
                "scheduled_anchor_count",
                manifest.get("selected_anchor_count", -1),
            )
        )
        if (
            not args.allow_incomplete
            and audit["observed_anchor_units"] != expected_anchors
        ):
            raise ValueError(
                f"K={bank_size} has {audit['observed_anchor_units']} observed "
                f"anchors but the frozen manifest requires {expected_anchors}"
            )
        if args.allow_incomplete:
            anchors = anchors.loc[anchors["complete_five_arm_pair"]].copy()
        if anchors.empty:
            continue
        anchor_frames.append(anchors)
        trial_frame = frame.copy()
        trial_frame["registered_bank_size"] = int(bank_size)
        trial_frames.append(trial_frame)
        run_audits.append(
            {
                **audit,
                "run_dir": str(runs[bank_size].resolve()),
                "manifest_sha256": _sha256(runs[bank_size] / "manifest.json"),
                "plan_sha256": str(manifest.get("plan_sha256")),
                "anchor_registry_sha256": str(
                    manifest.get("selected_anchor_registry_sha256")
                ),
                "anchor_registry_relation_to_clean_reference": registry_relations[
                    bank_size
                ],
            }
        )
    if not anchor_frames:
        raise ValueError("No complete five-arm anchors are available")

    anchors = pd.concat(anchor_frames, ignore_index=True)
    trials = pd.concat(trial_frames, ignore_index=True)
    estimands, arm_rates, seed_effects, sample_flow, failure_modes = _analyze(
        anchors,
        trials,
        bootstrap_samples=args.bootstrap_samples,
    )
    count_estimands, count_arm_rates = _analyze_count_strata(
        anchors,
        bootstrap_samples=args.bootstrap_samples,
    )
    selection_payload: dict[str, Any] | None = None
    selection_sha256: str | None = None
    selection_contract_validated = False
    if args.selection_config is not None:
        selection_payload = json.loads(
            args.selection_config.read_text(encoding="utf-8")
        )
        selection_sha256 = _sha256(args.selection_config)
        selected_registry_sha = str(selection_payload["anchor_registry"]["sha256"])
        if selected_registry_sha != reference_registry_sha256:
            raise ValueError(
                "Selection config and behavior runs disagree on anchor registry: "
                f"{selected_registry_sha} != {reference_registry_sha256}"
            )
        registered_plans = selection_payload["development_selection"]["plans"]
        registered_route_sha = str(selection_payload["routing"]["sha256"])
        for bank_size, (run_frame, run_manifest) in loaded.items():
            registered_plan = registered_plans[str(bank_size)]
            if str(run_manifest.get("plan_sha256")) != str(
                registered_plan["sha256"]
            ):
                raise ValueError(
                    f"K={bank_size} run plan hash is not the registered plan"
                )
            if str(run_manifest.get("anchor_routing_sha256")) != registered_route_sha:
                raise ValueError(
                    f"K={bank_size} run routing hash is not the registered route"
                )
            selected_hashes = set(
                run_frame.loc[
                    run_frame["condition"].astype(str).eq("selected_bank"),
                    "bank_sha256",
                ].astype(str)
            )
            if selected_hashes and selected_hashes != {
                str(registered_plan["selected_bank_sha256"])
            }:
                raise ValueError(
                    f"K={bank_size} selected trial bank hash is not registered: "
                    f"{selected_hashes}"
                )
        selection_contract_validated = True
        panel_flow = _registered_panel_flow(clean_reference_dir, selection_payload)
        sample_flow = pd.concat([panel_flow, sample_flow], ignore_index=True)
    output = args.output
    _atomic_csv(output / "anchor_level.csv", anchors)
    _atomic_csv(output / "seed_effects.csv", seed_effects)
    _atomic_csv(output / "estimands.csv", estimands)
    _atomic_csv(output / "raw_arm_rates.csv", arm_rates)
    _atomic_csv(output / "sample_flow.csv", sample_flow)
    _atomic_csv(output / "failure_modes.csv", failure_modes)
    _atomic_csv(output / "count_estimands.csv", count_estimands)
    _atomic_csv(output / "count_raw_arm_rates.csv", count_arm_rates)
    analysis_manifest = {
        "schema_version": "realistic_niah_v5_targeted_retrieval_analysis_v1",
        "analysis_status": "provisional_incomplete" if args.allow_incomplete else "complete",
        "bootstrap_samples": int(args.bootstrap_samples),
        "unit_of_inference": "seed_after_prompt_anchor_pairing",
        "primary_contrast": "selected_failure_minus_mean_of_three_layer_matched_random_failures",
        "population_order": ["all_examples", "clean_correct_only"],
        "scope_order": ["confirmation", "full_panel", "discovery"],
        "registered_confirmation_seeds": list(range(1254, 1264)),
        "primary_grammars": list(PRIMARY_GRAMMARS),
        "random_repeats": list(EXPECTED_RANDOM_REPEATS),
        "clean_reference_anchor_registry_sha256": reference_registry_sha256,
        "secondary_registry_policy": (
            "Each confirmation-only dose registry must be an exact row-wise "
            "subset of the K125 clean-reference registry."
        ),
        "selection_config": (
            str(args.selection_config.resolve()) if args.selection_config else None
        ),
        "selection_config_sha256": selection_sha256,
        "selection_contract_validated": selection_contract_validated,
        "runs": run_audits,
        "outputs": {
            name: _sha256(output / name)
            for name in (
                "anchor_level.csv",
                "seed_effects.csv",
                "estimands.csv",
                "raw_arm_rates.csv",
                "sample_flow.csv",
                "failure_modes.csv",
                "count_estimands.csv",
                "count_raw_arm_rates.csv",
            )
        },
    }
    _atomic_text(
        output / "analysis_manifest.json",
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(analysis_manifest, indent=2))


if __name__ == "__main__":
    main()
