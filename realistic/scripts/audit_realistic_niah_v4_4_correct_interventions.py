from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from realistic_niah_v4.correct_interventions import (
    ABLATION_STIMULUS_KEY_COLUMNS,
    ABLATION_TOP_NS,
    PARALLEL_ASSIGNMENT_METHOD,
    PARALLEL_WORKER_COUNT,
    PATCH_PAIR_KEY_COLUMNS,
    clean_correct_baselines,
    eligible_directed_pairs,
    parallel_plan_records,
    summarize_ablation_n_diagnostics,
    summarize_ablation_population,
    summarize_average_patching_accuracy,
)
from realistic_niah_v4.correct_only_slices import (
    clean_correct_ablation_rows,
    clean_correct_patching_rows,
)


EXACT_PATCH_GROUPS = (
    "model_label",
    "family",
    "site",
    "patch_protocol",
    "start_layer",
    "k",
    "target_direction",
)
AGGREGATE_PATCH_GROUPS = (
    "model_label",
    "family",
    "k",
    "target_direction",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _one_stage(run_root: Path, model: str) -> Path:
    parent = run_root / model / "numeric" / "correct_interventions"
    complete = sorted(parent.glob("confirmation_*/complete.json"))
    if len(complete) != 1:
        raise RuntimeError(
            f"Expected exactly one completed correct-intervention stage for {model}; "
            f"found {len(complete)}"
        )
    return complete[0].parent


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(
        self,
        *,
        model: str,
        category: str,
        name: str,
        condition: bool | Callable[[], bool],
        detail: str,
    ) -> None:
        try:
            passed = bool(condition() if callable(condition) else condition)
            error = "" if passed else detail
        except Exception as exc:  # audit must report every failed check
            passed = False
            error = f"{type(exc).__name__}: {exc}"
        self.rows.append(
            {
                "model_label": model,
                "category": category,
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "detail": error,
            }
        )


def _same_summary(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    groups: tuple[str, ...],
) -> bool:
    columns = [
        *groups,
        "pair_instances",
        "seed_clusters",
        "average_patching_acc",
        "average_post_patch_receiver_acc",
        "patching_acc_successes",
        "patching_acc_denominator",
    ]
    left = observed[columns].sort_values(list(groups)).reset_index(drop=True)
    right = expected[columns].sort_values(list(groups)).reset_index(drop=True)
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    for column in groups:
        if left[column].astype(str).tolist() != right[column].astype(str).tolist():
            return False
    numeric = [column for column in columns if column not in groups]
    return bool(
        np.allclose(
            left[numeric].to_numpy(dtype=float),
            right[numeric].to_numpy(dtype=float),
            equal_nan=True,
        )
    )


def _same_numeric_table(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    groups: tuple[str, ...],
    numeric: tuple[str, ...],
) -> bool:
    columns = [*groups, *numeric]
    left = observed[columns].sort_values(list(groups)).reset_index(drop=True)
    right = expected[columns].sort_values(list(groups)).reset_index(drop=True)
    if len(left) != len(right):
        return False
    for column in groups:
        if left[column].astype(str).tolist() != right[column].astype(str).tolist():
            return False
    return bool(
        np.allclose(
            left[list(numeric)].to_numpy(dtype=float),
            right[list(numeric)].to_numpy(dtype=float),
            equal_nan=True,
        )
    )


def _row_keys(frame: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    selected = frame[columns].copy()
    return {
        tuple("<NA>" if pd.isna(value) else str(value) for value in row)
        for row in selected.itertuples(index=False, name=None)
    }


def _same_detail_table(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: list[str],
) -> bool:
    """Compare a canonical merge with its worker union, ignoring row order."""

    if set(observed.columns) != set(expected.columns) or len(observed) != len(expected):
        return False
    columns = sorted(observed.columns)
    left = observed[columns].sort_values(keys, kind="stable").reset_index(drop=True)
    right = expected[columns].sort_values(keys, kind="stable").reset_index(drop=True)
    for column in columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(
            right[column]
        ):
            if not np.allclose(
                pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float),
                equal_nan=True,
            ):
                return False
        else:
            left_values = left[column].fillna("<NA>").astype(str).tolist()
            right_values = right[column].fillna("<NA>").astype(str).tolist()
            if left_values != right_values:
                return False
    return True


def _audit_model(
    audit: Audit,
    *,
    run_root: Path,
    model: str,
    definition: dict[str, Any],
) -> tuple[Path, list[dict[str, Any]]]:
    stage = _one_stage(run_root, model)
    completion = json.loads((stage / "complete.json").read_text(encoding="utf-8"))
    design = json.loads((stage / "design.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (stage / "supplement_selection.json").read_text(encoding="utf-8")
    )
    patch_target = int(
        definition["patching"]["minimum_seed_clusters_per_model_k_direction"]
    )
    ablation_target = int(
        definition["ablation"]["minimum_fresh_correct_seed_clusters_per_model"]
    )
    ablation_definition = definition["ablation"]
    ablation_counts = tuple(int(value) for value in ablation_definition["counts"])
    head_bank = str(ablation_definition["head_bank"])
    top_ns = tuple(int(value) for value in ablation_definition["top_n_candidates"])
    parallel_design = design.get("parallel_execution")
    parallel_plan: dict[str, Any] | None = None
    parallel_worker_details: dict[str, list[pd.DataFrame]] = {
        "prompt_patching": [],
        "answer_patching": [],
        "ablation": [],
    }
    if parallel_design is not None:
        plan_path = stage / "parallel_work_plan.json"
        prepare_path = stage / "prepare.complete.json"
        parallel_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
        plan_hash = _sha256(plan_path)
        prepare_hash = _sha256(prepare_path)
        worker_roots = sorted((stage / "workers").glob("worker_*_of_*"))
        audit.check(
            model=model,
            category="parallel",
            name="formal execution uses four isolated workers per model",
            condition=int(parallel_design.get("worker_count_per_model", 0))
            == PARALLEL_WORKER_COUNT
            and int(parallel_plan.get("worker_count", 0)) == PARALLEL_WORKER_COUNT
            and parallel_plan.get("assignment_method")
            == PARALLEL_ASSIGNMENT_METHOD
            and len(worker_roots) == PARALLEL_WORKER_COUNT,
            detail="parallel design, plan, or worker directory count differs from 4",
        )
        audit.check(
            model=model,
            category="parallel",
            name="prepare gate and merge completion share design and plan hashes",
            condition=prepare.get("status") == "complete"
            and prepare.get("design_hash") == stage.name.removeprefix("confirmation_")
            and parallel_plan.get("design_hash")
            == stage.name.removeprefix("confirmation_")
            and completion.get("parallel_execution", {}).get(
                "parallel_work_plan_sha256"
            )
            == plan_hash,
            detail="prepare, work plan, and merged completion provenance differ",
        )
        worker_markers_valid = True
        worker_designs_valid = True
        worker_runtimes_valid = True
        worker_outputs_valid = True
        for index in range(PARALLEL_WORKER_COUNT):
            root = stage / "workers" / f"worker_{index:03d}_of_004"
            marker_path = root / "complete.json"
            worker_design_path = root / "design.json"
            runtime_path = root / "runtime.json"
            if not (
                marker_path.is_file()
                and worker_design_path.is_file()
                and runtime_path.is_file()
            ):
                worker_markers_valid = False
                worker_designs_valid = False
                worker_runtimes_valid = False
                worker_outputs_valid = False
                continue
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            worker_design = json.loads(
                worker_design_path.read_text(encoding="utf-8")
            )
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            expected_pairs = parallel_plan_records(
                parallel_plan, work_kind="patch_pairs", worker_index=index
            )
            expected_ablation = parallel_plan_records(
                parallel_plan, work_kind="ablation_stimuli", worker_index=index
            )
            worker_markers_valid = worker_markers_valid and (
                marker.get("status") == "complete"
                and marker.get("design_hash")
                == stage.name.removeprefix("confirmation_")
                and marker.get("parallel_work_plan_sha256") == plan_hash
                and marker.get("prepare_sha256") == prepare_hash
                and int(marker.get("worker_index", -1)) == index
                and int(marker.get("worker_count", 0)) == PARALLEL_WORKER_COUNT
            )
            worker_designs_valid = worker_designs_valid and (
                worker_design.get("parallel_work_plan_sha256") == plan_hash
                and worker_design.get("prepare_sha256") == prepare_hash
                and int(worker_design.get("worker_index", -1)) == index
                and worker_design.get("assigned_patch_pairs") == expected_pairs
                and worker_design.get("assigned_ablation_stimuli")
                == expected_ablation
            )
            worker_runtimes_valid = worker_runtimes_valid and (
                bool(runtime.get("cuda_available"))
                and len(runtime.get("cuda_devices", [])) == 1
            )
            paths = {
                "prompt_patching": root
                / "patching"
                / "prompt_patching"
                / "detail.supplement.csv.gz",
                "answer_patching": root
                / "patching"
                / "answer_patching"
                / "detail.supplement.csv.gz",
                "ablation": root
                / "ablation"
                / "detail.all_examples.discovery.csv.gz",
            }
            marker_keys = {
                "prompt_patching": "prompt_detail",
                "answer_patching": "answer_detail",
                "ablation": "ablation_detail",
            }
            for family, path in paths.items():
                expected_hash = marker.get("output_sha256", {}).get(
                    marker_keys[family]
                )
                valid = path.is_file() and _sha256(path) == expected_hash
                worker_outputs_valid = worker_outputs_valid and valid
                if valid:
                    frame = pd.read_csv(path, compression="infer")
                    frame["parallel_worker_index"] = index
                    parallel_worker_details[family].append(frame)
        audit.check(
            model=model,
            category="parallel",
            name="all worker completion markers and assignments match the plan",
            condition=worker_markers_valid and worker_designs_valid,
            detail="a worker marker or assigned identity list differs from the plan",
        )
        audit.check(
            model=model,
            category="parallel",
            name="each GPU worker saw exactly one CUDA device",
            condition=worker_runtimes_valid,
            detail="at least one worker did not record exactly one visible GPU",
        )
        audit.check(
            model=model,
            category="parallel",
            name="all worker detail hashes are intact",
            condition=worker_outputs_valid
            and all(
                len(parallel_worker_details[family]) == PARALLEL_WORKER_COUNT
                for family in parallel_worker_details
            ),
            detail="a worker detail file is missing, modified, or unreadable",
        )
    audit.check(
        model=model,
        category="completion",
        name="completion marker",
        condition=completion.get("status") == "complete",
        detail="complete.json is not complete",
    )
    audit.check(
        model=model,
        category="completion",
        name="design hash agrees",
        condition=completion.get("design_hash")
        == stage.name.removeprefix("confirmation_"),
        detail="completion design hash differs from stage directory",
    )
    audit.check(
        model=model,
        category="selection",
        name="baseline-only stopping completed",
        condition=selection.get("selection_status") == "complete",
        detail="supplement reserve did not meet quotas",
    )
    scanned = [int(value) for value in selection["scanned_supplement_seeds"]]
    reserve = [int(value) for value in selection["reserve_seeds"]]
    expected_reserve = list(
        range(
            int(definition["reserve_seed_start"]),
            int(definition["reserve_seed_end_inclusive"]) + 1,
        )
    )
    unused = [int(value) for value in selection["unused_reserve_seeds"]]
    audit.check(
        model=model,
        category="selection",
        name="reserve seed catalog matches frozen definition",
        condition=reserve == expected_reserve,
        detail="selection manifest reserve differs from the frozen definition",
    )
    audit.check(
        model=model,
        category="selection",
        name="scanned seeds are an ordered prefix",
        condition=scanned == reserve[: len(scanned)],
        detail="scanned seeds are not the declared ascending prefix",
    )
    audit.check(
        model=model,
        category="selection",
        name="unused seeds are the untouched reserve suffix",
        condition=unused == reserve[len(scanned) :],
        detail="unused reserve suffix is incomplete or out of order",
    )
    final_support = selection["patching"]["final_support"]
    audit.check(
        model=model,
        category="selection",
        name="every k/direction reaches patch seed quota",
        condition=all(
            int(row["seed_clusters"]) >= patch_target for row in final_support
        ),
        detail="at least one k/direction remains below patch quota",
    )
    audit.check(
        model=model,
        category="selection",
        name="correct-only ablation reaches seed quota",
        condition=int(selection["correct_only_ablation"]["final_seed_clusters"])
        >= ablation_target,
        detail="correct-only ablation remains below seed quota",
    )
    shared_discovery_seeds = [
        int(value)
        for value in selection["correct_only_ablation"][
            "shared_discovery_seed_prefix"
        ]
    ]
    audit.check(
        model=model,
        category="selection",
        name="dual ablation populations use registered count-1--5 domain",
        condition=ablation_counts == (1, 2, 3, 4, 5)
        and tuple(int(value) for value in design.get("ablation_counts", []))
        == ablation_counts,
        detail="definition or stage design changed the shared ablation count domain",
    )
    audit.check(
        model=model,
        category="selection",
        name="shared ablation seeds form an ordered prefix",
        condition=bool(shared_discovery_seeds)
        and shared_discovery_seeds
        == reserve[: len(shared_discovery_seeds)]
        and set(shared_discovery_seeds).issubset(scanned),
        detail="dual-population discovery does not use the registered fresh-seed prefix",
    )
    labels = pd.read_csv(stage / "baseline_labels.scanned.csv")
    audit.check(
        model=model,
        category="baseline",
        name="baseline grid is complete for scanned seeds",
        condition=len(labels) == len(scanned) * 11
        and labels[["seed", "gold_count"]].drop_duplicates().shape[0]
        == len(scanned) * 11,
        detail="baseline seed/count grid is incomplete or duplicated",
    )
    added_pairs = pd.read_csv(stage / "selected_added_pairs.csv")
    eligible = eligible_directed_pairs(labels)
    pair_keys = ["model_label", "seed", "receiver_count", "donor_count"]
    checked_pairs = added_pairs.merge(
        eligible[pair_keys].drop_duplicates(),
        on=pair_keys,
        how="inner",
        validate="one_to_one",
    )
    audit.check(
        model=model,
        category="baseline",
        name="every added patch pair is clean-correct",
        condition=len(checked_pairs) == len(added_pairs),
        detail="an added pair lacks a correct receiver or donor baseline",
    )
    if parallel_plan is not None:
        planned_pair_keys = {
            tuple(str(record[column]) for column in PATCH_PAIR_KEY_COLUMNS)
            for record in parallel_plan_records(
                parallel_plan, work_kind="patch_pairs"
            )
        }
        selected_pair_keys = _row_keys(added_pairs, list(PATCH_PAIR_KEY_COLUMNS))
        audit.check(
            model=model,
            category="parallel",
            name="parallel patch plan is exhaustive and has no extra pairs",
            condition=planned_pair_keys == selected_pair_keys
            and len(planned_pair_keys) == len(added_pairs),
            detail="parallel patch assignments differ from baseline-gated pairs",
        )
    added_ablation = pd.read_csv(
        stage / "eligible_added_correct_ablation_baselines.csv"
    )
    clean_added = clean_correct_baselines(
        added_ablation, counts=definition["ablation"]["counts"]
    )
    audit.check(
        model=model,
        category="baseline",
        name="every added ablation stimulus is clean-correct",
        condition=len(clean_added) == len(added_ablation),
        detail="an added ablation stimulus is not clean-correct",
    )

    inventory: list[dict[str, Any]] = []
    for family in ("prompt_patching", "answer_patching"):
        root = stage / "patching" / family
        supplement = pd.read_csv(root / "detail.supplement.csv.gz")
        combined = pd.read_csv(root / "detail.clean_correct.combined.csv.gz")
        exact = pd.read_csv(root / "average_patching_acc.exact_groups.csv")
        aggregate = pd.read_csv(root / "average_patching_acc.aggregate_groups.csv")
        supplement_clean = clean_correct_patching_rows(supplement)
        combined_clean = clean_correct_patching_rows(combined)
        if parallel_plan is not None:
            worker_union = pd.concat(
                parallel_worker_details[family], ignore_index=True, sort=False
            )
            parallel_patch_row_keys = [
                *PATCH_PAIR_KEY_COLUMNS,
                "site",
                "patch_protocol",
                "start_layer",
                "condition",
            ]
            ownership = worker_union.groupby(list(PATCH_PAIR_KEY_COLUMNS))[
                "parallel_worker_index"
            ].nunique()
            audit.check(
                model=model,
                category="parallel",
                name=f"{family} worker rows are disjoint and equal the canonical merge",
                condition=ownership.eq(1).all()
                and len(worker_union) == len(supplement)
                and _row_keys(worker_union, parallel_patch_row_keys)
                == _row_keys(supplement, parallel_patch_row_keys)
                and _same_detail_table(
                    worker_union,
                    supplement,
                    keys=parallel_patch_row_keys,
                ),
                detail=(
                    f"{family} canonical detail omits, duplicates, or changes a "
                    "worker intervention row"
                ),
            )
        audit.check(
            model=model,
            category=family,
            name="supplement contains only clean-correct patch rows",
            condition=len(supplement_clean) == len(supplement),
            detail="supplement contains an incorrect receiver/donor/state source",
        )
        audit.check(
            model=model,
            category=family,
            name="combined detail remains clean-correct",
            condition=len(combined_clean) == len(combined),
            detail="combined patch detail contains an ineligible row",
        )
        audit.check(
            model=model,
            category=family,
            name="no patch intervention was skipped",
            condition=supplement["status"].astype(str).eq("ok").all(),
            detail="supplement includes skipped intervention rows",
        )
        expected_exact = summarize_average_patching_accuracy(
            combined,
            group_columns=EXACT_PATCH_GROUPS,
            bootstrap_repetitions=200,
        )
        expected_aggregate = summarize_average_patching_accuracy(
            combined,
            group_columns=AGGREGATE_PATCH_GROUPS,
            bootstrap_repetitions=200,
        )
        audit.check(
            model=model,
            category=family,
            name="every exact group has traceable average patching accuracy",
            condition=lambda: _same_summary(
                exact, expected_exact, groups=EXACT_PATCH_GROUPS
            ),
            detail="exact-group patching accuracy does not reproduce from detail",
        )
        audit.check(
            model=model,
            category=family,
            name="every aggregate group has traceable average patching accuracy",
            condition=lambda: _same_summary(
                aggregate, expected_aggregate, groups=AGGREGATE_PATCH_GROUPS
            ),
            detail="aggregate patching accuracy does not reproduce from detail",
        )
        audit.check(
            model=model,
            category=family,
            name="patching accuracy has explicit denominator",
            condition=(
                exact["patching_acc_denominator"].astype(int)
                == exact["pair_instances"].astype(int)
            ).all()
            and exact["patching_acc_definition"]
            .astype(str)
            .eq("patched_count_equals_donor_gold_count")
            .all(),
            detail="patching accuracy denominator or definition is missing",
        )

    ablation_root = stage / "ablation"
    all_detail = pd.read_csv(ablation_root / "detail.all_examples.discovery.csv.gz")
    new_correct = pd.read_csv(ablation_root / "detail.clean_correct.discovery.csv.gz")
    dual = pd.read_csv(ablation_root / "dual_population_ablation_summary.csv")
    diagnostics = pd.read_csv(ablation_root / "top_n_diagnostics.unfrozen.csv")
    all_stimuli = all_detail[
        ["model_label", "stimulus_id", "seed", "gold_count"]
    ].drop_duplicates()
    expected_all_stimuli = len(shared_discovery_seeds) * len(ablation_counts)
    if parallel_plan is not None:
        planned_ablation_keys = {
            tuple(str(record[column]) for column in ABLATION_STIMULUS_KEY_COLUMNS)
            for record in parallel_plan_records(
                parallel_plan, work_kind="ablation_stimuli"
            )
        }
        observed_ablation_keys = _row_keys(
            all_stimuli, list(ABLATION_STIMULUS_KEY_COLUMNS)
        )
        worker_union = pd.concat(
            parallel_worker_details["ablation"], ignore_index=True, sort=False
        )
        parallel_ablation_row_keys = [
            "stimulus_id",
            "head_bank",
            "top_n",
            "condition",
            "random_replicate",
        ]
        ownership = worker_union.groupby(list(ABLATION_STIMULUS_KEY_COLUMNS))[
            "parallel_worker_index"
        ].nunique()
        audit.check(
            model=model,
            category="parallel",
            name="parallel ablation plan exactly covers the shared stimulus grid",
            condition=planned_ablation_keys == observed_ablation_keys
            and len(planned_ablation_keys) == expected_all_stimuli,
            detail="parallel ablation assignments differ from the canonical grid",
        )
        audit.check(
            model=model,
            category="parallel",
            name="ablation worker rows are disjoint and equal the canonical merge",
            condition=ownership.eq(1).all()
            and len(worker_union) == len(all_detail)
            and _row_keys(worker_union, parallel_ablation_row_keys)
            == _row_keys(all_detail, parallel_ablation_row_keys)
            and _same_detail_table(
                worker_union,
                all_detail,
                keys=parallel_ablation_row_keys,
            ),
            detail="canonical ablation omits, duplicates, or changes a worker row",
        )
    audit.check(
        model=model,
        category="ablation",
        name="all-example discovery is the complete shared count-1--5 grid",
        condition=len(all_stimuli) == expected_all_stimuli
        and set(pd.to_numeric(all_stimuli["seed"]).astype(int))
        == set(shared_discovery_seeds)
        and set(pd.to_numeric(all_stimuli["gold_count"]).astype(int))
        == set(ablation_counts)
        and int(
            selection["correct_only_ablation"][
                "shared_discovery_expected_all_example_stimuli"
            ]
        )
        == expected_all_stimuli,
        detail="all-example discovery changed seeds, counts, or grid completeness",
    )
    audit.check(
        model=model,
        category="ablation",
        name="new ablation rows are clean-correct only",
        condition=len(clean_correct_ablation_rows(new_correct)) == len(new_correct),
        detail="new correct-only ablation contains an incorrect clean baseline",
    )
    filtered_correct = clean_correct_ablation_rows(all_detail)
    row_key_columns = ["stimulus_id", "top_n", "condition"]
    if "random_replicate" in all_detail.columns:
        row_key_columns.append("random_replicate")

    def ablation_row_keys(frame: pd.DataFrame) -> set[tuple[str, ...]]:
        selected = frame[row_key_columns].copy()
        return {
            tuple("<NA>" if pd.isna(value) else str(value) for value in row)
            for row in selected.itertuples(index=False, name=None)
        }

    audit.check(
        model=model,
        category="ablation",
        name="correct-only detail is the exact subset of all-example detail",
        condition=len(filtered_correct) == len(new_correct)
        and ablation_row_keys(filtered_correct) == ablation_row_keys(new_correct),
        detail="correct-only rows were rerun, omitted, or drawn from another seed pool",
    )
    audit.check(
        model=model,
        category="ablation",
        name="fresh correct-only discovery reaches seed quota",
        condition=int(new_correct["seed"].nunique()) >= ablation_target,
        detail="fresh correct-only discovery has too few seed clusters",
    )
    selected_ablation_seeds = {
        int(value)
        for value in selection["correct_only_ablation"]["added_eligible_seeds"]
    }
    audit.check(
        model=model,
        category="ablation",
        name="clean-correct discovery uses only the registered earliest seeds",
        condition=set(pd.to_numeric(new_correct["seed"]).astype(int))
        == selected_ablation_seeds
        and len(selected_ablation_seeds) == ablation_target,
        detail="clean-correct discovery includes an unselected or excess seed",
    )
    audit.check(
        model=model,
        category="ablation",
        name="top-n remains explicitly unfrozen",
        condition=ablation_definition.get("selection_status")
        == "unfrozen_discovery_only"
        and design.get("top_n_selection_status") == "unfrozen_discovery_only"
        and tuple(int(value) for value in design.get("ablation_top_n_candidates", []))
        == top_ns
        and diagnostics["selection_status"]
        .astype(str)
        .eq("discovery_only_unfrozen")
        .all(),
        detail="a top-n was frozen or the discovery status is ambiguous",
    )
    audit.check(
        model=model,
        category="ablation",
        name="clean-correct discovery sweeps every candidate top-n",
        condition=top_ns == ABLATION_TOP_NS
        and set(new_correct["head_bank"].astype(str)) == {head_bank}
        and set(pd.to_numeric(new_correct["top_n"]).astype(int)) == set(top_ns),
        detail="clean-correct discovery changed the head bank or omitted a top-n",
    )
    per_stimulus_n = new_correct.groupby(["stimulus_id", "top_n"]).size()
    expected_rows = 1 + int(ablation_definition["random_replicates"])
    audit.check(
        model=model,
        category="ablation",
        name="ranked and random controls are complete for every stimulus/top-n",
        condition=per_stimulus_n.eq(expected_rows).all(),
        detail="at least one correct stimulus/top-n lacks ranked/random rows",
    )
    all_per_stimulus_n = all_detail.groupby(["stimulus_id", "top_n"]).size()
    audit.check(
        model=model,
        category="ablation",
        name="all-example discovery has the same complete 1--32 dose grid",
        condition=set(all_detail["head_bank"].astype(str)) == {head_bank}
        and set(pd.to_numeric(all_detail["top_n"]).astype(int)) == set(top_ns)
        and all_per_stimulus_n.eq(expected_rows).all(),
        detail="all-example discovery omits a dose or ranked/random control",
    )
    expected_all = summarize_ablation_population(
        all_detail, population="all_examples_signed", bootstrap_repetitions=200
    )
    expected_correct = summarize_ablation_population(
        new_correct,
        population="clean_correct_only",
        bootstrap_repetitions=200,
    )
    observed_all = dual[dual["analysis_population"].eq("all_examples_signed")]
    observed_correct = dual[dual["analysis_population"].eq("clean_correct_only")]
    key_metrics = [
        "model_label",
        "head_bank",
        "top_n",
        "examples",
        "seed_clusters",
        "baseline_accuracy",
        "post_ablation_accuracy",
        "mean_accuracy_delta",
        "mean_signed_count_shift_valid",
        "mean_absolute_count_shift_valid",
        "mean_absolute_error_delta",
        "prediction_changed_rate",
        "correct_to_wrong_rate",
    ]

    def same_ablation(observed: pd.DataFrame, expected: pd.DataFrame) -> bool:
        left = (
            observed[key_metrics]
            .sort_values(["model_label", "head_bank", "top_n"])
            .reset_index(drop=True)
        )
        right = (
            expected[key_metrics]
            .sort_values(["model_label", "head_bank", "top_n"])
            .reset_index(drop=True)
        )
        return (
            len(left) == len(right)
            and bool(
                np.allclose(
                    left.drop(columns=["model_label", "head_bank"]).to_numpy(
                        dtype=float
                    ),
                    right.drop(columns=["model_label", "head_bank"]).to_numpy(
                        dtype=float
                    ),
                    equal_nan=True,
                )
            )
            and left[["model_label", "head_bank"]]
            .astype(str)
            .equals(right[["model_label", "head_bank"]].astype(str))
        )

    audit.check(
        model=model,
        category="ablation",
        name="all-example signed summary reproduces",
        condition=lambda: same_ablation(observed_all, expected_all),
        detail="all-example signed summary differs from raw detail",
    )
    audit.check(
        model=model,
        category="ablation",
        name="correct-only failure summary reproduces",
        condition=lambda: same_ablation(observed_correct, expected_correct),
        detail="correct-only summary differs from raw detail",
    )
    expected_all_diagnostics = summarize_ablation_n_diagnostics(
        all_detail,
        population="all_examples_signed",
        head_bank=head_bank,
        bootstrap_repetitions=200,
    )
    expected_correct_diagnostics = summarize_ablation_n_diagnostics(
        new_correct,
        population="clean_correct_only",
        head_bank=head_bank,
        bootstrap_repetitions=200,
    )
    expected_diagnostics = pd.concat(
        [expected_all_diagnostics, expected_correct_diagnostics],
        ignore_index=True,
        sort=False,
    )
    diagnostic_groups = [
        "model_label",
        "head_bank",
        "top_n",
        "analysis_population",
    ]
    diagnostic_numeric = [
        "examples",
        "seed_clusters",
        "ranked_mean_signed_count_shift",
        "random_mean_signed_count_shift",
        "ranked_minus_random_signed_count_shift",
        "ranked_mean_absolute_count_shift",
        "random_mean_absolute_count_shift",
        "ranked_minus_random_absolute_count_shift",
        "ranked_correct_to_wrong_rate",
        "random_correct_to_wrong_rate",
        "ranked_minus_random_correct_to_wrong",
        "ranked_minus_random_accuracy_delta",
        "ranked_minus_random_absolute_error_delta",
        "ranked_minus_random_prediction_changed",
        "ranked_valid_rate",
        "random_valid_rate",
        "primary_effect",
        "primary_rank_within_model_bank",
    ]
    audit.check(
        model=model,
        category="ablation",
        name="population-specific top-n diagnostics reproduce",
        condition=lambda: _same_numeric_table(
            diagnostics,
            expected_diagnostics,
            groups=tuple(diagnostic_groups),
            numeric=tuple(diagnostic_numeric),
        ),
        detail="top-n diagnostic endpoints differ from raw detail",
    )
    audit.check(
        model=model,
        category="ablation",
        name="two ablation populations are explicitly separated",
        condition=set(dual["analysis_population"].astype(str))
        == {"all_examples_signed", "clean_correct_only"},
        detail="dual summary lacks one of the two registered populations",
    )

    for path in sorted(stage.rglob("*")):
        if path.is_file():
            inventory.append(
                {
                    "model_label": model,
                    "path": str(path.relative_to(run_root)).replace("\\", "/"),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    audit.check(
        model=model,
        category="provenance",
        name="input source hashes are frozen in design",
        condition=isinstance(design.get("input_sha256"), dict)
        and len(design["input_sha256"]) == 11
        and all(len(str(value)) == 64 for value in design["input_sha256"].values()),
        detail="design lacks complete SHA-256 input provenance",
    )
    return stage, inventory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict audit for V4.4 correct patching and dual ablation."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--definition",
        default="configs/realistic_niah_v4_4_correct_interventions.json",
    )
    parser.add_argument("--models", nargs="+", default=["Qwen3-8B", "Gemma4-E4B"])
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    definition_path = Path(args.definition).resolve()
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    if definition.get("schema_version") != (
        "realistic_niah_v4_4_correct_interventions_v2"
    ):
        raise ValueError("Unexpected correct-intervention definition schema")
    audit = Audit()
    inventory: list[dict[str, Any]] = []
    stages: dict[str, str] = {}
    for model in args.models:
        stage, rows = _audit_model(
            audit,
            run_root=run_root,
            model=str(model),
            definition=definition,
        )
        stages[str(model)] = str(stage)
        inventory.extend(rows)
    audit_root = run_root / "audit" / "correct_interventions"
    checks = pd.DataFrame(audit.rows)
    _write_csv(checks, audit_root / "checks.csv")
    _write_csv(pd.DataFrame(inventory), audit_root / "file_inventory.csv")
    errors = checks[checks["status"].astype(str).eq("FAIL")]
    payload = {
        "schema_version": "realistic_niah_v4_4_correct_intervention_audit_v2",
        "status": "PASS" if errors.empty else "FAIL",
        "checks": int(len(checks)),
        "passed": int(checks["status"].eq("PASS").sum()),
        "errors": int(len(errors)),
        "models": [str(value) for value in args.models],
        "stages": stages,
        "definition_sha256": _sha256(definition_path),
    }
    _write_json(audit_root / "audit.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if not errors.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
