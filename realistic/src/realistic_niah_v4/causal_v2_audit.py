from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .causal_generation import causal_v2_monotonic_span_source_indices
from .causal_v2 import CausalV2Design, normalized_transport_metrics


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False}
    result = series.astype(str).str.lower().map(mapping)
    if result.isna().any():
        raise ValueError(
            f"Cannot parse boolean values: {series[result.isna()].unique()}"
        )
    return result.astype(bool)


@dataclass
class AuditCollector:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(
            {"name": str(name), "passed": bool(passed), "detail": str(detail)}
        )

    def require(self, name: str, condition: bool, detail: str) -> None:
        self.record(name, bool(condition), detail)

    @property
    def errors(self) -> list[dict[str, Any]]:
        return [item for item in self.checks if not item["passed"]]

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": "realistic_niah_v4_causal_v2_audit_v1",
            "status": "PASS" if not self.errors else "FAIL",
            "checks": self.checks,
            "check_count": len(self.checks),
            "error_count": len(self.errors),
        }


def _completed_formal_root(
    causal_root: Path,
    *,
    family: str,
    phase: str,
    required: bool,
    audit: AuditCollector,
) -> Path | None:
    parent = causal_root / family
    candidates: list[Path] = []
    if parent.is_dir():
        for root in sorted(parent.glob(f"{phase}_*")):
            design_path = root / "design.json"
            complete_path = root / "complete.json"
            if not design_path.is_file() or not complete_path.is_file():
                continue
            design = _read_json(design_path)
            complete = _read_json(complete_path)
            if (
                design.get("profile") == "formal"
                and complete.get("status") == "complete"
            ):
                candidates.append(root)
    audit.require(
        f"{family}.{phase}.unique_completed_formal_root",
        len(candidates) == (1 if required else min(1, len(candidates))),
        f"found={len(candidates)} roots={[str(path) for path in candidates]}",
    )
    if not candidates:
        return None
    if len(candidates) != 1:
        return None
    root = candidates[0]
    design = _read_json(root / "design.json")
    complete = _read_json(root / "complete.json")
    expected_hash = _json_hash(design)
    audit.require(
        f"{family}.{phase}.directory_hash",
        root.name == f"{phase}_{expected_hash}",
        f"directory={root.name}, expected={phase}_{expected_hash}",
    )
    audit.require(
        f"{family}.{phase}.complete_hash",
        str(complete.get("design_hash")) == expected_hash,
        f"complete={complete.get('design_hash')}, expected={expected_hash}",
    )
    audit.require(
        f"{family}.{phase}.strict_numeric_metric",
        design.get("behavior_metric") == "strict_greedy_complete_numeric_generation",
        str(design.get("behavior_metric")),
    )
    audit.require(
        f"{family}.{phase}.v4_4_only",
        design.get("design_variant") == "v4.4",
        str(design.get("design_variant")),
    )
    audit.require(
        f"{family}.{phase}.implementation_fingerprint",
        isinstance(design.get("implementation_sha256"), str)
        and len(design["implementation_sha256"]) == 64,
        str(design.get("implementation_sha256")),
    )
    return root


def _audit_shards(
    root: Path, audit: AuditCollector, label: str
) -> list[dict[str, Any]]:
    index_path = root / "capture" / "capture_index.jsonl"
    audit.require(
        f"{label}.capture_index_exists", index_path.is_file(), str(index_path)
    )
    if not index_path.is_file():
        return []
    rows = _read_jsonl(index_path)
    duplicate_paths = len({str(row.get("shard_path")) for row in rows}) != len(rows)
    audit.require(
        f"{label}.unique_shard_paths", not duplicate_paths, f"rows={len(rows)}"
    )
    mismatches: list[str] = []
    for row in rows:
        path = root / "capture" / str(row["shard_path"])
        if not path.is_file():
            mismatches.append(f"missing:{row['shard_path']}")
        elif _sha256_file(path) != str(row.get("sha256")):
            mismatches.append(f"sha256:{row['shard_path']}")
    audit.require(
        f"{label}.shard_checksums", not mismatches, f"mismatches={mismatches[:10]}"
    )
    return rows


def audit_stimulus_grid(
    stimuli_path: str | Path,
    *,
    design: CausalV2Design,
    audit: AuditCollector,
) -> None:
    rows = _read_jsonl(Path(stimuli_path))
    keys = [
        (str(row["design_variant"]), int(row["seed"]), int(row["gold_count"]))
        for row in rows
    ]
    expected_seeds = (
        set(design.centroid_fit_seeds)
        | set(design.screen_seeds)
        | set(design.confirmation_seeds)
    )
    audit.require(
        "stimuli.exact_row_count",
        len(rows) == len(expected_seeds) * len(design.valid_counts),
        f"rows={len(rows)}, expected={len(expected_seeds) * len(design.valid_counts)}",
    )
    audit.require(
        "stimuli.unique_cells", len(set(keys)) == len(keys), f"rows={len(keys)}"
    )
    audit.require(
        "stimuli.v4_4_only",
        {variant for variant, _seed, _count in keys} == {"v4.4"},
        str(sorted({variant for variant, _seed, _count in keys})),
    )
    audit.require(
        "stimuli.seed_grid",
        {seed for _variant, seed, _count in keys} == expected_seeds,
        str(sorted({seed for _variant, seed, _count in keys})),
    )
    by_seed = Counter(seed for _variant, seed, _count in keys)
    complete = all(by_seed[seed] == len(design.valid_counts) for seed in expected_seeds)
    audit.require(
        "stimuli.complete_count_grid", complete, str(dict(sorted(by_seed.items())))
    )


def _audit_baseline(root: Path, design: CausalV2Design, audit: AuditCollector) -> None:
    path = root / "generation_labels.csv"
    audit.require("baseline.labels_exists", path.is_file(), str(path))
    if not path.is_file():
        return
    frame = pd.read_csv(path)
    expected_seeds = (
        set(design.centroid_fit_seeds)
        | set(design.screen_seeds)
        | set(design.confirmation_seeds)
    )
    audit.require(
        "baseline.rows",
        len(frame) == len(expected_seeds) * len(design.valid_counts),
        f"rows={len(frame)}",
    )
    audit.require(
        "baseline.no_duplicate_stimuli",
        not frame["stimulus_id"].duplicated().any(),
        f"duplicates={int(frame['stimulus_id'].duplicated().sum())}",
    )
    audit.require(
        "baseline.counts_0_to_10",
        set(pd.to_numeric(frame["gold_count"]).astype(int)) == set(design.valid_counts),
        str(sorted(pd.to_numeric(frame["gold_count"]).astype(int).unique())),
    )
    audit.require(
        "baseline.seeds",
        set(pd.to_numeric(frame["seed"]).astype(int)) == expected_seeds,
        str(sorted(pd.to_numeric(frame["seed"]).astype(int).unique())),
    )


def _audit_full_span_mapping_table(
    frame: pd.DataFrame,
    design: CausalV2Design,
    audit: AuditCollector,
    label: str,
) -> None:
    required_columns = {
        "seed",
        "receiver_count",
        "donor_count",
        "k",
        "slot_index",
        "receiver_model_token_length",
        "donor_model_token_length",
        "absolute_model_token_length_delta",
        "exact_model_token_alignment",
        "alignment_policy",
        "mapping_supported",
        "source_index_map",
        "mapped_receiver_token_count",
        "distinct_source_token_count",
        "source_tokens_dropped",
        "source_tokens_reused",
        "max_normalized_position_error",
    }
    missing = sorted(required_columns - set(frame.columns))
    audit.require(
        f"{label}.schema",
        not missing,
        f"missing={missing}",
    )
    if missing:
        return
    policies = set(frame["alignment_policy"].astype(str))
    audit.require(
        f"{label}.registered_policy",
        policies == {design.prompt_full_span_alignment},
        str(sorted(policies)),
    )
    try:
        supported = _as_bool(frame["mapping_supported"])
    except ValueError as error:
        audit.require(f"{label}.all_changed_slots_supported", False, str(error))
        supported = pd.Series(False, index=frame.index)
    else:
        audit.require(
            f"{label}.all_changed_slots_supported",
            bool(supported.all()),
            f"unsupported={int((~supported).sum())}",
        )
    try:
        exact = _as_bool(frame["exact_model_token_alignment"])
    except ValueError as error:
        exact_parse_error = str(error)
        exact = pd.Series(False, index=frame.index)
    else:
        exact_parse_error = ""
    diagnostic_mismatches: list[int] = []
    exact_mismatches: list[int] = []
    for index, row in frame.iterrows():
        try:
            receiver_length = int(row["receiver_model_token_length"])
            source_length = int(row["donor_model_token_length"])
            source_indices = causal_v2_monotonic_span_source_indices(
                receiver_length,
                source_length,
            )
            distinct = len(set(source_indices))
            if receiver_length == 1 or source_length == 1:
                expected_error = 0.0
            else:
                expected_error = max(
                    abs(
                        source_index / (source_length - 1)
                        - receiver_index / (receiver_length - 1)
                    )
                    for receiver_index, source_index in enumerate(source_indices)
                )
            observed_map = tuple(
                int(value) for value in json.loads(str(row["source_index_map"]))
            )
            diagnostics_match = (
                observed_map == source_indices
                and int(row["absolute_model_token_length_delta"])
                == abs(receiver_length - source_length)
                and int(row["mapped_receiver_token_count"]) == receiver_length
                and int(row["distinct_source_token_count"]) == distinct
                and int(row["source_tokens_dropped"]) == source_length - distinct
                and int(row["source_tokens_reused"]) == receiver_length - distinct
                and math.isclose(
                    float(row["max_normalized_position_error"]),
                    float(expected_error),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            diagnostics_match = False
            receiver_length = -1
            source_length = -2
        if not diagnostics_match:
            diagnostic_mismatches.append(int(index))
        if bool(exact.loc[index]) != bool(receiver_length == source_length):
            exact_mismatches.append(int(index))
    audit.require(
        f"{label}.mapping_diagnostics_recomputed",
        not diagnostic_mismatches,
        f"mismatch_rows={diagnostic_mismatches[:10]}",
    )
    audit.require(
        f"{label}.exact_flag_recomputed",
        not exact_parse_error and not exact_mismatches,
        exact_parse_error or f"mismatch_rows={exact_mismatches[:10]}",
    )


def _audit_prompt_alignment(
    root: Path, design: CausalV2Design, audit: AuditCollector
) -> None:
    path = root / "prompt_full_span_alignment.csv"
    audit.require("prompt_alignment.table_exists", path.is_file(), str(path))
    if not path.is_file():
        return
    frame = pd.read_csv(path)
    _audit_full_span_mapping_table(frame, design, audit, "prompt_alignment")
    grid_columns = {"seed", "receiver_count", "donor_count"}
    if not grid_columns.issubset(frame.columns):
        return
    expected_rows = (len(design.screen_seeds) + len(design.confirmation_seeds)) * sum(
        2 * 3 * int(k) for k in design.k_values
    )
    audit.require(
        "prompt_alignment.rows",
        len(frame) == expected_rows,
        f"rows={len(frame)}, expected={expected_rows}",
    )
    observed_seeds = set(pd.to_numeric(frame["seed"], errors="raise").astype(int))
    audit.require(
        "prompt_alignment.seed_grid",
        observed_seeds == set(design.screen_seeds) | set(design.confirmation_seeds),
        str(sorted(observed_seeds)),
    )
    observed_pairs = set(
        zip(
            pd.to_numeric(frame["receiver_count"], errors="raise").astype(int),
            pd.to_numeric(frame["donor_count"], errors="raise").astype(int),
        )
    )
    audit.require(
        "prompt_alignment.all_directed_pairs",
        observed_pairs == set(design.directed_pairs),
        f"pairs={sorted(observed_pairs)}",
    )


def _audit_head_rankings(
    root: Path,
    design: CausalV2Design,
    audit: AuditCollector,
) -> None:
    stage_design = _read_json(root / "design.json")
    expected_model = str(stage_design.get("model_label"))
    registry_path = root / "head_phenotype_rankings.json"
    scores_path = root / "head_phenotype_scores.csv"
    source_manifest_path = root / "attention_source_manifest.json"
    audit.require(
        "head_rankings.registry_exists", registry_path.is_file(), str(registry_path)
    )
    audit.require("head_rankings.scores_exist", scores_path.is_file(), str(scores_path))
    audit.require(
        "head_rankings.source_manifest_exists",
        source_manifest_path.is_file(),
        str(source_manifest_path),
    )
    if source_manifest_path.is_file():
        source = _read_json(source_manifest_path)
        audit.require(
            "head_rankings.source_schema",
            source.get("schema_version")
            == "realistic_niah_v4_causal_v2_attention_source_v1",
            str(source.get("schema_version")),
        )
        audit.require(
            "head_rankings.source_model",
            str(source.get("model_label")) == expected_model,
            f"source={source.get('model_label')}, expected={expected_model}",
        )
        audit.require(
            "head_rankings.source_variant_split",
            source.get("design_variant") == "v4.4"
            and source.get("split") == "discovery",
            f"variant={source.get('design_variant')}, split={source.get('split')}",
        )
        audit.require(
            "head_rankings.source_seed_grid",
            set(int(value) for value in source.get("seeds", []))
            == set(design.centroid_fit_seeds),
            str(source.get("seeds")),
        )
        audit.require(
            "head_rankings.source_count_grid",
            set(int(value) for value in source.get("counts", [])) == set(range(1, 11)),
            str(source.get("counts")),
        )
        audit.require(
            "head_rankings.source_prompt_count",
            int(source.get("prompt_count", -1)) == len(design.centroid_fit_seeds) * 10,
            str(source.get("prompt_count")),
        )
        files = source.get("files", [])
        audit.require(
            "head_rankings.source_dimensions",
            int(source.get("file_count", -1)) == len(files)
            and int(source.get("rows", -1))
            == int(source.get("prompt_count", -1))
            * int(source.get("heads_per_prompt", -1))
            and (
                (
                    source.get("source_kind") == "attention_capture_index"
                    and len(files) == int(source.get("prompt_count", -1))
                )
                or (
                    source.get("source_kind") == "consolidated_attention_csv"
                    and len(files) == 1
                )
            ),
            f"files={len(files)}, registered_files={source.get('file_count')}, "
            f"rows={source.get('rows')}, prompts={source.get('prompt_count')}, "
            f"heads_per_prompt={source.get('heads_per_prompt')}",
        )
        file_errors: list[str] = []
        resolved = Path(str(source.get("resolved_source", "")))
        if not resolved.is_file() and source.get("portable_run_relative_source"):
            run_root = root.parents[4]
            resolved = run_root.joinpath(
                *str(source["portable_run_relative_source"]).split("/")
            )
        if source.get("source_kind") == "attention_capture_index":
            index_sha = _sha256_file(resolved) if resolved.is_file() else None
            audit.require(
                "head_rankings.source_index_verified",
                index_sha == str(source.get("source_index_sha256")),
                f"resolved={resolved}, registered_sha={source.get('source_index_sha256')}",
            )
            audit.require(
                "head_rankings.source_index_linked_to_design",
                stage_design.get("attention_source_index_sha256") == index_sha,
                f"observed={index_sha}, design="
                f"{stage_design.get('attention_source_index_sha256')}",
            )
        for row in files:
            if source.get("source_kind") == "attention_capture_index":
                path = resolved.parent / str(row.get("relative_shard_path", ""))
            else:
                path = resolved
            if not path.is_file():
                file_errors.append(f"missing:{path}")
            elif _sha256_file(path) != str(row.get("sha256")):
                file_errors.append(f"sha256:{path}")
        audit.require(
            "head_rankings.source_files_verified",
            bool(files) and not file_errors,
            f"files={len(files)}, errors={file_errors[:10]}",
        )
        fingerprint_payload = {
            "model_label": str(source.get("model_label")),
            "files": [
                {key: row[key] for key in sorted(row) if key != "bytes"}
                for row in files
            ],
        }
        observed_sha = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        audit.require(
            "head_rankings.source_aggregate_sha256",
            observed_sha == str(source.get("aggregate_sha256")),
            f"observed={observed_sha}, registered={source.get('aggregate_sha256')}",
        )
        audit.require(
            "head_rankings.source_sha_linked_to_design",
            stage_design.get("attention_source_aggregate_sha256") == observed_sha,
            str(stage_design.get("attention_source_aggregate_sha256")),
        )
    if not registry_path.is_file():
        return
    registry = _read_json(registry_path)
    audit.require(
        "head_rankings.registry_model",
        str(registry.get("model_label")) == expected_model,
        f"registry={registry.get('model_label')}, expected={expected_model}",
    )
    audit.require(
        "head_rankings.discovery_only",
        registry.get("selection_split") == "discovery",
        str(registry.get("selection_split")),
    )
    audit.require(
        "head_rankings.full_span_mass_definition",
        "every token" in str(registry.get("mass_definition", "")),
        str(registry.get("mass_definition")),
    )
    rankings = registry.get("rankings", {})
    errors: list[str] = []
    for bank in ("broad_aggregation", "first_locator"):
        rows = rankings.get(bank, [])
        heads = [(int(row["layer"]), int(row["head"])) for row in rows]
        if len(rows) != 32 or len(set(heads)) != 32:
            errors.append(f"{bank}:rows={len(rows)},unique={len(set(heads))}")
    audit.require("head_rankings.two_top32_banks", not errors, str(errors))
    if scores_path.is_file() and source_manifest_path.is_file():
        scores = pd.read_csv(scores_path)
        source = _read_json(source_manifest_path)
        audit.require(
            "head_rankings.scores_match_head_grid",
            len(scores) == int(source.get("heads_per_prompt", -1))
            and set(scores["model_label"].astype(str)) == {expected_model}
            and not scores.duplicated(["layer", "head"]).any(),
            f"scores={len(scores)}, expected={source.get('heads_per_prompt')}",
        )


def _parse_heads(value: Any) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if not token.startswith("L") or "H" not in token:
            raise ValueError(f"Malformed head label: {token}")
        layer_text, head_text = token[1:].split("H", 1)
        result.append((int(layer_text), int(head_text)))
    return tuple(result)


def _layer_counts(heads: Sequence[tuple[int, int]]) -> Counter[int]:
    return Counter(layer for layer, _head in heads)


def _audit_ablation(
    root: Path,
    *,
    design: CausalV2Design,
    audit: AuditCollector,
) -> None:
    index = _audit_shards(root, audit, "ablation")
    path = root / "detail.csv.gz"
    audit.require("ablation.detail_exists", path.is_file(), str(path))
    if not path.is_file():
        return
    frame = pd.read_csv(path, compression="gzip")
    expected_shards = len(design.screen_seeds) * len(design.ablation_counts)
    expected_per_shard = (
        2 * len(design.ablation_top_ns) * (1 + design.ablation_random_replicates)
    )
    audit.require(
        "ablation.shards", len(index) == expected_shards, f"rows={len(index)}"
    )
    audit.require(
        "ablation.rows",
        len(frame) == expected_shards * expected_per_shard,
        f"rows={len(frame)}, expected={expected_shards * expected_per_shard}",
    )
    audit.require(
        "ablation.answer_query_only",
        set(frame["scope"].astype(str)) == {"answer_query"},
        str(sorted(frame["scope"].astype(str).unique())),
    )
    audit.require(
        "ablation.top_1_to_32",
        set(pd.to_numeric(frame["top_n"]).astype(int)) == set(range(1, 33)),
        str(sorted(pd.to_numeric(frame["top_n"]).astype(int).unique())),
    )
    audit.require(
        "ablation.two_registered_banks",
        set(frame["head_bank"].astype(str)) == {"broad_aggregation", "first_locator"},
        str(sorted(frame["head_bank"].astype(str).unique())),
    )
    expected_conditions = {"ranked", "layer_matched_random"}
    audit.require(
        "ablation.conditions",
        set(frame["condition"].astype(str)) == expected_conditions,
        str(sorted(frame["condition"].astype(str).unique())),
    )
    random_rows = frame[frame["condition"].astype(str).eq("layer_matched_random")]
    audit.require(
        "ablation.unbiased_random_population_recorded",
        set(random_rows["random_sampling_population"].astype(str))
        == {"all_heads_in_matched_layers_without_replacement"},
        str(sorted(random_rows["random_sampling_population"].astype(str).unique())),
    )
    overlap = pd.to_numeric(
        random_rows["ranked_random_head_overlap"], errors="raise"
    ).astype(int)
    audit.require(
        "ablation.random_overlap_auditable",
        bool(
            (
                (overlap >= 0)
                & (overlap <= pd.to_numeric(random_rows["top_n"]).astype(int))
            ).all()
        ),
        f"overlap_range={overlap.min() if len(overlap) else None}..{overlap.max() if len(overlap) else None}",
    )
    pairing_errors: list[str] = []
    groups = ["stimulus_id", "head_bank", "top_n"]
    for keys, block in frame.groupby(groups, sort=False):
        ranked = block[block["condition"].astype(str).eq("ranked")]
        random = block[block["condition"].astype(str).eq("layer_matched_random")]
        if len(ranked) != 1 or random["random_replicate"].nunique() != 3:
            pairing_errors.append(f"multiplicity:{keys}")
            continue
        ranked_heads = _parse_heads(ranked.iloc[0]["heads"])
        top_n = int(keys[-1])
        if len(ranked_heads) != top_n:
            pairing_errors.append(f"ranked-size:{keys}")
        for row in random.itertuples(index=False):
            controls = _parse_heads(row.heads)
            if len(controls) != top_n or _layer_counts(controls) != _layer_counts(
                ranked_heads
            ):
                pairing_errors.append(f"layer-match:{keys}:rep={row.random_replicate}")
    audit.require(
        "ablation.layer_matched_random_integrity",
        not pairing_errors,
        f"errors={pairing_errors[:10]}",
    )


def _audit_transport_metrics(
    frame: pd.DataFrame, audit: AuditCollector, label: str
) -> None:
    successful = frame
    if "status" in frame.columns:
        successful = frame[frame["status"].astype(str).eq("ok")]
    mismatches: list[int] = []
    for index, row in successful.iterrows():
        baseline = row.get("baseline_predicted_count")
        intervened = row.get("patched_predicted_count")
        baseline_value = None if pd.isna(baseline) else int(baseline)
        intervened_value = None if pd.isna(intervened) else int(intervened)
        target_column = "donor_count" if "donor_count" in successful else "target_count"
        expected = normalized_transport_metrics(
            baseline_prediction=baseline_value,
            intervened_prediction=intervened_value,
            receiver_count=int(row["receiver_count"]),
            target_count=int(row[target_column]),
        )
        for column in (
            "normalized_transport",
            "target_conformity",
            "strict_normalized_transport",
            "strict_target_conformity",
        ):
            observed = row.get(column)
            target = expected[column]
            if pd.isna(target):
                equal = pd.isna(observed)
            else:
                equal = not pd.isna(observed) and math.isclose(
                    float(observed), float(target), rel_tol=1e-9, abs_tol=1e-9
                )
            if not equal:
                mismatches.append(int(index))
                break
        if int(index) not in mismatches:
            observed_valid = row.get("transport_numeric_valid")
            try:
                observed_valid_bool = bool(
                    _as_bool(pd.Series([observed_valid])).iloc[0]
                )
            except ValueError:
                observed_valid_bool = not bool(expected["transport_numeric_valid"])
            if observed_valid_bool != bool(expected["transport_numeric_valid"]):
                mismatches.append(int(index))
    audit.require(
        f"{label}.normalized_transport_recomputed",
        not mismatches,
        f"mismatch_rows={mismatches[:10]}",
    )


def _audit_patching(
    root: Path,
    *,
    family: str,
    phase: str,
    design: CausalV2Design,
    audit: AuditCollector,
) -> None:
    label = f"{family}.{phase}"
    index = _audit_shards(root, audit, label)
    if family == "prompt_patching":
        alignment_path = root / "prompt_full_span_alignment.csv"
        audit.require(
            f"{label}.full_span_alignment_exists",
            alignment_path.is_file(),
            str(alignment_path),
        )
        if alignment_path.is_file():
            alignment = pd.read_csv(alignment_path)
            audit.require(
                f"{label}.full_span_alignment_nonempty",
                not alignment.empty,
                f"rows={len(alignment)}",
            )
            _audit_full_span_mapping_table(
                alignment,
                design,
                audit,
                f"{label}.full_span_alignment",
            )
    path = root / "detail.csv.gz"
    audit.require(f"{label}.detail_exists", path.is_file(), str(path))
    if not path.is_file():
        return
    frame = pd.read_csv(path, compression="gzip")
    reuse_columns = {
        "generation_executed",
        "generation_reuse_mode",
        "full_span_alignment_policy",
    }
    audit.require(
        f"{label}.compute_reuse_schema",
        reuse_columns.issubset(frame.columns),
        str(sorted(frame.columns)),
    )
    if "full_span_alignment_policy" in frame.columns:
        observed_policies = set(frame["full_span_alignment_policy"].astype(str))
        audit.require(
            f"{label}.registered_full_span_alignment_policy",
            observed_policies == {design.prompt_full_span_alignment},
            str(sorted(observed_policies)),
        )
    reuse_summary_path = root / "compute_reuse_summary.csv"
    audit.require(
        f"{label}.compute_reuse_summary_exists",
        reuse_summary_path.is_file(),
        str(reuse_summary_path),
    )
    if reuse_summary_path.is_file():
        reuse_summary = pd.read_csv(reuse_summary_path)
        audit.require(
            f"{label}.compute_reuse_summary_rows",
            "logical_rows" in reuse_summary
            and int(pd.to_numeric(reuse_summary["logical_rows"]).sum()) == len(frame),
            f"summary_rows={reuse_summary.get('logical_rows', pd.Series(dtype=int)).sum()}, "
            f"detail_rows={len(frame)}",
        )
    if reuse_columns.issubset(frame.columns):
        executed = (
            frame["generation_executed"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
        )
        audit.require(
            f"{label}.compute_reuse_boolean",
            bool(executed.notna().all()),
            str(frame["generation_executed"].value_counts(dropna=False).to_dict()),
        )
        donor_rows = frame["condition"].astype(str).eq("donor_transport")
        audit.require(
            f"{label}.all_donor_transports_executed",
            bool(
                donor_rows.any()
                and executed.loc[donor_rows].all()
                and frame.loc[donor_rows, "generation_reuse_mode"]
                .astype(str)
                .eq("fresh_intervention")
                .all()
            ),
            "donor rows may never be synthesized from a control cache",
        )
        self_rows = frame[frame["condition"].astype(str).eq("self_patch")].copy()
        self_modes = set(self_rows["generation_reuse_mode"].astype(str))
        expected_self_modes = (
            {"baseline_identity_reuse", "executed_identity_preflight"}
            if phase == "screen"
            else {"baseline_identity_reuse", "executed_identity_preflight"}
        )
        audit.require(
            f"{label}.self_patch_reuse_modes",
            (
                self_modes == expected_self_modes
                if phase == "screen"
                else bool(self_modes) and self_modes.issubset(expected_self_modes)
            ),
            str(sorted(self_modes)),
        )
        self_executed = self_rows[
            self_rows["generation_reuse_mode"]
            .astype(str)
            .eq("executed_identity_preflight")
        ]
        audit.require(
            f"{label}.self_patch_identity_preflight",
            phase != "screen"
            or (
                not self_executed.empty
                and self_executed["seed"].nunique() == 1
                and self_executed[["receiver_count", "donor_count"]]
                .drop_duplicates()
                .shape[0]
                == 1
                and set(self_executed["site"].astype(str))
                == (
                    {"toggled_needle_end", "toggled_needle_span"}
                    if family == "prompt_patching"
                    else {"answer_query"}
                )
                and set(self_executed["patch_protocol"].astype(str))
                == set(design.patch_protocols)
                and self_executed["start_layer"].nunique() == 3
            ),
            f"rows={len(self_executed)}",
        )
        audit.require(
            f"{label}.self_patch_outputs_are_identity",
            bool(
                self_rows["patched_completion_text"]
                .astype(str)
                .eq(self_rows["baseline_completion_text"].astype(str))
                .all()
                and pd.to_numeric(
                    self_rows["strict_normalized_transport"], errors="coerce"
                )
                .fillna(0.0)
                .eq(0.0)
                .all()
            ),
            "self-patch must reproduce the registered greedy baseline",
        )
        if family == "answer_patching" and phase == "screen":
            same_count_modes = set(
                frame.loc[
                    frame["condition"].astype(str).eq("same_count_seed"),
                    "generation_reuse_mode",
                ].astype(str)
            )
            audit.require(
                f"{label}.same_count_equivalent_cache_used",
                same_count_modes
                == {"fresh_intervention", "equivalent_intervention_cache"},
                str(sorted(same_count_modes)),
            )
    audit.require(
        f"{label}.no_skipped_rows",
        set(frame["status"].astype(str)) == {"ok"},
        str(frame["status"].astype(str).value_counts().to_dict()),
    )
    observed_k = set(pd.to_numeric(frame["k"]).astype(int))
    expected_k_ok = (
        observed_k == set(design.k_values)
        if phase == "screen"
        else bool(observed_k) and observed_k.issubset(set(design.k_values))
    )
    audit.require(
        f"{label}.registered_k_values",
        expected_k_ok,
        str(sorted(observed_k)),
    )
    audit.require(
        f"{label}.changed_slot_count_equals_k",
        bool(
            (
                pd.to_numeric(frame["changed_slot_count"]).astype(int)
                == pd.to_numeric(frame["k"]).astype(int)
            ).all()
        ),
        "every row must patch all k changed nested slots",
    )
    sites = (
        {"toggled_needle_end", "toggled_needle_span"}
        if family == "prompt_patching"
        else {"answer_query"}
    )
    controls = (
        {"donor_transport", "self_patch"}
        if family == "prompt_patching"
        else {"donor_transport", "self_patch", "same_count_seed"}
    )
    observed_sites = set(frame["site"].astype(str))
    audit.require(
        f"{label}.sites",
        (
            observed_sites == sites
            if phase == "screen"
            else bool(observed_sites) and observed_sites.issubset(sites)
        ),
        str(sorted(observed_sites)),
    )
    audit.require(
        f"{label}.controls",
        set(frame["condition"].astype(str)) == controls,
        str(sorted(frame["condition"].astype(str).unique())),
    )
    observed_protocols = set(frame["patch_protocol"].astype(str))
    audit.require(
        f"{label}.protocols",
        (
            observed_protocols == set(design.patch_protocols)
            if phase == "screen"
            else bool(observed_protocols)
            and observed_protocols.issubset(set(design.patch_protocols))
        ),
        str(sorted(observed_protocols)),
    )
    layers = sorted(pd.to_numeric(frame["start_layer"]).astype(int).unique())
    if phase == "screen":
        audit.require(
            f"{label}.all_layers",
            layers == list(range(max(layers) + 1)) if layers else False,
            str(layers),
        )
        num_layers = max(layers) + 1 if layers else 0
    else:
        cumulative_depths = pd.to_numeric(
            frame.loc[
                frame["patch_protocol"].astype(str).eq("cumulative_from_layer"),
                "patched_layer_count",
            ]
        ).astype(int) + pd.to_numeric(
            frame.loc[
                frame["patch_protocol"].astype(str).eq("cumulative_from_layer"),
                "start_layer",
            ]
        ).astype(
            int
        )
        num_layers = (
            int(cumulative_depths.iloc[0])
            if not cumulative_depths.empty
            else max(layers) + 1
        )
        audit.require(
            f"{label}.selected_layers_registered",
            bool(layers) and all(0 <= layer < num_layers for layer in layers),
            f"layers={layers}, inferred_num_layers={num_layers}",
        )
    single = frame[frame["patch_protocol"].astype(str).eq("single_layer")]
    cumulative = frame[frame["patch_protocol"].astype(str).eq("cumulative_from_layer")]
    audit.require(
        f"{label}.single_layer_count",
        bool((pd.to_numeric(single["patched_layer_count"]).astype(int) == 1).all()),
        "single-layer rows must patch exactly one layer",
    )
    audit.require(
        f"{label}.cumulative_L_to_final",
        bool(
            cumulative.empty
            or (
                pd.to_numeric(cumulative["patched_layer_count"]).astype(int)
                == num_layers - pd.to_numeric(cumulative["start_layer"]).astype(int)
            ).all()
        ),
        f"num_layers={num_layers}",
    )
    expected_seeds = (
        design.screen_seeds if phase == "screen" else design.confirmation_seeds
    )
    audit.require(
        f"{label}.seed_split",
        set(pd.to_numeric(frame["seed"]).astype(int)) == set(expected_seeds),
        str(sorted(pd.to_numeric(frame["seed"]).astype(int).unique())),
    )
    if phase == "screen":
        audit.require(
            f"{label}.all_directed_pairs",
            set(
                zip(
                    pd.to_numeric(frame["receiver_count"]).astype(int),
                    pd.to_numeric(frame["donor_count"]).astype(int),
                )
            )
            == set(design.directed_pairs),
            (
                f"screen must cover all {len(design.directed_pairs)} registered "
                f"directions for k={design.k_values}"
            ),
        )
        expected_rows = (
            len(design.screen_seeds)
            * len(design.directed_pairs)
            * num_layers
            * len(sites)
            * len(design.patch_protocols)
            * len(controls)
        )
        audit.require(
            f"{label}.rows",
            len(frame) == expected_rows,
            f"rows={len(frame)}, expected={expected_rows}",
        )
        audit.require(
            f"{label}.shards",
            len(index) == len(design.screen_seeds) * len(design.directed_pairs),
            f"shards={len(index)}",
        )
    _audit_transport_metrics(frame, audit, label)


def _audit_centroids(root: Path, design: CausalV2Design, audit: AuditCollector) -> None:
    index = _audit_shards(root, audit, "steering_centroids.fit")
    expected = len(design.centroid_fit_seeds) * len(design.valid_counts)
    audit.require(
        "steering_centroids.fit.shards",
        len(index) == expected,
        f"shards={len(index)}, expected={expected}",
    )
    audit.require(
        "steering_centroids.fit.bundle",
        (root / "centroids.npz").is_file(),
        str(root / "centroids.npz"),
    )
    if index:
        audit.require(
            "steering_centroids.fit.seeds",
            {int(row["seed"]) for row in index} == set(design.centroid_fit_seeds),
            str(sorted({int(row["seed"]) for row in index})),
        )
        audit.require(
            "steering_centroids.fit.counts",
            {int(row["count"]) for row in index} == set(design.valid_counts),
            str(sorted({int(row["count"]) for row in index})),
        )


def _audit_steering(
    root: Path,
    *,
    phase: str,
    design: CausalV2Design,
    audit: AuditCollector,
) -> None:
    label = f"steering.{phase}"
    index = _audit_shards(root, audit, label)
    path = root / "detail.csv.gz"
    audit.require(f"{label}.detail_exists", path.is_file(), str(path))
    if not path.is_file():
        return
    frame = pd.read_csv(path, compression="gzip")
    audit.require(
        f"{label}.answer_query_only",
        set(frame["site"].astype(str)) == {"answer_query"},
        str(sorted(frame["site"].astype(str).unique())),
    )
    audit.require(
        f"{label}.conditions",
        set(frame["condition"].astype(str))
        == {"geometric", "orthogonal_norm_matched_random"},
        str(sorted(frame["condition"].astype(str).unique())),
    )
    observed_k = set(pd.to_numeric(frame["k"]).astype(int))
    audit.require(
        f"{label}.registered_k_values",
        (
            observed_k == set(design.k_values)
            if phase == "screen"
            else bool(observed_k) and observed_k.issubset(set(design.k_values))
        ),
        str(sorted(observed_k)),
    )
    expected_seeds = (
        design.screen_seeds if phase == "screen" else design.confirmation_seeds
    )
    audit.require(
        f"{label}.seed_split",
        set(pd.to_numeric(frame["seed"]).astype(int)) == set(expected_seeds),
        str(sorted(pd.to_numeric(frame["seed"]).astype(int).unique())),
    )
    if phase == "screen":
        audit.require(
            f"{label}.single_layer_only",
            set(frame["steering_protocol"].astype(str)) == {"single_layer"},
            str(sorted(frame["steering_protocol"].astype(str).unique())),
        )
        layers = sorted({int(value) for value in frame["layer_set"].astype(str)})
        audit.require(
            f"{label}.all_layers",
            layers == list(range(max(layers) + 1)) if layers else False,
            str(layers),
        )
        num_layers = max(layers) + 1 if layers else 0
        expected_rows = (
            len(design.screen_seeds)
            * len(design.directed_pairs)
            * num_layers
            * (1 + design.steering_random_replicates)
        )
        audit.require(
            f"{label}.rows",
            len(frame) == expected_rows,
            f"rows={len(frame)}, expected={expected_rows}",
        )
        audit.require(
            f"{label}.shards",
            len(index) == len(design.screen_seeds) * len(design.directed_pairs),
            f"shards={len(index)}",
        )
    _audit_transport_metrics(frame, audit, label)


def _audit_confirmation_matches_selection(
    *,
    family: str,
    selection_path: Path,
    confirmation_root: Path,
    design: CausalV2Design,
    audit: AuditCollector,
) -> None:
    selection = _read_json(selection_path)
    selected = selection.get("selected", [])
    detail_path = confirmation_root / "detail.csv.gz"
    if not detail_path.is_file():
        return
    frame = pd.read_csv(detail_path, compression="gzip")
    if family in {"prompt_patching", "answer_patching"}:
        expected = {
            (
                str(item["site"]),
                str(item["patch_protocol"]),
                int(item["start_layer"]),
                int(item["k"]),
            )
            for item in selected
        }
        observed = {
            (str(row.site), str(row.patch_protocol), int(row.start_layer), int(row.k))
            for row in frame.itertuples(index=False)
        }
        treatment_name = "donor_transport"
        treatment = frame[frame["condition"].astype(str).eq(treatment_name)]
        key_columns = ["site", "patch_protocol", "start_layer", "k"]
    else:
        singleton = {
            (
                str(item["steering_protocol"]),
                str(item["layer_set"]),
                int(item["k"]),
            )
            for item in selected
        }
        layers_by_k: dict[int, set[int]] = {}
        for _protocol, layer, k in singleton:
            layers_by_k.setdefault(k, set()).add(int(layer))
        multi = {
            ("multi_layer", "+".join(str(layer) for layer in sorted(layers)), k)
            for k, layers in layers_by_k.items()
            if len(layers) >= 2
        }
        expected = singleton | multi
        observed = {
            (str(row.steering_protocol), str(row.layer_set), int(row.k))
            for row in frame.itertuples(index=False)
        }
        treatment_name = "geometric"
        treatment = frame[frame["condition"].astype(str).eq(treatment_name)]
        key_columns = ["steering_protocol", "layer_set", "k"]
    audit.require(
        f"{family}.confirmation_exact_selected_conditions",
        observed == expected,
        f"expected={sorted(expected)}, observed={sorted(observed)}",
    )
    incomplete: list[str] = []
    for keys, block in treatment.groupby(key_columns, dropna=False, sort=False):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        k = int(keys_tuple[-1])
        seeds = set(pd.to_numeric(block["seed"]).astype(int))
        pairs = set(
            zip(
                pd.to_numeric(block["receiver_count"]).astype(int),
                pd.to_numeric(
                    block["donor_count" if "donor_count" in block else "target_count"]
                ).astype(int),
            )
        )
        if seeds != set(design.confirmation_seeds):
            incomplete.append(f"seeds:{keys_tuple}:{sorted(seeds)}")
        if pairs != set(design.pairs_for_k(k, directed=True)):
            incomplete.append(f"pairs:{keys_tuple}:{sorted(pairs)}")
    audit.require(
        f"{family}.confirmation_five_seeds_three_anchors_both_directions",
        not incomplete,
        f"errors={incomplete[:10]}",
    )


def audit_causal_v2_run(
    *,
    run_root: str | Path,
    model_label: str,
    stimuli_path: str | Path,
    design: CausalV2Design | None = None,
    require_confirmation: bool = False,
) -> dict[str, Any]:
    design = design or CausalV2Design()
    design.validate()
    audit = AuditCollector()
    audit_stimulus_grid(stimuli_path, design=design, audit=audit)
    causal_root = Path(run_root) / model_label / "numeric" / "causal_v2"
    audit.require("run.causal_root_exists", causal_root.is_dir(), str(causal_root))
    if not causal_root.is_dir():
        return audit.payload()

    roots: dict[str, Path | None] = {
        "prompt_alignment": _completed_formal_root(
            causal_root,
            family="prompt_span_alignment",
            phase="preflight",
            required=True,
            audit=audit,
        ),
        "baseline": _completed_formal_root(
            causal_root, family="baseline", phase="all", required=True, audit=audit
        ),
        "head_rankings": _completed_formal_root(
            causal_root,
            family="head_rankings",
            phase="discovery",
            required=True,
            audit=audit,
        ),
        "ablation": _completed_formal_root(
            causal_root,
            family="answer_query_head_ablation",
            phase="screen",
            required=True,
            audit=audit,
        ),
        "prompt_screen": _completed_formal_root(
            causal_root,
            family="prompt_patching",
            phase="screen",
            required=True,
            audit=audit,
        ),
        "answer_screen": _completed_formal_root(
            causal_root,
            family="answer_patching",
            phase="screen",
            required=True,
            audit=audit,
        ),
        "centroids": _completed_formal_root(
            causal_root,
            family="steering_centroids",
            phase="fit",
            required=True,
            audit=audit,
        ),
        "steering_screen": _completed_formal_root(
            causal_root, family="steering", phase="screen", required=True, audit=audit
        ),
    }
    confirmation_specs = (
        ("prompt", "prompt_patching", roots["prompt_screen"]),
        ("answer", "answer_patching", roots["answer_screen"]),
        ("steering", "steering", roots["steering_screen"]),
    )
    for key, family, screen_root in confirmation_specs:
        selected_count = 0
        if require_confirmation:
            selection_path = (
                screen_root / "selection" / f"{family}_selection.json"
                if screen_root is not None
                else Path("__missing_selection__")
            )
            audit.require(
                f"{family}.selection_manifest_exists",
                selection_path.is_file(),
                str(selection_path),
            )
            if selection_path.is_file():
                selection = _read_json(selection_path)
                selected_count = int(selection.get("selected_condition_count", -1))
                audit.require(
                    f"{family}.selection_seed_split",
                    selection.get("screen_seeds") == list(design.screen_seeds)
                    and selection.get("held_out_confirmation_seeds")
                    == list(design.confirmation_seeds),
                    f"screen={selection.get('screen_seeds')}, "
                    f"held_out={selection.get('held_out_confirmation_seeds')}",
                )
                audit.require(
                    f"{family}.selection_count_consistent",
                    selected_count == len(selection.get("selected", []))
                    and selected_count >= 0,
                    f"declared={selected_count}, listed={len(selection.get('selected', []))}",
                )
        confirmation_root = _completed_formal_root(
            causal_root,
            family=family,
            phase="confirmation",
            required=require_confirmation and selected_count > 0,
            audit=audit,
        )
        roots[f"{key}_confirmation"] = confirmation_root
        if require_confirmation and selected_count == 0:
            audit.require(
                f"{family}.zero_selection_has_no_confirmation_root",
                confirmation_root is None,
                str(confirmation_root),
            )
        if (
            require_confirmation
            and selected_count > 0
            and screen_root is not None
            and confirmation_root is not None
        ):
            selection_path = screen_root / "selection" / f"{family}_selection.json"
            confirmation_design = _read_json(confirmation_root / "design.json")
            audit.require(
                f"{family}.confirmation_selection_hash",
                confirmation_design.get("selection_json_sha256")
                == _sha256_file(selection_path),
                f"observed={confirmation_design.get('selection_json_sha256')}",
            )
            stats = (
                confirmation_root / "analysis" / f"{family}_confirmation_statistics.csv"
            )
            provenance = (
                confirmation_root / "analysis" / f"{family}_confirmation_inputs.json"
            )
            audit.require(
                f"{family}.confirmation_statistics_exists",
                stats.is_file(),
                str(stats),
            )
            audit.require(
                f"{family}.confirmation_provenance_exists",
                provenance.is_file(),
                str(provenance),
            )
            if stats.is_file():
                statistics = pd.read_csv(stats)
                required_statistics = {
                    "evidence_scope",
                    "is_primary_confirmation",
                    "screen_seeds",
                    "held_out_confirmation_seeds",
                    "exact_sign_flip_p",
                    "holm_p",
                }
                audit.require(
                    f"{family}.confirmation_statistics_schema",
                    required_statistics.issubset(statistics.columns),
                    str(sorted(statistics.columns)),
                )
                if required_statistics.issubset(statistics.columns):
                    primary = statistics[
                        _as_bool(statistics["is_primary_confirmation"])
                    ]
                    audit.require(
                        f"{family}.held_out_only_is_primary",
                        not primary.empty
                        and set(primary["evidence_scope"].astype(str))
                        == {"held_out_only"}
                        and bool(
                            (
                                pd.to_numeric(
                                    primary["held_out_confirmation_seeds"],
                                    errors="raise",
                                ).astype(int)
                                == len(design.confirmation_seeds)
                            ).all()
                        )
                        and bool(
                            (
                                pd.to_numeric(
                                    primary["screen_seeds"], errors="raise"
                                ).astype(int)
                                == 0
                            ).all()
                        ),
                        f"primary_rows={len(primary)} scopes="
                        f"{sorted(primary['evidence_scope'].astype(str).unique()) if not primary.empty else []}",
                    )
            if provenance.is_file():
                inputs = _read_json(provenance)
                audit.require(
                    f"{family}.confirmation_provenance_seed_split",
                    inputs.get("screen_seeds") == list(design.screen_seeds)
                    and inputs.get("held_out_confirmation_seeds")
                    == list(design.confirmation_seeds),
                    f"screen={inputs.get('screen_seeds')}, "
                    f"held_out={inputs.get('held_out_confirmation_seeds')}",
                )
            _audit_confirmation_matches_selection(
                family=family,
                selection_path=selection_path,
                confirmation_root=confirmation_root,
                design=design,
                audit=audit,
            )
        if require_confirmation and selected_count == 0 and screen_root is not None:
            marker = screen_root / "selection" / "no_confirmation_required.json"
            audit.require(
                f"{family}.zero_selection_marker",
                marker.is_file(),
                str(marker),
            )
    resolved_designs = {
        key: _read_json(root / "design.json")
        for key, root in roots.items()
        if root is not None
    }
    implementation_hashes = {
        str(payload.get("implementation_sha256"))
        for payload in resolved_designs.values()
    }
    audit.require(
        "run.one_implementation_fingerprint",
        len(implementation_hashes) == 1,
        str(sorted(implementation_hashes)),
    )
    stimulus_hashes = {
        str(payload.get("stimuli_sha256")) for payload in resolved_designs.values()
    }
    audit.require(
        "run.one_stimulus_fingerprint",
        stimulus_hashes == {_sha256_file(Path(stimuli_path))},
        str(sorted(stimulus_hashes)),
    )
    if roots.get("head_rankings") and roots.get("ablation"):
        ranking_path = roots["head_rankings"] / "head_phenotype_rankings.json"
        ablation_design = resolved_designs["ablation"]
        audit.require(
            "ablation.discovery_ranking_hash",
            ranking_path.is_file()
            and ablation_design.get("rankings_sha256") == _sha256_file(ranking_path),
            str(ablation_design.get("rankings_sha256")),
        )
    if roots.get("centroids") and roots.get("steering_screen"):
        centroid_path = roots["centroids"] / "centroids.npz"
        steering_design = resolved_designs["steering_screen"]
        audit.require(
            "steering.screen_centroid_hash",
            centroid_path.is_file()
            and steering_design.get("centroids_sha256") == _sha256_file(centroid_path),
            str(steering_design.get("centroids_sha256")),
        )
    if roots["prompt_alignment"]:
        _audit_prompt_alignment(roots["prompt_alignment"], design, audit)
    if roots["baseline"]:
        _audit_baseline(roots["baseline"], design, audit)
    if roots["head_rankings"]:
        _audit_head_rankings(roots["head_rankings"], design, audit)
    if roots["ablation"]:
        _audit_ablation(roots["ablation"], design=design, audit=audit)
    if roots["prompt_screen"]:
        _audit_patching(
            roots["prompt_screen"],
            family="prompt_patching",
            phase="screen",
            design=design,
            audit=audit,
        )
    if roots["answer_screen"]:
        _audit_patching(
            roots["answer_screen"],
            family="answer_patching",
            phase="screen",
            design=design,
            audit=audit,
        )
    if roots["centroids"]:
        _audit_centroids(roots["centroids"], design, audit)
    if roots["steering_screen"]:
        _audit_steering(
            roots["steering_screen"], phase="screen", design=design, audit=audit
        )
    if roots["prompt_confirmation"]:
        _audit_patching(
            roots["prompt_confirmation"],
            family="prompt_patching",
            phase="confirmation",
            design=design,
            audit=audit,
        )
    if roots["answer_confirmation"]:
        _audit_patching(
            roots["answer_confirmation"],
            family="answer_patching",
            phase="confirmation",
            design=design,
            audit=audit,
        )
    if roots["steering_confirmation"]:
        _audit_steering(
            roots["steering_confirmation"],
            phase="confirmation",
            design=design,
            audit=audit,
        )

    payload = audit.payload()
    payload.update(
        {
            "run_root": str(Path(run_root).resolve()),
            "model_label": str(model_label),
            "stimuli_path": str(Path(stimuli_path).resolve()),
            "require_confirmation": bool(require_confirmation),
            "resolved_stage_roots": {
                key: str(value) if value is not None else None
                for key, value in roots.items()
            },
        }
    )
    return payload


def render_audit_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# Realistic NIAH V4 causal-v2 audit — {payload['status']}",
        "",
        f"- Model: `{payload.get('model_label', 'unknown')}`",
        f"- Checks: {payload.get('check_count', 0)}",
        f"- Errors: {payload.get('error_count', 0)}",
        "",
        "## Checks",
        "",
    ]
    for item in payload.get("checks", []):
        mark = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- **{mark}** `{item['name']}` — {item['detail']}")
    return "\n".join(lines) + "\n"
