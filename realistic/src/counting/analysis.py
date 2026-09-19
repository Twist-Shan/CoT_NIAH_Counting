from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch

from dataset_generation.dynamic_niah_v2 import (
    DynamicNiahV2Config,
    dynamic_niah_v2_config_kwargs,
)
from dataset_generation.run_utils import slugify
from counting.cot import build_controlled_extended_input, load_cot_payload
from single_example import (
    DEFAULT_ABLATION_CONFIG_PATH,
    DEFAULT_REPRESENTATION_ABLATION_CONFIG_PATH,
    DEFAULT_REPRESENTATION_RESTORE_CONFIG_PATH,
    compute_single_example_hidden_states,
    create_single_example_run,
    load_ablation_config,
    load_model_and_tokenizer,
    load_representation_ablation_config,
    load_representation_restore_config,
    locate_uncontrolled_needle_segments,
    prepare_single_example_dataset,
    render_and_tokenize_messages,
    run_single_example_ablation,
    run_single_example_qk_outlier_analysis,
    run_single_example_representation_ablation,
    run_single_example_representation_restore,
    save_input_metadata,
    summarize_ablation_results_all,
)
from single_example.single_example_analysis import SingleExamplePaths

DEFAULT_COUNTING_DATASET_FILENAME = "dynamic_niah_v2.jsonl"
QK_OUTLIER_TABLE_FILENAMES = (
    "massive_tokens_all.csv",
    "massive_tokens_all.jsonl",
    "massive_tokens_outside_needles_all.csv",
    "massive_tokens_outside_needles_all.jsonl",
    "attention_sinks_topk.csv",
    "needle_attention_mass.csv",
    "outlier_attention_join.csv",
    "outlier_overlap_summary.csv",
)
QK_PATTERN_REQUIRED_TABLES = {
    "massive_activation": ("massive_tokens_outside_needles_all.csv",),
    "massive_activation_all": ("massive_tokens_all.csv",),
    "attention_sink": ("attention_sinks_topk.csv",),
    "attention_sink_all": ("attention_sinks_topk.csv",),
}


def _compact_model_name(model_name: str) -> str:
    """Return the non-redundant model identifier for counting run names."""

    text = str(model_name).strip()
    if not text:
        return "model"
    return text.rstrip("/").split("/")[-1]


def _position_slug(position: int | None) -> str:
    return "null" if position is None else slugify(position, max_length=24)


def build_counting_run_name(
    *,
    model_name: str,
    task_type: str,
    prompt_style: str,
    target_haystack_tokens: int,
    insertion_positions: Sequence[int | None],
    num_max_needles: int | None = None,
    start_time: datetime | None = None,
    randomize_needle_insertion: bool = False,
    randomize_needle_seed: int | None = None,
    sentence_level_insertion: bool = False,
    word_level_insertion: bool = False,
    fact_templates_path: str | None = None,
    counting_needle_kind: str = "city_score",
    uid_token_length: int | None = None,
) -> str:
    """Build the compact counting notebook run name.

    Example: ``run_20260609_205951_Qwen3-8B_match_count_easier_1000_needles_100_200_400``.
    """

    start = start_time or datetime.now()
    timestamp = start.strftime("%Y%m%d_%H%M%S")
    model_slug = slugify(_compact_model_name(model_name), max_length=80)
    parts = [
        f"run_{timestamp}",
        model_slug,
        slugify(task_type, max_length=40),
        slugify(prompt_style, max_length=40),
        slugify(target_haystack_tokens, max_length=24),
        "needles",
        *[_position_slug(pos) for pos in insertion_positions],
    ]
    if num_max_needles is not None:
        parts.extend(["num_max_needles", slugify(num_max_needles, max_length=24)])
    if randomize_needle_insertion:
        parts.extend(
            ["rand_insrt", "seed", slugify(randomize_needle_seed, max_length=24)]
        )
    if sentence_level_insertion:
        parts.append("sent_insrt")
    if word_level_insertion:
        parts.append("word_insrt")
    if counting_needle_kind != "city_score":
        parts.extend(["kind", slugify(counting_needle_kind, max_length=32)])
    if str(task_type) == "literal_count" and uid_token_length is not None:
        parts.extend(["uidtok", slugify(uid_token_length, max_length=24)])
    if fact_templates_path:
        parts.extend(
            [
                "tmpl",
                slugify(Path(str(fact_templates_path)).stem, max_length=40),
            ]
        )
    return "_".join(parts)


def build_counting_setting_name(
    *,
    model_name: str,
    task_type: str,
    prompt_style: str,
    target_haystack_tokens: int,
    num_examples: int,
    insertion_positions: Sequence[int | None],
    global_random_seed: int,
    haystack_seed: int | None,
    needle_seed: int | None,
    thinking_mode: bool,
    num_max_needles: int | None = None,
    randomize_needle_insertion: bool = False,
    randomize_needle_seed: int | None = None,
    sentence_level_insertion: bool = False,
    word_level_insertion: bool = False,
    fact_templates_path: str | None = None,
    counting_needle_kind: str = "city_score",
    uid_token_length: int | None = None,
) -> str:
    """Build a timestamp-free cache key for reusable counting datasets."""

    parts = [
        slugify(_compact_model_name(model_name), max_length=80),
        slugify(task_type, max_length=40),
        slugify(prompt_style, max_length=40),
        slugify(target_haystack_tokens, max_length=24),
        "examples",
        slugify(num_examples, max_length=24),
        "needles",
        *[_position_slug(pos) for pos in insertion_positions],
    ]
    if num_max_needles is not None:
        parts.extend(["num_max_needles", slugify(num_max_needles, max_length=24)])
    if randomize_needle_insertion:
        parts.extend(
            ["rand_insrt", "seed", slugify(randomize_needle_seed, max_length=24)]
        )
    if sentence_level_insertion:
        parts.append("sent_insrt")
    if word_level_insertion:
        parts.append("word_insrt")
    if counting_needle_kind != "city_score":
        parts.extend(["kind", slugify(counting_needle_kind, max_length=32)])
    if str(task_type) == "literal_count" and uid_token_length is not None:
        parts.extend(["uidtok", slugify(uid_token_length, max_length=24)])
    if fact_templates_path:
        parts.extend(
            [
                "tmpl",
                slugify(Path(str(fact_templates_path)).stem, max_length=40),
            ]
        )
    parts.extend(
        [
            "gseed",
            slugify(global_random_seed, max_length=24),
            "hseed",
            slugify("none" if haystack_seed is None else haystack_seed, max_length=24),
            "nseed",
            slugify("none" if needle_seed is None else needle_seed, max_length=24),
            "thinking",
            slugify(str(bool(thinking_mode)).lower(), max_length=8),
        ]
    )
    return "_".join(parts)


def _load_cache_config(cache_dir: Path) -> dict[str, Any]:
    config_path = cache_dir / "config.used.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Cached dataset is missing config.used.json: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def validate_counting_dataset_cache(
    cache_dir: str | Path, expected: dict[str, Any]
) -> dict[str, Any]:
    """Validate cached generated/scored counting artifacts against exact settings."""

    cache = Path(cache_dir)
    dataset_path = cache / DEFAULT_COUNTING_DATASET_FILENAME
    predictions_path = cache / "predictions.jsonl"
    metrics_path = cache / "metrics.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Cached dataset JSONL does not exist: {dataset_path}")
    config = _load_cache_config(cache)
    default_config = asdict(DynamicNiahV2Config())
    mismatches: dict[str, dict[str, Any]] = {}
    for key, expected_value in expected.items():
        actual = config.get(key, default_config.get(key))
        if isinstance(expected_value, tuple):
            expected_value = list(expected_value)
        if isinstance(actual, tuple):
            actual = list(actual)
        if actual != expected_value:
            mismatches[key] = {"expected": expected_value, "actual": actual}
    if mismatches:
        raise ValueError(
            "Cached counting dataset metadata mismatch: "
            + json.dumps(mismatches, indent=2, ensure_ascii=False)
        )
    return {
        "cache_dir": cache,
        "dataset_path": dataset_path,
        "config_path": cache / "config.used.json",
        "predictions_path": predictions_path if predictions_path.exists() else None,
        "metrics_path": metrics_path if metrics_path.exists() else None,
        "has_predictions": predictions_path.exists(),
    }


def save_counting_dataset_cache(
    *,
    cache_dir: str | Path,
    dataset_path: str | Path,
    config_path: str | Path,
    predictions_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, str]:
    """Copy small reusable dataset/evaluation artifacts into data/niah-example."""

    cache = Path(cache_dir)
    if cache.exists() and any(cache.iterdir()) and not overwrite:
        raise FileExistsError(f"Cached dataset directory already exists: {cache}")
    cache.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for source, name in [
        (dataset_path, DEFAULT_COUNTING_DATASET_FILENAME),
        (config_path, "config.used.json"),
        (predictions_path, "predictions.jsonl"),
        (metrics_path, "metrics.json"),
    ]:
        if source is None:
            continue
        src = Path(source)
        if not src.exists():
            continue
        dest = cache / name
        shutil.copyfile(src, dest)
        copied[name] = str(dest)
    return copied


def load_counting_dataset(dataset_path: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Counting dataset JSONL does not exist: {path}. Run the dataset generation cell first."
        )
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Counting dataset JSONL is empty: {path}")
    return rows


def validate_selected_example_ids(
    selected_example_ids: Sequence[int], *, num_examples: int
) -> list[int]:
    """Validate that ablation examples come from the first half of the dataset."""

    selected = [int(example_id) for example_id in selected_example_ids]
    if not selected:
        raise ValueError("SELECT_EXAMPLE_ID must contain at least one example id")
    max_examples = int(num_examples) // 2
    valid = set(range(max_examples))
    invalid = [example_id for example_id in selected if example_id not in valid]
    if invalid:
        raise ValueError(
            "SELECT_EXAMPLE_ID must be a subset of list(range(NUM_EXAMPLES // 2)); "
            f"valid ids are 0..{max_examples - 1}, got invalid ids {invalid}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError(f"SELECT_EXAMPLE_ID contains duplicate ids: {selected}")
    return selected


def cleanup_counting_archive_artifacts(
    run_dir: str | Path,
    *,
    delete_large_pt: bool = True,
    max_pt_bytes: int = 100 * 1024 * 1024,
) -> dict[str, list[Path]]:
    """Delete bulky counting-notebook intermediates before archiving.

    The counting notebook keeps table summaries and figures, but Q/K tensor caches,
    per-input attention-stat tensor folders, representation-restore corruption logs,
    and oversized ``.pt`` files are expensive to upload to Drive and are not needed in
    the final zip. Return paths grouped by cleanup rule for concise reporting.
    """

    root = Path(run_dir)
    removed: dict[str, list[Path]] = {
        "corrupted_needle_tokens": [],
        "attention_stats_inputs": [],
        "qk_cache": [],
        "large_pt": [],
    }
    if not root.exists():
        return removed

    for path in sorted(root.rglob("corrupted_needle_tokens.jsonl")):
        if path.is_file():
            path.unlink()
            removed["corrupted_needle_tokens"].append(path)

    for attention_stats_dir in sorted(root.rglob("attention_stats")):
        if not attention_stats_dir.is_dir():
            continue
        for input_dir in sorted(attention_stats_dir.glob("input_*")):
            if input_dir.is_dir():
                shutil.rmtree(input_dir)
                removed["attention_stats_inputs"].append(input_dir)

    for qk_cache_dir in sorted(root.rglob("qk_cache")):
        if qk_cache_dir.is_dir():
            shutil.rmtree(qk_cache_dir)
            removed["qk_cache"].append(qk_cache_dir)
        elif qk_cache_dir.exists():
            qk_cache_dir.unlink()
            removed["qk_cache"].append(qk_cache_dir)

    if delete_large_pt:
        seen: set[Path] = set()
        for pt in sorted(root.rglob("*.pt")):
            if pt in seen or not pt.is_file():
                continue
            seen.add(pt)
            if pt.stat().st_size > max_pt_bytes:
                pt.unlink()
                removed["large_pt"].append(pt)

    return removed


def format_counting_cleanup_summary(removed: dict[str, Sequence[Path]]) -> str:
    """Return a concise human-readable summary for notebook cleanup output."""

    return (
        "Deleted cleanup artifacts before zipping: "
        f"{len(removed.get('corrupted_needle_tokens', []))} corrupted_needle_tokens.jsonl file(s), "
        f"{len(removed.get('attention_stats_inputs', []))} attention_stats/input_* folder(s), "
        f"{len(removed.get('qk_cache', []))} qk_cache folder(s), "
        f"{len(removed.get('large_pt', []))} .pt file(s) larger than 100 MB."
    )


def _write_ablation_manifest(
    *,
    run_dir: Path,
    dataset_path: Path,
    selected_example_ids: Sequence[int],
    summaries: Sequence[dict[str, Any]],
) -> Path:
    payload = {
        "schema_version": "counting_ablation_manifest_v1",
        "dataset_path": str(dataset_path),
        "selected_example_ids": [int(x) for x in selected_example_ids],
        "example_run_dirs": [str(summary["paths"].run_dir) for summary in summaries],
        "summaries": [
            {k: v for k, v in summary.items() if k != "paths"} for summary in summaries
        ],
    }
    path = run_dir / "ablation_examples" / "counting_ablation_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return path


def _load_existing_dynamic_config(
    config_path: Path, *, dataset_path: Path, run_dir: Path
) -> DynamicNiahV2Config:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload.setdefault("data_save_path", str(dataset_path))
    payload.setdefault("output_dir", str(dataset_path.parent))
    payload.setdefault("run_dir", str(run_dir))
    return DynamicNiahV2Config(**dynamic_niah_v2_config_kwargs(payload))


def _merge_layers(
    base_layers: Sequence[int] | None, extra_layers: Sequence[int]
) -> list[int] | None:
    if base_layers is None:
        return None
    return sorted(
        {
            *[int(layer) for layer in base_layers],
            *[int(layer) for layer in extra_layers],
        }
    )


def _missing_required_qk_tables(
    *, paths: SingleExamplePaths, patterns: Sequence[str]
) -> list[Path]:
    required = {
        filename
        for pattern in patterns
        for filename in QK_PATTERN_REQUIRED_TABLES.get(str(pattern), ())
    }
    return [
        paths.tables_dir / filename
        for filename in sorted(required)
        if not (paths.tables_dir / filename).exists()
    ]


def _counting_qk_requirements(
    *,
    run_ablation: bool,
    ablation_config_path: str | Path,
    num_critical_tokens: int | None,
    ablation_random_seed: int | None,
    critical_token_calc_layer: int | None,
    run_representation_ablation: bool,
    representation_config_path: str | Path,
    representation_num_critical_tokens: int | None,
    randomize_from_top_layer: bool | None,
    run_representation_restore: bool,
    representation_restore_config_path: str | Path,
    restore_num_critical_tokens: int | None,
    restore_randomize_from_top_layer: bool | None,
) -> tuple[list[int], tuple[str, ...]]:
    """Return Q/K layers and patterns required by enabled ablation configs."""

    layers: set[int] = set()
    patterns: set[str] = set()

    if run_ablation:
        cfg = load_ablation_config(
            ablation_config_path,
            num_critical_tokens=num_critical_tokens,
            ablation_random_seed=ablation_random_seed,
            critical_token_calc_layer=critical_token_calc_layer,
        )
        qk_patterns = set(cfg.patterns) & set(QK_PATTERN_REQUIRED_TABLES)
        if qk_patterns:
            layers.add(int(cfg.critical_token_calc_layer))
            patterns.update(qk_patterns)

    if run_representation_ablation:
        cfg = load_representation_ablation_config(
            representation_config_path,
            num_critical_tokens=representation_num_critical_tokens,
            randomize_from_top_layer=randomize_from_top_layer,
            ablation_random_seed=ablation_random_seed,
            critical_token_calc_layer=critical_token_calc_layer,
        )
        qk_patterns = set(cfg.patterns) & set(QK_PATTERN_REQUIRED_TABLES)
        if qk_patterns:
            layers.add(int(cfg.critical_token_calc_layer))
            patterns.update(qk_patterns)

    if run_representation_restore:
        cfg = load_representation_restore_config(
            representation_restore_config_path,
            num_critical_tokens=restore_num_critical_tokens,
            randomize_from_top_layer=restore_randomize_from_top_layer,
            ablation_random_seed=ablation_random_seed,
            critical_token_calc_layer=critical_token_calc_layer,
        )
        qk_patterns = set(cfg.patterns) & set(QK_PATTERN_REQUIRED_TABLES)
        if qk_patterns:
            layers.add(int(cfg.critical_token_calc_layer))
            patterns.update(qk_patterns)

    return sorted(layers), tuple(sorted(patterns))


def _copy_shared_representation_stats_if_available(
    *, shared_stats_path: Path, paths: SingleExamplePaths
) -> None:
    if not shared_stats_path.exists():
        return
    target = paths.tensors_dir / "ablation_representation" / shared_stats_path.name
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(shared_stats_path, target)


def _capture_shared_representation_stats(
    *, shared_stats_path: Path, paths: SingleExamplePaths
) -> None:
    source = paths.tensors_dir / "ablation_representation" / shared_stats_path.name
    if shared_stats_path.exists() or not source.exists():
        return
    shared_stats_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, shared_stats_path)


def _copy_parent_qk_outlier_tables_if_available(
    *, parent_run_dir: Path, paths: SingleExamplePaths
) -> list[Path]:
    """Copy parent-run Q/K outlier tables needed by per-example ablations.

    Counting ablations execute in ``ablation_examples/example_id_*`` directories,
    while users may have already run Q/K outlier analysis against the parent
    counting run. Reusing those run-level tables avoids requiring an expensive
    per-example Q/K recapture just to select massive-activation and attention-sink
    tokens. Missing sources are allowed: the corresponding token selectors will
    return empty selections with a warning.
    """

    copied: list[Path] = []
    source_tables_dir = parent_run_dir / "tables"
    for filename in QK_OUTLIER_TABLE_FILENAMES:
        source = source_tables_dir / filename
        if not source.exists():
            continue
        target = paths.tables_dir / filename
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def _run_one_counting_example(
    *,
    row: dict[str, Any],
    example_id: int,
    dataset_path: Path,
    example_paths: SingleExamplePaths,
    model_name: str,
    config_path: Path,
    hidden_layers: Sequence[int] | None,
    run_ablation: bool,
    ablation_config_path: str | Path,
    num_critical_tokens: int | None,
    ablation_random_seed: int | None,
    critical_token_calc_layer: int | None,
    run_representation_ablation: bool,
    representation_config_path: str | Path,
    representation_num_critical_tokens: int | None,
    randomize_from_top_layer: bool | None,
    run_representation_restore: bool,
    representation_restore_config_path: str | Path,
    restore_num_critical_tokens: int | None,
    restore_randomize_from_top_layer: bool | None,
    shared_representation_stats_path: Path,
    analyze_reasoning_tokens: bool = False,
) -> dict[str, Any]:
    print(
        f"\n===== Counting ablation example {example_id}: {row.get('id')} =====",
        flush=True,
    )
    qk_layers, qk_patterns = _counting_qk_requirements(
        run_ablation=run_ablation,
        ablation_config_path=ablation_config_path,
        num_critical_tokens=num_critical_tokens,
        ablation_random_seed=ablation_random_seed,
        critical_token_calc_layer=critical_token_calc_layer,
        run_representation_ablation=run_representation_ablation,
        representation_config_path=representation_config_path,
        representation_num_critical_tokens=representation_num_critical_tokens,
        randomize_from_top_layer=randomize_from_top_layer,
        run_representation_restore=run_representation_restore,
        representation_restore_config_path=representation_restore_config_path,
        restore_num_critical_tokens=restore_num_critical_tokens,
        restore_randomize_from_top_layer=restore_randomize_from_top_layer,
    )
    analysis_hidden_layers = _merge_layers(hidden_layers, qk_layers)
    base_cfg = _load_existing_dynamic_config(
        config_path, dataset_path=dataset_path, run_dir=example_paths.run_dir
    )
    dataset_copy, cfg = prepare_single_example_dataset(
        row=row,
        example_id=example_id,
        dataset_path=dataset_path,
        paths=example_paths,
        model_name=model_name,
    )
    cfg = DynamicNiahV2Config(
        **{
            **asdict(cfg),
            "cache_dir": base_cfg.cache_dir,
            "trust_remote_code": base_cfg.trust_remote_code,
            "device_map": base_cfg.device_map,
            "torch_dtype": base_cfg.torch_dtype,
        }
    )
    print("Dataset source:", dataset_path, flush=True)
    print("Saved one-row ablation dataset:", dataset_copy, flush=True)

    model = None
    tokenizer = None
    qk_outlier_summary = None
    try:
        model, tokenizer = load_model_and_tokenizer(cfg, model_name=model_name)
        uncontrolled = render_and_tokenize_messages(
            tokenizer, row["uncontrolled_messages"], thinking_mode=cfg.thinking_mode
        )
        controlled = render_and_tokenize_messages(
            tokenizer, row["messages"], thinking_mode=cfg.thinking_mode
        )
        vertical_lines = None
        if analyze_reasoning_tokens:
            cot_payload = load_cot_payload(example_paths.run_dir.parent.parent / "tensors", example_id)
            uncontrolled_input_ids = cot_payload["input_ids"]
            controlled_input_ids, _ = build_controlled_extended_input(
                tokenizer=tokenizer,
                controlled_messages=row["messages"],
                thinking_mode=cfg.thinking_mode,
                reasoning_ids=cot_payload["reasoning_ids"],
            )
            cfg = DynamicNiahV2Config(
                **{
                    **asdict(cfg),
                    "max_new_tokens": int(cfg.max_new_tokens_for_cot),
                }
            )
            prompt_len = int(cot_payload["prompt_tokens"])
            extended_len = int(cot_payload["extended_input_tokens"])
            vertical_lines = [
                {"position": prompt_len, "label": "prompt ends / reasoning starts", "color": "cyan"},
                {"position": extended_len, "label": "reasoning ends / final answer starts", "color": "cyan"},
            ]
            print(
                f"Using CoT extended input for ablation: prompt_tokens={prompt_len} "
                f"extended_tokens={extended_len} max_new_tokens={cfg.max_new_tokens}",
                flush=True,
            )
        else:
            uncontrolled_input_ids = uncontrolled.input_ids
            controlled_input_ids = controlled.input_ids
        print(
            "Uncontrolled input shape:", tuple(uncontrolled_input_ids.shape), flush=True
        )
        print("Controlled input shape:", tuple(controlled_input_ids.shape), flush=True)

        needle_segments = locate_uncontrolled_needle_segments(
            row=row,
            uncontrolled_input_ids=uncontrolled_input_ids,
            prompt_text=uncontrolled.prompt_text,
            token_offsets=uncontrolled.token_offsets,
            expected_num_needles=sum(
                1
                for needle in row.get("needles", [])
                if needle.get("is_inserted", True)
            ),
        )
        input_metadata_path = save_input_metadata(
            path=example_paths.generate_data_dir / f"inputs_{example_id}.json",
            example_id=example_id,
            row=row,
            model_name=model_name,
            uncontrolled_input_ids=uncontrolled_input_ids,
            controlled_input_ids=controlled_input_ids,
            needle_segments=needle_segments,
        )
        print("Saved input metadata:", input_metadata_path, flush=True)

        hidden_outputs = compute_single_example_hidden_states(
            model=model,
            uncontrolled_input_ids=uncontrolled_input_ids,
            controlled_input_ids=controlled_input_ids,
            row=row,
            paths=example_paths,
            example_id=example_id,
            layers=analysis_hidden_layers,
            needle_segments=needle_segments,
            vertical_lines=vertical_lines,
        )
        for name, path in hidden_outputs.items():
            print(f"{name}: {path}", flush=True)

        missing_qk_tables = _missing_required_qk_tables(
            paths=example_paths, patterns=qk_patterns
        )
        if qk_layers and missing_qk_tables:
            print(
                "Running Q/K outlier analysis for counting ablation because "
                "required Q/K tables are missing:",
                [str(path) for path in missing_qk_tables],
                flush=True,
            )
            del model
            del tokenizer
            model = None
            tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            qk_outlier_summary = run_single_example_qk_outlier_analysis(
                paths=example_paths,
                layers=qk_layers,
                example_id=example_id,
                model_name=model_name,
            )
            print("Q/K outlier analysis summary:", qk_outlier_summary, flush=True)
            missing_qk_tables = _missing_required_qk_tables(
                paths=example_paths, patterns=qk_patterns
            )
            if missing_qk_tables:
                raise FileNotFoundError(
                    "Q/K outlier analysis completed but did not create required "
                    "table(s): "
                    + ", ".join(str(path) for path in missing_qk_tables)
                )
            model, tokenizer = load_model_and_tokenizer(cfg, model_name=model_name)

        ablation_summary = None
        if run_ablation:
            ablation_summary = run_single_example_ablation(
                paths=example_paths,
                row=row,
                example_id=example_id,
                model=model,
                tokenizer=tokenizer,
                uncontrolled_input_ids=uncontrolled_input_ids,
                needle_segments=needle_segments,
                config_path=ablation_config_path,
                num_critical_tokens=num_critical_tokens,
                ablation_random_seed=ablation_random_seed,
                critical_token_calc_layer=critical_token_calc_layer,
                dynamic_cfg=cfg,
            )
            print("Token-level ablation summary:", ablation_summary, flush=True)
        else:
            print(
                "Skipping token-level ablation because RUN_ABLATION=False", flush=True
            )

        representation_ablation_summary = None
        if run_representation_ablation:
            _copy_shared_representation_stats_if_available(
                shared_stats_path=shared_representation_stats_path, paths=example_paths
            )
            representation_ablation_summary = (
                run_single_example_representation_ablation(
                    paths=example_paths,
                    row=row,
                    dataset_path=dataset_path,
                    example_id=example_id,
                    model=model,
                    tokenizer=tokenizer,
                    uncontrolled_input_ids=uncontrolled_input_ids,
                    needle_segments=needle_segments,
                    config_path=representation_config_path,
                    num_critical_tokens=representation_num_critical_tokens,
                    randomize_from_top_layer=randomize_from_top_layer,
                    ablation_random_seed=ablation_random_seed,
                    critical_token_calc_layer=critical_token_calc_layer,
                    dynamic_cfg=cfg,
                )
            )
            _capture_shared_representation_stats(
                shared_stats_path=shared_representation_stats_path, paths=example_paths
            )
            print(
                "Representation-level ablation summary:",
                representation_ablation_summary,
                flush=True,
            )
        else:
            print(
                "Skipping representation-level ablation because RUN_REPRESENTATION_ABLATION=False",
                flush=True,
            )

        representation_restore_summary = None
        if run_representation_restore:
            representation_restore_summary = run_single_example_representation_restore(
                paths=example_paths,
                row=row,
                dataset_path=dataset_path,
                restore_dataset_run_name=None,
                example_id=example_id,
                model=model,
                tokenizer=tokenizer,
                uncontrolled_input_ids=uncontrolled_input_ids,
                needle_segments=needle_segments,
                config_path=representation_restore_config_path,
                num_critical_tokens=restore_num_critical_tokens,
                randomize_from_top_layer=restore_randomize_from_top_layer,
                ablation_random_seed=ablation_random_seed,
                critical_token_calc_layer=critical_token_calc_layer,
                dynamic_cfg=cfg,
            )
            print(
                "Representation-level restore summary:",
                representation_restore_summary,
                flush=True,
            )
        else:
            print(
                "Skipping representation-level restore because RUN_REPRESENTATION_RESTORE=False",
                flush=True,
            )
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "example_id": int(example_id),
        "row_id": row.get("id"),
        "paths": example_paths,
        "qk_outlier_summary": qk_outlier_summary,
        "ablation_summary": ablation_summary,
        "representation_ablation_summary": representation_ablation_summary,
        "representation_restore_summary": representation_restore_summary,
    }


def run_counting_ablation_examples(
    *,
    run_dir: str | Path,
    run_name: str,
    dataset_path: str | Path,
    config_path: str | Path,
    model_name: str,
    selected_example_ids: Sequence[int],
    num_examples: int,
    hidden_layers: Sequence[int] | None = None,
    run_ablation: bool = False,
    ablation_config_path: str | Path = DEFAULT_ABLATION_CONFIG_PATH,
    num_critical_tokens: int | None = None,
    ablation_random_seed: int | None = None,
    critical_token_calc_layer: int | None = None,
    run_representation_ablation: bool = False,
    representation_config_path: (
        str | Path
    ) = DEFAULT_REPRESENTATION_ABLATION_CONFIG_PATH,
    representation_num_critical_tokens: int | None = None,
    randomize_from_top_layer: bool | None = None,
    run_representation_restore: bool = False,
    representation_restore_config_path: (
        str | Path
    ) = DEFAULT_REPRESENTATION_RESTORE_CONFIG_PATH,
    restore_num_critical_tokens: int | None = None,
    restore_randomize_from_top_layer: bool | None = None,
    analyze_reasoning_tokens: bool = False,
) -> dict[str, Any]:
    """Run the requested ablation levels for selected counting examples."""

    selected = validate_selected_example_ids(
        selected_example_ids, num_examples=num_examples
    )
    if analyze_reasoning_tokens:
        base_cfg_for_check = _load_existing_dynamic_config(Path(config_path), dataset_path=Path(dataset_path), run_dir=Path(run_dir))
        if not base_cfg_for_check.thinking_mode:
            raise ValueError("analyze_reasoning_tokens=True requires thinking_mode=True")
    if not (run_ablation or run_representation_ablation or run_representation_restore):
        print(
            "No counting ablation level enabled; skipping selected examples.",
            flush=True,
        )
        return {"selected_example_ids": selected, "example_summaries": []}

    run_dir_path = Path(run_dir)
    dataset_path = Path(dataset_path)
    config_path = Path(config_path)
    rows = load_counting_dataset(dataset_path)
    if len(rows) < int(num_examples):
        raise ValueError(
            f"Generated dataset at {dataset_path} has {len(rows)} rows, expected at least NUM_EXAMPLES={num_examples}."
        )
    missing = [example_id for example_id in selected if example_id >= len(rows)]
    if missing:
        raise ValueError(
            f"Selected example ids are missing from dataset {dataset_path}: {missing}"
        )

    ablation_root = run_dir_path / "ablation_examples"
    ablation_root.mkdir(parents=True, exist_ok=True)
    shared_representation_stats_path = (
        ablation_root / "shared" / "hidden_state_distribution_stats.pt"
    )
    summaries: list[dict[str, Any]] = []
    for example_id in selected:
        row = rows[example_id]
        example_paths = create_single_example_run(
            row=row,
            example_id=example_id,
            model_name=model_name,
            run_root=ablation_root,
            user_run_name=f"example_id_{example_id}",
        )
        copied_qk_tables = _copy_parent_qk_outlier_tables_if_available(
            parent_run_dir=run_dir_path, paths=example_paths
        )
        if copied_qk_tables:
            print(
                "Copied parent Q/K outlier tables for counting ablation:",
                [str(path) for path in copied_qk_tables],
                flush=True,
            )
        summaries.append(
            _run_one_counting_example(
                row=row,
                example_id=example_id,
                dataset_path=dataset_path,
                example_paths=example_paths,
                model_name=model_name,
                config_path=config_path,
                hidden_layers=hidden_layers,
                run_ablation=run_ablation,
                ablation_config_path=ablation_config_path,
                num_critical_tokens=num_critical_tokens,
                ablation_random_seed=ablation_random_seed,
                critical_token_calc_layer=critical_token_calc_layer,
                run_representation_ablation=run_representation_ablation,
                representation_config_path=representation_config_path,
                representation_num_critical_tokens=representation_num_critical_tokens,
                randomize_from_top_layer=randomize_from_top_layer,
                run_representation_restore=run_representation_restore,
                representation_restore_config_path=representation_restore_config_path,
                restore_num_critical_tokens=restore_num_critical_tokens,
                restore_randomize_from_top_layer=restore_randomize_from_top_layer,
                shared_representation_stats_path=shared_representation_stats_path,
                analyze_reasoning_tokens=analyze_reasoning_tokens,
            )
        )

    aggregate_ablation_summary = None
    if run_ablation and summaries:
        aggregate_ablation_summary = summarize_ablation_results_all(
            run_dir=ablation_root,
            example_dirs=[summary["paths"].run_dir for summary in summaries],
        )
        print(
            "Saved selected-example token ablation summary:", aggregate_ablation_summary
        )

    manifest_path = _write_ablation_manifest(
        run_dir=run_dir_path,
        dataset_path=dataset_path,
        selected_example_ids=selected,
        summaries=summaries,
    )
    print("Saved counting ablation manifest:", manifest_path, flush=True)
    return {
        "selected_example_ids": selected,
        "example_summaries": summaries,
        "manifest_path": manifest_path,
        "aggregate_ablation_summary": aggregate_ablation_summary,
        "shared_representation_stats_path": (
            shared_representation_stats_path
            if shared_representation_stats_path.exists()
            else None
        ),
    }
