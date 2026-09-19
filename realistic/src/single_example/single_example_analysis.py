from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch

from dataset_generation.chat_templates import apply_generation_chat_template
from dataset_generation.dynamic_niah_v2 import (
    DynamicNiahV2Config,
    dynamic_niah_v2_config_kwargs,
    load_config_file,
    parse_insertion_position,
)
from dataset_generation.hidden_state_analysis import (
    build_prompt_needle_spans,
    build_uncontrolled_needle_insertions,
    compare_hidden_states,
    compute_alignment_offset,
    expand_needle_segments,
    plot_measurements,
    save_hidden_states,
    save_measurements,
    save_model_input_ids_table,
)
from dataset_generation.qk_hook_attention.outlier_analysis import (
    run_qk_outlier_analysis,
)
from dataset_generation.run_utils import archive_directory, build_run_name

DEFAULT_NIAH_EXAMPLE_ROOT = Path("data/niah-example")
DEFAULT_DATASET_FILENAME = "dynamic_niah_v2.jsonl"
DEFAULT_DATASET_PATH = DEFAULT_NIAH_EXAMPLE_ROOT / DEFAULT_DATASET_FILENAME
DEFAULT_RESULTS_ROOT = Path("results/single-example")


@dataclass(frozen=True)
class SingleExamplePaths:
    """Filesystem layout for one single-example run."""

    run_name: str
    run_dir: Path
    figures_dir: Path
    tensors_dir: Path
    generate_data_dir: Path
    tables_dir: Path
    logs_path: Path
    analyze_config_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class TokenizedPrompt:
    """Rendered chat-template text plus tokenized model input IDs."""

    input_ids: torch.Tensor
    prompt_text: str
    token_offsets: torch.Tensor | None


def resolve_niah_example_dataset_path(
    run_name: str | None = None,
    *,
    base_dir: str | Path = DEFAULT_NIAH_EXAMPLE_ROOT,
    dataset_filename: str = DEFAULT_DATASET_FILENAME,
) -> Path:
    """Resolve the Dynamic NIAH v2 JSONL path for the default or a named example run.

    The default dataset lives directly under ``data/niah-example``. Alternative
    datasets live under ``data/niah-example/{run_name}`` and must use the same
    Dynamic NIAH v2 JSONL filename.
    """

    root = Path(base_dir)
    selected_run = None if run_name is None else str(run_name).strip()
    dataset_path = (
        root / dataset_filename
        if not selected_run
        else root / selected_run / dataset_filename
    )
    if not dataset_path.exists():
        run_hint = (
            "default dataset" if not selected_run else f"dataset run {selected_run!r}"
        )
        raise FileNotFoundError(
            f"Could not find Dynamic NIAH v2 JSONL for {run_hint}: {dataset_path}. "
            f"Expected filename: {dataset_filename}."
        )
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Resolved Dynamic NIAH v2 dataset path is not a file: {dataset_path}"
        )
    return dataset_path


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset JSONL does not exist: {dataset_path}")
    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {dataset_path}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected object on line {line_no} of {dataset_path}, got {type(row).__name__}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Dataset JSONL contains no rows: {dataset_path}")
    return rows


def load_jsonl_example(
    dataset_path: str | Path = DEFAULT_DATASET_PATH, example_id: int = 0
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load one row by zero-based example index from the Dynamic NIAH v2 JSONL."""

    rows = _read_jsonl(dataset_path)
    idx = int(example_id)
    if idx < 0 or idx >= len(rows):
        raise IndexError(
            f"EXAMPLE_ID={idx} is out of range for {dataset_path}; valid range is 0..{len(rows) - 1}"
        )
    return rows[idx], rows


def _load_config_near_dataset(dataset_path: str | Path) -> dict[str, Any]:
    config_path = Path(dataset_path).parent / "config.used.json"
    if not config_path.exists():
        return {}
    return load_config_file(config_path)


def build_single_example_config(
    row: dict[str, Any],
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_name: str | None = None,
    run_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    data_save_path: str | Path | None = None,
) -> DynamicNiahV2Config:
    """Build a DynamicNiahV2Config suitable for analysis of a saved row."""

    config_kwargs = _load_config_near_dataset(dataset_path)
    controls = row.get("controls") if isinstance(row.get("controls"), dict) else {}
    haystack = row.get("haystack") if isinstance(row.get("haystack"), dict) else {}
    tokenizer = row.get("tokenizer") if isinstance(row.get("tokenizer"), dict) else {}

    config_kwargs.update(
        {
            "task_type": row.get("task_type", config_kwargs.get("task_type", "argmax")),
            "tokenizer_name": model_name
            or tokenizer.get("name")
            or config_kwargs.get("tokenizer_name", DynamicNiahV2Config.tokenizer_name),
            "target_haystack_tokens": int(
                haystack.get(
                    "target_haystack_tokens",
                    config_kwargs.get(
                        "target_haystack_tokens",
                        DynamicNiahV2Config.target_haystack_tokens,
                    ),
                )
            ),
            "num_needles": len(row.get("needles", []))
            or int(config_kwargs.get("num_needles", DynamicNiahV2Config.num_needles)),
            "insertion_positions": tuple(
                parse_insertion_position(x)
                for x in controls.get(
                    "insertion_positions",
                    config_kwargs.get(
                        "insertion_positions", DynamicNiahV2Config.insertion_positions
                    ),
                )
            ),
            "prompt_style": controls.get(
                "prompt_style",
                config_kwargs.get("prompt_style", DynamicNiahV2Config.prompt_style),
            ),
            "thinking_mode": bool(
                controls.get(
                    "thinking_mode",
                    config_kwargs.get(
                        "thinking_mode", DynamicNiahV2Config.thinking_mode
                    ),
                )
            ),
            "control_switch": list(
                controls.get(
                    "control_switch",
                    config_kwargs.get("control_switch")
                    or [
                        bool(n.get("is_control", False)) for n in row.get("needles", [])
                    ],
                )
            ),
            "num_examples": 1,
            "save_data": True,
        }
    )
    if run_dir is not None:
        config_kwargs["run_dir"] = str(run_dir)
    if output_dir is not None:
        config_kwargs["output_dir"] = str(output_dir)
    if data_save_path is not None:
        config_kwargs["data_save_path"] = str(data_save_path)
    return DynamicNiahV2Config(**dynamic_niah_v2_config_kwargs(config_kwargs))


def _run_params(row: dict[str, Any], example_id: int) -> dict[str, Any]:
    controls = row.get("controls") if isinstance(row.get("controls"), dict) else {}
    haystack = row.get("haystack") if isinstance(row.get("haystack"), dict) else {}
    return {
        "task": row.get("task_type", "unknown"),
        "example": int(example_id),
        "prompt": controls.get("prompt_style", "unknown"),
        "len": haystack.get("target_haystack_tokens", "unknown"),
        "needles": len(row.get("needles", [])),
    }


def create_single_example_run(
    *,
    row: dict[str, Any],
    example_id: int,
    model_name: str,
    run_root: str | Path = "/content",
    user_run_name: str | None = None,
    start_time: datetime | None = None,
) -> SingleExamplePaths:
    """Create the /content/{RUN_NAME} directory tree for one example."""

    run_name = user_run_name or build_run_name(
        model_name=model_name,
        params=_run_params(row, example_id),
        start_time=start_time,
    )
    run_dir = Path(run_root) / run_name
    paths = SingleExamplePaths(
        run_name=run_name,
        run_dir=run_dir,
        figures_dir=run_dir / "figures",
        tensors_dir=run_dir / "tensors",
        generate_data_dir=run_dir / "generate_data",
        tables_dir=run_dir / "tables",
        logs_path=run_dir / "logs.txt",
        analyze_config_path=run_dir / "analyze_hidden_states_config.json",
        metadata_path=run_dir / "run_metadata.json",
    )
    for path in [
        paths.run_dir,
        paths.figures_dir,
        paths.tensors_dir,
        paths.generate_data_dir,
        paths.tables_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def prepare_single_example_dataset(
    *,
    row: dict[str, Any],
    example_id: int,
    dataset_path: str | Path,
    paths: SingleExamplePaths,
    model_name: str,
) -> tuple[Path, DynamicNiahV2Config]:
    """Write the selected row and metadata into the run's generate_data directory."""

    out_path = paths.generate_data_dir / "dynamic_niah_v2.jsonl"
    out_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    source_schema = Path(dataset_path).parent / "dataset.schema.json"
    if source_schema.exists():
        shutil.copyfile(source_schema, paths.generate_data_dir / "dataset.schema.json")

    cfg = build_single_example_config(
        row,
        dataset_path=dataset_path,
        model_name=model_name,
        run_dir=paths.run_dir,
        output_dir=paths.generate_data_dir,
        data_save_path=out_path,
    )
    config_payload = asdict(cfg)
    (paths.generate_data_dir / "config.used.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metadata = {
        "run_dir": str(paths.run_dir),
        "run_name": paths.run_name,
        "model": model_name,
        "example_id": int(example_id),
        "source_dataset_path": str(dataset_path),
        "selected_row_id": row.get("id"),
        "params": _run_params(row, example_id),
        "resolved_config": config_payload,
    }
    paths.metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths.analyze_config_path.write_text(
        json.dumps(
            {
                **metadata,
                "figures_dir": str(paths.figures_dir),
                "tensors_dir": str(paths.tensors_dir),
                "generate_data_dir": str(paths.generate_data_dir),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out_path, cfg


def render_and_tokenize_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    thinking_mode: bool = False,
) -> TokenizedPrompt:
    """Apply the generation chat template and tokenize a saved message list."""

    prompt_text = apply_generation_chat_template(
        tokenizer, messages, thinking_mode=thinking_mode
    )
    tokenize_kwargs: dict[str, Any] = {"return_tensors": "pt"}
    if getattr(tokenizer, "is_fast", False):
        tokenize_kwargs["return_offsets_mapping"] = True
    encoded = tokenizer(prompt_text, **tokenize_kwargs)
    offsets = encoded.get("offset_mapping") if "offset_mapping" in encoded else None
    return TokenizedPrompt(
        input_ids=encoded.input_ids,
        prompt_text=prompt_text,
        token_offsets=offsets,
    )


def _as_consecutive_positions(start: int, end: int) -> list[int]:
    if end <= start:
        raise ValueError(
            f"Needle segment must be non-empty; got start={start}, end={end}"
        )
    return list(range(start, end))


def locate_uncontrolled_needle_segments(
    *,
    row: dict[str, Any],
    uncontrolled_input_ids: torch.Tensor | list[int],
    prompt_text: str | None = None,
    token_offsets: torch.Tensor | list[Any] | None = None,
    expected_num_needles: int | None = 3,
) -> list[dict[str, Any]]:
    """Find and validate original needle token spans in the uncontrolled model input."""

    insertions = build_uncontrolled_needle_insertions(
        row.get("realized_insertions", []), row.get("needles", [])
    )
    spans = build_prompt_needle_spans(
        uncontrolled_input_ids,
        insertions,
        prompt_text=prompt_text,
        token_offsets=token_offsets,
    )
    if expected_num_needles is not None and len(spans) != int(expected_num_needles):
        raise ValueError(
            f"Expected {expected_num_needles} uncontrolled needle spans, found {len(spans)}"
        )
    segments: list[dict[str, Any]] = []
    previous_end = -1
    for span in sorted(spans, key=lambda item: int(item["start"])):
        start = int(span["start"])
        end = int(span["end"])
        if start < previous_end:
            raise ValueError(f"Needle spans overlap or are out of order: {spans}")
        positions = _as_consecutive_positions(start, end)
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError(f"Needle positions are not consecutive for span {span}")
        previous_end = end
        segment = dict(span)
        segment["positions"] = positions
        segment["length"] = len(positions)
        segments.append(segment)
    return segments


def _flat_input_ids(input_ids: torch.Tensor | list[int]) -> list[int]:
    if isinstance(input_ids, torch.Tensor):
        return [int(x) for x in input_ids.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in input_ids]


def save_input_metadata(
    *,
    path: str | Path,
    example_id: int,
    row: dict[str, Any],
    model_name: str,
    uncontrolled_input_ids: torch.Tensor | list[int],
    controlled_input_ids: torch.Tensor | list[int],
    needle_segments: list[dict[str, Any]],
) -> Path:
    """Save tokenized input IDs and needle segments for inspection."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "single_example_inputs_v1",
        "example_id": int(example_id),
        "row_id": row.get("id"),
        "model_name": model_name,
        "uncontrolled_input_ids": _flat_input_ids(uncontrolled_input_ids),
        "controlled_input_ids": _flat_input_ids(controlled_input_ids),
        "needle_segments": needle_segments,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path


def _model_input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def _torch_dtype_from_config(cfg: DynamicNiahV2Config) -> torch.dtype:
    if cfg.torch_dtype == "bfloat16_if_cuda_else_float32":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, cfg.torch_dtype)


def compute_single_example_hidden_states(
    *,
    model: Any,
    uncontrolled_input_ids: torch.Tensor,
    controlled_input_ids: torch.Tensor,
    row: dict[str, Any],
    paths: SingleExamplePaths,
    example_id: int,
    layers: Sequence[int] | None = None,
    needle_segments: list[dict[str, Any]] | None = None,
    vertical_lines: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Run model hidden-state comparison for one controlled/uncontrolled prompt pair."""

    input_device = _model_input_device(model)
    inputs = uncontrolled_input_ids.to(input_device)
    inputs_control = controlled_input_ids.to(input_device)
    segments = needle_segments or locate_uncontrolled_needle_segments(
        row=row, uncontrolled_input_ids=uncontrolled_input_ids
    )
    insertion_position = min(int(segment["start"]) for segment in segments)
    control_segment = next(
        (s for s in segments if bool(s.get("is_control", False))), None
    )
    if control_segment is not None:
        insertion_position = int(control_segment["start"])
    pca_start_position = min(int(segment["start"]) for segment in segments)
    offset = compute_alignment_offset(inputs, inputs_control, insertion_position)

    with torch.no_grad():
        out = model(inputs, output_hidden_states=True)
        out_control = model(inputs_control, output_hidden_states=True)
    hidden = torch.stack(list(out.hidden_states), dim=0).squeeze(1)
    hidden_control = torch.stack(list(out_control.hidden_states), dim=0).squeeze(1)

    selected_layers = None if layers is None else [int(layer) for layer in layers]
    expanded_segments = expand_needle_segments(
        segments,
        sequence_length=int(uncontrolled_input_ids.detach().cpu().reshape(-1).numel()),
        expansion=0,
    )
    measurements = compare_hidden_states(
        hidden, hidden_control, insertion_position, offset, selected_layers
    )
    measurements["needle_spans"] = segments
    measurements["expanded_needle_segments"] = expanded_segments
    measurements_path = save_measurements(measurements, paths.tensors_dir, example_id)
    hidden_path = save_hidden_states(
        hidden,
        hidden_control,
        paths.tensors_dir,
        example_id,
        layers=selected_layers,
        input_ids=uncontrolled_input_ids,
        input_ids_control=controlled_input_ids,
        insertion_position=insertion_position,
        offset=offset,
        pca_start_position=pca_start_position,
        needle_spans=segments,
        expanded_needle_segments=expanded_segments,
    )
    figure_path = plot_measurements(
        measurements,
        paths.figures_dir,
        example_id,
        needle_spans=segments,
        vertical_lines=vertical_lines,
    )
    input_ids_path = save_model_input_ids_table(
        [
            {
                "sample_idx": int(example_id),
                "uncontrolled_input_ids": _flat_input_ids(uncontrolled_input_ids),
                "controlled_input_ids": _flat_input_ids(controlled_input_ids),
            }
        ],
        paths.tables_dir,
    )
    return {
        "measurements": measurements_path,
        "hidden": hidden_path,
        "figure": figure_path,
        "input_ids_table": input_ids_path,
    }


def load_model_and_tokenizer(
    cfg: DynamicNiahV2Config, *, model_name: str | None = None
) -> tuple[Any, Any]:
    """Load the configured model and tokenizer for Colab execution."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_model = model_name or cfg.tokenizer_name
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        trust_remote_code=cfg.trust_remote_code,
        device_map=cfg.device_map,
        torch_dtype=_torch_dtype_from_config(cfg),
        cache_dir=cfg.cache_dir,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name,
        trust_remote_code=cfg.trust_remote_code,
        cache_dir=cfg.cache_dir,
    )
    return model, tokenizer


def run_single_example_qk_outlier_analysis(
    *,
    paths: SingleExamplePaths,
    layers: Sequence[int],
    example_id: int,
    model_name: str | None = None,
    repo_root: str | Path = ".",
    heads: Sequence[int] | None = None,
    force_capture: bool = False,
    capture_attn_implementation: str = "sdpa",
) -> dict[str, Any]:
    """Run Q/K outlier analysis using the notebook-selected example id."""

    return run_qk_outlier_analysis(
        run_dir=paths.run_dir,
        layers=layers,
        heads=heads,
        model=model_name,
        repo_root=repo_root,
        force_capture=force_capture,
        capture_attn_implementation=capture_attn_implementation,
        example_indices=[int(example_id)],
    )


def cleanup_large_tensor_artifacts(
    paths: SingleExamplePaths,
    *,
    example_id: int,
    remove_qk_cache: bool = True,
    remove_hidden_states: bool = True,
    max_pt_file_size_mb: int | None = 200,
) -> list[Path]:
    """Delete large intermediate hidden-state and Q/K cache tensors."""

    removed: list[Path] = []
    if remove_hidden_states:
        hidden_paths = [
            paths.tensors_dir / f"hidden_inputs_{int(example_id)}.pt",
            paths.tensors_dir
            / "ablation_representation"
            / "hidden_state_distribution_stats.pt",
            paths.tensors_dir
            / "ablation_representation"
            / f"hidden_states_unablated_{int(example_id)}.pt",
            paths.tensors_dir
            / "ablation_representation_restore"
            / f"hidden_states_clean_{int(example_id)}.pt",
        ]
        for hidden_path in hidden_paths:
            if hidden_path.exists():
                hidden_path.unlink()
                removed.append(hidden_path)
    if max_pt_file_size_mb is not None:
        size_limit = int(max_pt_file_size_mb) * 1024 * 1024
        for artifact_path in paths.run_dir.rglob("*.pt"):
            if artifact_path.is_file() and artifact_path.stat().st_size > size_limit:
                artifact_path.unlink()
                removed.append(artifact_path)
    if remove_qk_cache:
        for relative in [
            "qk_cache",
            "attention_stats",
            "massive_activations",
            "attention_scores",
        ]:
            candidate = paths.tensors_dir / relative
            if candidate.exists():
                if candidate.is_dir():
                    shutil.rmtree(candidate)
                else:
                    candidate.unlink()
                removed.append(candidate)
    return removed


def zip_single_example_results(
    *,
    paths: SingleExamplePaths,
    results_path: str | Path | None = None,
) -> Path:
    """Zip a run directory and move the archive to the requested results path."""

    # These representation-ablation tensors are useful only as intermediates and
    # can make the final Colab archive too large. Remove them immediately before
    # zipping so reruns that skipped cleanup do not upload oversized archives. Also
    # drop any remaining .pt file larger than 200 MB, as requested for Colab runs.
    for pattern in (
        "**/tensors/ablation_representation/hidden_state_distribution_stats.pt",
        "**/tensors/ablation_representation/hidden_states_unablated_*.pt",
        "**/tensors/ablation_representation_restore/hidden_states_clean_*.pt",
    ):
        for artifact_path in paths.run_dir.glob(pattern):
            if artifact_path.is_file():
                artifact_path.unlink()
    size_limit = 200 * 1024 * 1024
    for artifact_path in paths.run_dir.rglob("*.pt"):
        if artifact_path.is_file() and artifact_path.stat().st_size > size_limit:
            artifact_path.unlink()

    destination = (
        Path(results_path)
        if results_path is not None
        else DEFAULT_RESULTS_ROOT / paths.run_name
    )
    return archive_directory(paths.run_dir, destination, archive_name=paths.run_name)
