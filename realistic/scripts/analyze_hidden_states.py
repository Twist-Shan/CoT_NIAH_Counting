from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

from dataset_generation.chat_templates import apply_generation_chat_template
from dataset_generation.dynamic_niah_v2 import (
    DynamicNiahV2Config,
    build_uncontrolled_context,
    dynamic_niah_v2_config_kwargs,
    load_config_file,
    parse_insertion_position,
)
from dataset_generation.hidden_state_analysis import (
    build_prompt_needle_spans,
    build_uncontrolled_needle_insertions,
    compare_hidden_states,
    compute_alignment_offset,
    compute_needle_sensitive_tokens,
    expand_needle_segments,
    generate_control_dataset_with_logging,
    load_hidden_state_records,
    plot_measurements,
    plot_saved_hidden_pca,
    prune_large_pt_files,
    save_hidden_states,
    save_measurements,
    save_model_input_ids_table,
    save_needle_sensitive_outputs,
    write_dataset_jsonl,
)
from dataset_generation.niah_prompt_utils import (
    build_messages_easier,
    build_messages_vanilla,
    response_schema_for_task,
)
from dataset_generation.response_eval import canonical_task_type
from counting.cot import build_controlled_extended_input, load_cot_payload
from dataset_generation.run_utils import (
    archive_directory,
    create_run_paths,
    is_google_drive_path,
    tee_output,
)


def _build_tokenized_prompt(
    cfg: DynamicNiahV2Config, context: str, query: str
) -> tuple[torch.Tensor, str, torch.Tensor | None]:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name,
        trust_remote_code=cfg.trust_remote_code,
        cache_dir=cfg.cache_dir,
    )
    response_schema = response_schema_for_task(canonical_task_type(cfg.task_type))
    if cfg.prompt_style == "easier":
        messages = build_messages_easier(
            context,
            query,
            thinking_mode=cfg.thinking_mode,
            response_schema=response_schema,
            task_type=canonical_task_type(cfg.task_type),
            counting_needle_kind=cfg.counting_needle_kind,
            marker_text=cfg.marker_text,
        )
    elif cfg.prompt_style == "vanilla_no_cue":
        messages = build_messages_vanilla(
            context,
            query,
            thinking_mode=cfg.thinking_mode,
            response_schema=response_schema,
            task_type=canonical_task_type(cfg.task_type),
            counting_needle_kind=cfg.counting_needle_kind,
            marker_text=cfg.marker_text,
            include_reasoning_ban=False,
            include_memorization_instruction=False,
        )
    else:
        messages = build_messages_vanilla(
            context,
            query,
            thinking_mode=cfg.thinking_mode,
            response_schema=response_schema,
            task_type=canonical_task_type(cfg.task_type),
            counting_needle_kind=cfg.counting_needle_kind,
            marker_text=cfg.marker_text,
        )
    text = apply_generation_chat_template(
        tokenizer, messages, thinking_mode=cfg.thinking_mode
    )
    tokenize_kwargs: dict[str, object] = {"return_tensors": "pt"}
    if getattr(tokenizer, "is_fast", False):
        tokenize_kwargs["return_offsets_mapping"] = True
    encoded = tokenizer(text, **tokenize_kwargs)
    offsets = encoded.get("offset_mapping") if "offset_mapping" in encoded else None
    return encoded.input_ids, text, offsets


def _build_inputs(cfg: DynamicNiahV2Config, context: str, query: str) -> torch.Tensor:
    input_ids, _, _ = _build_tokenized_prompt(cfg, context, query)
    return input_ids


def _resolve_input_device(model: PreTrainedModel) -> torch.device:
    # Use the embedding matrix device because it is what input_ids index into first.
    return model.get_input_embeddings().weight.device


def _run_params(cfg: DynamicNiahV2Config) -> dict[str, object]:
    return {
        "task": cfg.task_type,
        "needle_kind": cfg.counting_needle_kind,
        "prompt": cfg.prompt_style,
        "len": cfg.target_haystack_tokens,
        "needles": cfg.num_needles,
    }


def _write_run_metadata(cfg: DynamicNiahV2Config, paths, model_name: str) -> None:
    payload = {
        "run_dir": str(paths.run_dir),
        "model": model_name,
        "params": _run_params(cfg),
        "resolved_config": asdict(cfg),
    }
    paths.metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _read_dataset_jsonl(path: str | Path) -> list[dict]:
    dataset_path = Path(path)
    rows = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Existing hidden-analysis dataset is empty: {dataset_path}")
    return rows


def _load_or_generate_hidden_rows(cfg: DynamicNiahV2Config) -> list[dict]:
    control_count = sum(bool(x) for x in (cfg.control_switch or []))
    if control_count != 1:
        raise ValueError(
            "Expected exactly one True in control_switch for hidden-state analysis"
        )
    if cfg.data_save_path is not None and Path(cfg.data_save_path).exists():
        rows = _read_dataset_jsonl(cfg.data_save_path)
        print(
            f"[hidden-analysis] loaded existing dataset rows={len(rows)} path={cfg.data_save_path}"
        )
        return rows
    return generate_control_dataset_with_logging(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/niah_dynamic.json")
    parser.add_argument("--model")
    parser.add_argument(
        "--task-type",
        choices=[
            "argmax",
            "count_avg",
            "count_average_score",
            "match_count",
            "literal_count",
        ],
    )
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--target-haystack-tokens", type=int)
    parser.add_argument("--num-needles", type=int)
    parser.add_argument("--positions", nargs="+", type=parse_insertion_position)
    parser.add_argument("--output-dir")
    parser.add_argument("--data-save-path")
    parser.add_argument("--results-root")
    parser.add_argument("--run-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--global-random-seed", type=int)
    parser.add_argument("--haystack-seed", type=int)
    parser.add_argument("--needle-seed", type=int)
    parser.add_argument("--save_data", type=lambda x: x.lower() == "true")
    parser.add_argument("--prompt-style", choices=["vanilla", "vanilla_no_cue", "easier"])
    parser.add_argument("--layers", nargs="*", type=int, default=None)
    parser.add_argument(
        "--pca-from-saved-hidden",
        action="store_true",
        help="Load hidden_inputs_*.pt from the run tensors directory and run train/test PCA plots only.",
    )
    parser.add_argument(
        "--pca-test-count",
        type=int,
        default=None,
        help="Number of earliest examples reserved for PCA plots; defaults to num_examples // 2.",
    )
    parser.add_argument("--pca-filter-top-frac", type=float, default=0.10)
    parser.add_argument("--pca-output-dir")
    parser.add_argument("--needle-sensitive-top-m", type=int, default=20)
    parser.add_argument("--needle-sensitive-expansion", type=int, default=5)
    parser.add_argument("--analyze-reasoning-tokens", action="store_true")
    parser.add_argument("--skip-needle-sensitive", action="store_true")
    parser.add_argument(
        "--control_switch",
        nargs="*",
        type=lambda x: x.lower() == "true",
        default=None,
        help=(
            "Boolean control flags aligned to needle index. Defaults to "
            "[True, False, ...] sized to num_needles."
        ),
    )
    args = parser.parse_args()

    payload = load_config_file(args.config) if args.config else {}
    kwargs = dict(payload)
    if args.task_type is not None:
        kwargs["task_type"] = args.task_type
    if args.num_examples is not None:
        kwargs["num_examples"] = args.num_examples
    if args.target_haystack_tokens is not None:
        kwargs["target_haystack_tokens"] = args.target_haystack_tokens
    if args.num_needles is not None:
        kwargs["num_needles"] = args.num_needles
    if args.positions is not None:
        kwargs["insertion_positions"] = tuple(args.positions)
    # Hidden-state analysis expects exactly one control needle by default.
    if args.control_switch is not None:
        kwargs["control_switch"] = args.control_switch
    if args.output_dir is not None:
        kwargs["output_dir"] = args.output_dir
    if args.data_save_path is not None:
        kwargs["data_save_path"] = args.data_save_path
    if args.results_root is not None:
        kwargs["results_root"] = args.results_root
    if args.run_dir is not None:
        kwargs["run_dir"] = args.run_dir
    if args.run_name is not None:
        kwargs["run_name"] = args.run_name
    if args.global_random_seed is not None:
        kwargs["global_random_seed"] = args.global_random_seed
    if args.haystack_seed is not None:
        kwargs["haystack_seed"] = args.haystack_seed
    if args.needle_seed is not None:
        kwargs["needle_seed"] = args.needle_seed
    if args.save_data is not None:
        kwargs["save_data"] = args.save_data
    if args.analyze_reasoning_tokens:
        kwargs["analyze_reasoning_tokens"] = True
    if args.prompt_style is not None:
        kwargs["prompt_style"] = args.prompt_style

    if kwargs.get("control_switch") is None:
        num_needles = int(kwargs.get("num_needles", DynamicNiahV2Config.num_needles))
        kwargs["control_switch"] = [True] + [False] * (num_needles - 1)

    cfg = DynamicNiahV2Config(
        **dynamic_niah_v2_config_kwargs(kwargs, warn_unknown=True)
    )
    if cfg.analyze_reasoning_tokens and not cfg.thinking_mode:
        raise ValueError("analyze_reasoning_tokens=True requires thinking_mode=True")
    extra_output_dir = cfg.output_dir
    extra_data_save_path = cfg.data_save_path
    config_layers = payload.get("layers")
    layers = args.layers if args.layers is not None else config_layers
    model_name = args.model or cfg.tokenizer_name

    paths = create_run_paths(
        results_root=cfg.results_root,
        model_name=model_name,
        params=_run_params(cfg),
        run_dir=cfg.run_dir,
        run_name=cfg.run_name,
    )
    primary_cfg = replace(
        cfg,
        output_dir=str(paths.generate_data_dir),
        data_save_path=str(paths.generate_data_dir / "dynamic_niah_v2.jsonl"),
        run_dir=str(paths.run_dir),
    )

    with tee_output(paths.logs_path):
        print(f"[run] run_dir={paths.run_dir}")
        print(f"[run] logs={paths.logs_path}")
        _write_run_metadata(primary_cfg, paths, model_name)
        print(f"[run] metadata={paths.metadata_path}")

        if args.pca_from_saved_hidden:
            records = load_hidden_state_records(paths.tensors_dir)
            pca_output_dir = (
                Path(args.pca_output_dir) if args.pca_output_dir else paths.figures_dir
            )
            pca_paths = plot_saved_hidden_pca(
                records,
                pca_output_dir,
                layers,
                test_count=args.pca_test_count,
                filter_top_frac=args.pca_filter_top_frac,
            )
            print(
                f"[hidden-analysis] saved {len(pca_paths)} PCA figures to {pca_output_dir}"
            )
            prune_large_pt_files(paths.tensors_dir, delete=False)
            return

        rows = _load_or_generate_hidden_rows(primary_cfg)

        run_config = {
            "config_path": args.config,
            "model": args.model,
            "num_examples": args.num_examples,
            "run_dir": str(paths.run_dir),
            "figures_dir": str(paths.figures_dir),
            "tensors_dir": str(paths.tensors_dir),
            "generate_data_dir": str(paths.generate_data_dir),
            "layers": layers,
            "control_switch": args.control_switch,
            "needle_sensitive_top_m": args.needle_sensitive_top_m,
            "needle_sensitive_expansion": args.needle_sensitive_expansion,
            "skip_needle_sensitive": args.skip_needle_sensitive,
            "analyze_reasoning_tokens": primary_cfg.analyze_reasoning_tokens,
            "pca_test_count": args.pca_test_count,
            "pca_test_count_resolved": (
                args.pca_test_count
                if args.pca_test_count is not None
                else primary_cfg.num_examples // 2
            ),
            "resolved_config": asdict(primary_cfg),
        }
        paths.analyze_config_path.write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[hidden-analysis] saved config={paths.analyze_config_path}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=primary_cfg.trust_remote_code,
            device_map=primary_cfg.device_map,
            torch_dtype=(
                (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
                if primary_cfg.torch_dtype == "bfloat16_if_cuda_else_float32"
                else getattr(torch, primary_cfg.torch_dtype)
            ),
            cache_dir=primary_cfg.cache_dir,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            primary_cfg.tokenizer_name,
            trust_remote_code=primary_cfg.trust_remote_code,
            cache_dir=primary_cfg.cache_dir,
        )

        copied_pt_paths: list[Path] = []
        copied_png_paths: list[Path] = []
        model_input_id_records: list[dict[str, object]] = []
        needle_sensitive_records: list[dict[str, object]] = []
        for i, row in enumerate(rows):
            control_idx = next(
                idx for idx, n in enumerate(row["needles"]) if n["is_control"]
            )
            context_control = row["context"]
            context = build_uncontrolled_context(row)

            inputs, prompt_text, token_offsets = _build_tokenized_prompt(
                primary_cfg, context, row["query"]
            )
            inputs_control, _, _ = _build_tokenized_prompt(
                primary_cfg, context_control, row["query"]
            )

            cot_payload = None
            vertical_lines = None
            if primary_cfg.analyze_reasoning_tokens:
                cot_payload = load_cot_payload(paths.tensors_dir, i)
                inputs = cot_payload["input_ids"]
                inputs_control, _controlled_cot_text = build_controlled_extended_input(
                    tokenizer=tokenizer,
                    controlled_messages=row["messages"],
                    thinking_mode=primary_cfg.thinking_mode,
                    reasoning_ids=cot_payload["reasoning_ids"],
                )
                prompt_len = int(cot_payload["prompt_tokens"])
                extended_len = int(cot_payload["extended_input_tokens"])
                vertical_lines = [
                    {"position": prompt_len, "label": "prompt ends / reasoning starts", "color": "cyan"},
                    {"position": extended_len, "label": "reasoning ends / final answer starts", "color": "cyan"},
                ]
                print(
                    f"[hidden-analysis] using CoT extended input sample={i} "
                    f"prompt_tokens={prompt_len} extended_tokens={extended_len}",
                    flush=True,
                )

            input_device = _resolve_input_device(model)
            inputs = inputs.to(input_device)
            inputs_control = inputs_control.to(input_device)

            model_input_id_records.append(
                {
                    "sample_idx": i,
                    "uncontrolled_input_ids": [
                        int(x) for x in inputs.detach().cpu().reshape(-1).tolist()
                    ],
                    "controlled_input_ids": [
                        int(x)
                        for x in inputs_control.detach().cpu().reshape(-1).tolist()
                    ],
                }
            )

            uncontrolled_insertions = build_uncontrolled_needle_insertions(
                row["realized_insertions"], row["needles"]
            )

            prompt_needle_spans = build_prompt_needle_spans(
                inputs,
                uncontrolled_insertions,
                prompt_text=prompt_text,
                token_offsets=token_offsets,
            )
            control_needle_id = row["needles"][control_idx]["needle_id"]
            control_uncontrolled_insertion = next(
                (
                    insertion
                    for insertion in uncontrolled_insertions
                    if insertion.get("needle_id") == control_needle_id
                ),
                uncontrolled_insertions[control_idx],
            )
            control_span = next(
                (span for span in prompt_needle_spans if span.get("is_control")),
                None,
            )
            insertion_position = (
                int(control_span["start"])
                if control_span is not None
                else int(control_uncontrolled_insertion["final_position"])
            )
            pca_start_position = (
                min(int(span["start"]) for span in prompt_needle_spans)
                if prompt_needle_spans
                else min(
                    int(insertion["final_position"])
                    for insertion in uncontrolled_insertions
                )
            )
            offset = compute_alignment_offset(
                inputs, inputs_control, insertion_position
            )
            with torch.no_grad():
                out = model(inputs, output_hidden_states=True)
                out_ctrl = model(inputs_control, output_hidden_states=True)

            hidden = torch.stack(list(out.hidden_states), dim=0)[..., :].squeeze(1)
            hidden_control = torch.stack(list(out_ctrl.hidden_states), dim=0)[
                ..., :
            ].squeeze(1)

            expanded_segments = expand_needle_segments(
                prompt_needle_spans,
                sequence_length=int(inputs.detach().cpu().reshape(-1).numel()),
                expansion=args.needle_sensitive_expansion,
            )
            if not args.skip_needle_sensitive:
                layer_indices = (
                    list(range(hidden.shape[0]))
                    if layers is None
                    else [int(x) for x in layers]
                )
                needle_sensitive_records.append(
                    {
                        "sample_idx": i,
                        "top_m": args.needle_sensitive_top_m,
                        "expanded_segment_extra": args.needle_sensitive_expansion,
                        "insertion_position": insertion_position,
                        "offset": offset,
                        "needle_spans": prompt_needle_spans,
                        "expanded_needle_segments": expanded_segments,
                        "layers": compute_needle_sensitive_tokens(
                            hidden,
                            hidden_control,
                            inputs,
                            layer_indices=layer_indices,
                            layer_labels=layer_indices,
                            expanded_segments=expanded_segments,
                            insertion_position=insertion_position,
                            offset=offset,
                            top_m=args.needle_sensitive_top_m,
                            decode_token=tokenizer.decode,
                        ),
                    }
                )

            measurements = compare_hidden_states(
                hidden, hidden_control, insertion_position, offset, layers
            )
            measurements["needle_spans"] = prompt_needle_spans
            measurements["expanded_needle_segments"] = expanded_segments
            if cot_payload is not None:
                measurements["cot_prompt_tokens"] = torch.tensor(int(cot_payload["prompt_tokens"]), dtype=torch.int64)
                measurements["cot_extended_input_tokens"] = torch.tensor(int(cot_payload["extended_input_tokens"]), dtype=torch.int64)
            pt_path = save_measurements(measurements, paths.tensors_dir, i)
            hidden_path = save_hidden_states(
                hidden,
                hidden_control,
                paths.tensors_dir,
                i,
                layers=layers,
                input_ids=inputs,
                input_ids_control=inputs_control,
                insertion_position=insertion_position,
                offset=offset,
                pca_start_position=pca_start_position,
                needle_spans=prompt_needle_spans,
                expanded_needle_segments=expanded_segments,
            )
            png_path = plot_measurements(
                measurements,
                paths.figures_dir,
                i,
                needle_spans=prompt_needle_spans,
                vertical_lines=vertical_lines,
            )
            copied_pt_paths.extend([pt_path, hidden_path])
            copied_png_paths.append(png_path)
            print(
                f"[hidden-analysis] saved sample={i} tensor={pt_path} hidden={hidden_path} "
                f"figure={png_path}"
            )

        input_ids_path = save_model_input_ids_table(
            model_input_id_records, paths.tables_dir
        )
        print(f"[hidden-analysis] saved model input ids={input_ids_path}")
        needle_sensitive_paths: tuple[Path, Path] | None = None
        if needle_sensitive_records:
            needle_sensitive_paths = save_needle_sensitive_outputs(
                needle_sensitive_records, paths.tables_dir
            )
            missing_needle_sensitive_paths = [
                path for path in needle_sensitive_paths if not path.exists()
            ]
            if missing_needle_sensitive_paths:
                raise FileNotFoundError(
                    "Needle-sensitive token output save failed; missing: "
                    + ", ".join(str(path) for path in missing_needle_sensitive_paths)
                )
            print(
                "[hidden-analysis] saved needle-sensitive tokens="
                f"{needle_sensitive_paths[0]} and {needle_sensitive_paths[1]}"
            )
        elif not args.skip_needle_sensitive:
            raise RuntimeError(
                "Needle-sensitive token generation was enabled, but no records were "
                "created. Check that the hidden-state analysis generated at least one "
                "controlled example, or rerun with --skip-needle-sensitive to opt out."
            )

        records = load_hidden_state_records(paths.tensors_dir)
        pca_output_dir = (
            Path(args.pca_output_dir) if args.pca_output_dir else paths.figures_dir
        )
        try:
            pca_paths = plot_saved_hidden_pca(
                records,
                pca_output_dir,
                layers,
                test_count=args.pca_test_count,
                filter_top_frac=args.pca_filter_top_frac,
            )
        except ValueError as exc:
            pca_paths = []
            print(f"[hidden-analysis] skipped PCA visualization: {exc}")
        copied_png_paths.extend(pca_paths)
        prune_large_pt_files(copied_pt_paths, delete=False)

        if extra_output_dir is not None:
            extra_root = Path(extra_output_dir)
            if is_google_drive_path(extra_root):
                archive_path = archive_directory(paths.run_dir, extra_root)
                print(
                    "[run] output_dir is on Google Drive; skipped per-file copying "
                    f"and moved one archive instead: {archive_path}"
                )
            else:
                extra_figures = extra_root / "figures"
                extra_tensors = extra_root / "tensors"
                extra_tables = extra_root / "tables"
                extra_generate_data = extra_root / "generate_data"
                extra_figures.mkdir(parents=True, exist_ok=True)
                extra_tensors.mkdir(parents=True, exist_ok=True)
                extra_tables.mkdir(parents=True, exist_ok=True)
                extra_generate_data.mkdir(parents=True, exist_ok=True)
                for path in copied_pt_paths:
                    shutil.copy2(path, extra_tensors / path.name)
                for path in copied_png_paths:
                    shutil.copy2(path, extra_figures / path.name)
                shutil.copy2(
                    paths.analyze_config_path,
                    extra_root / paths.analyze_config_path.name,
                )
                shutil.copy2(prompt_diff_path, extra_tables / prompt_diff_path.name)
                shutil.copy2(input_ids_path, extra_tables / input_ids_path.name)
                if needle_sensitive_paths is not None:
                    for path in needle_sensitive_paths:
                        shutil.copy2(path, extra_tables / path.name)
                if primary_cfg.save_data:
                    extra_jsonl = extra_generate_data / "dynamic_niah_v2.jsonl"
                    write_dataset_jsonl(rows, extra_jsonl)
                print(
                    f"[run] additional analysis copy saved because output_dir was specified: {extra_root}"
                )

        if extra_data_save_path is not None:
            if is_google_drive_path(extra_data_save_path):
                archive_path = archive_directory(
                    paths.generate_data_dir,
                    Path(extra_data_save_path).parent,
                    archive_name=Path(extra_data_save_path).stem
                    or f"{paths.run_dir.name}_generate_data",
                )
                print(
                    "[run] data_save_path is on Google Drive; skipped direct writes "
                    f"and moved one archive instead: {archive_path}"
                )
            else:
                extra_path = write_dataset_jsonl(rows, extra_data_save_path)
                print(
                    f"[run] additional dataset copy saved because data_save_path was specified: {extra_path}"
                )


if __name__ == "__main__":
    main()
