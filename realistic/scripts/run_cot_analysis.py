from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from counting.cot_analysis import (
    archive_cot_results,
    build_cot_analysis_run_config,
    cache_mode_outputs,
    cleanup_cot_mode_artifacts,
    create_cot_run_paths,
    dynamic_config_from_cot_config,
    ensure_selected_full_sequence_artifacts,
    generate_or_load_dataset,
    plot_cot_attention_tables,
    restore_mode_outputs_from_cache,
    run_generation_for_mode,
    run_mode_qk_outlier_analysis,
    save_selected_hidden_states_for_examples,
    select_analysis_example_ids,
    write_cot_response_checkpoint_archive,
    write_cot_attention_projection_tables,
    write_selected_mode_run,
)
from counting.feature_analysis import (
    StageTimer,
    print_gpu_memory_snapshot,
    release_torch_memory,
)


def _load_overrides(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
        return dict(payload["config"])
    if not isinstance(payload, dict):
        raise ValueError(f"Config override file must contain a JSON object: {path}")
    return {
        str(key): value
        for key, value in payload.items()
        if not str(key).startswith("_") and str(key) not in {"notes", "comments"}
    }


def _resolve_dtype(cfg: dict[str, Any]) -> torch.dtype:
    value = str(cfg["ANALYSIS_DTYPE"])
    if value == "bfloat16_if_cuda_else_float32":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, value)


def _load_model_and_tokenizer(cfg: dict[str, Any], *, attn_implementation: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["TOKENIZER_NAME"],
        trust_remote_code=True,
    )
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": _resolve_dtype(cfg),
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(cfg["MODEL_NAME"], **kwargs)
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CoT NIAH outlier/attention analysis.")
    parser.add_argument("--config", default=None, help="Optional JSON config override file.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="JSON object merged after --config, e.g. '{\"NUM_EXAMPLES\": 5}'.",
    )
    args = parser.parse_args()

    overrides = _load_overrides(args.config)
    for raw in args.override:
        item = json.loads(raw)
        if not isinstance(item, dict):
            raise ValueError("--override must decode to a JSON object")
        overrides.update(item)
    cfg = build_cot_analysis_run_config(overrides)
    paths = create_cot_run_paths(cfg)
    timer = StageTimer(
        json_path=paths.timing_json_path,
        csv_path=paths.timing_csv_path,
    )

    print("CoT analysis run:", paths.run_name, flush=True)
    print("Run dir:", paths.run_dir, flush=True)
    print("Modes:", cfg["THINKING_MODES"], flush=True)

    with timer.stage("dataset_generation_or_reuse"):
        rows = generate_or_load_dataset(
            cfg,
            paths,
            force=bool(cfg["FORCE_REGENERATE_DATASET"]),
        )
        print(f"Dataset rows: {len(rows)} path={paths.dataset_path}", flush=True)

    results_by_mode: dict[str, list[dict[str, Any]]] = {}
    metrics_by_mode: dict[str, dict[str, Any]] = {}
    if cfg["RUN_RESPONSE_GENERATION"]:
        with timer.stage("response_generation_and_scoring"):
            modes_to_generate: list[str] = []
            for mode in cfg["THINKING_MODES"]:
                if (
                    not paths.predictions_path(mode).exists()
                    or not paths.metrics_path(mode).exists()
                ) and not cfg["FORCE_REGENERATE_RESPONSES"]:
                    if restore_mode_outputs_from_cache(cfg=cfg, paths=paths, mode=mode):
                        print(
                            f"Restored cached responses for mode={mode} from data cache: {paths.cache_dir}",
                            flush=True,
                        )
                if (
                    paths.predictions_path(mode).exists()
                    and paths.metrics_path(mode).exists()
                    and not cfg["FORCE_REGENERATE_RESPONSES"]
                ):
                    print(f"Using existing responses for mode={mode}", flush=True)
                    results_by_mode[mode] = [
                        json.loads(line)
                        for line in paths.predictions_path(mode).read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    metrics_by_mode[mode] = json.loads(
                        paths.metrics_path(mode).read_text(encoding="utf-8")
                    )
                else:
                    modes_to_generate.append(mode)

            print(f"Modes needing generation: {modes_to_generate}", flush=True)
            if modes_to_generate:
                model, tokenizer = _load_model_and_tokenizer(cfg)
                print_gpu_memory_snapshot("cot-analysis after generation model load")
                try:
                    for mode in modes_to_generate:
                        results, metrics = run_generation_for_mode(
                            cfg=cfg,
                            paths=paths,
                            mode=mode,
                            rows=rows,
                            model=model,
                            tokenizer=tokenizer,
                        )
                        results_by_mode[mode] = results
                        metrics_by_mode[mode] = metrics
                        cache_dir = cache_mode_outputs(cfg=cfg, paths=paths, mode=mode)
                        print(f"Cached mode={mode} outputs to {cache_dir}", flush=True)
                finally:
                    del model
                    release_torch_memory(collect_garbage=True)
            else:
                print(
                    "All requested modes restored from run folder or stable data cache; model generation skipped.",
                    flush=True,
                )
    else:
        for mode in cfg["THINKING_MODES"]:
            results_by_mode[mode] = [
                json.loads(line)
                for line in paths.predictions_path(mode).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            metrics_by_mode[mode] = json.loads(paths.metrics_path(mode).read_text(encoding="utf-8"))

    if cfg["ZIP_RESULTS_AFTER_RESPONSE_STAGE"]:
        with timer.stage("response_checkpoint_zip"):
            write_cot_response_checkpoint_archive(paths=paths, modes=cfg["THINKING_MODES"])
            print("Response checkpoint zip:", paths.response_checkpoint_zip, flush=True)

    selected_ids = select_analysis_example_ids(
        rows=rows,
        results_by_mode=results_by_mode,
        max_examples=int(cfg["MAX_ANALYSIS_EXAMPLES"]),
    )
    print("Selected analysis example ids:", selected_ids, flush=True)

    if cfg["RUN_OUTLIER_ANALYSIS"] or cfg["RUN_ATTENTION_ANALYSIS"]:
        for mode in cfg["THINKING_MODES"]:
            with timer.stage(f"prepare_mode_run_{mode}"):
                write_selected_mode_run(
                    cfg=cfg,
                    paths=paths,
                    mode=mode,
                    rows=rows,
                    selected_ids=selected_ids,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    cfg["TOKENIZER_NAME"],
                    trust_remote_code=True,
                )
                materialized = ensure_selected_full_sequence_artifacts(
                    cfg=cfg,
                    paths=paths,
                    mode=mode,
                    rows=rows,
                    selected_ids=selected_ids,
                    tokenizer=tokenizer,
                )
                print(
                    f"Prepared {len(materialized)} full-sequence artifact(s) for mode={mode}",
                    flush=True,
                )

            with timer.stage(f"hidden_state_capture_{mode}"):
                model, _tokenizer = _load_model_and_tokenizer(cfg)
                try:
                    save_selected_hidden_states_for_examples(
                        model=model,
                        mode_dir=paths.mode_dir(mode),
                        example_ids=selected_ids,
                        layers=cfg["LAYERS"],
                        hidden_state_dtype=_resolve_dtype(cfg),
                    )
                finally:
                    del model
                    release_torch_memory(collect_garbage=True)

            with timer.stage(f"qk_outlier_attention_{mode}"):
                summary = run_mode_qk_outlier_analysis(
                    cfg=cfg,
                    paths=paths,
                    mode=mode,
                    selected_ids=selected_ids,
                )
                print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

            if cfg["RUN_ATTENTION_ANALYSIS"]:
                with timer.stage(f"cot_attention_tables_{mode}"):
                    table_paths = write_cot_attention_projection_tables(
                        cfg=cfg,
                        paths=paths,
                        mode=mode,
                        selected_ids=selected_ids,
                    )
                    print(f"Wrote {len(table_paths)} CoT attention table(s)", flush=True)
            if cfg["RUN_PLOTTING"]:
                with timer.stage(f"cot_attention_plots_{mode}"):
                    figure_paths = plot_cot_attention_tables(
                        cfg=cfg,
                        paths=paths,
                        mode=mode,
                        selected_ids=selected_ids,
                    )
                    print(f"Wrote {len(figure_paths)} CoT attention plot(s)", flush=True)

    with timer.stage("cleanup_before_archive"):
        removed = cleanup_cot_mode_artifacts(
            paths.run_dir,
            keep_hidden_states=bool(cfg["SAVE_FULL_HIDDEN_STATES"]),
            keep_full_sequence_tensors=bool(cfg["SAVE_FULL_HIDDEN_STATES"]),
        )
        print(
            "Cleanup removed: "
            + ", ".join(f"{key}={len(value)}" for key, value in removed.items()),
            flush=True,
        )

    if cfg["ZIP_RESULTS_AFTER_ANALYSIS_STAGE"]:
        with timer.stage("final_archive"):
            archive = archive_cot_results(paths, results_path=cfg["RESULTS_PATH"])
            print("Final archive:", archive, flush=True)


if __name__ == "__main__":
    main()
