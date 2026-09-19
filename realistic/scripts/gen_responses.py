from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

from dataset_generation.chat_templates import apply_generation_chat_template
from dataset_generation.dynamic_niah_v2 import (
    DynamicNiahV2Config,
    build_prediction_messages,
    generate_dynamic_niah_dataset_v2,
    write_dynamic_niah_v2,
    dynamic_niah_v2_config_kwargs,
    load_config_file,
    parse_insertion_position,
)
from dataset_generation.response_eval import (
    build_response_result,
    summarize_results,
    write_jsonl,
)
from counting.cot import save_cot_generation_artifacts, write_cot_info
from counting.feature_analysis import print_gpu_memory_snapshot, release_torch_memory
from dataset_generation.run_utils import (
    archive_directory,
    create_run_paths,
    is_google_drive_path,
    localize_runtime_path,
    tee_output,
)


def _run_params(cfg: DynamicNiahV2Config) -> dict[str, object]:
    return {
        "task": cfg.task_type,
        "prompt": cfg.prompt_style,
        "len": cfg.target_haystack_tokens,
        "needles": cfg.num_needles,
    }


def _resolve_input_device(model: PreTrainedModel) -> torch.device:
    return model.get_input_embeddings().weight.device


def _resolve_torch_dtype(cfg: DynamicNiahV2Config) -> torch.dtype:
    if cfg.torch_dtype == "bfloat16_if_cuda_else_float32":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, cfg.torch_dtype)


def _resolve_max_new_tokens(cfg: DynamicNiahV2Config) -> int:
    if cfg.max_new_tokens is not None:
        return int(cfg.max_new_tokens)
    return 1024 if cfg.thinking_mode else 64


def _tqdm(iterable: Any, *, total: int, desc: str) -> Any:
    try:
        from tqdm import tqdm

        return tqdm(
            iterable,
            total=total,
            desc=desc,
            dynamic_ncols=True,
            file=sys.stdout,
            ascii=True,
            mininterval=0.5,
            miniters=1,
            leave=True,
        )
    except Exception:
        return iterable


def _apply_cli_overrides(
    payload: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    kwargs = dict(payload)
    for attr, key in (
        ("num_examples", "num_examples"),
        ("target_haystack_tokens", "target_haystack_tokens"),
        ("num_needles", "num_needles"),
        ("tokenizer", "tokenizer_name"),
        ("output_dir", "output_dir"),
        ("data_save_path", "data_save_path"),
        ("output_pred_jsonl", "output_pred_jsonl"),
        ("output_metrics_json", "output_metrics_json"),
        ("results_root", "results_root"),
        ("run_dir", "run_dir"),
        ("run_name", "run_name"),
        ("prompt_style", "prompt_style"),
        ("task_type", "task_type"),
        ("max_new_tokens", "max_new_tokens"),
        ("max_new_tokens_for_cot", "max_new_tokens_for_cot"),
        ("temperature", "temperature"),
        ("global_random_seed", "global_random_seed"),
        ("haystack_seed", "haystack_seed"),
        ("needle_seed", "needle_seed"),
        ("fact_templates_path", "fact_templates_path"),
        ("counting_needle_kind", "counting_needle_kind"),
        ("marker_text", "marker_text"),
    ):
        value = getattr(args, attr)
        if value is not None:
            kwargs[key] = value
    if args.positions is not None:
        kwargs["insertion_positions"] = tuple(args.positions)
    if args.save_data is not None:
        kwargs["save_data"] = args.save_data
    if args.control_switch is not None:
        kwargs["control_switch"] = args.control_switch
    if getattr(args, "analyze_reasoning_tokens", False):
        kwargs["analyze_reasoning_tokens"] = True
    if getattr(args, "sentence_level_insertion", False):
        kwargs["sentence_level_insertion"] = True
    if getattr(args, "word_level_insertion", False):
        kwargs["word_level_insertion"] = True
    return kwargs


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and score model responses for dynamic NIAH v2 rows."
    )
    parser.add_argument("--config", default="configs/niah_dynamic.json")
    parser.add_argument("--model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--target-haystack-tokens", type=int)
    parser.add_argument("--num-needles", type=int)
    parser.add_argument("--positions", nargs="+", type=parse_insertion_position)
    parser.add_argument("--output-dir")
    parser.add_argument("--data-save-path")
    parser.add_argument("--output-pred-jsonl")
    parser.add_argument("--output-metrics-json")
    parser.add_argument("--results-root")
    parser.add_argument("--run-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--save_data", type=lambda x: x.lower() == "true")
    parser.add_argument("--prompt-style", choices=["vanilla", "vanilla_no_cue", "easier"])
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
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-new-tokens-for-cot", type=int)
    parser.add_argument(
        "--no-use-kv-cache-for-nonthinkg",
        action="store_true",
        help=(
            "Disable KV cache during non-thinking generation. The spelling "
            "matches USE_KV_CACHE_FOR_NONTHINKG in the counting-feature notebook."
        ),
    )
    parser.add_argument("--analyze-reasoning-tokens", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--global-random-seed", type=int)
    parser.add_argument("--haystack-seed", type=int)
    parser.add_argument("--needle-seed", type=int)
    parser.add_argument("--fact-templates-path")
    parser.add_argument("--sentence-level-insertion", action="store_true")
    parser.add_argument("--word-level-insertion", action="store_true")
    parser.add_argument("--counting-needle-kind", choices=["city_score", "marker"])
    parser.add_argument("--marker-text")
    parser.add_argument(
        "--control_switch", nargs="*", type=lambda x: x.lower() == "true"
    )
    args = parser.parse_args()

    payload = load_config_file(args.config) if args.config else {}
    cfg = DynamicNiahV2Config(
        **dynamic_niah_v2_config_kwargs(
            _apply_cli_overrides(payload, args), warn_unknown=True
        )
    )
    model_name = args.model or cfg.tokenizer_name
    if cfg.analyze_reasoning_tokens and not cfg.thinking_mode:
        raise ValueError("analyze_reasoning_tokens=True requires thinking_mode=True")

    paths = create_run_paths(
        results_root=cfg.results_root,
        model_name=model_name,
        params=_run_params(cfg),
        run_dir=cfg.run_dir,
        run_name=cfg.run_name,
    )
    requested_drive_outputs = [
        Path(p)
        for p in (
            cfg.data_save_path,
            cfg.output_pred_jsonl,
            cfg.output_metrics_json,
            cfg.output_dir,
        )
        if p and is_google_drive_path(p)
    ]
    data_path = (
        localize_runtime_path(cfg.data_save_path)
        if cfg.data_save_path
        else paths.generate_data_dir / "dynamic_niah_v2.jsonl"
    )
    pred_path = (
        localize_runtime_path(cfg.output_pred_jsonl)
        if cfg.output_pred_jsonl
        else paths.predictions_path
    )
    metrics_path = (
        localize_runtime_path(cfg.output_metrics_json)
        if cfg.output_metrics_json
        else paths.metrics_path
    )
    primary_cfg = replace(
        cfg,
        output_dir=str(paths.generate_data_dir),
        data_save_path=str(data_path),
        output_pred_jsonl=str(pred_path),
        output_metrics_json=str(metrics_path),
        run_dir=str(paths.run_dir),
        max_new_tokens=_resolve_max_new_tokens(cfg),
    )
    use_kv_cache_for_nonthinkg = not bool(args.no_use_kv_cache_for_nonthinkg)

    with tee_output(paths.logs_path):
        print(f"[run] run_dir={paths.run_dir}", flush=True)
        print(f"[run] logs={paths.logs_path}", flush=True)
        print(f"[gen-responses] dataset={primary_cfg.data_save_path}", flush=True)
        print(f"[gen-responses] predictions={primary_cfg.output_pred_jsonl}", flush=True)
        print(f"[gen-responses] metrics={primary_cfg.output_metrics_json}", flush=True)
        print(
            "[gen-responses] use_kv_cache_for_nonthinkg="
            f"{use_kv_cache_for_nonthinkg}",
            flush=True,
        )
        _write_run_metadata(primary_cfg, paths, model_name)
        print(f"[run] metadata={paths.metadata_path}", flush=True)

        rows = generate_dynamic_niah_dataset_v2(primary_cfg)
        dataset_paths = write_dynamic_niah_v2(rows, primary_cfg)
        print(
            f"[gen-responses] saved dataset rows={len(rows)} path={primary_cfg.data_save_path}",
            flush=True,
        )
        print(
            "[gen-responses] saved dataset metadata: "
            + json.dumps(dataset_paths, indent=2, ensure_ascii=False),
            flush=True,
        )

        print(
            f"[gen-responses] loading tokenizer={primary_cfg.tokenizer_name}",
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            primary_cfg.tokenizer_name,
            trust_remote_code=primary_cfg.trust_remote_code,
            cache_dir=primary_cfg.cache_dir,
        )
        print("[gen-responses] tokenizer loaded", flush=True)
        torch_dtype = _resolve_torch_dtype(primary_cfg)
        print(
            f"[gen-responses] loading model={model_name} "
            f"dtype={torch_dtype} device_map={primary_cfg.device_map}",
            flush=True,
        )
        print_gpu_memory_snapshot("gen-responses before model load")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=primary_cfg.trust_remote_code,
            device_map=primary_cfg.device_map,
            torch_dtype=torch_dtype,
            cache_dir=primary_cfg.cache_dir,
        )
        model.eval()
        input_device = _resolve_input_device(model)
        print_gpu_memory_snapshot("gen-responses after model load")
        print(
            f"[gen-responses] model loaded; input_device={input_device}; "
            f"starting generation for {len(rows)} examples",
            flush=True,
        )

        results: list[dict[str, Any]] = []
        cot_infos: list[dict[str, Any]] = []
        for idx, row in _tqdm(
            enumerate(rows), total=len(rows), desc="[gen-responses] generating"
        ):
            if idx == 0 or idx % 10 == 0:
                print_gpu_memory_snapshot(f"gen-responses before sample {idx}")
            eval_messages = build_prediction_messages(primary_cfg, row)
            text = apply_generation_chat_template(
                tokenizer, eval_messages, thinking_mode=primary_cfg.thinking_mode
            )
            inputs = tokenizer(text, return_tensors="pt").to(input_device)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": primary_cfg.max_new_tokens,
                "do_sample": primary_cfg.temperature > 0,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if not primary_cfg.thinking_mode and not use_kv_cache_for_nonthinkg:
                generation_kwargs["use_cache"] = False
            if primary_cfg.temperature > 0:
                generation_kwargs["temperature"] = primary_cfg.temperature

            with torch.inference_mode():
                out = model.generate(**inputs, **generation_kwargs)

            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = out[0][prompt_len:]
            if primary_cfg.analyze_reasoning_tokens:
                cot_info = save_cot_generation_artifacts(
                    tensors_dir=paths.tensors_dir,
                    tables_dir=paths.tables_dir,
                    example_id=idx,
                    row=row,
                    prompt_input_ids=inputs["input_ids"].detach().cpu(),
                    generated_ids=gen_ids.detach().cpu(),
                    tokenizer=tokenizer,
                    task_type=primary_cfg.task_type,
                    max_new_tokens=primary_cfg.max_new_tokens,
                    prompt_text=text,
                )
                cot_infos.append(cot_info)
                if cot_info.get("hit_max_new_tokens_without_eos"):
                    print(
                        "\n*** WARNING: CoT generation hit max_new_tokens without EOS "
                        f"for sample={idx}; max_new_tokens={primary_cfg.max_new_tokens} ***\n",
                        flush=True,
                    )
                if not cot_info.get("final_answer_found"):
                    print(
                        "\n*** WARNING: Could not locate final JSON answer start "
                        f"for sample={idx}; extended input includes all generated non-EOS tokens ***\n",
                        flush=True,
                    )
            model_output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            (paths.tables_dir / f"input_generate_{idx}.txt").write_text(
                text + "\n" + model_output, encoding="utf-8"
            )
            result = build_response_result(row, model_output)
            results.append(result)
            print(
                f"[gen-responses] sample={idx} id={row.get('id')} exact={result['exact_match']} "
                f"parse_mode={result.get('parse_mode')}",
                flush=True,
            )
            del inputs
            del out
            del gen_ids
            release_torch_memory(collect_garbage=True)
            if idx == 0 or idx % 10 == 0 or idx == len(rows) - 1:
                print_gpu_memory_snapshot(f"gen-responses after sample {idx}")

        if primary_cfg.analyze_reasoning_tokens:
            cot_info_path = write_cot_info(cot_infos, paths.tables_dir)
            print(
                f"[gen-responses] saved CoT extended input metadata to: {cot_info_path}",
                flush=True,
            )
        write_jsonl(results, primary_cfg.output_pred_jsonl)
        metrics = summarize_results(results) | {
            "model_name": model_name,
            "tokenizer_name": primary_cfg.tokenizer_name,
            "thinking_mode": primary_cfg.thinking_mode,
            "max_new_tokens": primary_cfg.max_new_tokens,
            "temperature": primary_cfg.temperature,
            "analyze_reasoning_tokens": primary_cfg.analyze_reasoning_tokens,
            "max_new_tokens_for_cot": primary_cfg.max_new_tokens_for_cot,
            "use_kv_cache_for_nonthinkg": use_kv_cache_for_nonthinkg,
            "dataset_path": primary_cfg.data_save_path,
            "output_predictions_file": primary_cfg.output_pred_jsonl,
            "output_metrics_json": primary_cfg.output_metrics_json,
            "run_dir": str(paths.run_dir),
        }
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
        print(
            f"[gen-responses] saved per-example outputs to: {primary_cfg.output_pred_jsonl}",
            flush=True,
        )
        print(
            f"[gen-responses] saved metrics to: {primary_cfg.output_metrics_json}",
            flush=True,
        )
        if cfg.output_dir and is_google_drive_path(cfg.output_dir):
            archive_path = archive_directory(paths.run_dir, cfg.output_dir)
            print(
                "[run] output_dir is on Google Drive; skipped per-file copying "
                f"and moved one archive instead: {archive_path}",
                flush=True,
            )
        elif requested_drive_outputs:
            archive_path = archive_directory(
                paths.run_dir, requested_drive_outputs[0].parent
            )
            print(
                "[run] one or more requested output files are on Google Drive; "
                f"kept runtime writes local and moved one archive instead: {archive_path}",
                flush=True,
            )

        print(
            "[next] python scripts/analyze_hidden_states.py "
            f"--config {args.config} --model {model_name} --run-dir {paths.run_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
