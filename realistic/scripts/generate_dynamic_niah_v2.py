from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from dataset_generation.dynamic_niah_v2 import (
    DynamicNiahV2Config,
    generate_dynamic_niah_dataset_v2,
    dynamic_niah_v2_config_kwargs,
    load_config_file,
    parse_insertion_position,
    write_dynamic_niah_v2,
)
from dataset_generation.run_utils import (
    archive_directory,
    create_run_paths,
    is_google_drive_path,
    tee_output,
)


def _parse_needle_seeds(raw: str) -> dict[int, int | None] | None:
    if not raw:
        return None
    payload = json.loads(raw)
    out: dict[int, int | None] = {}
    for k, v in payload.items():
        out[int(k)] = None if v is None else int(v)
    return out


def _run_params(cfg: DynamicNiahV2Config) -> dict[str, object]:
    params: dict[str, object] = {
        "task": cfg.task_type,
        "needle_kind": cfg.counting_needle_kind,
        "prompt": cfg.prompt_style,
        "len": cfg.target_haystack_tokens,
        "needles": cfg.num_needles,
    }
    if cfg.num_max_needles is not None:
        params["num_max_needles"] = cfg.num_max_needles
    if cfg.task_type == "literal_count":
        params["uid_token_length"] = cfg.uid_token_length
    return params


def _write_run_metadata(cfg: DynamicNiahV2Config, paths) -> None:
    payload = {
        "run_dir": str(paths.run_dir),
        "model": cfg.tokenizer_name,
        "params": _run_params(cfg),
        "resolved_config": asdict(cfg),
    }
    paths.metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="")
    p.add_argument("--task-type")
    p.add_argument("--tokenizer")
    p.add_argument("--num-examples", type=int)
    p.add_argument("--target-haystack-tokens", type=int)
    p.add_argument("--num-needles", type=int)
    p.add_argument("--num-max-needles", type=int)
    p.add_argument("--positions", nargs="+", type=parse_insertion_position)
    p.add_argument("--randomize-needle-insertion", action="store_true")
    p.add_argument("--randomize-needle-seed", type=int)
    p.add_argument("--sentence-level-insertion", action="store_true")
    p.add_argument("--word-level-insertion", action="store_true")
    p.add_argument("--prompt-style", choices=["vanilla", "vanilla_no_cue", "easier"])
    p.add_argument("--thinking-mode", action="store_true")
    p.add_argument("--output-dir")
    p.add_argument("--data-save-path")
    p.add_argument("--results-root")
    p.add_argument("--run-dir")
    p.add_argument("--run-name")
    p.add_argument("--global-random-seed", type=int)
    p.add_argument("--haystack-seed", type=int)
    p.add_argument("--needle-seed", type=int)
    p.add_argument("--fact-templates-path")
    p.add_argument("--counting-needle-kind", choices=["city_score", "marker"])
    p.add_argument("--marker-text")
    p.add_argument("--uid-token-length", type=int)
    p.add_argument("--needle-seeds-json", default="")
    p.add_argument("--control-switch-json", default="")
    args = p.parse_args()

    kwargs = load_config_file(args.config) if args.config else {}
    if args.task_type is not None:
        kwargs["task_type"] = args.task_type
    if args.tokenizer is not None:
        kwargs["tokenizer_name"] = args.tokenizer
    if args.num_examples is not None:
        kwargs["num_examples"] = args.num_examples
    if args.target_haystack_tokens is not None:
        kwargs["target_haystack_tokens"] = args.target_haystack_tokens
    if args.num_needles is not None:
        kwargs["num_needles"] = args.num_needles
    if args.num_max_needles is not None:
        kwargs["num_max_needles"] = args.num_max_needles
    if args.positions is not None:
        kwargs["insertion_positions"] = tuple(args.positions)
    if args.randomize_needle_insertion:
        kwargs["randomize_needle_insertion"] = True
    if args.randomize_needle_seed is not None:
        kwargs["randomize_needle_seed"] = args.randomize_needle_seed
    if args.sentence_level_insertion:
        kwargs["sentence_level_insertion"] = True
    if args.word_level_insertion:
        kwargs["word_level_insertion"] = True
    if args.prompt_style is not None:
        kwargs["prompt_style"] = args.prompt_style
    if args.thinking_mode:
        kwargs["thinking_mode"] = True
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
    if args.fact_templates_path is not None:
        kwargs["fact_templates_path"] = args.fact_templates_path
    if args.counting_needle_kind is not None:
        kwargs["counting_needle_kind"] = args.counting_needle_kind
    if args.marker_text is not None:
        kwargs["marker_text"] = args.marker_text
    if args.uid_token_length is not None:
        kwargs["uid_token_length"] = args.uid_token_length
    if args.needle_seeds_json:
        kwargs["needle_seeds"] = _parse_needle_seeds(args.needle_seeds_json)
    if args.control_switch_json:
        kwargs["control_switch"] = json.loads(args.control_switch_json)

    cfg = DynamicNiahV2Config(
        **dynamic_niah_v2_config_kwargs(kwargs, warn_unknown=True)
    )
    extra_output_dir = cfg.output_dir
    extra_data_save_path = cfg.data_save_path
    paths = create_run_paths(
        results_root=cfg.results_root,
        model_name=cfg.tokenizer_name,
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
        _write_run_metadata(primary_cfg, paths)
        print(f"[run] metadata={paths.metadata_path}")
        rows = generate_dynamic_niah_dataset_v2(primary_cfg)
        paths_written = write_dynamic_niah_v2(rows, primary_cfg)
        print(json.dumps({**paths_written, "num_examples": len(rows)}, indent=2))

        if extra_output_dir is not None:
            if is_google_drive_path(extra_output_dir):
                archive_path = archive_directory(
                    paths.generate_data_dir,
                    extra_output_dir,
                    archive_name=f"{paths.run_dir.name}_generate_data",
                )
                print(
                    "[run] output_dir is on Google Drive; skipped per-file copying "
                    f"and moved one archive instead: {archive_path}"
                )
            else:
                extra_jsonl = extra_data_save_path or str(
                    Path(extra_output_dir) / "dynamic_niah_v2.jsonl"
                )
                extra_cfg = replace(
                    primary_cfg, output_dir=extra_output_dir, data_save_path=extra_jsonl
                )
                extra_paths = write_dynamic_niah_v2(rows, extra_cfg)
                print(
                    f"[run] additional copy saved because output_dir was specified: {extra_output_dir}"
                )
                if extra_data_save_path is not None:
                    print(
                        f"[run] additional dataset copy used specified data_save_path: {extra_data_save_path}"
                    )
                print(json.dumps(extra_paths, indent=2))
        elif extra_data_save_path is not None:
            extra_path = Path(extra_data_save_path)
            if is_google_drive_path(extra_path):
                archive_path = archive_directory(
                    paths.generate_data_dir,
                    extra_path.parent,
                    archive_name=extra_path.stem
                    or f"{paths.run_dir.name}_generate_data",
                )
                print(
                    "[run] data_save_path is on Google Drive; skipped direct writes "
                    f"and moved one archive instead: {archive_path}"
                )
            else:
                extra_path.parent.mkdir(parents=True, exist_ok=True)
                with extra_path.open("w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"[run] additional dataset copy saved because data_save_path was specified: {extra_path}"
                )


if __name__ == "__main__":
    main()
