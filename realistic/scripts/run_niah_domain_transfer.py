#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.domain_transfer import (
    capture_native_domain_shards,
    capture_nonthinking_domain_shards,
    generate_native_domain_shards,
    prepare_domain_transfer_panel,
    read_jsonl,
    write_domain_transfer_panel,
)


DEFAULT_V4_CONFIG = ROOT / "configs" / "realistic_niah_v4.json"
DEFAULT_V5_CONFIG = ROOT / "configs" / "realistic_niah_v5.json"
MODEL_CHOICES = ("Qwen3-8B", "Gemma4-E4B")
DOMAIN_CHOICES = ("flower", "animal")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
        "status_short": status,
    }


def _write_runtime_provenance(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    started: float,
) -> None:
    import numpy as np
    import torch
    import transformers

    payload = {
        "schema_version": "realistic_niah_domain_transfer_runtime_v1",
        "command": list(sys.argv),
        "resolved_args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "func"
        },
        "working_directory": str(Path.cwd()),
        "repo_root": str(ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "git": _git_state(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(output_dir / "runtime_provenance.json", payload)


def _load_model(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    from realistic_niah_v4.modeling import load_registered_model
    from realistic_niah_v4.spec import resolve_model_spec

    spec = resolve_model_spec(args.model)
    model, tokenizer, adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    return model, tokenizer, adapter, spec


def _selected_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    domain_set = set(args.domains)
    seed_set = None if args.seeds is None else {int(value) for value in args.seeds}
    count_set = None if args.counts is None else {int(value) for value in args.counts}
    rows = [
        row
        for row in read_jsonl(args.stimuli)
        if str(row.get("entity_domain")) in domain_set
        and (seed_set is None or int(row.get("seed", -1)) in seed_set)
        and (count_set is None or int(row.get("gold_count", -1)) in count_set)
    ]
    rows.sort(
        key=lambda row: (
            args.domains.index(str(row["entity_domain"])),
            int(row["seed"]),
            int(row["gold_count"]),
        )
    )
    if args.max_rows_per_domain is not None:
        limited: list[dict[str, Any]] = []
        for domain in args.domains:
            limited.extend(
                [row for row in rows if row["entity_domain"] == domain][
                    : args.max_rows_per_domain
                ]
            )
        rows = limited
    if not rows:
        raise ValueError("Stimulus filters selected no domain-transfer rows")
    return rows


def command_prepare(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    rows = prepare_domain_transfer_panel(
        read_jsonl(args.source),
        entity_domains=args.domains,
    )
    paths = write_domain_transfer_panel(
        args.output,
        rows,
        source_path=args.source,
    )
    _write_runtime_provenance(args.output, args=args, started=started)
    print(
        json.dumps({key: str(value) for key, value in paths.items()}, indent=2),
        flush=True,
    )


def command_nonthinking_capture(args: argparse.Namespace) -> None:
    from realistic_niah_v4.spec import V4Config

    started = time.perf_counter()
    rows = _selected_rows(args)
    model, tokenizer, adapter, spec = _load_model(args)
    config = V4Config.from_json(args.config)
    index = capture_nonthinking_domain_shards(
        model,
        tokenizer,
        adapter,
        rows,
        model_spec=spec,
        config=config,
        output_dir=args.output,
        layers=args.layers,
        save_dtype=args.save_dtype,
        overwrite=args.overwrite,
    )
    _write_runtime_provenance(args.output, args=args, started=started)
    print(f"[domain-transfer] non-thinking index: {index}", flush=True)


def command_native_generate(args: argparse.Namespace) -> None:
    from realistic_niah_v5.spec import V5Config

    started = time.perf_counter()
    rows = _selected_rows(args)
    model, tokenizer, _adapter, spec = _load_model(args)
    config = V5Config.load(args.config)
    generations = generate_native_domain_shards(
        model,
        tokenizer,
        rows,
        model_spec=spec,
        decoding=config.decoding,
        output_dir=args.output,
        overwrite=args.overwrite,
    )
    _write_runtime_provenance(args.output, args=args, started=started)
    print(f"[domain-transfer] native generations: {generations}", flush=True)


def command_native_capture(args: argparse.Namespace) -> None:
    from realistic_niah_v5.spec import V5Config

    started = time.perf_counter()
    records = read_jsonl(args.generations)
    domain_set = set(args.domains)
    seed_set = None if args.seeds is None else {int(value) for value in args.seeds}
    count_set = None if args.counts is None else {int(value) for value in args.counts}
    records = [
        row
        for row in records
        if str(row.get("entity_domain")) in domain_set
        and str(row.get("model_label")) == args.model
        and (seed_set is None or int(row.get("seed", -1)) in seed_set)
        and (count_set is None or int(row.get("gold_count", -1)) in count_set)
    ]
    if args.max_rows_per_domain is not None:
        records = [
            row
            for domain in args.domains
            for row in [
                value for value in records if value["entity_domain"] == domain
            ][: args.max_rows_per_domain]
        ]
    if not records:
        raise ValueError("Generation filters selected no native records")
    model, tokenizer, adapter, _spec = _load_model(args)
    config = V5Config.load(args.config)
    index = capture_native_domain_shards(
        model,
        tokenizer,
        adapter,
        records,
        config=config,
        output_dir=args.output,
        layers=args.layers,
        overwrite=args.overwrite,
    )
    _write_runtime_provenance(args.output, args=args, started=started)
    print(f"[domain-transfer] native capture index: {index}", flush=True)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")


def _add_panel_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--domains", nargs="+", choices=DOMAIN_CHOICES, default=list(DOMAIN_CHOICES))
    parser.add_argument(
        "--max-rows-per-domain",
        type=int,
        help="Smoke-test limiter applied independently to each requested domain.",
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--counts", type=int, nargs="+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Entity-domain transfer experiment for NiaH final-answer and "
            "running-index representations"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--domains", nargs="+", choices=DOMAIN_CHOICES, default=list(DOMAIN_CHOICES))
    prepare.set_defaults(func=command_prepare)

    nonthinking = subparsers.add_parser("nonthinking-capture")
    _add_model_args(nonthinking)
    _add_panel_args(nonthinking)
    nonthinking.add_argument("--stimuli", type=Path, required=True)
    nonthinking.add_argument("--config", type=Path, default=DEFAULT_V4_CONFIG)
    nonthinking.add_argument("--output", type=Path, required=True)
    nonthinking.add_argument("--layers", type=int, nargs="+")
    nonthinking.add_argument(
        "--save-dtype", choices=("float16", "float32"), default="float16"
    )
    nonthinking.add_argument("--overwrite", action="store_true")
    nonthinking.set_defaults(func=command_nonthinking_capture)

    native_generate = subparsers.add_parser("native-generate")
    _add_model_args(native_generate)
    _add_panel_args(native_generate)
    native_generate.add_argument("--stimuli", type=Path, required=True)
    native_generate.add_argument("--config", type=Path, default=DEFAULT_V5_CONFIG)
    native_generate.add_argument("--output", type=Path, required=True)
    native_generate.add_argument("--overwrite", action="store_true")
    native_generate.set_defaults(func=command_native_generate)

    native_capture = subparsers.add_parser("native-capture")
    _add_model_args(native_capture)
    _add_panel_args(native_capture)
    native_capture.add_argument("--generations", type=Path, required=True)
    native_capture.add_argument("--config", type=Path, default=DEFAULT_V5_CONFIG)
    native_capture.add_argument("--output", type=Path, required=True)
    native_capture.add_argument("--layers", type=int, nargs="+")
    native_capture.add_argument("--overwrite", action="store_true")
    native_capture.set_defaults(func=command_native_capture)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "max_rows_per_domain", None) is not None:
        if int(args.max_rows_per_domain) < 1:
            raise ValueError("--max-rows-per-domain must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
