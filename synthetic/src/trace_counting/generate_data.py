from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .io_utils import ensure_dir, save_json
from .tokenizer import DEFAULT_POSITIVE_VOCAB, build_default_tokenizer

SPLIT_OFFSETS = {
    "train": 11,
    "val_id": 101,
    "val_length_ood": 211,
    "val_density_shift_low": 307,
    "val_density_shift_high": 401,
    "val_count_ood": 509,
}

TASK_FORMATS = {
    "think_trace",
    "answer_only",
    "think_trace_repeat_count",
    "answer_only_repeat_count",
}


def is_think_format(task_format: str) -> bool:
    return task_format in {"think_trace", "think_trace_repeat_count"}


def is_repeat_count_format(task_format: str) -> bool:
    return task_format in {"think_trace_repeat_count", "answer_only_repeat_count"}


def parse_int_list(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    value = str(value).strip()
    if not value:
        return []
    return [int(part) for part in value.split(",") if part]


def parse_count_spec(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    value = str(value).strip()
    if ":" in value:
        left, right = value.split(":", maxsplit=1)
        start, end = int(left), int(right)
        if end < start:
            raise ValueError(f"Invalid count range {value!r}")
        return list(range(start, end + 1))
    return parse_int_list(value)


def make_example(
    *,
    split: str,
    seed: int,
    seq_len: int,
    count: int,
    rng: random.Random,
    example_index: int,
    max_count: int = 64,
    positive_vocab: list[str] | tuple[str, ...] = tuple(DEFAULT_POSITIVE_VOCAB),
    noise_vocab: list[str] | tuple[str, ...] | None = None,
    task_format: str = "think_trace",
) -> dict:
    if task_format not in TASK_FORMATS:
        raise ValueError(f"task_format must be one of {sorted(TASK_FORMATS)}")
    if count < 0 or count > min(seq_len, max_count):
        raise ValueError(f"Invalid count={count} for seq_len={seq_len}, max_count={max_count}")
    if noise_vocab is None:
        noise_vocab = [f"N{i}" for i in range(64)]

    positions = sorted(rng.sample(range(seq_len), count))
    position_set = set(positions)
    source_tokens: list[str] = []
    positive_markers: list[str] = []
    for pos in range(seq_len):
        if pos in position_set:
            marker = rng.choice(list(positive_vocab))
            source_tokens.append(marker)
            positive_markers.append(marker)
        else:
            source_tokens.append(rng.choice(list(noise_vocab)))

    trace_tokens: list[str] = []
    for k, marker in enumerate(positive_markers, start=1):
        trace_tokens.extend(["<TICK>" if is_repeat_count_format(task_format) else f"<I{k}>", marker])

    source_start = 1
    source_end_exclusive = 1 + seq_len
    answer_tokens = ["<CNT>"] * count if is_repeat_count_format(task_format) else [f"<C{count}>"]
    if is_think_format(task_format):
        full_tokens = ["<BOS>"] + source_tokens + ["<Think>"] + trace_tokens + ["<Think>", "<ANS>"] + answer_tokens + [
            "<EOS>",
        ]
        think_open_idx = source_end_exclusive
        trace_start = think_open_idx + 1
        trace_end_exclusive = trace_start + len(trace_tokens)
        think_close_idx = trace_end_exclusive
        ans_idx = think_close_idx + 1
        count_start_idx = ans_idx + 1
        count_end_exclusive = count_start_idx + len(answer_tokens)
        count_idx = count_start_idx if answer_tokens else count_end_exclusive
        eos_idx = count_end_exclusive
    else:
        full_tokens = ["<BOS>"] + source_tokens + ["<ANS>"] + answer_tokens + ["<EOS>"]
        think_open_idx = None
        trace_start = source_end_exclusive
        trace_end_exclusive = source_end_exclusive
        think_close_idx = None
        ans_idx = source_end_exclusive
        count_start_idx = ans_idx + 1
        count_end_exclusive = count_start_idx + len(answer_tokens)
        count_idx = count_start_idx if answer_tokens else count_end_exclusive
        eos_idx = count_end_exclusive

    trace_pairs = []
    for pair_idx, (source_pos, marker) in enumerate(zip(positions, positive_markers), start=1):
        index_idx = trace_start + 2 * (pair_idx - 1) if is_think_format(task_format) else None
        trace_pairs.append(
            {
                "k": pair_idx,
                "index_idx": index_idx,
                "marker_idx": index_idx + 1 if index_idx is not None else None,
                "marker": marker,
                "source_idx": source_start + source_pos,
            }
        )

    example = {
        "example_id": f"{split}_L{seq_len}_n{count}_seed{seed}_{example_index:06d}",
        "split": split,
        "seed": seed,
        "seq_len": seq_len,
        "count": count,
        "source_tokens": source_tokens,
        "positive_positions_source": positions,
        "positive_markers": positive_markers,
        "trace_tokens": trace_tokens,
        "answer_token": f"<C{count}>" if not is_repeat_count_format(task_format) else "<CNT>",
        "answer_tokens": answer_tokens,
        "full_tokens": full_tokens,
        "task_format": task_format,
        "spans": {
            "source_start": source_start,
            "source_end_exclusive": source_end_exclusive,
            "think_open_idx": think_open_idx,
            "trace_start": trace_start,
            "trace_end_exclusive": trace_end_exclusive,
            "think_close_idx": think_close_idx,
            "ans_idx": ans_idx,
            "count_start_idx": count_start_idx,
            "count_end_exclusive": count_end_exclusive,
            "count_idx": count_idx,
            "eos_idx": eos_idx,
            "trace_pairs": trace_pairs,
        },
    }
    validate_example(example)
    return example


def validate_example(example: dict) -> None:
    full_tokens = example["full_tokens"]
    spans = example["spans"]
    count = int(example["count"])
    task_format = example.get("task_format", "think_trace")
    assert full_tokens[0] == "<BOS>"
    assert full_tokens[-1] == "<EOS>"
    if is_think_format(task_format):
        assert full_tokens[spans["think_open_idx"]] == "<Think>"
        assert full_tokens[spans["think_close_idx"]] == "<Think>"
        assert len(example["trace_tokens"]) == 2 * count
        expected_index_token = "<TICK>" if is_repeat_count_format(task_format) else None
        if expected_index_token is not None:
            assert example["trace_tokens"][0::2] == [expected_index_token] * count
    elif task_format in {"answer_only", "answer_only_repeat_count"}:
        assert spans["think_open_idx"] is None
        assert spans["think_close_idx"] is None
        assert spans["trace_start"] == spans["trace_end_exclusive"]
    else:
        raise AssertionError(f"Unknown task_format={task_format!r}")
    assert full_tokens[spans["ans_idx"]] == "<ANS>"
    if is_repeat_count_format(task_format):
        assert full_tokens[spans["count_start_idx"] : spans["count_end_exclusive"]] == ["<CNT>"] * count
    else:
        assert full_tokens[spans["count_idx"]] == f"<C{count}>"
        assert spans["count_end_exclusive"] == spans["count_start_idx"] + 1
    assert spans["eos_idx"] == len(full_tokens) - 1
    assert len(example["positive_positions_source"]) == count
    source_marker_tokens = [full_tokens[pair["source_idx"]] for pair in spans["trace_pairs"]]
    trace_marker_tokens = [pair["marker"] for pair in spans["trace_pairs"]]
    assert trace_marker_tokens == source_marker_tokens


def write_split(
    *,
    out_dir: Path,
    split: str,
    lengths: list[int],
    counts: list[int],
    examples_per_pair: int,
    seeds: list[int],
    max_count: int,
    noise_vocab_size: int,
    positive_vocab: list[str] | tuple[str, ...] = tuple(DEFAULT_POSITIVE_VOCAB),
    task_format: str = "think_trace",
) -> int:
    path = out_dir / f"{split}.jsonl"
    noise_vocab = [f"N{i}" for i in range(noise_vocab_size)]
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for seed in seeds:
            rng = random.Random(seed * 1_000_003 + SPLIT_OFFSETS[split])
            example_index = 0
            for seq_len in lengths:
                for count in counts:
                    if count > min(seq_len, max_count):
                        raise ValueError(f"count={count} is invalid for split={split}, seq_len={seq_len}.")
                    for _ in range(examples_per_pair):
                        example = make_example(
                            split=split,
                            seed=seed,
                            seq_len=seq_len,
                            count=count,
                            rng=rng,
                            example_index=example_index,
                            max_count=max_count,
                            positive_vocab=positive_vocab,
                            noise_vocab=noise_vocab,
                            task_format=task_format,
                        )
                        f.write(json.dumps(example, sort_keys=True) + "\n")
                        written += 1
                        example_index += 1
    return written


def generate_dataset(
    *,
    out_dir: str | Path,
    max_count: int = 64,
    noise_vocab_size: int = 64,
    train_lengths: list[int] | None = None,
    train_counts: list[int] | None = None,
    val_id_lengths: list[int] | None = None,
    val_id_counts: list[int] | None = None,
    val_length_ood_lengths: list[int] | None = None,
    val_length_ood_counts: list[int] | None = None,
    val_density_shift_low_lengths: list[int] | None = None,
    val_density_shift_low_counts: list[int] | None = None,
    val_density_shift_high_lengths: list[int] | None = None,
    val_density_shift_high_counts: list[int] | None = None,
    val_count_ood_lengths: list[int] | None = None,
    val_count_ood_counts: list[int] | None = None,
    examples_per_pair_train: int = 512,
    examples_per_pair_val: int = 128,
    seeds: list[int] | None = None,
    task_format: str = "think_trace",
    include_legacy_shifts: bool = True,
) -> dict:
    out_dir = ensure_dir(out_dir)
    seeds = seeds or [0, 1, 2]
    train_lengths = train_lengths or [32, 64, 128]
    train_counts = train_counts or list(range(25))
    split_specs = {
        "train": (train_lengths, train_counts, examples_per_pair_train),
        "val_id": (val_id_lengths or train_lengths, val_id_counts or train_counts, examples_per_pair_val),
    }
    if val_count_ood_lengths is not None or val_count_ood_counts is not None:
        split_specs["val_count_ood"] = (
            val_count_ood_lengths or train_lengths,
            val_count_ood_counts or list(range(5, 11)),
            examples_per_pair_val,
        )
    if include_legacy_shifts:
        split_specs.update(
            {
                "val_length_ood": (
                    val_length_ood_lengths or [256, 512],
                    val_length_ood_counts or train_counts,
                    examples_per_pair_val,
                ),
                "val_density_shift_low": (
                    val_density_shift_low_lengths or [512],
                    val_density_shift_low_counts or list(range(9)),
                    examples_per_pair_val,
                ),
                "val_density_shift_high": (
                    val_density_shift_high_lengths or [64],
                    val_density_shift_high_counts or list(range(16, 25)),
                    examples_per_pair_val,
                ),
            }
        )

    tokenizer = build_default_tokenizer(max_count=max_count, noise_vocab_size=noise_vocab_size)
    tokenizer.save(out_dir / "vocab.json")

    split_counts: dict[str, int] = {}
    for split, (lengths, counts, examples_per_pair) in split_specs.items():
        split_counts[split] = write_split(
            out_dir=out_dir,
            split=split,
            lengths=list(lengths),
            counts=list(counts),
            examples_per_pair=int(examples_per_pair),
            seeds=seeds,
            max_count=max_count,
            noise_vocab_size=noise_vocab_size,
            task_format=task_format,
        )

    metadata = {
        "max_count": max_count,
        "noise_vocab_size": noise_vocab_size,
        "seeds": seeds,
        "task_format": task_format,
        "split_counts": split_counts,
        "split_specs": {
            split: {
                "lengths": list(lengths),
                "counts": list(counts),
                "examples_per_pair": int(examples_per_pair),
            }
            for split, (lengths, counts, examples_per_pair) in split_specs.items()
        },
    }
    save_json(metadata, out_dir / "dataset_metadata.json")
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate trace-enumeration counting JSONL data.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_count", type=int, default=64)
    parser.add_argument("--noise_vocab_size", type=int, default=64)
    parser.add_argument("--train_lengths", default="32,64,128")
    parser.add_argument("--train_counts", default="0:24")
    parser.add_argument("--val_id_lengths", default=None)
    parser.add_argument("--val_id_counts", default=None)
    parser.add_argument("--val_length_ood_lengths", default="256,512")
    parser.add_argument("--val_length_ood_counts", default=None)
    parser.add_argument("--val_density_shift_low_lengths", default="512")
    parser.add_argument("--val_density_shift_low_counts", default="0:8")
    parser.add_argument("--val_density_shift_high_lengths", default="64")
    parser.add_argument("--val_density_shift_high_counts", default="16:24")
    parser.add_argument("--val_count_ood_lengths", default=None)
    parser.add_argument("--val_count_ood_counts", default=None)
    parser.add_argument("--examples_per_pair_train", type=int, default=512)
    parser.add_argument("--examples_per_pair_val", type=int, default=128)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--task_format", default="think_trace", choices=sorted(TASK_FORMATS))
    parser.add_argument("--no_legacy_shifts", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Use the tiny debug split from the pipeline spec.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.debug:
        args.train_lengths = "16,32"
        args.train_counts = "0:4"
        args.val_id_lengths = "16,32"
        args.val_id_counts = "0:4"
        args.val_length_ood_lengths = "64"
        args.val_length_ood_counts = "0:4"
        args.val_density_shift_low_lengths = "64"
        args.val_density_shift_low_counts = "0:2"
        args.val_density_shift_high_lengths = "16"
        args.val_density_shift_high_counts = "3:4"
        args.val_count_ood_lengths = None
        args.val_count_ood_counts = None
        args.examples_per_pair_train = 8
        args.examples_per_pair_val = 4

    metadata = generate_dataset(
        out_dir=args.out_dir,
        max_count=args.max_count,
        noise_vocab_size=args.noise_vocab_size,
        train_lengths=parse_int_list(args.train_lengths),
        train_counts=parse_count_spec(args.train_counts),
        val_id_lengths=parse_int_list(args.val_id_lengths) if args.val_id_lengths else None,
        val_id_counts=parse_count_spec(args.val_id_counts) if args.val_id_counts else None,
        val_length_ood_lengths=parse_int_list(args.val_length_ood_lengths),
        val_length_ood_counts=parse_count_spec(args.val_length_ood_counts) if args.val_length_ood_counts else None,
        val_density_shift_low_lengths=parse_int_list(args.val_density_shift_low_lengths),
        val_density_shift_low_counts=parse_count_spec(args.val_density_shift_low_counts),
        val_density_shift_high_lengths=parse_int_list(args.val_density_shift_high_lengths),
        val_density_shift_high_counts=parse_count_spec(args.val_density_shift_high_counts),
        val_count_ood_lengths=parse_int_list(args.val_count_ood_lengths) if args.val_count_ood_lengths else None,
        val_count_ood_counts=parse_count_spec(args.val_count_ood_counts) if args.val_count_ood_counts else None,
        examples_per_pair_train=args.examples_per_pair_train,
        examples_per_pair_val=args.examples_per_pair_val,
        seeds=parse_int_list(args.seeds),
        task_format=args.task_format,
        include_legacy_shifts=not args.no_legacy_shifts,
    )
    print(json.dumps(metadata["split_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
