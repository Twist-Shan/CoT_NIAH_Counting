from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .io_utils import read_jsonl
from .loss_masks import build_labels_and_weights, segment_ids_for_example
from .tokenizer import VocabTokenizer


class JsonlTraceDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: VocabTokenizer,
        *,
        loss_mask: str,
        final_weight: float = 10.0,
        eos_weight: float = 1.0,
        final_count_only_include_eos: bool = False,
        limit: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.loss_mask = loss_mask
        self.final_weight = final_weight
        self.eos_weight = eos_weight
        self.final_count_only_include_eos = final_count_only_include_eos
        self.examples = read_jsonl(self.path, limit=limit)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        input_ids = self.tokenizer.encode(example["full_tokens"])
        labels, loss_weights = build_labels_and_weights(
            example,
            self.tokenizer,
            loss_mask=self.loss_mask,
            final_weight=self.final_weight,
            eos_weight=self.eos_weight,
            final_count_only_include_eos=self.final_count_only_include_eos,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "loss_weights": loss_weights,
            "segment_ids": segment_ids_for_example(example),
            "metadata": example,
        }


class TraceCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        attention_mask = []
        labels = []
        loss_weights = []
        segment_ids = []
        metadata = []
        for item in batch:
            length = len(item["input_ids"])
            pad = max_len - length
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
            loss_weights.append(item["loss_weights"] + [0.0] * pad)
            segment_ids.append(item["segment_ids"] + [-1] * pad)
            metadata.append(item["metadata"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
            "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
            "metadata": metadata,
        }


def dataset_for_split(
    data_dir: str | Path,
    split: str,
    tokenizer: VocabTokenizer,
    *,
    loss_mask: str,
    final_weight: float = 10.0,
    eos_weight: float = 1.0,
    final_count_only_include_eos: bool = False,
    limit: int | None = None,
) -> JsonlTraceDataset:
    return JsonlTraceDataset(
        Path(data_dir) / f"{split}.jsonl",
        tokenizer,
        loss_mask=loss_mask,
        final_weight=final_weight,
        eos_weight=eos_weight,
        final_count_only_include_eos=final_count_only_include_eos,
        limit=limit,
    )
