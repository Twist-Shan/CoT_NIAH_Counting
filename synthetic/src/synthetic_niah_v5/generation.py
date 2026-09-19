from __future__ import annotations

import torch

from .vocab import Vocab


def pad_sequences(sequences: list[list[int]], pad_id: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), int(pad_id), dtype=torch.long, device=device)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long, device=device)
    for idx, seq in enumerate(sequences):
        input_ids[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return input_ids, lengths


@torch.no_grad()
def greedy_generate_one(
    model,
    prefix: list[int],
    vocab: Vocab,
    device: str | torch.device,
    max_new_tokens: int,
    continue_after_close: int = 2,
) -> list[int]:
    model.eval()
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    generated: list[int] = []
    after_close = 0
    saw_close = False
    for _ in range(int(max_new_tokens)):
        out = model(ids)
        next_id = int(out.logits[0, -1].argmax().detach().cpu())
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if next_id == vocab.eos_id:
            break
        if saw_close:
            after_close += 1
            if after_close >= int(continue_after_close):
                break
        elif next_id == vocab.think_close_id:
            saw_close = True
    return generated


@torch.no_grad()
def next_token_logits(
    model,
    prefixes: list[list[int]],
    vocab: Vocab,
    device: str | torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    model.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, len(prefixes), int(batch_size)):
        chunk = prefixes[start : start + int(batch_size)]
        input_ids, lengths = pad_sequences(chunk, vocab.pad_id, device)
        out = model(input_ids)
        logits = out.logits[torch.arange(input_ids.size(0), device=input_ids.device), lengths - 1]
        chunks.append(logits.detach().cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, len(vocab.id_to_token))
