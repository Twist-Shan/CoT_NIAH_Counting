"""Offline CPU checks of parsers, source integrity, and the synthetic model."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "realistic/src"), str(ROOT / "synthetic/src")]


def main() -> None:
    started = time.perf_counter()
    import torch
    import torch.nn.functional as F
    from realistic_niah.parsing import parse_total
    from synthetic_counting_v20.data import V20Example, V20Vocab, character_token, render_v20
    from synthetic_counting_v20.model import build_model
    from synthetic_counting_v58.config import preset_config

    assert parse_total("Total: 3") == 3
    assert parse_total("No final count") is None
    provenance = json.loads((ROOT / "realistic/provenance/NIAH_PARSER_V5.json").read_text())
    for rel, expected in provenance["files"].items():
        assert hashlib.sha256((ROOT / "realistic" / rel).read_bytes()).hexdigest() == expected, rel
    corpus = ROOT / "synthetic/src/synthetic_counting_v11/resources/tiny_shakespeare"
    expected = json.loads((corpus / "SOURCE.json").read_text())["sha256"]
    assert hashlib.sha256((corpus / "input.txt").read_bytes()).hexdigest() == expected

    # Compare the complete saved configuration to the current paper preset.
    saved = json.loads((ROOT / "synthetic/configs/paper_v58_saved_config.json").read_text())
    actual = json.loads(json.dumps(preset_config("main", device="cuda").to_dict()))
    differences = {k: {"saved": v, "preset": actual.get(k)} for k, v in saved.items() if actual.get(k) != v}
    if differences:
        raise AssertionError(f"Paper preset differs from saved configuration: {differences}")

    torch.manual_seed(1234)
    torch.set_num_threads(1)
    # This deliberately smaller model is a smoke configuration, not a paper run.
    cfg = replace(preset_config("main", device="cpu"), n_layer=2, n_head=2,
                  n_embd=32, n_inner=64, n_positions=64, precision="float32")
    vocab = V20Vocab.build(cfg, "abc xyz\n")
    char = character_token("a")
    example = V20Example(
        example_kind="counting_task", seq_tokens=[char, character_token("x"), char],
        corpus_region="train", corpus_start=0, corpus_end=3, prompt_sha256="smoke",
        set_id="smoke", needle_characters=("a", "b", "c"),
        rendered_set_order=("a", "b", "c"), needle_positions=(0, 2),
        needle_markers=(char, char), count=2, per_character_counts=(2, 0, 0),
    )
    losses = {}
    for mode in ("nonthinking", "thinking"):
        rendered = render_v20(example, vocab, mode)
        assert rendered.count == 2
        model = build_model(cfg, vocab, device="cpu").eval()
        ids = torch.tensor([rendered.input_ids], dtype=torch.long)
        with torch.no_grad():
            fast = model(ids).logits
            explicit = model(ids, output_attentions=True)
        torch.testing.assert_close(fast, explicit.logits, rtol=1e-4, atol=1e-5)
        for attention in explicit.attentions:
            assert torch.count_nonzero(torch.triu(attention, diagonal=1)) == 0
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = F.cross_entropy(model(ids).logits[:, :-1].reshape(-1, len(vocab.id_to_token)), ids[:, 1:].reshape(-1))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        losses[mode] = float(loss.detach())
    result = {"status": "passed", "device": "cpu", "model": "random 2-layer smoke model",
              "checks": ["count parser", "frozen parser checksums", "corpus checksum", "saved paper preset",
                         "trace serialization", "SDPA/explicit attention agreement", "causal mask", "forward/backward/optimizer"],
              "loss": losses, "elapsed_seconds": round(time.perf_counter() - started, 3)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
