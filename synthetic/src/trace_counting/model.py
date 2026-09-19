from __future__ import annotations

from pathlib import Path

from transformers import GPT2Config, GPT2LMHeadModel

from .io_utils import load_yaml
from .tokenizer import VocabTokenizer


def load_model_config(path: str | Path) -> dict:
    return load_yaml(path)


def build_model_from_config(config: dict, tokenizer: VocabTokenizer) -> GPT2LMHeadModel:
    cfg = dict(config)
    cfg.pop("model_name", None)
    vocab_size = cfg.pop("vocab_size", "auto")
    if vocab_size == "auto":
        vocab_size = len(tokenizer)
    gpt2_config = GPT2Config(
        vocab_size=int(vocab_size),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **cfg,
    )
    return GPT2LMHeadModel(gpt2_config)


def load_model_from_checkpoint(checkpoint: str | Path, **kwargs) -> GPT2LMHeadModel:
    return GPT2LMHeadModel.from_pretrained(str(checkpoint), **kwargs)
