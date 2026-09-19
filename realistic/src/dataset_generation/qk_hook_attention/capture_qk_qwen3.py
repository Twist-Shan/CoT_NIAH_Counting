#!/usr/bin/env python3
"""
Capture raw q_proj/k_proj outputs for selected layers of Qwen/Qwen3-8B while
running the normal Hugging Face forward pass, e.g. with FlashAttention-2.

This script saves pre-Qwen3-q_norm, pre-RoPE Q/K projection outputs. The
companion analysis script reconstructs the exact per-head attention logits by
applying Qwen3 q_norm/k_norm, RoPE, GQA head mapping, and causal masks offline.

Target: text-only Qwen3 causal LMs, especially Qwen/Qwen3-8B.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "auto": "auto",
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "same": None,
}

SAVE_DTYPE_MAP = {
    "same": None,
    "auto": None,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def parse_layers(spec: str) -> List[int]:
    """
    Convert a layer-selection string into a sorted list of unique layer indices.

    Accepts comma-separated indices such as ``"0,8,16"`` and Python-like
    range specs such as ``"0:36:4"``. This controls which transformer layers
    receive Q/K hooks, so reducing this list directly reduces saved-cache size.
    """
    layers: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = [p.strip() for p in part.split(":")]
            if len(pieces) not in (2, 3):
                raise ValueError(f"Bad slice in --layers: {part!r}")
            start = int(pieces[0]) if pieces[0] else 0
            stop = int(pieces[1])
            step = int(pieces[2]) if len(pieces) == 3 and pieces[2] else 1
            layers.extend(range(start, stop, step))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def read_prompt(args: argparse.Namespace) -> str:
    """
    Read the prompt from exactly one supported source.

    The prompt can come from ``--prompt``, ``--prompt-file``, or stdin. The
    function validates that the caller did not provide conflicting prompt
    sources, then returns raw text before tokenization or chat-template wrapping.
    """
    if args.prompt is not None and args.prompt_file is not None:
        raise ValueError("Provide only one of --prompt or --prompt-file.")
    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("Provide --prompt, --prompt-file, or pipe text on stdin.")


def find_all_subsequences(haystack: List[int], needle: List[int]) -> List[Tuple[int, int]]:
    """
    Find every exact occurrence of a token-id sequence inside another sequence.

    This is used to locate user-provided marker strings, such as needle tokens,
    after tokenization. It returns half-open spans ``(start, end)`` in token
    positions. The scan is intentionally simple and easy to debug.
    """
    if not needle:
        return []
    spans: List[Tuple[int, int]] = []
    n = len(needle)
    # Naive scan is fine for prompt construction/debugging. For huge prompts
    # and many markers, replace with KMP/Aho-Corasick.
    for i in range(0, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            spans.append((i, i + n))
    return spans


def load_markers(path: str | None) -> Dict[str, str]:
    """
    Load optional marker text definitions from JSON.

    The expected file format is ``{"marker_name": "exact marker text"}``.
    These markers are tokenized later and used to seed ``analysis_spec.json``
    with named spans and after-marker query positions.
    """
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("markers JSON must be an object: {name: marker_text}")
    return {str(k): str(v) for k, v in data.items()}


def build_analysis_spec(
    *,
    input_ids: List[int],
    tokenizer,
    markers: Dict[str, str],
) -> Dict:
    """
    Build a starter offline-analysis specification from the tokenized input.

    The returned dictionary contains named spans, critical query positions, and
    default window-received metrics. It is only a convenience file for the
    analysis script; users can edit ``analysis_spec.json`` manually afterward.
    """
    seq_len = len(input_ids)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    special_positions = [i for i, tok_id in enumerate(input_ids) if tok_id in special_ids]

    spans: Dict[str, List[List[int]]] = {
        "bos": [[0, 1]],
    }
    if special_positions:
        spans["special_tokens"] = [[p, p + 1] for p in special_positions]

    query_positions: Dict[str, List[int]] = {
        "last": [seq_len - 1],
    }
    if special_positions:
        query_positions["special_tokens"] = special_positions

    window_received = []

    for name, marker_text in markers.items():
        candidate_texts = [marker_text]
        if marker_text and not marker_text.startswith(" "):
            candidate_texts.append(" " + marker_text)
        if marker_text.startswith(" "):
            candidate_texts.append(marker_text.lstrip())

        found: List[Tuple[int, int]] = []
        seen = set()
        for text in candidate_texts:
            marker_ids = tokenizer(text, add_special_tokens=False).input_ids
            for span in find_all_subsequences(input_ids, marker_ids):
                if span not in seen:
                    found.append(span)
                    seen.add(span)

        found = sorted(found)
        if found:
            spans[name] = [[s, e] for s, e in found]
            after_positions = [e for _, e in found if e < seq_len]
            if after_positions:
                query_positions[f"after_{name}"] = after_positions
            # Default "received by later tokens" metric for this marker.
            later_start = max(e for _, e in found)
            if later_start < seq_len:
                window_received.append(
                    {
                        "name": f"{name}_received_from_later_tokens",
                        "span_name": name,
                        "query_start": later_start,
                        "query_end": None,
                    }
                )
        else:
            print(
                f"[warn] Marker {name!r} was not found as an exact token subsequence. "
                "You can still edit analysis_spec.json manually.",
                file=sys.stderr,
            )

    return {
        "spans": spans,
        "query_positions": query_positions,
        "window_received": window_received,
        "local_windows": [32, 128, 512],
    }


def prepare_text(tokenizer, prompt: str, args: argparse.Namespace) -> str:
    """
    Optionally wrap the raw prompt with the model tokenizer's chat template.

    For plain causal-LM prompts, this returns the prompt unchanged. For chat
    experiments, it calls ``tokenizer.apply_chat_template`` and passes Qwen3's
    optional thinking flag when supported by the installed tokenizer version.
    """
    if not args.chat_template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if args.enable_thinking is not None:
        kwargs["enable_thinking"] = args.enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Some tokenizer versions may not accept enable_thinking.
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def install_qk_hooks(
    *,
    model,
    target_layers: Iterable[int],
    out_dir: Path,
    save_dtype: torch.dtype | None,
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Attach forward hooks to q_proj and k_proj in the requested Qwen3 layers.

    Each hook saves the raw projection output to disk during the normal model
    forward pass. The tensors are pre-Qwen3-q_norm and pre-RoPE, which is why
    the analysis script later applies q_norm/k_norm and RoPE offline. The
    returned handles should be removed after the capture pass.
    """
    handles = []

    def make_hook(layer_idx: int, kind: str):
        """
        Create one closure that knows which layer and projection kind it is saving.

        ``layer_idx`` and ``kind`` are captured in the nested hook so the same
        hook body can save files such as ``layer_08_q_raw.pt`` and
        ``layer_08_k_raw.pt`` without relying on global mutable state.
        """
        def hook(_module, _inputs, output):
            """
            Save one q_proj/k_proj output when PyTorch calls the forward hook.

            PyTorch passes the module, inputs, and output to every forward hook.
            This capture hook only uses ``output``: it detaches the tensor,
            optionally casts it, moves it to CPU, writes it to disk, and returns
            nothing so the model's actual forward output is not modified.
            """
            # output shape:
            #   q_proj: [batch, seq_len, num_attention_heads * head_dim]
            #   k_proj: [batch, seq_len, num_key_value_heads * head_dim]
            tensor = output.detach()
            if save_dtype is not None:
                tensor = tensor.to(dtype=save_dtype)
            tensor_cpu = tensor.cpu()
            path = out_dir / f"layer_{layer_idx:02d}_{kind}_raw.pt"
            torch.save(tensor_cpu, path)
            print(
                f"[hook] saved layer={layer_idx} {kind}_raw "
                f"shape={tuple(tensor_cpu.shape)} dtype={tensor_cpu.dtype} -> {path}",
                flush=True,
            )
            del tensor_cpu

        return hook

    for layer_idx in target_layers:
        attn = model.model.layers[layer_idx].self_attn
        handles.append(attn.q_proj.register_forward_hook(make_hook(layer_idx, "q")))
        handles.append(attn.k_proj.register_forward_hook(make_hook(layer_idx, "k")))
    return handles


def save_layer_norms(model, target_layers: Iterable[int], out_dir: Path) -> None:
    """
    Save Qwen3 q_norm/k_norm weights and eps values for selected layers.

    Qwen3 normalizes Q and K after projection and head reshaping. Because the
    hook saves raw projection outputs, these normalization parameters are needed
    later to reconstruct the exact post-norm Q/K used by attention.
    """
    for layer_idx in target_layers:
        attn = model.model.layers[layer_idx].self_attn
        payload = {
            "q_norm_weight": attn.q_norm.weight.detach().cpu().float(),
            "k_norm_weight": attn.k_norm.weight.detach().cpu().float(),
            "q_norm_eps": float(attn.q_norm.variance_epsilon),
            "k_norm_eps": float(attn.k_norm.variance_epsilon),
        }
        torch.save(payload, out_dir / f"layer_{layer_idx:02d}_qk_norms.pt")


@torch.no_grad()
def capture_qk_cache(
    *,
    model,
    tokenizer,
    model_name: str,
    prompt: str,
    out_dir: Path,
    target_layers: Sequence[int],
    markers: Dict[str, str] | None = None,
    chat_template: bool = False,
    enable_thinking: bool | None = None,
    attn_implementation_requested: str | None = None,
    save_dtype: torch.dtype | None = torch.bfloat16,
    save_token_strings: bool = True,
    write_metadata_files: bool = True,
    input_ids_override: torch.Tensor | Sequence[int] | None = None,
) -> Dict[str, Any]:
    """Capture selected-layer Q/K tensors with an already-loaded model/tokenizer."""

    out_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(chat_template=chat_template, enable_thinking=enable_thinking)
    text = prepare_text(tokenizer, prompt, args)
    if input_ids_override is None:
        encoded = tokenizer([text], return_tensors="pt", add_special_tokens=False)
        if encoded["input_ids"].shape[0] != 1:
            raise ValueError("This simple implementation assumes batch size 1.")
        input_ids_cpu = encoded["input_ids"].cpu()
        attention_mask_cpu = encoded.get("attention_mask", torch.ones_like(input_ids_cpu)).cpu()
    else:
        input_ids_cpu = torch.as_tensor(input_ids_override, dtype=torch.long).detach().cpu()
        if input_ids_cpu.ndim == 1:
            input_ids_cpu = input_ids_cpu.unsqueeze(0)
        if input_ids_cpu.shape[0] != 1:
            raise ValueError("This simple implementation assumes batch size 1.")
        attention_mask_cpu = torch.ones_like(input_ids_cpu, dtype=torch.long)
    seq_len = int(input_ids_cpu.shape[1])
    position_ids_cpu = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    markers = markers or {}
    analysis_spec = build_analysis_spec(
        input_ids=input_ids_cpu[0].tolist(),
        tokenizer=tokenizer,
        markers=markers,
    )
    tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu[0].tolist()) if save_token_strings else None

    if write_metadata_files:
        (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (out_dir / "model_text.txt").write_text(text, encoding="utf-8")
        (out_dir / "analysis_spec.json").write_text(json.dumps(analysis_spec, indent=2), encoding="utf-8")
        if tokens is not None:
            (out_dir / "tokens.json").write_text(json.dumps(tokens, ensure_ascii=False, indent=2), encoding="utf-8")

    torch.save(input_ids_cpu, out_dir / "input_ids.pt")
    torch.save(attention_mask_cpu, out_dir / "attention_mask.pt")
    torch.save(position_ids_cpu, out_dir / "position_ids.pt")

    num_layers = int(model.config.num_hidden_layers)
    bad_layers = [i for i in target_layers if i < 0 or i >= num_layers]
    if bad_layers:
        raise ValueError(f"Invalid target layers {bad_layers}; model has {num_layers} layers.")

    save_layer_norms(model, target_layers, out_dir)

    input_device = model.get_input_embeddings().weight.device
    model_inputs = {
        "input_ids": input_ids_cpu.to(input_device),
        "attention_mask": attention_mask_cpu.to(input_device),
        "position_ids": position_ids_cpu.to(input_device),
    }

    metadata = {
        "model_name": model_name,
        "created_unix_time": time.time(),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "attn_implementation_requested": attn_implementation_requested,
        "attn_implementation_model_config": getattr(model.config, "_attn_implementation", None),
        "model_dtype": str(getattr(model, "dtype", None)),
        "save_dtype": str(save_dtype),
        "target_layers": [int(x) for x in target_layers],
        "seq_len": seq_len,
        "batch_size": 1,
        "model_config": model.config.to_dict(),
        "special_token_ids": getattr(tokenizer, "all_special_ids", []),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "files": {
            "input_ids": "input_ids.pt",
            "attention_mask": "attention_mask.pt",
            "position_ids": "position_ids.pt",
            "analysis_spec": "analysis_spec.json" if write_metadata_files else None,
            "tokens": "tokens.json" if write_metadata_files and save_token_strings else None,
        },
    }
    if write_metadata_files:
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    handles = install_qk_hooks(model=model, target_layers=target_layers, out_dir=out_dir, save_dtype=save_dtype)
    inference_start = time.perf_counter()
    try:
        with torch.inference_mode():
            _ = model(
                **model_inputs,
                use_cache=False,
                output_attentions=False,
                return_dict=True,
                logits_to_keep=1,
            )
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_elapsed_seconds = time.perf_counter() - inference_start
        for handle in handles:
            handle.remove()
    metadata["model_inference_elapsed_seconds"] = inference_elapsed_seconds
    if write_metadata_files:
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return {
        "metadata": metadata,
        "prompt": prompt,
        "model_text": text,
        "markers": markers,
        "analysis_spec": analysis_spec,
        "tokens": tokens,
    }


def main() -> None:
    """
    Command-line entry point for capturing selected-layer Q/K caches.

    This function tokenizes the input, writes metadata and analysis scaffolding,
    loads Qwen3 with the requested Hugging Face attention implementation, installs
    the Q/K hooks, runs one inference-only forward pass, and removes the hooks.
    """
    parser = argparse.ArgumentParser(description="Capture selected-layer Q/K projection outputs for Qwen3.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layers", default="0,8,16,24,35", help="e.g. '0,8,16,24,35' or '0:36:4'")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--markers-json", default=None, help="Optional JSON: {name: exact_marker_text}")
    parser.add_argument("--chat-template", action="store_true", help="Wrap prompt with tokenizer.apply_chat_template.")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Passed to Qwen3 chat template if supported. Omit for tokenizer default.",
    )
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--model-dtype", default="bf16", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--save-dtype", default="bf16", choices=list(SAVE_DTYPE_MAP.keys()))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--save-token-strings", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_layers = parse_layers(args.layers)

    model_dtype = DTYPE_MAP[args.model_dtype]
    save_dtype = SAVE_DTYPE_MAP[args.save_dtype]

    prompt = read_prompt(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    print(f"[load] {args.model_name} with attn_implementation={args.attn_implementation}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=model_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    capture_qk_cache(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name,
        prompt=prompt,
        out_dir=out_dir,
        target_layers=target_layers,
        markers=load_markers(args.markers_json),
        chat_template=args.chat_template,
        enable_thinking=args.enable_thinking,
        attn_implementation_requested=args.attn_implementation,
        save_dtype=save_dtype,
        save_token_strings=args.save_token_strings,
        write_metadata_files=True,
    )

    print(f"[done] Q/K cache written to {out_dir}")


if __name__ == "__main__":
    main()
