from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.behavior import label_generated_completion  # noqa: E402
from realistic_niah_v4.counter_channel_interventions import (  # noqa: E402
    run_counter_subspace_conditions,
    stable_intervention_seed,
)
from realistic_niah_v4.modeling import (  # noqa: E402
    capture_post_block_states,
    generate_answer_completion,
    load_registered_model,
)
from realistic_niah_v4.prompts import PromptEncoding, render_v4_prompt  # noqa: E402
from realistic_niah_v4.spec import V4Config, resolve_model_spec  # noqa: E402
from realistic_niah_v4.stimuli import load_stimuli  # noqa: E402


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _site_positions(encoding: PromptEncoding, site: dict[str, Any]) -> tuple[int, ...]:
    kind = str(site["kind"])
    if kind == "answer_query":
        return (int(encoding.query_position),)
    if kind == "slot_end":
        raw_occurrence = site["occurrence"]
        occurrence = (
            len(encoding.slot_spans)
            if str(raw_occurrence).lower() == "last"
            else int(raw_occurrence)
        )
        if not 1 <= occurrence <= len(encoding.slot_spans):
            raise ValueError(f"slot occurrence outside 1..{len(encoding.slot_spans)}")
        return (int(encoding.slot_spans[occurrence - 1].end) - 1,)
    if kind == "needle_end":
        raw_occurrence = site["occurrence"]
        occurrence = (
            len(encoding.needle_spans)
            if str(raw_occurrence).lower() == "last"
            else int(raw_occurrence)
        )
        if not 1 <= occurrence <= len(encoding.needle_spans):
            raise ValueError(
                f"active needle occurrence outside 1..{len(encoding.needle_spans)}"
            )
        return (int(encoding.needle_spans[occurrence - 1].end) - 1,)
    raise ValueError(f"Unknown intervention site kind: {kind}")


def _load_basis(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as payload:
        missing = {"basis", "center"} - set(payload.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        result = {
            "basis": torch.from_numpy(np.asarray(payload["basis"], dtype=np.float32)),
            "center": torch.from_numpy(np.asarray(payload["center"], dtype=np.float32)),
        }
    return result


def _resolve(jobs_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else jobs_path.parent / path


def _label(completion: dict[str, Any], gold_count: int) -> dict[str, Any]:
    return label_generated_completion(
        str(completion["completion_text"]), gold_count=int(gold_count)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU runner for projected counter-subspace patch/removal/mediation"
    )
    parser.add_argument("--v4-config", type=Path, required=True)
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    config = V4Config.from_json(args.v4_config)
    jobs_path = args.jobs.resolve()
    job_document = json.loads(jobs_path.read_text(encoding="utf-8"))
    jobs = job_document.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs JSON must contain a non-empty jobs list")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_design = {
        "schema_version": "realistic_niah_v4_4_counter_subspace_intervention_v1",
        "v4_config": str(args.v4_config.resolve()),
        "stimuli": str(args.stimuli.resolve()),
        "jobs": str(jobs_path),
        "max_new_tokens": args.max_new_tokens,
        "job_count": len(jobs),
        "resource": "GPU",
    }
    design_path = output / "design.json"
    if design_path.exists() and not args.overwrite:
        observed = json.loads(design_path.read_text(encoding="utf-8"))
        if observed != resolved_design:
            raise RuntimeError(f"Existing design mismatch: {design_path}")
    else:
        design_path.write_text(
            json.dumps(resolved_design, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    stimulus_rows = {
        (int(row["seed"]), int(row["gold_count"])): row
        for row in load_stimuli(args.stimuli)
        if str(row.get("design_variant")) == "v4.4"
    }
    by_model: dict[str, list[dict[str, Any]]] = {}
    for index, job in enumerate(jobs):
        job = dict(job)
        job.setdefault("job_id", f"job_{index:05d}")
        by_model.setdefault(str(job["model_label"]), []).append(job)

    completed = 0
    for model_label, model_jobs in by_model.items():
        model, tokenizer, adapter = load_registered_model(
            resolve_model_spec(model_label),
            cache_dir=args.cache_dir,
            device_map=args.device_map,
            torch_dtype=config.model_torch_dtype,
            attention_backend=config.attention_prefix_backend,
        )
        encoding_cache: dict[tuple[int, int], PromptEncoding] = {}

        def encoding(seed: int, count: int) -> PromptEncoding:
            key = (int(seed), int(count))
            if key not in encoding_cache:
                if key not in stimulus_rows:
                    raise KeyError(f"Missing V4.4 stimulus seed={seed}, count={count}")
                encoding_cache[key] = render_v4_prompt(
                    stimulus_rows[key],
                    tokenizer=tokenizer,
                    model_spec=resolve_model_spec(model_label),
                    config=config,
                    answer_format="numeric",
                )
            return encoding_cache[key]

        for job in model_jobs:
            job_id = str(job["job_id"])
            destination = output / model_label / f"{job_id}.json"
            if destination.is_file() and not args.overwrite:
                completed += 1
                continue
            seed = int(job["seed"])
            receiver_count = int(job["receiver_count"])
            donor_count = int(job["donor_count"])
            receiver = encoding(seed, receiver_count)
            donor = encoding(seed, donor_count)
            source_layer = int(job["source_layer"])
            source_site = dict(job["source_site"])
            donor_source_site = dict(job.get("donor_source_site", source_site))
            receiver_positions = _site_positions(receiver, source_site)
            donor_positions = _site_positions(donor, donor_source_site)
            _, receiver_capture = capture_post_block_states(
                model,
                adapter,
                receiver,
                receiver_positions,
                layers=[source_layer],
            )
            _, donor_capture = capture_post_block_states(
                model,
                adapter,
                donor,
                donor_positions,
                layers=[source_layer],
            )
            source_bundle = _load_basis(
                _resolve(jobs_path, str(job["source_basis_path"]))
            )
            mediator_layer = job.get("mediator_layer")
            mediator_site = dict(job.get("mediator_site", {"kind": "answer_query"}))
            mediator_positions = (
                None if mediator_layer is None else _site_positions(receiver, mediator_site)
            )
            mediator_bundle = (
                None
                if mediator_layer is None
                else _load_basis(
                    _resolve(jobs_path, str(job["mediator_basis_path"]))
                )
            )

            clean_receiver = generate_answer_completion(
                model,
                tokenizer,
                receiver,
                max_new_tokens=args.max_new_tokens,
            )
            clean_donor = generate_answer_completion(
                model,
                tokenizer,
                donor,
                max_new_tokens=args.max_new_tokens,
            )
            conditions = run_counter_subspace_conditions(
                model,
                tokenizer,
                adapter,
                receiver,
                source_layer=source_layer,
                source_positions=receiver_positions,
                receiver_source_state=receiver_capture[source_layer],
                donor_source_state=donor_capture[source_layer],
                source_basis=source_bundle["basis"],
                mediator_layer=None if mediator_layer is None else int(mediator_layer),
                mediator_positions=mediator_positions,
                mediator_center=None if mediator_bundle is None else mediator_bundle["center"],
                mediator_basis=None if mediator_bundle is None else mediator_bundle["basis"],
                removal_dose=float(job.get("removal_dose", 1.0)),
                random_seed=stable_intervention_seed(
                    f"{model_label}:{job_id}:{seed}:{receiver_count}:{donor_count}"
                ),
                max_new_tokens=args.max_new_tokens,
            )
            labeled_conditions = {
                name: {
                    "generation": completion,
                    "outcome": _label(completion, receiver_count),
                }
                for name, completion in conditions.items()
                if name != "_audit"
            }
            payload = {
                "schema_version": "realistic_niah_v4_4_counter_subspace_intervention_shard_v1",
                "job": job,
                "model_label": model_label,
                "seed": seed,
                "receiver_count": receiver_count,
                "donor_count": donor_count,
                "receiver_stimulus_id": receiver.stimulus_id,
                "donor_stimulus_id": donor.stimulus_id,
                "source_positions": list(receiver_positions),
                "donor_source_positions": list(donor_positions),
                "mediator_positions": None if mediator_positions is None else list(mediator_positions),
                "clean_receiver": {
                    "generation": clean_receiver,
                    "outcome": _label(clean_receiver, receiver_count),
                },
                "clean_donor": {
                    "generation": clean_donor,
                    "outcome": _label(clean_donor, donor_count),
                },
                "conditions": labeled_conditions,
                "intervention_audit": conditions["_audit"],
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(destination)
            completed += 1
            print(
                f"[counter-subspace] {model_label} {job_id} "
                f"{receiver_count}->{donor_count} ({completed}/{len(jobs)})",
                flush=True,
            )
        del model, tokenizer, adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    complete = {
        **resolved_design,
        "completed_jobs": completed,
        "elapsed_seconds": time.perf_counter() - started,
        "status": "PASS" if completed == len(jobs) else "INCOMPLETE",
    }
    (output / "complete.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(complete, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
