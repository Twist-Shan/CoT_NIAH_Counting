from __future__ import annotations

"""Build the frozen confirmation-seed counter-subspace intervention design."""

import argparse
import json
from pathlib import Path


MODEL_LAYERS = {
    "Qwen3-8B": {"source": 28, "mediator": 29},
    "Gemma4-E4B": {"source": 36, "mediator": 37},
}
PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--basis-root", default="../analysis/subspaces_discovery/bases")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1254, 1264)))
    parser.add_argument("--bidirectional", action="store_true")
    args = parser.parse_args()
    pairs = [*PAIRS, *([(right, left) for left, right in PAIRS] if args.bidirectional else [])]
    jobs = []
    for model, layers in MODEL_LAYERS.items():
        for seed in args.seeds:
            for receiver, donor in pairs:
                job_id = f"{model.replace('-', '_')}__seed{seed}__{receiver}_to_{donor}"
                jobs.append({
                    "job_id": job_id,
                    "model_label": model,
                    "seed": seed,
                    "receiver_count": receiver,
                    "donor_count": donor,
                    "source_layer": layers["source"],
                    "source_site": {"kind": "needle_end", "occurrence": "last"},
                    "source_basis_path": (
                        f"{args.basis_root}/{model}__prompt_running__L{layers['source']:02d}.npz"
                    ),
                    "mediator_layer": layers["mediator"],
                    "mediator_site": {"kind": "answer_query"},
                    "mediator_basis_path": (
                        f"{args.basis_root}/{model}__answer_query__L{layers['mediator']:02d}.npz"
                    ),
                    "removal_dose": 1.0,
                })
    document = {
        "schema_version": "realistic_niah_v4_4_counter_subspace_jobs_v1",
        "fit_split": "discovery",
        "test_split": "confirmation",
        "test_seeds": args.seeds,
        "pair_directionality": "bidirectional" if args.bidirectional else "increasing_only",
        "layer_freeze": MODEL_LAYERS,
        "layer_rationale": {
            "Qwen3-8B": "late prompt counter stability at L28; established answer integration at L29",
            "Gemma4-E4B": "late prompt counter stability at L36; established answer integration at L37",
        },
        "conditions": [
            "projected_patch",
            "orthogonal_norm_matched",
            "projected_patch_plus_removal",
            "removal_only",
        ],
        "jobs": jobs,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "jobs": len(jobs)}, indent=2))


if __name__ == "__main__":
    main()
