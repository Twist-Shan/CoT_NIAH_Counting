from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class HybridOrchestrationTest(unittest.TestCase):
    def test_job_ledger_and_validator_cover_empty_confirmation_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "runs"
            model_root = run_root / "ToyModel"
            registry = model_root / "registries" / "all" / "selected_anchor_registry.jsonl"
            registry_rows = [
                {
                    "request_id": "request-a",
                    "anchor_equivalence_id": "anchor-a",
                    "seed": 1,
                    "target_grammar_class": "grammar_a",
                },
                {
                    "request_id": "request-b",
                    "anchor_equivalence_id": "anchor-b",
                    "seed": 2,
                    "target_grammar_class": "grammar_b",
                },
            ]
            _write_jsonl(registry, registry_rows)
            spec = {
                "scientific_contract": {
                    "selection_metric": "target_source_attention_mass",
                    "selection_aggregation": "equal_seed_mean_of_within_seed_event_means",
                    "intervention_start_anchor_role": "p0_item_end",
                    "decode_head_ablation_steps": -1,
                    "random_control_repeats": 3,
                },
                "models": {
                    "ToyModel": {
                        "registry_rows": 2,
                        "registry_sha256": hashlib.sha256(
                            registry.read_bytes()
                        ).hexdigest(),
                        "doses": [1, 2],
                        "primary_bank_size": 2,
                        "grammars": {
                            "grammar_a": "p0_item_end",
                            "grammar_b": "post_marker",
                        },
                    }
                },
            }
            spec_path = root / "spec.json"
            config_path = root / "config.json"
            _write_json(spec_path, spec)
            _write_json(
                config_path,
                {
                    "causal_development_seeds": [1],
                    "causal_confirmation_seeds": [2],
                },
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/partition_v5_native_hybrid_anchor_registry.py"),
                    "--spec",
                    str(spec_path),
                    "--model",
                    "ToyModel",
                    "--registry",
                    str(registry),
                    "--output",
                    str(model_root / "registries" / "by_grammar"),
                ],
                check=True,
                cwd=ROOT,
            )
            for grammar, role in spec["models"]["ToyModel"]["grammars"].items():
                for k in (1, 2):
                    plan = (
                        model_root
                        / "plans"
                        / grammar
                        / role
                        / f"k{k}"
                        / "retrieval_anchor_bank_plan.csv"
                    )
                    plan.parent.mkdir(parents=True, exist_ok=True)
                    plan.write_text(
                        "condition,repeat\n"
                        "selected_bank,0\n"
                        "global_random,1\n"
                        "global_random,2\n"
                        "global_random,3\n",
                        encoding="utf-8",
                    )
            jobs = root / "jobs.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/build_v5_native_hybrid_behavior_jobs.py"),
                    "--spec",
                    str(spec_path),
                    "--config",
                    str(config_path),
                    "--run-root",
                    str(run_root),
                    "--output",
                    str(jobs),
                ],
                check=True,
                cwd=ROOT,
            )
            job_rows = [
                json.loads(line)
                for line in jobs.read_text(encoding="utf-8").splitlines()
            ]
            empty = [
                row
                for row in job_rows
                if row["grammar"] == "grammar_a" and row["bank_size"] == 1
            ]
            self.assertEqual(empty[0]["execution_status"], "skipped_empty_split")

            _write_jsonl(
                model_root / "behaviors" / "clean_full" / "shards" / "clean.jsonl",
                [
                    {"condition": "clean", "behavior_outcome": "correct_next_needle"},
                    {"condition": "clean", "behavior_outcome": "correct_next_needle"},
                ],
            )
            for job in job_rows:
                if job["execution_status"] == "skipped_empty_split":
                    continue
                split = (
                    "discovery"
                    if job["grammar"] == "grammar_a"
                    else "confirmation"
                )
                result_rows = []
                for condition, repeats in (("selected_bank", 1), ("global_random", 3)):
                    for _repeat in range(repeats):
                        result_rows.append(
                            {
                                "condition": condition,
                                "split": split,
                                "behavior_outcome": "correct_next_needle",
                                "routed_target_grammar_class": job["grammar"],
                                "head_selection_anchor_role": job[
                                    "selection_anchor_role"
                                ],
                                "intervention_start_anchor_role": "p0_item_end",
                                "head_ablation_decode_steps_requested": -1,
                                "selection_intervention_site_decoupled": job[
                                    "selection_intervention_site_decoupled"
                                ],
                                "head_ablation_selected_post_zero_max_abs": 0.0,
                            }
                        )
                _write_jsonl(
                    Path(job["output"]) / "shards" / "results.jsonl", result_rows
                )

            analysis = root / "analysis"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/validate_v5_native_hybrid_localizer_p0_ablation.py"),
                    "--spec",
                    str(spec_path),
                    "--config",
                    str(config_path),
                    "--run-root",
                    str(run_root),
                    "--model",
                    "ToyModel",
                    "--jobs",
                    str(jobs),
                    "--output",
                    str(analysis),
                ],
                check=True,
                cwd=ROOT,
            )
            completion = json.loads(
                (analysis / "hybrid_localizer_p0_ablation_complete.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(completion["status"], "PASS")
            self.assertEqual(completion["registry_rows"], 2)
            self.assertEqual(len(completion["overall"]), 4)


if __name__ == "__main__":
    unittest.main()
