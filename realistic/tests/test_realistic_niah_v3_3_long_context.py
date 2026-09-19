from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import pytest

from realistic_niah.runner import load_vllm_runtime, request_id
from realistic_niah.spec import QUERY_LAYOUT
from realistic_niah_v3_1.spec import MODEL_SPECS
from realistic_niah_v3_3_long_context.engine import formal_engine_config
from realistic_niah_v3_3_long_context.integrity import (
    SEALED_DATASET_FILES,
    seal_frozen_dataset,
    sha256_file,
    validate_frozen_dataset,
)
from realistic_niah_v3_3_long_context.runner import (
    _validate_preflight_subset,
    finalize_run,
    validate_output,
)
from realistic_niah_v3_3_long_context.spec import (
    EXPECTED_REQUESTS,
    EXPECTED_REQUESTS_PER_MODEL,
    EXPECTED_STIMULI,
    FORMAL_PROMPT_MODES,
    MAX_MODEL_LEN,
    MODEL_CONTEXT_ENGINE_OVERRIDES,
    MODEL_IDS,
    MODEL_LABELS,
    MODEL_REVISIONS,
    NEEDLE_COUNTS,
    PASSAGE_LENGTHS,
    PROTOCOL_VERSION,
    SEEDS,
    V33_LONG_CONTEXT_FREEZE_PROTOCOL,
    V33_LONG_CONTEXT_RUN_PROTOCOL,
    WORKER_LENGTHS,
)
from realistic_niah_v3_3_long_context.stimuli import default_freeze_spec


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_registered_stimuli(path: Path) -> list[dict]:
    rows = [
        {
            "stimulus_id": f"V33LC_T{length}_N{count}_seed{seed}",
            "target_passage_tokens": length,
            "num_needles": count,
            "seed": seed,
        }
        for length in PASSAGE_LENGTHS
        for count in NEEDLE_COUNTS
        for seed in SEEDS
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def _make_worker_output(
    *,
    run_root: Path,
    stimuli: list[dict],
    model_label: str,
    worker_index: int,
) -> None:
    lengths = WORKER_LENGTHS[worker_index]
    output = run_root / "formal" / model_label / f"worker-{worker_index}" / "main"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    spec = MODEL_SPECS[model_label]
    for stimulus in stimuli:
        if stimulus["target_passage_tokens"] not in lengths:
            continue
        for mode in FORMAL_PROMPT_MODES:
            rows.append(
                {
                    "schema_version": V33_LONG_CONTEXT_RUN_PROTOCOL.request_schema_version,
                    "protocol_version": PROTOCOL_VERSION,
                    "request_id": request_id(
                        model_spec=spec,
                        prompt_mode=mode,
                        query_layout=QUERY_LAYOUT,
                        stimulus_id=stimulus["stimulus_id"],
                        namespace=V33_LONG_CONTEXT_RUN_PROTOCOL.request_id_namespace,
                    ),
                    "model_label": model_label,
                    "model_id": MODEL_IDS[model_label],
                    "model_revision": MODEL_REVISIONS[model_label],
                    "target_passage_tokens": stimulus["target_passage_tokens"],
                    "num_needles": stimulus["num_needles"],
                    "seed": stimulus["seed"],
                    "prompt_mode": mode,
                    "query_layout": QUERY_LAYOUT,
                }
            )
    requests_path = output / "requests.jsonl"
    with requests_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema_version": V33_LONG_CONTEXT_RUN_PROTOCOL.run_manifest_schema_version,
        "protocol_version": PROTOCOL_VERSION,
        "model": asdict(spec),
        "model_revision": MODEL_REVISIONS[model_label],
        "query_layout": QUERY_LAYOUT,
        "prompt_modes": list(FORMAL_PROMPT_MODES),
        "engine": asdict(formal_engine_config(model_label)),
        "model_engine_overrides": MODEL_CONTEXT_ENGINE_OVERRIDES[model_label],
        "expected_requests": len(rows),
        "completed_requests": len(rows),
        "completed_at_utc": "2026-09-06T00:00:00+00:00",
    }
    manifest_path = output / "run_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(output / "_SUCCESS.json", {"manifest_sha256": sha256_file(manifest_path)})


def _make_preflight_output(
    *,
    output: Path,
    model_label: str,
    corrupt_request_id: bool = False,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    spec = MODEL_SPECS[model_label]
    stimulus_id = "V33LC_T100000_N20_seed1234"
    rows = []
    for index, mode in enumerate(FORMAL_PROMPT_MODES):
        identifier = request_id(
            model_spec=spec,
            prompt_mode=mode,
            query_layout=QUERY_LAYOUT,
            stimulus_id=stimulus_id,
            namespace=V33_LONG_CONTEXT_RUN_PROTOCOL.request_id_namespace,
        )
        if corrupt_request_id and index == 0:
            identifier = "incorrect-preflight-request-id"
        rows.append(
            {
                "schema_version": V33_LONG_CONTEXT_RUN_PROTOCOL.request_schema_version,
                "protocol_version": PROTOCOL_VERSION,
                "request_id": identifier,
                "model_label": model_label,
                "model_id": MODEL_IDS[model_label],
                "model_revision": MODEL_REVISIONS[model_label],
                "target_passage_tokens": 100_000,
                "num_needles": 20,
                "seed": 1234,
                "prompt_mode": mode,
                "query_layout": QUERY_LAYOUT,
            }
        )
    with (output / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write_json(
        output / "run_manifest.json",
        {
            "schema_version": (
                V33_LONG_CONTEXT_RUN_PROTOCOL.run_manifest_schema_version
            ),
            "protocol_version": PROTOCOL_VERSION,
            "model": asdict(spec),
            "model_revision": MODEL_REVISIONS[model_label],
            "prompt_modes": list(FORMAL_PROMPT_MODES),
            "engine": asdict(formal_engine_config(model_label)),
            "model_engine_overrides": MODEL_CONTEXT_ENGINE_OVERRIDES[model_label],
            "expected_requests": 2,
            "completed_requests": 2,
            "completed_at_utc": "2026-09-06T00:00:00+00:00",
        },
    )


def test_registered_grid_models_and_request_accounting() -> None:
    assert PASSAGE_LENGTHS == (
        25_000,
        30_000,
        40_000,
        50_000,
        60_000,
        70_000,
        80_000,
        90_000,
        100_000,
    )
    assert MODEL_LABELS == ("Gemma4-31B", "Qwen3-32B")
    assert EXPECTED_STIMULI == 3_780
    assert EXPECTED_REQUESTS_PER_MODEL == 7_560
    assert EXPECTED_REQUESTS == 15_120
    assert MAX_MODEL_LEN == 131_072
    assert V33_LONG_CONTEXT_RUN_PROTOCOL.request_id_namespace == "v3.3-long-context"
    assert V33_LONG_CONTEXT_FREEZE_PROTOCOL.stimulus_id_prefix == "V33LC_"
    assigned = [value for values in WORKER_LENGTHS.values() for value in values]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == set(PASSAGE_LENGTHS)
    freeze = default_freeze_spec()
    assert freeze.passage_lengths == PASSAGE_LENGTHS
    assert freeze.needle_counts == NEEDLE_COUNTS
    assert freeze.seeds == SEEDS


def test_config_matches_registered_implementation() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "realistic_niah_v3_3_long_context.json").read_text(
            encoding="utf-8"
        )
    )
    assert tuple(config["target_passage_tokens"]) == PASSAGE_LENGTHS
    assert tuple(config["needle_counts"]) == NEEDLE_COUNTS
    assert tuple(config["seeds"]) == SEEDS
    assert tuple(item["label"] for item in config["models"]) == MODEL_LABELS
    assert config["expected_stimuli"] == EXPECTED_STIMULI
    assert config["expected_requests_per_model"] == EXPECTED_REQUESTS_PER_MODEL
    assert config["expected_requests"] == EXPECTED_REQUESTS
    assert config["engine"] == {
        key: value
        for key, value in asdict(formal_engine_config("Gemma4-31B")).items()
        if key
        in {
            "dtype",
            "tensor_parallel_size",
            "max_model_len",
            "gpu_memory_utilization",
            "max_num_seqs",
            "request_batch_size",
            "enforce_eager",
            "disable_custom_all_reduce",
            "enable_prefix_caching",
        }
    }
    qwen = next(item for item in config["models"] if item["label"] == "Qwen3-32B")
    assert qwen["context_extension"] == {
        "method": "YaRN",
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32_768,
        "validated_context_claim_tokens": 131_072,
    }


def test_vllm_runtime_forwards_registered_qwen_yarn(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            return object()

    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = FakeLLM  # type: ignore[attr-defined]
    fake_vllm.SamplingParams = object  # type: ignore[attr-defined]
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "vllm", fake_vllm)
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        "realistic_niah.runner.resolve_model_revision",
        lambda model_id, revision: MODEL_REVISIONS["Qwen3-32B"],
    )

    runtime = load_vllm_runtime(
        model_spec=MODEL_SPECS["Qwen3-32B"],
        revision=MODEL_REVISIONS["Qwen3-32B"],
        engine_config=formal_engine_config("Qwen3-32B"),
        additional_engine_overrides=MODEL_CONTEXT_ENGINE_OVERRIDES["Qwen3-32B"],
    )
    assert captured["max_model_len"] == 131_072
    assert captured["tensor_parallel_size"] == 2
    assert "rope_scaling" not in captured
    assert captured["hf_overrides"] == {
        "rope_parameters": {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32_768,
            "rope_theta": 1_000_000,
        },
    }
    assert runtime.engine_overrides == MODEL_CONTEXT_ENGINE_OVERRIDES["Qwen3-32B"]


def test_preflight_requires_exact_registered_request_ids(tmp_path: Path) -> None:
    output = tmp_path / "preflight"
    _make_preflight_output(output=output, model_label="Qwen3-32B")
    assert _validate_preflight_subset(
        model_label="Qwen3-32B", output_dir=output
    )["requests"] == 2

    _make_preflight_output(
        output=output,
        model_label="Qwen3-32B",
        corrupt_request_id=True,
    )
    with pytest.raises(RuntimeError, match="Preflight request IDs"):
        _validate_preflight_subset(model_label="Qwen3-32B", output_dir=output)


def test_anvil_layout_uses_two_tp2_tasks_and_independent_model_jobs() -> None:
    root = Path(__file__).resolve().parents[1]
    infra = root / "infra" / "anvil" / "realistic_niah_v3_3_long_context"
    slurm = (infra / "v3_3_long_context_inference.slurm").read_text(
        encoding="utf-8"
    )
    submit = (infra / "submit_anvil.sh").read_text(encoding="utf-8")
    assert "--ntasks=2" in slurm
    assert "--gpus-per-task=2" in slurm
    assert "--gpu-bind=per_task:2" in slurm
    assert slurm.count("srun --exact --exclusive") == 1
    assert 'models=("Gemma4-31B" "Qwen3-32B")' in submit
    assert "--export=" in submit
    assert "REALISTIC_NIAH_MODEL_LABEL=${model_label}" in submit


def test_anvil_freeze_job_requires_full_prompt_audit() -> None:
    root = Path(__file__).resolve().parents[1]
    infra = root / "infra" / "anvil" / "realistic_niah_v3_3_long_context"
    freeze = (infra / "v3_3_long_context_freeze.slurm").read_text(
        encoding="utf-8"
    )
    prepare = (
        root / "scripts" / "prepare_realistic_niah_v3_3_long_context.py"
    ).read_text(encoding="utf-8")
    assert "freeze_realistic_niah_v3_3_long_context.py" in freeze
    assert "audit_realistic_niah_v3_3_long_context_prompts.py" in freeze
    assert "V3.3_DATASET_AND_PROMPT_AUDIT_OK" in freeze
    assert 'run_root / "orchestration" / "prompt_audit.json"' in prepare
    assert "unique_request_ids" in prepare
    assert "maximum_total_budget" in prepare


def test_sealed_dataset_detects_tampering(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "stimuli.jsonl").write_text("{}\n" * EXPECTED_STIMULI, encoding="utf-8")
    _write_json(dataset / "manifest.json", {
        "protocol_version": PROTOCOL_VERSION,
        "rows": EXPECTED_STIMULI,
    })
    _write_json(dataset / "audit_report.json", {
        "protocol_version": PROTOCOL_VERSION,
        "passed": True,
        "rows_checked": EXPECTED_STIMULI,
    })
    for filename in SEALED_DATASET_FILES:
        path = dataset / filename
        if not path.exists():
            path.write_text(f"{filename}\n", encoding="utf-8")
    seal_frozen_dataset(dataset)
    seal_sha256 = sha256_file(dataset / "dataset_seal.json")
    assert validate_frozen_dataset(
        dataset, expected_seal_sha256=seal_sha256
    )["rows"] == EXPECTED_STIMULI
    (dataset / "cell_counts.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Frozen dataset mismatch"):
        validate_frozen_dataset(dataset, expected_seal_sha256=seal_sha256)


def test_finalizer_requires_both_models_and_exact_grid(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "realistic_niah_v3_3_long_context" / "test"
    stimuli_path = run_root / "dataset" / "stimuli.jsonl"
    stimuli = _write_registered_stimuli(stimuli_path)
    for model_label in MODEL_LABELS:
        for worker_index in WORKER_LENGTHS:
            _make_worker_output(
                run_root=run_root,
                stimuli=stimuli,
                model_label=model_label,
                worker_index=worker_index,
            )
            audit = validate_output(
                model_label=model_label,
                stimuli_path=stimuli_path,
                output_dir=(
                    run_root
                    / "formal"
                    / model_label
                    / f"worker-{worker_index}"
                    / "main"
                ),
                lengths=WORKER_LENGTHS[worker_index],
            )
            assert audit["requests"] in {3_360, 4_200}

    audit = finalize_run(stimuli_path=stimuli_path, run_root=run_root)
    assert audit["passed"]
    assert audit["requests"] == EXPECTED_REQUESTS
    assert audit["unique_request_ids"] == EXPECTED_REQUESTS
    assert audit["cells"] == len(MODEL_LABELS) * 2 * 9 * 14
    assert (run_root / "results" / "_SUCCESS.json").is_file()
