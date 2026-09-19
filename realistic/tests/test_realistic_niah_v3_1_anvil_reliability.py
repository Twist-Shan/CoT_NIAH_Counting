from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from realistic_niah_v3_1 import integrity
from realistic_niah_v3_1.shard_state import audit_shard_state
from realistic_niah_v3_1.spec import PROTOCOL_VERSION
from scripts import prepare_realistic_niah_v3_1


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_frozen_dataset_validator_records_revision_and_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {
        "stimuli.jsonl": b'{"id": 1}\n{"id": 2}\n',
        "manifest.json": json.dumps({"protocol_version": PROTOCOL_VERSION}).encode(),
        "audit_report.json": json.dumps(
            {"protocol_version": PROTOCOL_VERSION, "passed": True, "rows_checked": 2}
        ).encode(),
    }
    for filename, payload in payloads.items():
        (tmp_path / filename).write_bytes(payload)
    monkeypatch.setattr(integrity, "EXPECTED_STIMULI", 2)
    monkeypatch.setattr(
        integrity,
        "EXPECTED_DATASET_FILES",
        {
            filename: {"bytes": len(payload), "sha256": _sha256(payload)}
            for filename, payload in payloads.items()
        },
    )

    result = integrity.validate_frozen_dataset(
        tmp_path, require_source_revision=True, record_source_revision=True
    )
    assert result["dataset_id"] == integrity.DATASET_ID
    assert result["revision"] == integrity.DATASET_REVISION
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dataset mismatch"):
        integrity.validate_frozen_dataset(tmp_path)


def _write_tsv(path: Path, header: tuple[str, ...], values: tuple[object, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\t".join(header) + "\n" + "\t".join(map(str, values)) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _complete_state(root: Path) -> tuple[list[str], list[str]]:
    plan_path = root / "orchestration" / "formal_bundles.tsv"
    plan_path.parent.mkdir(parents=True)
    plan_header = (
        "bundle_id",
        "priority",
        "model_label",
        "expected_logical_shards",
        "expected_requests",
        "model_revision",
        "prompt_modes",
        "logical_task_ids",
    )
    plan_lines = ["\t".join(plan_header)]
    all_tasks: list[str] = []
    all_bundles: list[str] = []
    for index in range(14):
        mode_count = 4 if index < 6 else 3
        bundle_id = f"bundle-{index:02d}"
        model = f"model-{index:02d}"
        modes = [f"mode-{slot}" for slot in range(mode_count)]
        task_ids = [f"{model}__{mode}" for mode in modes]
        requests = mode_count * 3_360
        plan_lines.append(
            "\t".join(
                (
                    bundle_id,
                    str(index),
                    model,
                    str(mode_count),
                    str(requests),
                    "a" * 40,
                    ",".join(modes),
                    ",".join(task_ids),
                )
            )
        )
        all_bundles.append(bundle_id)
        all_tasks.extend(task_ids)
        for task_id, mode in zip(task_ids, modes, strict=True):
            _write_tsv(
                root / "orchestration" / "shard_state" / "completed" / f"{task_id}.tsv",
                (
                    "task_id",
                    "model",
                    "prompt_mode",
                    "worker_id",
                    "attempt_id",
                    "completed_at_utc",
                ),
                (task_id, model, mode, "worker", "attempt", "2026-08-19T00:00:00Z"),
            )
            manifest_path = root / "shards" / task_id / "main" / "run_manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "completed_requests": 3_360,
                        "expected_requests": 3_360,
                        "prompt_payload_storage": "sha256_only",
                    }
                ),
                encoding="utf-8",
            )
        _write_tsv(
            root
            / "orchestration"
            / "shard_state"
            / "completed_bundles"
            / f"{bundle_id}.tsv",
            (
                "bundle_id",
                "model",
                "logical_shards",
                "requests",
                "worker_id",
                "attempt_id",
                "completed_at_utc",
            ),
            (
                bundle_id,
                model,
                mode_count,
                requests,
                "worker",
                "attempt",
                "2026-08-19T00:00:00Z",
            ),
        )
    plan_path.write_text("\n".join(plan_lines) + "\n", encoding="utf-8", newline="\n")
    (root / "orchestration" / "shard_state" / "failed_bundles").mkdir(parents=True)
    return all_tasks, all_bundles


def test_state_audit_accepts_only_complete_exact_marker_sets(tmp_path: Path) -> None:
    tasks, bundles = _complete_state(tmp_path)
    result = audit_shard_state(tmp_path)
    assert result == {
        "passed": True,
        "physical_bundles": 14,
        "logical_shards": 48,
        "requests": 161_280,
    }

    truncated = (
        tmp_path
        / "orchestration"
        / "shard_state"
        / "completed_bundles"
        / f"{bundles[0]}.tsv"
    )
    truncated.write_text(
        "bundle_id\tmodel\tlogical_shards\trequests\tworker_id\tattempt_id\tcompleted_at_utc\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="exactly one full row"):
        audit_shard_state(tmp_path, bundles[0])

    truncated.unlink()
    with pytest.raises(RuntimeError, match="Missing completion marker"):
        audit_shard_state(tmp_path)
    assert len(tasks) == 48


def test_anvil_adapter_has_bounded_finalization_and_explicit_exports() -> None:
    root = Path(__file__).resolve().parents[1]
    finalizer = (root / "scripts" / "finalize_realistic_niah_v3_1.sh").read_text()
    worker = (root / "scripts" / "run_realistic_niah_v3_1_worker.sh").read_text()
    submit = (
        root / "infra" / "anvil" / "realistic_niah_v3_1" / "submit_anvil.sh"
    ).read_text()
    slurm = (
        root / "infra" / "anvil" / "realistic_niah_v3_1" / "v3_1_inference.slurm"
    ).read_text()
    task_launcher = (
        root / "infra" / "anvil" / "realistic_niah_v3_1" / "run_slurm_task.sh"
    ).read_text()
    mixed_submit = (
        root / "infra" / "anvil" / "realistic_niah_v3_1" / "submit_anvil_mixed.sh"
    ).read_text()
    mixed_slurm = (
        root
        / "infra"
        / "anvil"
        / "realistic_niah_v3_1"
        / "v3_1_mixed_inference.slurm"
    ).read_text()
    split_submit = (
        root / "infra" / "anvil" / "realistic_niah_v3_1" / "submit_anvil_split.sh"
    ).read_text()
    split_slurm = (
        root
        / "infra"
        / "anvil"
        / "realistic_niah_v3_1"
        / "v3_1_split_inference.slurm"
    ).read_text()
    split_finalizer = (
        root / "scripts" / "finalize_realistic_niah_v3_1_split_group.sh"
    ).read_text()
    assert "while true" not in finalizer
    assert "audit_realistic_niah_v3_1_shard_state.py" in finalizer
    assert "write_two_row_marker" in worker
    assert "--export=ALL" not in submit
    assert "--expected-commit" in submit
    assert "source /etc/profile.d/modules.sh" in slurm
    assert slurm.index("source /etc/profile.d/modules.sh") < slurm.index("set -u")
    assert (
        'cuda_home="${REALISTIC_NIAH_CUDA_HOME:-/apps/anvilgpu/external/apps/cuda-toolkit/12.8.0}"'
        in slurm
    )
    assert 'export CUDA_HOME="${cuda_home}"' in slurm
    assert 'export CUDA_PATH="${cuda_home}"' in slurm
    assert 'export PATH="${env_bin}:${CUDA_HOME}/bin:${PATH}"' in slurm
    assert '[[ -e "${CUDA_HOME}/lib64/libcudart.so.12" ]]' in slurm
    assert (
        'export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"'
        in slurm
    )
    assert "-gencode=arch=compute_90a,code=sm_90a -c -o /dev/null -" in slurm
    assert "command -v ninja" in slurm
    assert "export VLLM_USE_FLASHINFER_SAMPLER=0" in slurm
    assert (
        'task_script="${repo}/infraoutputs/external/anvil_realistic_niah_v3_1_run_slurm_task.sh"'
        in slurm
    )
    assert 'bash "${task_script}" "${run_root}"' in slurm
    assert "--export=HOME,USER,PATH,SHELL" in slurm
    assert "CUDA_HOME,CUDA_PATH,LD_LIBRARY_PATH,VLLM_USE_FLASHINFER_SAMPLER" in slurm
    assert "CUDA runtime environment did not reach task" in task_launcher
    assert 'ctypes.CDLL("libcudart.so.12")' in task_launcher
    assert "Visible GPU count does not match GPUs per task" in task_launcher
    assert "Tensor parallel size does not match GPUs per task" in task_launcher
    assert "VLLM_TP2_PREFLIGHT_OK" in task_launcher
    assert 'version("vllm")=="0.25.1"' in task_launcher
    assert "REALISTIC_NIAH_WORKER_NAMESPACE" in task_launcher
    assert "REALISTIC_NIAH_WORKER_OFFSET" in task_launcher
    assert "-m realistic_niah_v3_1.engine" in worker
    assert 'bundle_command+=(--enforce-eager)' in worker
    assert 'bundle_command+=(--disable-custom-all-reduce)' in worker
    assert '[[ -z "${model_filter}" || "${model}" == "${model_filter}" ]]' in worker
    assert '[[ -z "${model_exclude}" || "${model}" != "${model_exclude}" ]]' in worker
    assert '--tensor-parallel-size "${tensor_parallel_size}"' in worker
    assert 'gpus_per_task="${REALISTIC_NIAH_GPUS_PER_TASK:-1}"' in slurm
    assert (
        'tensor_parallel_size="${REALISTIC_NIAH_TENSOR_PARALLEL_SIZE:-1}"'
        in slurm
    )
    assert '[[ "${tensor_parallel_size}" == "${gpus_per_task}" ]]' in slurm
    assert '[[ "${model_filter}" != "Gemma4-31B" ]]' in slurm
    assert '--gpus-per-task="${gpus_per_task}"' in slurm
    assert '--gpu-bind="per_task:${gpus_per_task}"' in slurm
    assert 'if [[ "${finalize_mode}" == "1" ]]' in slurm
    assert 'dirname -- "${BASH_SOURCE[0]}"' not in slurm
    assert "--gpus-per-node=4" in mixed_submit
    assert "--nodes=2" in mixed_submit
    assert "--dependency" in mixed_submit
    assert "--resume-from-commits" in mixed_submit
    assert 'layout=mixed_8gpu' in mixed_slurm
    assert "REALISTIC_NIAH_RESUME_FROM_COMMITS" in mixed_slurm
    assert 'REALISTIC_NIAH_MODEL_FILTER=Gemma4-31B' in mixed_slurm
    assert mixed_slurm.count('REALISTIC_NIAH_MODEL_EXCLUDE=Gemma4-31B') == 3
    assert 'REALISTIC_NIAH_TENSOR_PARALLEL_SIZE=2' in mixed_slurm
    assert 'REALISTIC_NIAH_WORKER_NAMESPACE=topup' in mixed_slurm
    assert 'REALISTIC_NIAH_WORKER_OFFSET=6' in mixed_slurm
    assert mixed_slurm.index('wait "${gemma_pid}"') < mixed_slurm.index(
        'REALISTIC_NIAH_WORKER_NAMESPACE=topup'
    )
    assert mixed_slurm.index('wait "${topup_pid}"') < mixed_slurm.index(
        'finalize_realistic_niah_v3_1.sh'
    )
    assert "two independent four-H100" in split_submit
    assert split_submit.count("--nodes=1") == 1
    assert split_submit.count("--gpus-per-node=4") == 1
    assert "REALISTIC_NIAH_SPLIT_ROLE=gemma4" in split_submit
    assert "REALISTIC_NIAH_SPLIT_ROLE=general4" in split_submit
    assert "--dependency" not in split_submit
    assert 'layout=split_4gpu' in split_slurm
    assert '[[ "${SLURM_NNODES}" == "1" ]]' in split_slurm
    assert 'REALISTIC_NIAH_MODEL_FILTER=Gemma4-31B' in split_slurm
    assert split_slurm.count('REALISTIC_NIAH_MODEL_EXCLUDE=Gemma4-31B') == 3
    assert 'REALISTIC_NIAH_TENSOR_PARALLEL_SIZE=2' in split_slurm
    assert 'REALISTIC_NIAH_WORKER_NAMESPACE=gemma-topup' in split_slurm
    assert 'flock -x 8' in split_slurm
    assert 'audit_realistic_niah_v3_1_resume_manifests.py' in split_slurm
    assert 'audit_realistic_niah_v3_1_resume_manifests.py' in split_submit
    merge = (root / "src" / "realistic_niah_v3_1" / "merge.py").read_text()
    assert "audit_resume_manifests" in merge
    assert "permitted_manifest_commits" in merge
    assert 'finalize_realistic_niah_v3_1_split_group.sh' in split_slurm
    assert 'flock -x 9' in split_finalizer
    assert 'gemma4.done' in split_finalizer
    assert 'general4.done' in split_finalizer
    assert split_finalizer.index('gemma4.done') < split_finalizer.index(
        'finalize_realistic_niah_v3_1.sh'
    )


def test_prepare_script_exports_the_frozen_protocol_version() -> None:
    assert prepare_realistic_niah_v3_1.PROTOCOL_VERSION == PROTOCOL_VERSION
