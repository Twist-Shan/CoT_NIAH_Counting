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




def test_prepare_script_exports_the_frozen_protocol_version() -> None:
    assert prepare_realistic_niah_v3_1.PROTOCOL_VERSION == PROTOCOL_VERSION
