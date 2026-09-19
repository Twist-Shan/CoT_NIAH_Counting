from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from realistic_niah_v3_1.engine import formal_engine_config, shell_settings
from realistic_niah.runner import _ordered_id_digest
from realistic_niah_v3_1.resume import (
    audit_resume_manifests,
    expected_resume_signature,
    parse_resume_commits,
    validate_resume_manifest,
)
from realistic_niah_v3_1.sharding import formal_shard_plan
from realistic_niah_v3_1.spec import MODEL_LABELS


def _manifest_for(task: dict, commit: str) -> dict:
    signature = expected_resume_signature(
        task,
        stimuli_sha256="a" * 64,
        request_ids_sha256="b" * 64,
    )
    manifest = deepcopy(signature)
    model_id = manifest.pop("model_id")
    manifest["model"] = {
        "label": task["model_label"],
        "model_id": model_id,
    }
    manifest["model_revision"] = task["model_revision"]
    manifest["prompt_modes"] = [task["prompt_mode"]]
    manifest["git"] = {"commit": commit, "dirty": False}
    return manifest


def test_every_v31_model_has_one_registered_engine_configuration() -> None:
    configs = {label: formal_engine_config(label) for label in MODEL_LABELS}
    assert set(configs) == set(MODEL_LABELS)
    assert configs["Gemma4-31B"].tensor_parallel_size == 2
    assert configs["Gemma4-31B"].enforce_eager is True
    assert configs["Gemma4-31B"].disable_custom_all_reduce is True
    assert shell_settings("Qwen3-32B", 1) == "1 1 1 0.92 0 0"
    assert shell_settings("Gemma4-31B", 2) == "2 1 1 0.92 1 1"
    with pytest.raises(ValueError, match="requires tensor_parallel_size=2"):
        formal_engine_config("Gemma4-31B", tensor_parallel_size=1)


def test_legacy_false_engine_fields_are_compatible_only_with_allowlisted_commit() -> None:
    task = formal_shard_plan()["tasks"][0]
    old_commit = "1" * 40
    current_commit = "2" * 40
    manifest = _manifest_for(task, old_commit)
    manifest["engine"].pop("enforce_eager")
    manifest["engine"].pop("disable_custom_all_reduce")

    assert validate_resume_manifest(
        manifest,
        task,
        stimuli_sha256="a" * 64,
        request_ids_sha256="b" * 64,
        current_commit=current_commit,
        allowed_commits={old_commit},
    ) == old_commit
    with pytest.raises(RuntimeError, match="not explicitly authorized"):
        validate_resume_manifest(
            manifest,
            task,
            stimuli_sha256="a" * 64,
            request_ids_sha256="b" * 64,
            current_commit=current_commit,
            allowed_commits=set(),
        )


def test_semantic_engine_or_protocol_changes_are_never_normalized_away() -> None:
    task = formal_shard_plan()["tasks"][0]
    commit = "3" * 40
    baseline = _manifest_for(task, commit)
    for mutate in (
        lambda value: value["engine"].__setitem__(
            "max_num_seqs", value["engine"]["max_num_seqs"] + 1
        ),
        lambda value: value.__setitem__("query_layout", "wrong_layout"),
        lambda value: value.__setitem__("request_ids_sha256", "c" * 64),
        lambda value: value.__setitem__("checkpoint_strategy", "legacy"),
    ):
        manifest = deepcopy(baseline)
        mutate(manifest)
        with pytest.raises(RuntimeError, match="incompatible with the frozen run"):
            validate_resume_manifest(
                manifest,
                task,
                stimuli_sha256="a" * 64,
                request_ids_sha256="b" * 64,
                current_commit=commit,
                allowed_commits=set(),
            )


def test_resume_commit_parser_rejects_partial_or_malformed_shas() -> None:
    first = "a" * 40
    second = "b" * 40
    assert parse_resume_commits(f"{first}:{second}") == {first, second}
    assert parse_resume_commits("") == set()
    for raw in ("abc", f"{first}:", f"{first}:UPPERCASE"):
        with pytest.raises(ValueError, match="full Git SHAs"):
            parse_resume_commits(raw)


def test_resume_audit_accepts_exact_legacy_manifest_and_rejects_extra_shard(
    tmp_path: Path,
) -> None:
    task = formal_shard_plan()["tasks"][0]
    old_commit = "4" * 40
    current_commit = "5" * 40
    stimuli = b'{"stimulus_id":"fixture-stimulus"}\n'
    stimuli_path = tmp_path / "dataset" / "stimuli.jsonl"
    stimuli_path.parent.mkdir(parents=True)
    stimuli_path.write_bytes(stimuli)
    request_ids = (
        f"v3.1/{task['model_label']}/{task['prompt_mode']}"
        "/cue_before_query_after/fixture-stimulus",
    )
    request_ids_sha256 = _ordered_id_digest(request_ids)
    manifest = _manifest_for(task, old_commit)
    manifest["stimuli_sha256"] = hashlib.sha256(stimuli).hexdigest()
    manifest["request_ids_sha256"] = request_ids_sha256
    manifest["engine"].pop("enforce_eager")
    manifest["engine"].pop("disable_custom_all_reduce")
    manifest_path = (
        tmp_path / "shards" / task["task_id"] / "main" / "run_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = audit_resume_manifests(
        tmp_path,
        current_commit=current_commit,
        allowed_commits={old_commit},
    )
    assert result["passed"] is True
    assert result["manifests"] == 1
    assert result["observed_commits"] == [old_commit]

    extra = tmp_path / "shards" / "not-in-the-frozen-plan" / "main"
    extra.mkdir(parents=True)
    (extra / "run_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unexpected V3.1 shard manifests"):
        audit_resume_manifests(
            tmp_path,
            current_commit=current_commit,
            allowed_commits={old_commit},
        )
