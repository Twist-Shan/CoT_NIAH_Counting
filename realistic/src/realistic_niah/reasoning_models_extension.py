from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

from .spec import (
    FORMAL_PROMPT_MODES,
    NONTHINKING_PROMPT_MODES,
    QUERY_LAYOUT,
    REASONING_EXTENSION_MODEL_REVISIONS,
    REASONING_EXTENSION_MODEL_SPECS,
)

REASONING_EXTENSION_PLAN_SCHEMA = (
    "realistic_niah_reasoning_models_extension_plan_v1"
)
REASONING_EXTENSION_PROTOCOL = (
    "realistic_niah_v2_reasoning_models_extension_v1"
)
SOURCE_FORMAL_STIMULI_SHA256 = (
    "b739122c96adf73ec6df4abe0266af239a026b4de6f09f309933231f604c7f71"
)
PLAN_BASENAME = "reasoning_models_extension_shards"
RUN_NAME_PREFIXES = (
    "reasoning_models_extension_",
    "reasoning_models_smoke_",
)
EXPECTED_STIMULI_PER_SHARD = 500
EXPECTED_EXTENSION_SHARDS = 20
EXPECTED_EXTENSION_REQUESTS = 10_000

LOGICAL_GROUPS: dict[str, dict[str, Any]] = {
    "Nemotron-Nano-v2-9B": {
        "comparison_type": "same_checkpoint_thinking_toggle",
        "variants": ("Nemotron-Nano-v2-9B",),
    },
    "Nemotron-3-Nano-4B": {
        "comparison_type": "same_checkpoint_thinking_toggle",
        "variants": ("Nemotron-3-Nano-4B",),
    },
    "Granite-3.3-Instruct-8B": {
        "comparison_type": "same_checkpoint_thinking_toggle",
        "variants": ("Granite-3.3-Instruct-8B",),
    },
    "Cogito-v1-Preview-8B": {
        "comparison_type": "same_checkpoint_thinking_toggle",
        "variants": ("Cogito-v1-Preview-8B",),
    },
    "Ministral-3-8B": {
        "comparison_type": "separate_instruct_reasoning_checkpoints",
        "variants": (
            "Ministral-3-Instruct-8B",
            "Ministral-3-Reasoning-8B",
        ),
    },
}

_MODES_BY_LABEL = {
    "Nemotron-Nano-v2-9B": FORMAL_PROMPT_MODES,
    "Nemotron-3-Nano-4B": FORMAL_PROMPT_MODES,
    "Granite-3.3-Instruct-8B": FORMAL_PROMPT_MODES,
    "Cogito-v1-Preview-8B": FORMAL_PROMPT_MODES,
    "Ministral-3-Instruct-8B": NONTHINKING_PROMPT_MODES,
    "Ministral-3-Reasoning-8B": ("native_thinking",),
}

_LOGICAL_LABEL_BY_MODEL = {
    model_label: logical_label
    for logical_label, definition in LOGICAL_GROUPS.items()
    for model_label in definition["variants"]
}

# Operational priorities only; they do not change prompts, decoding, or IDs.
_TASK_PRIORITIES = {
    ("Nemotron-Nano-v2-9B", "native_thinking"): 100,
    ("Ministral-3-Reasoning-8B", "native_thinking"): 95,
    ("Granite-3.3-Instruct-8B", "native_thinking"): 90,
    ("Cogito-v1-Preview-8B", "native_thinking"): 85,
    ("Nemotron-3-Nano-4B", "native_thinking"): 80,
    ("Nemotron-Nano-v2-9B", "enumeration_index"): 75,
    ("Nemotron-Nano-v2-9B", "enumeration_bullet"): 74,
    ("Granite-3.3-Instruct-8B", "enumeration_index"): 70,
    ("Granite-3.3-Instruct-8B", "enumeration_bullet"): 69,
    ("Cogito-v1-Preview-8B", "enumeration_index"): 65,
    ("Cogito-v1-Preview-8B", "enumeration_bullet"): 64,
    ("Ministral-3-Instruct-8B", "enumeration_index"): 60,
    ("Ministral-3-Instruct-8B", "enumeration_bullet"): 59,
    ("Nemotron-3-Nano-4B", "enumeration_index"): 55,
    ("Nemotron-3-Nano-4B", "enumeration_bullet"): 54,
    ("Nemotron-Nano-v2-9B", "direct"): 45,
    ("Granite-3.3-Instruct-8B", "direct"): 40,
    ("Cogito-v1-Preview-8B", "direct"): 35,
    ("Ministral-3-Instruct-8B", "direct"): 30,
    ("Nemotron-3-Nano-4B", "direct"): 25,
}


def _task_id(model_label: str, prompt_mode: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_label).strip("-")
    return f"{normalized}__{prompt_mode}"


def reasoning_extension_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for model_label, prompt_modes in _MODES_BY_LABEL.items():
        spec = REASONING_EXTENSION_MODEL_SPECS[model_label]
        if tuple(prompt_modes) != spec.prompt_modes:
            raise RuntimeError(
                f"Prompt-mode registration differs for {model_label}"
            )
        for prompt_mode in prompt_modes:
            key = (model_label, prompt_mode)
            tasks.append(
                {
                    "task_id": _task_id(model_label, prompt_mode),
                    "priority": _TASK_PRIORITIES[key],
                    "logical_model_label": _LOGICAL_LABEL_BY_MODEL[
                        model_label
                    ],
                    "model_label": model_label,
                    "model_id": spec.model_id,
                    "model_revision": REASONING_EXTENSION_MODEL_REVISIONS[
                        model_label
                    ],
                    "prompt_mode": prompt_mode,
                    "reasoning_policy": spec.reasoning_policy,
                    "chat_template_control": spec.chat_template_control,
                    "system_prompt_strategy": spec.system_prompt_strategy,
                    "engine_profile": spec.engine_profile,
                    "output_collection": "models",
                    "expected_requests": EXPECTED_STIMULI_PER_SHARD,
                }
            )
    tasks.sort(key=lambda task: (-int(task["priority"]), str(task["task_id"])))

    task_ids = [str(task["task_id"]) for task in tasks]
    if (
        len(tasks) != EXPECTED_EXTENSION_SHARDS
        or len(task_ids) != len(set(task_ids))
    ):
        raise RuntimeError("The reasoning-model extension needs 20 unique shards")
    if sum(int(task["expected_requests"]) for task in tasks) != (
        EXPECTED_EXTENSION_REQUESTS
    ):
        raise RuntimeError(
            "The reasoning-model extension must contain 10,000 requests"
        )
    return tasks


def reasoning_extension_plan() -> dict[str, Any]:
    tasks = reasoning_extension_tasks()
    canonical = json.dumps(
        tasks,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": REASONING_EXTENSION_PLAN_SCHEMA,
        "protocol_version": REASONING_EXTENSION_PROTOCOL,
        "source_protocol_version": "realistic_niah_v2",
        "query_layout": QUERY_LAYOUT,
        "source_stimuli_sha256": SOURCE_FORMAL_STIMULI_SHA256,
        "expected_stimuli_per_shard": EXPECTED_STIMULI_PER_SHARD,
        "expected_shards": EXPECTED_EXTENSION_SHARDS,
        "expected_requests": EXPECTED_EXTENSION_REQUESTS,
        "logical_groups": {
            label: {
                **definition,
                "variants": list(definition["variants"]),
            }
            for label, definition in LOGICAL_GROUPS.items()
        },
        "tasks_sha256": hashlib.sha256(canonical).hexdigest(),
        "tasks": tasks,
    }


def expected_request_ids(
    stimulus_ids: Iterable[str],
    task: dict[str, Any],
) -> tuple[str, ...]:
    return tuple(
        f"{task['model_label']}/{task['prompt_mode']}/"
        f"{QUERY_LAYOUT}/{stimulus_id}"
        for stimulus_id in stimulus_ids
    )
