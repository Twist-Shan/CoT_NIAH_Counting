from __future__ import annotations

from realistic_niah.runner import RunProtocol
from realistic_niah.spec import (
    FORMAL_PROMPT_MODES,
    NONTHINKING_PROMPT_MODES,
    QUERY_LAYOUT,
    REASONING_ONLY_PROMPT_MODES,
    ModelSpec,
)
from realistic_niah.stimuli import FreezeProtocol

PROTOCOL_VERSION = "realistic_niah_v3"
CANONICAL_TOKENIZER = "Qwen/Qwen3-8B"
CANONICAL_TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
PASSAGE_LENGTHS = (2_000, 3_000, 5_000, 8_000, 10_000, 15_000, 20_000)
NEEDLE_COUNTS = tuple(range(1, 11)) + (12, 15, 18, 20)
SEEDS = tuple(range(1234, 1244))
INSERTION_DEPTH_MIN_FRACTION = 0.05
INSERTION_DEPTH_MAX_FRACTION = 0.95
EXPECTED_STIMULI = 980

V3_RUN_PROTOCOL = RunProtocol(
    protocol_version=PROTOCOL_VERSION,
    run_manifest_schema_version="realistic_niah_run_manifest_v3",
    request_schema_version="realistic_niah_request_v3",
    request_id_namespace="v3",
)
V3_FREEZE_PROTOCOL = FreezeProtocol(
    protocol_version=PROTOCOL_VERSION,
    stimulus_schema_version="realistic_niah_master_v3",
    manifest_schema_version="realistic_niah_manifest_v3",
    audit_schema_version="realistic_niah_audit_v3",
    stimulus_id_prefix="V3_",
)

SWITCHABLE_MODEL_LABELS = (
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "Gemma4-E4B",
    "Gemma4-12B",
    "Gemma4-26B-A4B",
    "Gemma4-31B",
    "Nemotron-Nano-v2-9B",
    "Nemotron-3-Nano-4B",
)
MATCHED_CONTROL_MODEL_LABELS = (
    "GLM-4-9B-0414",
    "Ministral-3-Instruct-8B",
)
MATCHED_REASONING_MODEL_LABELS = (
    "GLM-Z1-9B-0414",
    "Ministral-3-Reasoning-8B",
)
MODEL_LABELS = (
    *SWITCHABLE_MODEL_LABELS,
    *MATCHED_CONTROL_MODEL_LABELS,
    *MATCHED_REASONING_MODEL_LABELS,
)
MATCHED_REASONING_PAIRS = {
    "GLM-Z1-9B-0414": "GLM-4-9B-0414",
    "Ministral-3-Reasoning-8B": "Ministral-3-Instruct-8B",
}

MODEL_REVISIONS = {
    "Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen3-8B": CANONICAL_TOKENIZER_REVISION,
    "Qwen3-14B": "40c069824f4251a91eefaf281ebe4c544efd3e18",
    "Qwen3-32B": "9216db5781bf21249d130ec9da846c4624c16137",
    "Gemma4-E4B": "ee0ef6023621cff504d758262d4e04895a5af4a2",
    "Gemma4-12B": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    "Gemma4-26B-A4B": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
    "Gemma4-31B": "842da3794eaa0b77d5f08bae87a17459d91ff475",
    "Nemotron-Nano-v2-9B": "6533e8de2c68e4536bf7c411d7a3ce5734111476",
    "Nemotron-3-Nano-4B": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
    "GLM-4-9B-0414": "645b8482494e31b6b752272bf7f7f273ef0f3caf",
    "GLM-Z1-9B-0414": "b221b06fefb23ca320922cf6e68ab5f2fb82de81",
    "Ministral-3-Instruct-8B": "5b26027e7b19eeb4b7352e1fed3926375dd2cb4d",
    "Ministral-3-Reasoning-8B": (
        "81eaece1948f3875421d9a45bc55487d10e2d894"
    ),
}

MODEL_SPECS = {
    spec.label: spec
    for spec in (
        ModelSpec(
            "Qwen3-4B",
            "Qwen/Qwen3-4B",
            "qwen3",
            True,
            FORMAL_PROMPT_MODES,
            chat_template_control="enable_thinking_kwarg",
        ),
        ModelSpec(
            "Qwen3-8B",
            "Qwen/Qwen3-8B",
            "qwen3",
            True,
            FORMAL_PROMPT_MODES,
            chat_template_control="enable_thinking_kwarg",
        ),
        ModelSpec(
            "Qwen3-14B",
            "Qwen/Qwen3-14B",
            "qwen3",
            True,
            FORMAL_PROMPT_MODES,
            chat_template_control="enable_thinking_kwarg",
        ),
        ModelSpec(
            "Qwen3-32B",
            "Qwen/Qwen3-32B",
            "qwen3",
            True,
            FORMAL_PROMPT_MODES,
            chat_template_control="enable_thinking_kwarg",
        ),
        ModelSpec(
            "Gemma4-E4B",
            "google/gemma-4-E4B-it",
            "gemma4",
            True,
            FORMAL_PROMPT_MODES,
            chat_template_control="enable_thinking_kwarg",
        ),
        ModelSpec(
            "Gemma4-12B",
            "google/gemma-4-12B-it",
            "gemma4",
            True,
            FORMAL_PROMPT_MODES,
            chat_template_control="enable_thinking_kwarg",
        ),
        ModelSpec(
            "Gemma4-26B-A4B",
            "google/gemma-4-26B-A4B-it",
            "gemma4",
            True,
            FORMAL_PROMPT_MODES,
            chat_template_control="enable_thinking_kwarg",
        ),
        ModelSpec(
            "Gemma4-31B",
            "google/gemma-4-31B-it",
            "gemma4",
            True,
            FORMAL_PROMPT_MODES,
            chat_template_control="enable_thinking_kwarg",
        ),
        ModelSpec(
            "Nemotron-Nano-v2-9B",
            "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
            "nemotron_nano_v2",
            True,
            FORMAL_PROMPT_MODES,
            chat_template_control="system_reasoning_signal",
            system_prompt_strategy="nemotron_reasoning_signal",
            engine_profile="mamba_float32",
        ),
        ModelSpec(
            "Nemotron-3-Nano-4B",
            "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            "nemotron3_nano",
            True,
            FORMAL_PROMPT_MODES,
            chat_template_control="enable_thinking_kwarg",
            engine_profile="mamba_float32",
        ),
        ModelSpec(
            "GLM-4-9B-0414",
            "zai-org/GLM-4-9B-0414",
            "glm4",
            False,
            NONTHINKING_PROMPT_MODES,
            reasoning_policy="off_only",
        ),
        ModelSpec(
            "GLM-Z1-9B-0414",
            "zai-org/GLM-Z1-9B-0414",
            "glm_z1",
            True,
            REASONING_ONLY_PROMPT_MODES,
            reasoning_policy="always_on",
        ),
        ModelSpec(
            "Ministral-3-Instruct-8B",
            "mistralai/Ministral-3-8B-Instruct-2512",
            "ministral3_instruct",
            False,
            NONTHINKING_PROMPT_MODES,
            reasoning_policy="off_only",
            engine_profile="mistral_common",
        ),
        ModelSpec(
            "Ministral-3-Reasoning-8B",
            "mistralai/Ministral-3-8B-Reasoning-2512",
            "ministral3_reasoning",
            True,
            REASONING_ONLY_PROMPT_MODES,
            reasoning_policy="always_on",
            system_prompt_strategy="ministral3_reasoning",
            engine_profile="mistral_common",
        ),
    )
}

EXPECTED_SHARDS = 48
EXPECTED_REQUESTS = EXPECTED_STIMULI * EXPECTED_SHARDS


def resolve_model_spec(model: str) -> ModelSpec:
    if model in MODEL_SPECS:
        return MODEL_SPECS[model]
    for spec in MODEL_SPECS.values():
        if spec.model_id == model:
            return spec
    raise ValueError(f"Unknown registered V3 model: {model}")


def validate_v3_spec() -> None:
    if len(PASSAGE_LENGTHS) != 7:
        raise ValueError("V3 must contain seven passage lengths")
    if len(NEEDLE_COUNTS) != 14 or 0 in NEEDLE_COUNTS:
        raise ValueError("V3 must contain fourteen positive needle counts")
    if len(SEEDS) != 10:
        raise ValueError("V3 must contain ten paired seeds")
    if (
        len(PASSAGE_LENGTHS) * len(NEEDLE_COUNTS) * len(SEEDS)
        != EXPECTED_STIMULI
    ):
        raise ValueError("V3 must contain exactly 980 shared stimuli")
    if not (
        0.0
        <= INSERTION_DEPTH_MIN_FRACTION
        < INSERTION_DEPTH_MAX_FRACTION
        <= 1.0
    ):
        raise ValueError("Invalid V3 insertion-depth interval")
    if set(MODEL_LABELS) != set(MODEL_SPECS):
        raise ValueError("V3 model labels do not cover the registry")
    if set(MODEL_REVISIONS) != set(MODEL_SPECS):
        raise ValueError("Every V3 model must have an immutable revision")
    if any(
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        for revision in MODEL_REVISIONS.values()
    ):
        raise ValueError("V3 model revisions must be lowercase 40-character SHAs")
    if any(
        MODEL_SPECS[label].prompt_modes != FORMAL_PROMPT_MODES
        for label in SWITCHABLE_MODEL_LABELS
    ):
        raise ValueError("Every switchable checkpoint must use all four modes")
    if any(
        MODEL_SPECS[label].prompt_modes != NONTHINKING_PROMPT_MODES
        for label in MATCHED_CONTROL_MODEL_LABELS
    ):
        raise ValueError("Matched controls must use the three visible modes")
    if any(
        MODEL_SPECS[label].prompt_modes != REASONING_ONLY_PROMPT_MODES
        for label in MATCHED_REASONING_MODEL_LABELS
    ):
        raise ValueError("Matched reasoning checkpoints must use native thinking")
    if EXPECTED_REQUESTS != 47_040:
        raise ValueError("V3 request accounting must equal 47,040")
    if QUERY_LAYOUT != "cue_before_query_after":
        raise ValueError("V3 uses one frozen cue-before/query-after layout")


validate_v3_spec()
