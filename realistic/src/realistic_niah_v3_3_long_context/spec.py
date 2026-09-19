from __future__ import annotations

from realistic_niah.runner import RunProtocol
from realistic_niah.spec import QUERY_LAYOUT
from realistic_niah.stimuli import FreezeProtocol
from realistic_niah_v3_1.spec import (
    CANONICAL_TOKENIZER,
    CANONICAL_TOKENIZER_REVISION,
    MODEL_REVISIONS as V31_MODEL_REVISIONS,
    MODEL_SPECS,
    NEEDLE_COUNTS,
    SEEDS,
)

PROTOCOL_VERSION = "realistic_niah_v3_3_long_context"
MODEL_LABELS = ("Gemma4-31B", "Qwen3-32B")
MODEL_IDS = {label: MODEL_SPECS[label].model_id for label in MODEL_LABELS}
MODEL_REVISIONS = {label: V31_MODEL_REVISIONS[label] for label in MODEL_LABELS}
PASSAGE_LENGTHS = (
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
FORMAL_PROMPT_MODES = ("direct", "native_thinking")
INSERTION_DEPTH_MIN_FRACTION = 0.05
INSERTION_DEPTH_MAX_FRACTION = 0.95
EXPECTED_STIMULI = 3_780
EXPECTED_REQUESTS_PER_MODEL = 7_560
EXPECTED_REQUESTS = 15_120
EXPECTED_REQUESTS_PER_CELL = len(SEEDS)
MAX_MODEL_LEN = 131_072
MODEL_CONFIG_MAX_POSITION_EMBEDDINGS = {
    "Gemma4-31B": 262_144,
    "Qwen3-32B": 40_960,
}
MODEL_CONTEXT_ENGINE_OVERRIDES = {
    "Gemma4-31B": {},
    "Qwen3-32B": {
        "hf_overrides": {
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32_768,
                "rope_theta": 1_000_000,
            },
        },
    },
}
PREFLIGHT_LENGTH = 100_000
PREFLIGHT_NEEDLE_COUNT = 20
PREFLIGHT_SEED = 1234

# The split is fixed prospectively. The length sets are disjoint and their union
# is the complete registered grid. They balance the projected generation time
# more closely than splitting by request count because long contexts are slower.
WORKER_LENGTHS = {
    0: (30_000, 60_000, 70_000, 100_000),
    1: (25_000, 40_000, 50_000, 80_000, 90_000),
}

V33_LONG_CONTEXT_RUN_PROTOCOL = RunProtocol(
    protocol_version=PROTOCOL_VERSION,
    run_manifest_schema_version=(
        "realistic_niah_run_manifest_v3_3_long_context"
    ),
    request_schema_version="realistic_niah_request_v3_3_long_context",
    request_id_namespace="v3.3-long-context",
    store_prompt_payload=False,
)
V33_LONG_CONTEXT_FREEZE_PROTOCOL = FreezeProtocol(
    protocol_version=PROTOCOL_VERSION,
    stimulus_schema_version="realistic_niah_master_v3_3_long_context",
    manifest_schema_version="realistic_niah_manifest_v3_3_long_context",
    audit_schema_version="realistic_niah_audit_v3_3_long_context",
    stimulus_id_prefix="V33LC_",
)


def validate_spec() -> None:
    if len(PASSAGE_LENGTHS) != 9 or PASSAGE_LENGTHS != tuple(
        sorted(PASSAGE_LENGTHS)
    ):
        raise ValueError("V3.3 long-context requires nine sorted length levels")
    if PASSAGE_LENGTHS[0] != 25_000 or PASSAGE_LENGTHS[-1] != 100_000:
        raise ValueError("V3.3 long-context length endpoints are immutable")
    if NEEDLE_COUNTS != tuple(range(1, 11)) + (12, 15, 18, 20):
        raise ValueError("V3.3 long-context must reuse the V3.1 needle grid")
    if SEEDS != tuple(range(1234, 1264)):
        raise ValueError("V3.3 long-context must reuse paired seeds 1234-1263")
    if FORMAL_PROMPT_MODES != ("direct", "native_thinking"):
        raise ValueError("Only direct and native-thinking are registered")
    if QUERY_LAYOUT != "cue_before_query_after":
        raise ValueError("The V3.1 cue-before/query-after layout is immutable")
    expected_revisions = {
        "Gemma4-31B": "842da3794eaa0b77d5f08bae87a17459d91ff475",
        "Qwen3-32B": "9216db5781bf21249d130ec9da846c4624c16137",
    }
    if MODEL_REVISIONS != expected_revisions:
        raise ValueError("Registered long-context model revisions changed")
    expected_stimuli = len(PASSAGE_LENGTHS) * len(NEEDLE_COUNTS) * len(SEEDS)
    if expected_stimuli != EXPECTED_STIMULI:
        raise ValueError("V3.3 long-context stimulus accounting is inconsistent")
    if (
        EXPECTED_STIMULI * len(FORMAL_PROMPT_MODES)
        != EXPECTED_REQUESTS_PER_MODEL
    ):
        raise ValueError("Per-model request accounting is inconsistent")
    if EXPECTED_REQUESTS_PER_MODEL * len(MODEL_LABELS) != EXPECTED_REQUESTS:
        raise ValueError("V3.3 long-context request accounting is inconsistent")
    assigned = [length for values in WORKER_LENGTHS.values() for length in values]
    if len(assigned) != len(set(assigned)):
        raise ValueError("Worker length assignments overlap")
    if set(assigned) != set(PASSAGE_LENGTHS):
        raise ValueError("Worker length assignments do not cover the grid")
    if MAX_MODEL_LEN <= PREFLIGHT_LENGTH + 4096:
        raise ValueError("The registered engine budget cannot cover preflight")
    if MODEL_CONFIG_MAX_POSITION_EMBEDDINGS["Gemma4-31B"] < MAX_MODEL_LEN:
        raise ValueError("Gemma engine context exceeds its immutable config")
    qwen_rope = (
        MODEL_CONTEXT_ENGINE_OVERRIDES["Qwen3-32B"]
        .get("hf_overrides", {})
        .get("rope_parameters")
    )
    if qwen_rope != {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32_768,
        "rope_theta": 1_000_000,
    }:
        raise ValueError("Qwen3-32B requires the registered YaRN override")
    if not (
        0.0
        <= INSERTION_DEPTH_MIN_FRACTION
        < INSERTION_DEPTH_MAX_FRACTION
        <= 1.0
    ):
        raise ValueError("Invalid insertion-depth interval")


validate_spec()
