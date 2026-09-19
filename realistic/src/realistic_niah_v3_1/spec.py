from __future__ import annotations

from realistic_niah.runner import RunProtocol
from realistic_niah.spec import (
    FORMAL_PROMPT_MODES,
    NONTHINKING_PROMPT_MODES,
    QUERY_LAYOUT,
    REASONING_ONLY_PROMPT_MODES,
)
from realistic_niah.stimuli import FreezeProtocol
from realistic_niah_v3.spec import (
    CANONICAL_TOKENIZER as CANONICAL_TOKENIZER,
    CANONICAL_TOKENIZER_REVISION as CANONICAL_TOKENIZER_REVISION,
    MATCHED_CONTROL_MODEL_LABELS,
    MATCHED_REASONING_MODEL_LABELS,
    MATCHED_REASONING_PAIRS as MATCHED_REASONING_PAIRS,
    MODEL_LABELS,
    MODEL_REVISIONS,
    MODEL_SPECS,
    SWITCHABLE_MODEL_LABELS,
)

PROTOCOL_VERSION = "realistic_niah_v3_1"
PASSAGE_LENGTHS = (
    1_000,
    2_000,
    3_000,
    5_000,
    8_000,
    10_000,
    15_000,
    20_000,
)
NEEDLE_COUNTS = tuple(range(1, 11)) + (12, 15, 18, 20)
SEEDS = tuple(range(1234, 1264))
INSERTION_DEPTH_MIN_FRACTION = 0.05
INSERTION_DEPTH_MAX_FRACTION = 0.95
EXPECTED_STIMULI = 3_360
EXPECTED_SHARDS = 48
EXPECTED_REQUESTS = 161_280
BIAS_TRIM_PROPORTION = 0.10
MINIMUM_PARSEABLE_PER_BIAS_CELL = 20
BOOTSTRAP_REPLICATES = 2_000

V31_RUN_PROTOCOL = RunProtocol(
    protocol_version=PROTOCOL_VERSION,
    run_manifest_schema_version="realistic_niah_run_manifest_v3_1",
    request_schema_version="realistic_niah_request_v3_1",
    request_id_namespace="v3.1",
    store_prompt_payload=False,
)
V31_FREEZE_PROTOCOL = FreezeProtocol(
    protocol_version=PROTOCOL_VERSION,
    stimulus_schema_version="realistic_niah_master_v3_1",
    manifest_schema_version="realistic_niah_manifest_v3_1",
    audit_schema_version="realistic_niah_audit_v3_1",
    stimulus_id_prefix="V31_",
)


def resolve_model_spec(model: str):
    if model in MODEL_SPECS:
        return MODEL_SPECS[model]
    for spec in MODEL_SPECS.values():
        if spec.model_id == model:
            return spec
    raise ValueError(f"Unknown registered V3.1 model: {model}")


def validate_v31_spec() -> None:
    if len(PASSAGE_LENGTHS) != 8 or PASSAGE_LENGTHS[0] != 1_000:
        raise ValueError("V3.1 must contain the eight registered lengths")
    if len(NEEDLE_COUNTS) != 14 or 0 in NEEDLE_COUNTS:
        raise ValueError("V3.1 must contain fourteen positive needle counts")
    if len(SEEDS) != 30 or SEEDS != tuple(range(1234, 1264)):
        raise ValueError("V3.1 must contain paired seeds 1234 through 1263")
    calculated = len(PASSAGE_LENGTHS) * len(NEEDLE_COUNTS) * len(SEEDS)
    if calculated != EXPECTED_STIMULI:
        raise ValueError("V3.1 must contain exactly 3,360 shared stimuli")
    if EXPECTED_STIMULI * EXPECTED_SHARDS != EXPECTED_REQUESTS:
        raise ValueError("V3.1 request accounting must equal 161,280")
    if not (0.0 <= INSERTION_DEPTH_MIN_FRACTION < INSERTION_DEPTH_MAX_FRACTION <= 1.0):
        raise ValueError("Invalid V3.1 insertion-depth interval")
    if set(MODEL_LABELS) != set(MODEL_SPECS):
        raise ValueError("V3.1 model labels do not cover the V3 registry")
    if set(MODEL_REVISIONS) != set(MODEL_SPECS):
        raise ValueError("Every V3.1 model must have an immutable revision")
    if any(
        MODEL_SPECS[label].prompt_modes != FORMAL_PROMPT_MODES
        for label in SWITCHABLE_MODEL_LABELS
    ):
        raise ValueError("Every switchable checkpoint must use all four modes")
    if any(
        MODEL_SPECS[label].prompt_modes != NONTHINKING_PROMPT_MODES
        for label in MATCHED_CONTROL_MODEL_LABELS
    ):
        raise ValueError("Matched controls must use three non-thinking modes")
    if any(
        MODEL_SPECS[label].prompt_modes != REASONING_ONLY_PROMPT_MODES
        for label in MATCHED_REASONING_MODEL_LABELS
    ):
        raise ValueError("Matched reasoning checkpoints must use native thinking")
    if QUERY_LAYOUT != "cue_before_query_after":
        raise ValueError("V3.1 uses the frozen cue-before/query-after layout")


validate_v31_spec()
