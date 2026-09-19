from __future__ import annotations

from dataclasses import dataclass

CANONICAL_TOKENIZER = "Qwen/Qwen3-8B"
PASSAGE_LENGTHS = (2_000, 3_000, 5_000, 10_000, 20_000)
NEEDLE_COUNTS = (1, 2, 3, 4, 5, 6, 8, 10, 20, 30)
SEEDS = tuple(range(1234, 1244))
QUERY_LAYOUT = "cue_before_query_after"
FORMAL_PROMPT_MODES = (
    "direct",
    "enumeration_index",
    "enumeration_bullet",
    "native_thinking",
)
NONTHINKING_PROMPT_MODES = FORMAL_PROMPT_MODES[:-1]
REASONING_ONLY_PROMPT_MODES = ("native_thinking",)
THINKING_PROMPT_MODES = frozenset(("native_thinking",))
ENUMERATION_PROMPT_MODES = frozenset(
    ("enumeration_index", "enumeration_bullet")
)

SMOKE_PASSAGE_LENGTHS = (2_000, 20_000)
SMOKE_NEEDLE_COUNTS = (6, 20, 30)
SMOKE_SEEDS = (2234, 2235)

DECODING_CONTROL_PASSAGE_LENGTHS = (2_000, 20_000)
DECODING_CONTROL_NEEDLE_COUNTS = (5, 6, 20, 30)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_id: str
    family: str
    native_thinking: bool
    prompt_modes: tuple[str, ...]
    reasoning_policy: str = "switchable"
    chat_template_control: str = "none"
    system_prompt_strategy: str = "none"
    engine_profile: str = "standard"


PRIMARY_MODEL_LABELS = (
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-32B",
    "Gemma4-E4B",
    "Gemma4-12B",
    "DeepSeek-R1-0528-Qwen3-8B",
    "GLM-Z1-9B-0414",
)
FULL_MODE_MODEL_LABELS = PRIMARY_MODEL_LABELS[:6]
REASONING_ONLY_MODEL_LABELS = PRIMARY_MODEL_LABELS[6:]
MATCHED_CONTROL_MODEL_LABELS = ("GLM-4-9B-0414",)
MATCHED_NONTHINKING_CONTROLS = {
    "DeepSeek-R1-0528-Qwen3-8B": "Qwen3-8B",
    "GLM-Z1-9B-0414": "GLM-4-9B-0414",
}
MODEL_REVISIONS = {
    "Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "Qwen3-32B": "9216db5781bf21249d130ec9da846c4624c16137",
    "Gemma4-E4B": "ee0ef6023621cff504d758262d4e04895a5af4a2",
    "Gemma4-12B": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    "DeepSeek-R1-0528-Qwen3-8B": (
        "6e8885a6ff5c1dc5201574c8fd700323f23c25fa"
    ),
    "GLM-Z1-9B-0414": "b221b06fefb23ca320922cf6e68ab5f2fb82de81",
    "GLM-4-9B-0414": "645b8482494e31b6b752272bf7f7f273ef0f3caf",
}


MODEL_SPECS = {
    spec.label: spec
    for spec in (
        ModelSpec(
            "Qwen3-1.7B",
            "Qwen/Qwen3-1.7B",
            "qwen3",
            True,
            FORMAL_PROMPT_MODES,
            chat_template_control="enable_thinking_kwarg",
        ),
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
            "DeepSeek-R1-0528-Qwen3-8B",
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            "deepseek_r1_qwen3",
            True,
            REASONING_ONLY_PROMPT_MODES,
            "always_on",
        ),
        ModelSpec(
            "GLM-Z1-9B-0414",
            "zai-org/GLM-Z1-9B-0414",
            "glm_z1",
            True,
            REASONING_ONLY_PROMPT_MODES,
            "always_on",
        ),
        ModelSpec(
            "GLM-4-9B-0414",
            "zai-org/GLM-4-9B-0414",
            "glm4",
            False,
            NONTHINKING_PROMPT_MODES,
            "off_only",
        ),
    )
}

EXTENSION_MODEL_LABELS = (
    "Olmo3-7B-Instruct",
    "Olmo3-7B-Think",
)
EXTENSION_MODEL_REVISIONS = {
    "Olmo3-7B-Instruct": "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
    "Olmo3-7B-Think": "d97e442d7cc678210054dbcc9b440894d62c89a4",
}
EXTENSION_MODEL_SPECS = {
    spec.label: spec
    for spec in (
        ModelSpec(
            "Olmo3-7B-Instruct",
            "allenai/Olmo-3-7B-Instruct",
            "olmo3",
            False,
            NONTHINKING_PROMPT_MODES,
            "off_only",
        ),
        ModelSpec(
            "Olmo3-7B-Think",
            "allenai/Olmo-3-7B-Think",
            "olmo3",
            True,
            REASONING_ONLY_PROMPT_MODES,
            "always_on",
        ),
    )
}

REASONING_EXTENSION_MODEL_LABELS = (
    "Nemotron-Nano-v2-9B",
    "Nemotron-3-Nano-4B",
    "Granite-3.3-Instruct-8B",
    "Cogito-v1-Preview-8B",
    "Ministral-3-Instruct-8B",
    "Ministral-3-Reasoning-8B",
)
REASONING_EXTENSION_MODEL_REVISIONS = {
    "Nemotron-Nano-v2-9B": "6533e8de2c68e4536bf7c411d7a3ce5734111476",
    "Nemotron-3-Nano-4B": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
    "Granite-3.3-Instruct-8B": "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
    "Cogito-v1-Preview-8B": "64c42369b3f322fbffb277bfff146551dd2823cc",
    "Ministral-3-Instruct-8B": "5b26027e7b19eeb4b7352e1fed3926375dd2cb4d",
    "Ministral-3-Reasoning-8B": (
        "81eaece1948f3875421d9a45bc55487d10e2d894"
    ),
}
REASONING_EXTENSION_MODEL_SPECS = {
    spec.label: spec
    for spec in (
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
            "Granite-3.3-Instruct-8B",
            "ibm-granite/granite-3.3-8b-instruct",
            "granite33",
            True,
            FORMAL_PROMPT_MODES,
            chat_template_control="thinking_kwarg",
        ),
        ModelSpec(
            "Cogito-v1-Preview-8B",
            "deepcogito/cogito-v1-preview-llama-8B",
            "cogito_v1",
            True,
            FORMAL_PROMPT_MODES,
            chat_template_control="enable_thinking_kwarg",
        ),
        ModelSpec(
            "Ministral-3-Instruct-8B",
            "mistralai/Ministral-3-8B-Instruct-2512",
            "ministral3_instruct",
            False,
            NONTHINKING_PROMPT_MODES,
            "off_only",
            engine_profile="mistral_common",
        ),
        ModelSpec(
            "Ministral-3-Reasoning-8B",
            "mistralai/Ministral-3-8B-Reasoning-2512",
            "ministral3_reasoning",
            True,
            REASONING_ONLY_PROMPT_MODES,
            "always_on",
            system_prompt_strategy="ministral3_reasoning",
            engine_profile="mistral_common",
        ),
    )
}

# The completed V2 formal panel remains frozen in MODEL_SPECS. New checkpoint
# families are resolved through this combined registry without changing the
# original 29-shard / 14,500-request design.
ALL_MODEL_SPECS = {
    **MODEL_SPECS,
    **EXTENSION_MODEL_SPECS,
    **REASONING_EXTENSION_MODEL_SPECS,
}
ALL_MODEL_REVISIONS = {
    **MODEL_REVISIONS,
    **EXTENSION_MODEL_REVISIONS,
    **REASONING_EXTENSION_MODEL_REVISIONS,
}


def validate_experiment_spec() -> None:
    if 0 in NEEDLE_COUNTS:
        raise ValueError("N=0 is not part of the registered experiment")
    if len(NEEDLE_COUNTS) != 10:
        raise ValueError("The main grid must contain exactly 10 needle counts")
    if len(PASSAGE_LENGTHS) != 5:
        raise ValueError("The V2 grid must contain exactly five passage lengths")
    if len(SEEDS) != 10:
        raise ValueError("The V2 grid must contain exactly ten seeds")
    if len(PASSAGE_LENGTHS) * len(NEEDLE_COUNTS) * len(SEEDS) != 500:
        raise ValueError("The V2 master grid must contain exactly 500 stimuli")
    if len(PRIMARY_MODEL_LABELS) != 8:
        raise ValueError("The V2 primary panel must contain exactly eight models")
    if (
        len(FULL_MODE_MODEL_LABELS) != 6
        or len(REASONING_ONLY_MODEL_LABELS) != 2
        or set(FULL_MODE_MODEL_LABELS).intersection(REASONING_ONLY_MODEL_LABELS)
        or set(FULL_MODE_MODEL_LABELS).union(REASONING_ONLY_MODEL_LABELS)
        != set(PRIMARY_MODEL_LABELS)
    ):
        raise ValueError(
            "The V2 panel must contain six full-mode and two reasoning-only models"
        )
    if set(PRIMARY_MODEL_LABELS) | set(MATCHED_CONTROL_MODEL_LABELS) != set(
        MODEL_SPECS
    ):
        raise ValueError("Primary and matched-control labels must cover the registry")
    if set(MODEL_REVISIONS) != set(MODEL_SPECS):
        raise ValueError("Every registered model must have one immutable revision")
    if any(
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        for revision in MODEL_REVISIONS.values()
    ):
        raise ValueError("Model revisions must be lowercase 40-character Git SHAs")
    if any(
        target not in MODEL_SPECS or control not in MODEL_SPECS
        for target, control in MATCHED_NONTHINKING_CONTROLS.items()
    ):
        raise ValueError("Matched non-thinking controls must be registered")
    invalid_policies = {
        spec.reasoning_policy
        for spec in MODEL_SPECS.values()
        if spec.reasoning_policy
        not in {"switchable", "always_on", "off_only"}
    }
    if invalid_policies:
        raise ValueError(f"Invalid reasoning policies: {invalid_policies}")


def validate_extension_spec() -> None:
    if set(MODEL_SPECS).intersection(EXTENSION_MODEL_SPECS):
        raise ValueError("Extension model labels must not alter the formal registry")
    if set(EXTENSION_MODEL_LABELS) != set(EXTENSION_MODEL_SPECS):
        raise ValueError("Extension labels must cover the extension registry")
    if set(EXTENSION_MODEL_REVISIONS) != set(EXTENSION_MODEL_SPECS):
        raise ValueError("Every extension model must have one immutable revision")
    if any(
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        for revision in EXTENSION_MODEL_REVISIONS.values()
    ):
        raise ValueError(
            "Extension revisions must be lowercase 40-character Git SHAs"
        )
    if (
        EXTENSION_MODEL_SPECS["Olmo3-7B-Instruct"].prompt_modes
        != NONTHINKING_PROMPT_MODES
        or EXTENSION_MODEL_SPECS["Olmo3-7B-Think"].prompt_modes
        != REASONING_ONLY_PROMPT_MODES
    ):
        raise ValueError("OLMo 3 checkpoints must use their registered mode split")


def validate_reasoning_extension_spec() -> None:
    if set(ALL_MODEL_SPECS) != (
        set(MODEL_SPECS)
        | set(EXTENSION_MODEL_SPECS)
        | set(REASONING_EXTENSION_MODEL_SPECS)
    ):
        raise ValueError("Combined model registry is incomplete")
    if set(REASONING_EXTENSION_MODEL_LABELS) != set(
        REASONING_EXTENSION_MODEL_SPECS
    ):
        raise ValueError("Reasoning-extension labels must cover the registry")
    if set(REASONING_EXTENSION_MODEL_REVISIONS) != set(
        REASONING_EXTENSION_MODEL_SPECS
    ):
        raise ValueError(
            "Every reasoning-extension model must have one immutable revision"
        )
    if any(
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        for revision in REASONING_EXTENSION_MODEL_REVISIONS.values()
    ):
        raise ValueError(
            "Reasoning-extension revisions must be lowercase 40-character SHAs"
        )
    registries = (
        set(MODEL_SPECS),
        set(EXTENSION_MODEL_SPECS),
        set(REASONING_EXTENSION_MODEL_SPECS),
    )
    if any(
        left.intersection(right)
        for index, left in enumerate(registries)
        for right in registries[index + 1 :]
    ):
        raise ValueError("Model registries must have disjoint labels")
    valid_controls = {
        "none",
        "enable_thinking_kwarg",
        "thinking_kwarg",
        "system_reasoning_signal",
    }
    valid_system_strategies = {
        "none",
        "nemotron_reasoning_signal",
        "ministral3_reasoning",
    }
    valid_engine_profiles = {
        "standard",
        "mamba_float32",
        "mistral_common",
    }
    for spec in ALL_MODEL_SPECS.values():
        if spec.chat_template_control not in valid_controls:
            raise ValueError(
                f"Unknown chat-template control for {spec.label}: "
                f"{spec.chat_template_control}"
            )
        if spec.system_prompt_strategy not in valid_system_strategies:
            raise ValueError(
                f"Unknown system-prompt strategy for {spec.label}: "
                f"{spec.system_prompt_strategy}"
            )
        if spec.engine_profile not in valid_engine_profiles:
            raise ValueError(
                f"Unknown engine profile for {spec.label}: "
                f"{spec.engine_profile}"
            )
        if (
            spec.reasoning_policy == "switchable"
            and spec.chat_template_control == "none"
        ):
            raise ValueError(
                f"Switchable model lacks a thinking control: {spec.label}"
            )


validate_experiment_spec()
validate_extension_spec()
validate_reasoning_extension_spec()
