from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from realistic_niah.prompts import build_messages
from realistic_niah_v4.spec import V4ModelSpec, resolve_model_spec
from realistic_niah_v6.encoding import build_structured_trace_encoding
from realistic_niah_v6.completion import (
    FRAME_EVIDENCE,
    _audit_foundation_recovery_evidence,
    _ordinary_failed_reserve_attempts,
    _resolve_coherent_policy_lineage,
)
from realistic_niah_v6.generation import (
    build_v6_user_text,
    render_structured_prompt,
)
from realistic_niah_v6.parsing import (
    PARSER_SCHEMA_VERSION,
    formal_cohort_eligible,
    parse_and_align_record,
    parse_trace_record,
)
from realistic_niah_v6.reporting import (
    RESULT_SOURCES,
    interesting_scalars,
    validate_report_html,
)
from realistic_niah_v6.pipeline import (
    EXPECTED_SOURCE_STIMULI_SHA256,
    audit_stimulus_panel,
    read_jsonl,
    registered_records,
    sha256_file,
    validate_generation_contracts,
)
from realistic_niah_v6.spec import (
    CONFIRMATION_SEEDS,
    DISCOVERY_SEEDS,
    GENERATION_SCHEMA_VERSION,
    PROMPT_MODES,
    PROTOCOL_VERSION,
    V6Config,
)
from realistic_niah_v6.suite import (
    EXPERIMENTS,
    NATIVE_ANALYSIS_PATH,
    freeze_confirmation,
    validate_confirmation_freeze,
    validate_experiment_registry,
)


ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    eos_token_id = ord("§")
    pad_token_id = ord("§")

    def __init__(self) -> None:
        self.template_calls: list[dict[str, object]] = []

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, object]:
        assert not add_special_tokens
        result: dict[str, object] = {
            "input_ids": [ord(value) for value in text],
            "attention_mask": [1] * len(text),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(value) for value in text]

    def decode(
        self,
        values,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert not clean_up_tokenization_spaces
        text = "".join(chr(int(value)) for value in values)
        return text.replace("§", "") if skip_special_tokens else text

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.template_calls.append(dict(kwargs))
        contents = "".join(str(message["content"]) for message in messages)
        if kwargs.get("add_generation_prompt"):
            return f"<chat>{contents}<assistant>"
        return f"<chat>{contents}<eos>"


MODEL_SPEC = V4ModelSpec(
    label="Qwen3-8B",
    model_id="Qwen/Qwen3-8B",
    revision="test-revision",
    family="qwen3",
    loader_class="AutoModelForCausalLM",
)


def _stimulus() -> dict[str, object]:
    passage = "Paris received a score of 7. Lima received a score of -2."
    records = []
    pairs = []
    for slot, (city, score) in enumerate((("Paris", 7), ("Lima", -2)), start=1):
        start = passage.index(city)
        end = passage.index(".", start) + 1
        records.append(
            {
                "slot_index": slot,
                "city": city,
                "score": score,
                "char_start": start,
                "char_end": end,
            }
        )
        pairs.append({"city": city, "score": score})
    return {
        "stimulus_id": "v4.4/test/seed-1234/count-2",
        "design_variant": "v4.4",
        "seed": 1234,
        "split": "discovery",
        "passage": passage,
        "gold_count": 2,
        "gold_pairs": pairs,
        "active_needle_spans": records,
    }


def _generation_row(mode: str) -> dict[str, object]:
    if mode == "enumeration_index":
        raw = "1. Paris: 7\n2. Lima: -2\nTotal: 2"
    else:
        raw = "- Paris: 7\n- Lima: -2\nTotal: 2"
    prompt = "Paris record. Lima record."
    registered_model = resolve_model_spec("Qwen3-8B")
    decoding = {
        "max_new_tokens": 4096,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
    }
    row: dict[str, object] = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "request_id": f"test/{mode}",
        "stimulus_id": f"test/{mode}",
        "design_variant": "v4.4",
        "model_label": "Qwen3-8B",
        "model_family": "qwen3",
        "model_id": registered_model.model_id,
        "model_revision": registered_model.revision,
        "model_loader_class": registered_model.loader_class,
        "chat_template_control": registered_model.chat_template_control,
        "prompt_mode": mode,
        "native_thinking": False,
        "chat_template_thinking_enabled": False,
        "seed": 1234,
        "split": "discovery",
        "gold_count": 2,
        "gold_records": [
            {"city": "Paris", "score": 7},
            {"city": "Lima", "score": -2},
        ],
        "raw_output_text": raw,
        "output_token_ids": [ord(value) for value in raw],
        "input_ids": [ord(value) for value in prompt],
        "attention_mask": [1] * len(prompt),
        "user_text": "test user",
        "rendered_prompt": prompt,
        "prompt_record_spans": [
            {"slot_index": 1, "city": "Paris", "score": 7, "start": 0, "end": 13},
            {"slot_index": 2, "city": "Lima", "score": -2, "start": 14, "end": 26},
        ],
        "decoding": decoding,
        "finish_reason": "stop",
        "source_stimulus_sha256": "0" * 64,
    }
    row["user_text_sha256"] = hashlib.sha256(
        str(row["user_text"]).encode("utf-8")
    ).hexdigest()
    row["rendered_prompt_sha256"] = hashlib.sha256(
        str(row["rendered_prompt"]).encode("utf-8")
    ).hexdigest()
    row["generation_contract_sha256"] = hashlib.sha256(
        json.dumps(
            {"prompt_mode": mode, "thinking": False, "decoding": decoding},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return row


@pytest.mark.parametrize("mode", PROMPT_MODES)
def test_v6_config_is_frozen_to_each_enumeration_mode(mode: str) -> None:
    config = V6Config(prompt_mode=mode)
    config.validate()
    assert config.discovery_seeds == DISCOVERY_SEEDS
    assert config.confirmation_seeds == CONFIRMATION_SEEDS
    assert config.counts == tuple(range(1, 11))
    assert config.expected_marker_kind == (
        "indexed" if mode == "enumeration_index" else "bullet"
    )
    assert config.decoding.max_new_tokens == 4096
    assert config.targeted_bank_grid("Qwen3-8B") == (32, 64, 80, 96, 112, 125, 128)
    assert config.targeted_bank_grid("Gemma4-E4B") == (1, 2, 4, 6, 8)
    assert config.report_reference_bank_size("Qwen3-8B") == 128
    assert config.report_reference_bank_size("Gemma4-E4B") == 6


def test_v6_rejects_native_thinking() -> None:
    with pytest.raises(ValueError, match="prompt_mode"):
        V6Config(prompt_mode="native_thinking").validate()


@pytest.mark.parametrize("mode", PROMPT_MODES)
def test_generation_provenance_is_fail_closed(mode: str) -> None:
    row = _generation_row(mode)
    audit = validate_generation_contracts(
        [row], V6Config(prompt_mode=mode), model_label="Qwen3-8B"
    )
    assert audit["status"] == "PASS"
    row["chat_template_thinking_enabled"] = True
    with pytest.raises(ValueError, match="chat_template_thinking_enabled"):
        validate_generation_contracts(
            [row], V6Config(prompt_mode=mode), model_label="Qwen3-8B"
        )


def test_registered_cohort_reparses_stale_cached_payload() -> None:
    row = _generation_row("enumeration_index")
    row["trace_parse"] = {
        "schema_version": PARSER_SCHEMA_VERSION,
        "strict_causal_eligible": False,
    }
    selected = registered_records(
        [row], V6Config(prompt_mode="enumeration_index"), formal_only=True
    )
    assert len(selected) == 1
    assert selected[0]["trace_parse"]["strict_causal_eligible"] is True


@pytest.mark.parametrize("mode", PROMPT_MODES)
def test_prompt_is_the_shared_registered_prompt_with_thinking_disabled(mode: str) -> None:
    stimulus = _stimulus()
    tokenizer = CharacterTokenizer()
    rendered = render_structured_prompt(
        stimulus,
        tokenizer=tokenizer,
        model_spec=MODEL_SPEC,
        prompt_mode=mode,
    )
    expected = build_messages(str(stimulus["passage"]), prompt_mode=mode)[0]["content"]
    assert build_v6_user_text(str(stimulus["passage"]), prompt_mode=mode) == expected
    assert rendered.user_text == expected
    assert rendered.prompt_mode == mode
    assert rendered.model_id == MODEL_SPEC.model_id
    assert rendered.model_revision == MODEL_SPEC.revision
    assert rendered.chat_template_control == "enable_thinking_kwarg"
    assert len(rendered.prompt_record_spans) == 2
    assert tokenizer.template_calls[-1]["enable_thinking"] is False


@pytest.mark.parametrize(
    ("mode", "marker", "sequence_source"),
    [
        ("enumeration_index", "indexed", "rank_supported_episode"),
        ("enumeration_bullet", "bullet", "structural_fallback"),
    ],
)
def test_strict_parser_and_sites(mode: str, marker: str, sequence_source: str) -> None:
    tokenizer = CharacterTokenizer()
    parsed = parse_and_align_record(_generation_row(mode), tokenizer)
    assert parsed["schema_version"] == PARSER_SCHEMA_VERSION
    assert parsed["strict_causal_eligible"] is True
    assert parsed["parser"]["marker_kind"] == marker
    assert parsed["sequence_source"] == sequence_source
    assert parsed["exact_ordered_gold_pairs"] is True
    assert parsed["alignment_summary"]["eligible"] == len(parsed["token_sites"])
    assert formal_cohort_eligible(_generation_row(mode)) is True
    assert [
        site["occurrence"]
        for site in parsed["token_sites"]
        if site["site_kind"] == "item_end"
    ] == [1, 2]


def test_strict_parser_rejects_wrong_marker_order_and_extra_text() -> None:
    wrong_marker = _generation_row("enumeration_index")
    wrong_marker["raw_output_text"] = "- Paris: 7\n- Lima: -2\nTotal: 2"
    wrong_marker["output_token_ids"] = [
        ord(value) for value in str(wrong_marker["raw_output_text"])
    ]
    assert parse_trace_record(wrong_marker)["strict_causal_eligible"] is False

    wrong_order = _generation_row("enumeration_bullet")
    wrong_order["raw_output_text"] = "- Lima: -2\n- Paris: 7\nTotal: 2"
    wrong_order["output_token_ids"] = [
        ord(value) for value in str(wrong_order["raw_output_text"])
    ]
    assert parse_trace_record(wrong_order)["strict_causal_eligible"] is False

    extra = _generation_row("enumeration_index")
    extra["raw_output_text"] = str(extra["raw_output_text"]) + "\nDone."
    extra["output_token_ids"] = [ord(value) for value in str(extra["raw_output_text"])]
    assert parse_trace_record(extra)["strict_causal_eligible"] is False


@pytest.mark.parametrize("mode", PROMPT_MODES)
def test_answer_candidate_termination_uses_nonthinking_container(mode: str) -> None:
    tokenizer = CharacterTokenizer()
    encoding = build_structured_trace_encoding(
        _generation_row(mode),
        tokenizer,
        site_id="answer_query_v3",
        candidate_counts=tuple(range(1, 11)),
    )
    assert dict(encoding.count_candidate_answer_token_ids)[2] == (ord("2"),)
    assert "".join(
        chr(value) for value in dict(encoding.count_candidate_token_ids)[2]
    ).endswith("<eos>")
    assert tokenizer.template_calls[-1]["enable_thinking"] is False


def test_frozen_real_stimulus_panel_is_exactly_300_cells_per_mode() -> None:
    source = ROOT / "work" / "nonthinking_report_filestream_stage3" / "stimuli.jsonl"
    if not source.is_file():
        pytest.skip("Original frozen stimulus artifact is not bundled; regenerate or restore it to run this integration check")
    assert sha256_file(source) == EXPECTED_SOURCE_STIMULI_SHA256
    rows = read_jsonl(source)
    assert len(rows) == 1200
    for mode in PROMPT_MODES:
        audit = audit_stimulus_panel(rows, V6Config(prompt_mode=mode))
        assert audit["status"] == "PASS"
        assert audit["selected_rows"] == 300
        assert audit["discovery_rows"] == 200
        assert audit["confirmation_rows"] == 100


def test_read_jsonl_preserves_unicode_paragraph_separators(tmp_path: Path) -> None:
    expected = [
        {"request_id": "first", "text": "left\u2029right"},
        {"request_id": "second", "text": "before\u2028after"},
    ]
    source = tmp_path / "unicode-separators.jsonl"
    source.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in expected
        ),
        encoding="utf-8",
    )

    # str.splitlines() treats U+2028/U+2029 inside a valid JSON string as record
    # boundaries. The production reader must iterate physical JSONL lines instead.
    assert len(source.read_text(encoding="utf-8").splitlines()) == 4
    assert read_jsonl(source) == expected


def test_report_tail_supervisor_has_unicode_safe_minimal_resume() -> None:
    script = (
        ROOT / "scripts" / "supervise_realistic_niah_v6_report_tail_confirmation.sh"
    ).read_text(encoding="utf-8")

    assert "rows = read_jsonl(generations)" in script
    assert "cells = read_jsonl(registry)" in script
    assert "generations.read_text().splitlines()" not in script
    assert 'with path.open("r", encoding="utf-8") as handle:' in script
    assert "V6_REPORT_TAIL_RESUME_FROM" in script
    assert "PASS_REUSE_PRE_WALKTHROUGH_OUTPUTS" in script


def test_suite_registry_maps_every_native_report_frame() -> None:
    audit = validate_experiment_registry()
    assert audit["status"] == "PASS"
    assert audit["report_frames"] == list(range(1, 21))
    assert len(EXPERIMENTS) == 20
    assert set(NATIVE_ANALYSIS_PATH) == {
        row.experiment_id for row in EXPERIMENTS
    }
    representation = next(
        row for row in EXPERIMENTS if row.experiment_id == "layerwise_representation"
    )
    assert representation.all_sample_analysis is True
    assert representation.correct_only_sensitivity is False
    assert not all(row.all_sample_analysis for row in EXPERIMENTS)
    assert NATIVE_ANALYSIS_PATH["layerwise_representation"] == {
        "native_report_section": "2_representation_measurement",
        "analysis_population": "original_item_end_exact_four_cell_common_support_plus_answer_query_v3_full300",
        "evidence_tier": "primary",
    }
    assert all(
        "run_realistic_niah_v5" not in row.kernel_entrypoint
        for row in EXPERIMENTS
    )


def test_confirmation_freeze_hash_locks_discovery_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "discovery.json"
    artifact.write_text('{"selected": true}\n', encoding="utf-8")
    needed = {
        row.experiment_id
        for row in EXPERIMENTS
        if row.discovery_selection and row.confirmation_required
    }
    ledger = {
        "schema_version": "realistic_niah_v6_discovery_ledger_v1",
        "status": "DISCOVERY_FROZEN",
        "prompt_mode": "enumeration_index",
        "model_label": "Qwen3-8B",
        "confirmation_outcomes_read": False,
        "experiments": {
            experiment_id: {
                "status": "FROZEN",
                "choice": "registered",
                "negative_result_retained": False,
                "artifact_paths": [str(artifact)],
            }
            for experiment_id in needed
        }
    }
    freeze = freeze_confirmation(
        prompt_mode="enumeration_index",
        model_label="Qwen3-8B",
        discovery_ledger=ledger,
        artifact_paths=[artifact],
    )
    assert freeze["status"] == "CONFIRMATION_FROZEN"
    assert freeze["confirmation_outcomes_read"] is False
    assert freeze["artifact_sha256"][str(artifact.resolve())] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    assert len(freeze["freeze_sha256"]) == 64
    validated = validate_confirmation_freeze(
        freeze,
        prompt_mode="enumeration_index",
        model_label="Qwen3-8B",
    )
    assert validated == freeze
    tampered = dict(freeze)
    tampered["confirmation_outcomes_read"] = True
    with pytest.raises(ValueError, match="outcomes were read"):
        validate_confirmation_freeze(
            tampered,
            prompt_mode="enumeration_index",
            model_label="Qwen3-8B",
        )


def test_count_stream_adapter_accepts_both_v6_mechanism_configs() -> None:
    from realistic_niah_v6.kernel import install_v6_kernel_adapters

    audit = install_v6_kernel_adapters()
    assert audit["status"] == "INSTALLED"
    assert audit["v5_source_files_modified"] is False
    assert "realistic_niah_v5.causal_sites.parse_trace_record" in audit[
        "patched_process_globals"
    ]
    assert "realistic_niah_v5.pipeline.parse_trace_record" in audit[
        "patched_process_globals"
    ]
    assert audit["structured_progress_transition_adapter"] == {
        "status": "INSTALLED",
        "mechanism": "progress_transition",
        "boundary_policy": "strict_registered",
        "compiler": (
            "realistic_niah_v5.causal._registered_mechanism_continuations"
        ),
        "numerical_patch_implementation_changed": False,
    }
    assert "realistic_niah_v5.count_stream.mechanism_continuations" in audit[
        "patched_process_globals"
    ]
    assert "realistic_niah_v5.native_loop.mechanism_continuations" in audit[
        "patched_process_globals"
    ]
    from realistic_niah_v5.count_stream import NativeCountMechanismSpec

    for mode in PROMPT_MODES:
        path = ROOT / "configs" / f"realistic_niah_v6_{mode}_count_stream_dev.json"
        mechanism = NativeCountMechanismSpec.load(path)
        assert mode in mechanism.experiment_id
        assert mechanism.status == "development_only"
        assert mechanism.formal_inference_eligible is False


def test_v5_filtered_source_hashes_are_unchanged_by_v6_adapter() -> None:
    # A downloadable anonymous archive has no Git history. Check the same
    # source immutability property directly, covering every V5 Python file.
    source = ROOT / "src" / "realistic_niah_v5"
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")}
    from realistic_niah_v6.kernel import install_v6_kernel_adapters

    install_v6_kernel_adapters()
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")}
    assert after == before


def test_foundation_marker_recovery_requires_complete_consistent_outputs(
    tmp_path: Path,
) -> None:
    from scripts.audit_realistic_niah_v6_foundation_marker_recovery import (
        recover_foundation_marker,
    )

    root = tmp_path / "enumeration_bullet" / "Gemma4-E4B"

    def write(relative: str, value: object) -> Path:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, str):
            path.write_text(value, encoding="utf-8")
        else:
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return path

    capture = {
        "schema_version": "realistic_niah_v6_trace_capture_v1",
        "parser_implementation": "realistic_niah_v6.parse_structured_enumeration_trace",
        "excluded_rows": 0,
        "rows": 173,
    }
    write("capture/formal/capture_manifest.json", capture)
    write("capture/all_sample/capture_manifest.json", {**capture, "rows": 200})
    adapter = {
        "status": "INSTALLED",
        "prompt_mode": "enumeration_bullet",
        "v5_source_files_modified": False,
    }
    write("capture/formal/v6_adapter_manifest.json", adapter)
    write("capture/all_sample/v6_adapter_manifest.json", adapter)
    attention_base = {
        "model_label": "Gemma4-E4B",
        "prompt_mode": "enumeration_bullet",
        "seed_role": "discovery",
        "rows": 1,
    }
    write(
        "attention/discovery_formal.manifest.json",
        {**attention_base, "formal_cohort": True, "requests": 173},
    )
    write(
        "attention/discovery_answer_query_formal.manifest.json",
        {**attention_base, "formal_cohort": True, "requests": 173},
    )
    write(
        "attention/discovery_all_sample.manifest.json",
        {**attention_base, "formal_cohort": False, "requests": 200},
    )
    for relative in (
        "capture/formal/capture_index.jsonl",
        "capture/all_sample/capture_index.jsonl",
        "attention/discovery_formal.csv",
        "attention/discovery_all_sample.csv",
        "attention/discovery_answer_query_formal.csv",
    ):
        write(relative, "artifact\n")

    audit = recover_foundation_marker(
        model_root=root,
        model_label="Gemma4-E4B",
        prompt_mode="enumeration_bullet",
        reason="test launch-time marker loss",
    )
    assert audit["status"] == "PASS_OUTPUTS_COMPLETE_MARKER_RECOVERED"
    assert audit["model_outputs_recomputed"] is False
    assert audit["seed_failure_recorded"] is False
    assert (root / "discovery-foundation.COMPLETE").read_text().strip() == "PASS"
    assert recover_foundation_marker(
        model_root=root,
        model_label="Gemma4-E4B",
        prompt_mode="enumeration_bullet",
        reason="ignored on idempotent replay",
    ) == audit


def test_targeted_retrieval_selection_is_seed_equal_and_discovery_only(
    tmp_path: Path,
) -> None:
    import subprocess
    import sys

    causal = tmp_path / "causal"
    bank_sizes = (32, 64, 80, 96, 112, 128)
    for bank_size in bank_sizes:
        behavior = causal / "behavior" / f"k{bank_size}"
        shards = behavior / "shards"
        shards.mkdir(parents=True)
        (behavior / "manifest.json").write_text(
            json.dumps({"status": "COMPLETE", "bank_size": bank_size}) + "\n",
            encoding="utf-8",
        )
        plan = causal / "plans" / f"k{bank_size}"
        plan.mkdir(parents=True)
        plan_rows = []
        random_condition = (
            "global_random" if bank_size == 128 else "layer_matched_random"
        )
        for condition, repeat, head_block in (
            ("selected_bank", 0, 0),
            (random_condition, 1, 8),
            (random_condition, 2, 16),
            (random_condition, 3, 24),
        ):
            heads = [
                [index % 16, head_block + index // 16]
                for index in range(bank_size)
            ]
            serialized = json.dumps(heads)
            plan_rows.append(
                ",".join(
                    [
                        "Qwen3-8B",
                        condition,
                        str(repeat),
                        str(bank_size),
                        hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
                        '"' + serialized.replace('"', '""') + '"',
                    ]
                )
            )
        (plan / "retrieval_anchor_bank_plan.csv").write_text(
            "model_label,condition,repeat,bank_size,bank_sha256,heads\n"
            + "\n".join(plan_rows)
            + "\n",
            encoding="utf-8",
        )
        (plan / "causal_plan_audit.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "registered_bank_size": bank_size,
                    "full_panel_plan": True,
                    "confirmation_used_for_selection": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        shard_index = 0
        for seed in range(1234, 1254):
            arms = [
                ("clean", 0, True),
                ("selected_bank", 0, bank_size != 64),
                (random_condition, 1, True),
                (random_condition, 2, True),
                (random_condition, 3, True),
            ]
            for condition, repeat, correct in arms:
                row = {
                    "status": "ok",
                    "trial_complete": True,
                    "split": "discovery",
                    "seed": seed,
                    "request_id": f"seed-{seed}",
                    "anchor_equivalence_id": f"{seed}:9->10",
                    "condition": condition,
                    "repeat": repeat,
                    "correct_next_needle": correct,
                    "planned_bank_size": bank_size,
                }
                (shards / f"trial_{shard_index:04d}.jsonl").write_text(
                    json.dumps(row) + "\n", encoding="utf-8"
                )
                shard_index += 1

    output = tmp_path / "analysis"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_realistic_niah_v6_targeted_retrieval.py"),
            "--model",
            "Qwen3-8B",
            "--prompt-mode",
            "enumeration_index",
            "--causal-root",
            str(causal),
            "--bank-sizes",
            *[str(value) for value in bank_sizes],
            "--expected-seeds",
            "20",
            "--bootstrap-samples",
            "200",
            "--report-reference-k",
            "128",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    assert selection["status"] == "DISCOVERY_FROZEN_CHOICE"
    assert selection["selected_k"] == 64
    assert selection["selected_random_condition"] == "layer_matched_random"
    assert selection["dose_argmax_k"] == 64
    assert selection["dose_argmax_used_for_downstream_bank"] is True
    assert selection["selected_by_v6_discovery_dose_rule"] is True
    assert selection["selected_effect"] == 1.0
    assert selection["negative_result_retained"] is False
    audit = json.loads((output / "analysis_audit.json").read_text(encoding="utf-8"))
    assert [row["seed_count"] for row in audit["dose_rows"]] == [20] * 6
    assert [row["condition_rows"] for row in audit["dose_rows"]] == [100] * 6

    model_root = tmp_path / "model_root"
    audited_selection = (
        model_root
        / "causal"
        / "targeted_retrieval"
        / "discovery_formal"
        / "analysis"
        / "selection.json"
    )
    audited_selection.parent.mkdir(parents=True)
    audited_selection.write_bytes((output / "selection.json").read_bytes())
    confirmation_analysis = (
        model_root
        / "causal"
        / "targeted_retrieval"
        / "confirmation_formal"
        / "analysis.json"
    )
    confirmation_analysis.parent.mkdir(parents=True)
    confirmation_analysis.write_text(
        json.dumps(
            {
                "status": "CONFIRMATION_EVALUATED_FROZEN_K",
                "model_label": "Qwen3-8B",
                "prompt_mode": "enumeration_index",
                "selected_k": 64,
                "selected_random_condition": "layer_matched_random",
                "confirmation_used_for_selection": False,
                "bank_size_reselected": False,
                "confirmation_reselected_k": False,
                "selection_sha256": hashlib.sha256(
                    audited_selection.read_bytes()
                ).hexdigest(),
                "report_contract_sha256": selection["report_contract_sha256"],
                "discovery_dose_response_sha256": selection[
                    "dose_response_sha256"
                ],
                "result": {
                    "bank_size": 64,
                    "random_condition": "layer_matched_random",
                    "split": "confirmation",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    from realistic_niah_v6.completion import audit_targeted_retrieval_selection

    selection_audit = audit_targeted_retrieval_selection(
        model_root,
        prompt_mode="enumeration_index",
        model_label="Qwen3-8B",
    )
    assert selection_audit["status"] == (
        "PASS_DISCOVERY_SELECTED_CONFIRMATION_FROZEN_K"
    )
    assert selection_audit["selected_k"] == 64
    assert selection_audit["report_reference_k_forced"] is False

    from realistic_niah_v6.kernel import install_v6_selected_bank_contract
    from scripts import run_realistic_niah_v5_generated_suffix_state_bridge as bridge
    from realistic_niah_v5 import stratified_targeted_counter_ncc as stratified_ncc
    from realistic_niah_v5 import targeted_counter_logit_margin as logit_margin
    from realistic_niah_v5 import targeted_counter_ncc as targeted_ncc
    from realistic_niah_v5 import targeted_counter_write as targeted_write

    previous = {
        "bridge_contract": dict(bridge.MODEL_CONTRACTS["Qwen3-8B"]),
        "write_contract": dict(targeted_write._MODEL_CONTRACTS["Qwen3-8B"]),
        "bridge_loader": bridge._load_banks,
        "write_runner": targeted_write.run_targeted_counter_write_trials,
        "ncc_normalize": targeted_ncc._normalize_banks,
        "stratified_normalize": stratified_ncc._normalize_banks,
        "stratified_reach": stratified_ncc._validate_causal_reach,
        "logit_normalize": logit_margin._normalize_banks,
        "logit_factorial": logit_margin._validate_factorial_banks,
    }
    try:
        contract = install_v6_selected_bank_contract(
            output / "selection.json",
            causal / "plans" / "k64" / "retrieval_anchor_bank_plan.csv",
            config=V6Config(prompt_mode="enumeration_index"),
            model_label="Qwen3-8B",
        )
        assert contract["status"] == "INSTALLED"
        assert contract["legacy_report_reference_bank_size"] == 128
        assert contract["selected_bank_size"] == 64
        assert contract["selected_random_condition"] == "layer_matched_random"
        assert contract["source_layer_preserved"] == 19
        assert contract["base_config_bank_grid"] == [32, 64, 80, 96, 112, 125, 128]
        assert contract["effective_registered_bank_grid"] == [32, 64, 80, 96, 112, 128]
        assert contract["head_identities_changed"] is False
        assert contract["numerical_intervention_implementation_changed"] is False
        assert bridge.MODEL_CONTRACTS["Qwen3-8B"]["bank_size"] == 64
        assert targeted_write._MODEL_CONTRACTS["Qwen3-8B"]["bank_size"] == 64

        layer_banks = bridge._load_banks(
            causal / "plans" / "k64" / "retrieval_anchor_bank_plan.csv",
            model_label="Qwen3-8B",
        )
        assert [row["condition"] for row in layer_banks] == [
            "clean",
            "layer_matched_random",
            "layer_matched_random",
            "layer_matched_random",
            "selected_bank",
        ]

        global_selection = dict(selection)
        global_plan = causal / "plans" / "k128" / "retrieval_anchor_bank_plan.csv"
        global_selection.update(
            {
                "selected_k": 128,
                "selected_random_condition": "global_random",
                "selected_control_matching": "global",
                "dose_argmax_k": 128,
                "frozen_plan": str(global_plan.resolve()),
                "frozen_plan_sha256": hashlib.sha256(
                    global_plan.read_bytes()
                ).hexdigest(),
            }
        )
        global_selection_path = tmp_path / "global_selection.json"
        global_selection_path.write_text(
            json.dumps(global_selection) + "\n", encoding="utf-8"
        )
        global_contract = install_v6_selected_bank_contract(
            global_selection_path,
            global_plan,
            config=V6Config(prompt_mode="enumeration_index"),
            model_label="Qwen3-8B",
        )
        assert global_contract["selected_bank_size"] == 128
        assert global_contract["selected_random_condition"] == "global_random"
        assert global_contract["random_control_kernel_adapter"] == (
            "v6_process_local_global_random_label_compatibility"
        )
        global_banks = bridge._load_banks(global_plan, model_label="Qwen3-8B")
        assert [row["condition"] for row in global_banks] == [
            "clean",
            "global_random",
            "global_random",
            "global_random",
            "selected_bank",
        ]

        class DummyAdapter:
            num_layers = 36
            num_heads = (32,) * 36

        normalized = targeted_ncc._normalize_banks(
            DummyAdapter(), global_banks, selected_size=128
        )
        assert [row["condition"] for row in normalized] == [
            "clean",
            "selected_bank",
            "global_random",
            "global_random",
            "global_random",
        ]
        assert stratified_ncc._validate_causal_reach(
            normalized, capture_start_layer=16
        ) == 15
        assert logit_margin._validate_factorial_banks(normalized) is None

        # The report-matched K128 global controls can reach the final model
        # layer even when the selected treatment ends one layer earlier.  The
        # specialized adapter replaces only those unreachable controls and
        # must leave the discovery-selected treatment row exactly unchanged.
        unreachable_plan = tmp_path / "unreachable" / "retrieval_anchor_bank_plan.csv"
        unreachable_plan.parent.mkdir(parents=True)
        with global_plan.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or ())
            unreachable_rows = [dict(row) for row in reader]
        for row in unreachable_rows:
            if row["condition"] != "global_random":
                continue
            heads = json.loads(row["heads"])
            heads[0] = [16, int(row["repeat"])]
            serialized = json.dumps(heads)
            row["heads"] = serialized
            row["bank_sha256"] = hashlib.sha256(
                serialized.encode("utf-8")
            ).hexdigest()
        with unreachable_plan.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(unreachable_rows)
        unreachable_selection = dict(global_selection)
        unreachable_selection.update(
            {
                "frozen_plan": str(unreachable_plan.resolve()),
                "frozen_plan_sha256": hashlib.sha256(
                    unreachable_plan.read_bytes()
                ).hexdigest(),
            }
        )
        unreachable_selection_path = tmp_path / "unreachable_selection.json"
        unreachable_selection_path.write_text(
            json.dumps(unreachable_selection) + "\n", encoding="utf-8"
        )
        universe_path = tmp_path / "head_universe.csv"
        with universe_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["model_label", "layer", "head"],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(
                {
                    "model_label": "Qwen3-8B",
                    "layer": layer,
                    "head": head,
                }
                for layer in range(17)
                for head in range(32)
            )
        from scripts.build_realistic_niah_v6_specialized_bank_plan import (
            build_specialized_bank_plan,
        )

        specialized_plan_root = tmp_path / "specialized_plan"
        plan_audit = build_specialized_bank_plan(
            selection_path=unreachable_selection_path,
            source_plan_path=unreachable_plan,
            head_universe_path=universe_path,
            model_label="Qwen3-8B",
            prompt_mode="enumeration_index",
            output=specialized_plan_root,
        )
        assert plan_audit["status"] == (
            "PASS_CAPTURE_REACHABLE_GLOBAL_CONTROL_ADAPTER"
        )
        assert plan_audit["replacement_count"] == 3
        assert plan_audit["selected_treatment_heads_unchanged"] is True
        assert plan_audit["sample_failure"] is False
        derived_plan = specialized_plan_root / "retrieval_anchor_bank_plan.csv"
        derived_contract = install_v6_selected_bank_contract(
            unreachable_selection_path,
            derived_plan,
            config=V6Config(prompt_mode="enumeration_index"),
            model_label="Qwen3-8B",
        )
        assert derived_contract["head_identities_changed"] is True
        assert derived_contract["selected_head_identities_changed"] is False
        assert derived_contract["random_control_head_identities_changed"] is True
        assert derived_contract["specialized_bank_plan_adapter"]["replacement_count"] == 3
        derived_banks = bridge._load_banks(
            derived_plan, model_label="Qwen3-8B"
        )
        derived_normalized = targeted_ncc._normalize_banks(
            DummyAdapter(), derived_banks, selected_size=128
        )
        assert max(
            layer
            for bank in derived_normalized
            for layer, _head in bank["heads"]
        ) == 15
        assert stratified_ncc._validate_causal_reach(
            derived_normalized, capture_start_layer=16
        ) == 15
    finally:
        bridge.MODEL_CONTRACTS["Qwen3-8B"] = previous["bridge_contract"]
        targeted_write._MODEL_CONTRACTS["Qwen3-8B"] = previous["write_contract"]
        bridge._load_banks = previous["bridge_loader"]
        targeted_write.run_targeted_counter_write_trials = previous["write_runner"]
        targeted_ncc._normalize_banks = previous["ncc_normalize"]
        stratified_ncc._normalize_banks = previous["stratified_normalize"]
        stratified_ncc._validate_causal_reach = previous["stratified_reach"]
        logit_margin._normalize_banks = previous["logit_normalize"]
        logit_margin._validate_factorial_banks = previous["logit_factorial"]


def test_pool_exhaustion_amendment_is_an_outcome_blind_discovery_suffix() -> None:
    from realistic_niah_v6 import replacement

    config = V6Config(prompt_mode="enumeration_bullet")
    base = replacement.load_replacement_policy(
        ROOT / "configs" / "realistic_niah_v6_replacement_policy.json", config
    )
    amended = replacement.load_replacement_policy(
        ROOT
        / "configs"
        / "realistic_niah_v6_replacement_policy_amendment1.json",
        config,
    )
    base_discovery = list(map(int, base["discovery_replacement_seed_pool"]))
    amended_discovery = list(
        map(int, amended["discovery_replacement_seed_pool"])
    )
    assert amended_discovery[: len(base_discovery)] == base_discovery
    assert amended_discovery[len(base_discovery) :] == list(range(1414, 1514))
    assert amended["confirmation_replacement_seed_pool"] == base[
        "confirmation_replacement_seed_pool"
    ]
    audit = amended["pool_exhaustion_amendment"]
    assert audit["accepted_complete_trajectories_before_exhaustion"] == 14
    assert audit["remaining_complete_trajectory_shortfall"] == 4
    assert audit["frozen_before_extension_model_outputs"] is True
    assert audit["intervention_outcomes_read"] is False
    assert audit["hidden_states_read"] is False
    assert audit["attention_scores_read"] is False


def test_confirmation_pool_exhaustion_amendment_is_a_cumulative_outcome_blind_suffix() -> None:
    from realistic_niah_v6 import replacement

    config = V6Config(prompt_mode="enumeration_bullet")
    discovery_amendment = replacement.load_replacement_policy(
        ROOT
        / "configs"
        / "realistic_niah_v6_replacement_policy_amendment1.json",
        config,
    )
    confirmation_amendment = replacement.load_replacement_policy(
        ROOT
        / "configs"
        / "realistic_niah_v6_replacement_policy_amendment2.json",
        config,
    )
    assert confirmation_amendment["discovery_replacement_seed_pool"] == (
        discovery_amendment["discovery_replacement_seed_pool"]
    )
    base_confirmation = list(
        map(int, discovery_amendment["confirmation_replacement_seed_pool"])
    )
    amended_confirmation = list(
        map(int, confirmation_amendment["confirmation_replacement_seed_pool"])
    )
    assert amended_confirmation[: len(base_confirmation)] == base_confirmation
    assert amended_confirmation[len(base_confirmation) :] == list(range(1514, 1614))
    assert confirmation_amendment["pool_exhaustion_amendment"] == (
        discovery_amendment["pool_exhaustion_amendment"]
    )
    audit = confirmation_amendment["confirmation_pool_exhaustion_amendment"]
    assert audit["accepted_complete_trajectories_before_exhaustion"] == 7
    assert audit["remaining_complete_trajectory_shortfall"] == 2
    assert audit["remaining_analysis_slots"] == [1262, 1263]
    assert audit["confirmation_extension_seeds"] == list(range(1514, 1614))
    assert audit["frozen_before_extension_model_outputs"] is True
    assert audit["intervention_outcomes_read"] is False
    assert audit["hidden_states_read"] is False
    assert audit["attention_scores_read"] is False

    from scripts.run_realistic_niah_v6_broad_panel_replacement import (
        _load_native_loop_policy,
        _replacement_policy_lineage_names,
    )

    lineage = _replacement_policy_lineage_names(
        ROOT
        / "configs"
        / "realistic_niah_v6_replacement_policy_amendment2.json",
        confirmation_amendment,
    )
    assert lineage == {
        "realistic_niah_v6_replacement_policy.json",
        "realistic_niah_v6_replacement_policy_amendment1.json",
        "realistic_niah_v6_replacement_policy_amendment2.json",
    }
    assert "unrelated_policy.json" not in lineage
    coherent = _load_native_loop_policy(
        ROOT
        / "configs"
        / "realistic_niah_v6_coherent_native_loop_replacement_policy.json",
        replacement_policy=(
            ROOT
            / "configs"
            / "realistic_niah_v6_replacement_policy_amendment2.json"
        ),
        replacement_policy_value=confirmation_amendment,
    )
    assert coherent["base_replacement_policy"].endswith(
        "realistic_niah_v6_replacement_policy.json"
    )
    with pytest.raises(ValueError, match="different base policy"):
        _load_native_loop_policy(
            ROOT
            / "configs"
            / "realistic_niah_v6_coherent_native_loop_replacement_policy.json",
            replacement_policy=ROOT / "configs" / "unrelated_policy.json",
            replacement_policy_value={},
        )


def test_incremental_pool_extension_reuses_the_exact_frozen_amendment1_grid(
    tmp_path: Path,
) -> None:
    from realistic_niah_v6 import replacement
    from scripts.build_realistic_niah_v6_replacement_seed_pool import (
        _load_frozen_base_pool,
    )

    config = V6Config(prompt_mode="enumeration_bullet")
    base_policy_path = (
        ROOT
        / "configs"
        / "realistic_niah_v6_replacement_policy_amendment1.json"
    )
    current_policy_path = (
        ROOT
        / "configs"
        / "realistic_niah_v6_replacement_policy_amendment2.json"
    )
    base_policy = replacement.load_replacement_policy(base_policy_path, config)
    current_policy = replacement.load_replacement_policy(current_policy_path, config)
    base_pool = tmp_path / "base_pool"
    base_pool.mkdir()
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")

    discovery = list(map(int, base_policy["discovery_replacement_seed_pool"]))
    confirmation = list(
        map(int, base_policy["confirmation_replacement_seed_pool"])
    )
    rows = [
        {
            "seed": seed,
            "gold_count": count,
            "split": role,
            "v6_replacement_seed_role": role,
            "v6_replacement_candidate": True,
            "v6_replacement_policy_sha256": sha256_file(base_policy_path),
            "v6_replacement_selection_outcomes_available": False,
        }
        for role, seeds in (
            ("discovery", discovery),
            ("confirmation", confirmation),
        )
        for seed in seeds
        for count in config.counts
    ]
    stimuli = base_pool / "stimuli.jsonl"
    stimuli.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": replacement.POOL_SCHEMA_VERSION,
        "status": "PASS_AMENDMENT_RESERVE_POOL",
        "stimuli_sha256": sha256_file(stimuli),
        "replacement_policy_sha256": sha256_file(base_policy_path),
        "source_stimuli_sha256": sha256_file(source),
        "discovery_seeds": discovery,
        "confirmation_seeds": confirmation,
        "v6_stimulus_contract": {
            "design_variant": config.design_variant,
            "counts": list(map(int, config.counts)),
            "original_discovery_seeds": list(map(int, config.discovery_seeds)),
            "original_confirmation_seeds": list(
                map(int, config.confirmation_seeds)
            ),
        },
        "families": [],
    }
    (base_pool / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )

    reused_rows, families, metadata = _load_frozen_base_pool(
        base_pool=base_pool,
        base_policy_path=base_policy_path,
        current_policy=current_policy,
        config=config,
        source_stimuli=source,
    )
    assert reused_rows == rows
    assert families == []
    assert metadata["base_rows_reused"] == 2500
    assert metadata["extension_discovery_seeds"] == []
    assert metadata["extension_confirmation_seeds"] == list(range(1514, 1614))
    assert metadata["construction_mode"] == "incremental_frozen_pool_extension"

    current_discovery = list(
        map(int, current_policy["discovery_replacement_seed_pool"])
    )
    current_confirmation = list(
        map(int, current_policy["confirmation_replacement_seed_pool"])
    )
    current_policy_sha256 = sha256_file(current_policy_path)
    extension_rows = [
        {
            "seed": seed,
            "gold_count": count,
            "split": "confirmation",
            "v6_replacement_seed_role": "confirmation",
            "v6_replacement_candidate": True,
            "v6_replacement_policy_sha256": current_policy_sha256,
            "v6_replacement_selection_outcomes_available": False,
        }
        for seed in range(1514, 1614)
        for count in config.counts
    ]
    current_pool = tmp_path / "current_pool"
    current_pool.mkdir()
    current_stimuli = current_pool / "stimuli.jsonl"
    current_stimuli.write_text(
        stimuli.read_text(encoding="utf-8")
        + "".join(json.dumps(row, sort_keys=True) + "\n" for row in extension_rows),
        encoding="utf-8",
    )
    current_manifest = {
        "schema_version": replacement.POOL_SCHEMA_VERSION,
        "status": "PASS_AMENDMENT_RESERVE_POOL",
        "counts": list(map(int, config.counts)),
        "discovery_seeds": current_discovery,
        "confirmation_seeds": current_confirmation,
        "rows": len(reused_rows) + len(extension_rows),
        "rows_per_seed": len(config.counts),
        "stimuli_sha256": sha256_file(current_stimuli),
        "replacement_policy_sha256": current_policy_sha256,
        "families": [
            {"seed": seed, "seed_role": role}
            for role, seeds in (
                ("discovery", current_discovery),
                ("confirmation", current_confirmation),
            )
            for seed in seeds
        ],
        "intervention_outcomes_read": False,
        "causal_intervention_outcomes_read_during_pool_construction": False,
        "reserve_model_outputs_available_during_pool_construction": False,
        "anchor_regeneration_audits": [
            {
                "role": role,
                "seed": seed,
                "status": "EXACT_MATCH",
                "matched_counts": list(map(int, config.counts)),
            }
            for role, seed in (
                ("discovery", int(config.discovery_seeds[0])),
                ("confirmation", int(config.confirmation_seeds[0])),
            )
        ],
        **metadata,
        "extension_rows_generated": len(extension_rows),
        "historical_rows_preserved_byte_for_byte": True,
    }
    (current_pool / "manifest.json").write_text(
        json.dumps(current_manifest) + "\n", encoding="utf-8"
    )
    import subprocess
    import sys

    audited = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_realistic_niah_v6_replacement_seed_pool.py"),
            "--pool",
            str(current_pool),
            "--replacement-policy",
            str(current_policy_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    audit = json.loads(audited.stdout)
    assert audit["status"] == "PASS_AMENDMENT_RESERVE_POOL_INDEPENDENT_AUDIT"
    assert audit["base_rows_reused"] == 2500
    assert audit["extension_rows_generated"] == 1000
    assert audit["historical_rows_preserved_byte_for_byte"] is True
    assert audit["row_policy_sha256_counts"] == {
        sha256_file(base_policy_path): 2500,
        current_policy_sha256: 1000,
    }


def test_replacement_resolution_preserves_slots_without_aliasing_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from realistic_niah_v6 import replacement

    config = V6Config(prompt_mode="enumeration_index")
    policy = replacement.load_replacement_policy(
        ROOT / "configs" / "realistic_niah_v6_replacement_policy.json", config
    )
    rows = []
    for seed in DISCOVERY_SEEDS:
        for count in config.counts:
            rows.append(
                {
                    "request_id": f"original/{seed}/{count}",
                    "stimulus_id": f"original/{seed}/{count}",
                    "model_label": "Qwen3-8B",
                    "prompt_mode": "enumeration_index",
                    "seed": seed,
                    "split": "discovery",
                    "gold_count": count,
                    "test_strict": not (seed == 1234 and count == 10),
                }
            )
    rows.extend(
        [
            {
                "request_id": "replacement/1264/10",
                "stimulus_id": "replacement/1264/10",
                "model_label": "Qwen3-8B",
                "prompt_mode": "enumeration_index",
                "seed": 1264,
                "split": "discovery",
                "gold_count": 10,
                "test_strict": False,
            },
            {
                "request_id": "replacement/1265/10",
                "stimulus_id": "replacement/1265/10",
                "model_label": "Qwen3-8B",
                "prompt_mode": "enumeration_index",
                "seed": 1265,
                "split": "discovery",
                "gold_count": 10,
                "test_strict": True,
            },
        ]
    )

    def eligibility(row):
        strict = bool(row["test_strict"])
        return {
            "eligible": strict,
            "failure_reasons": [] if strict else ["fresh_v6_strict_parser_failure"],
            "trace_parse": {"strict_causal_eligible": strict},
        }

    monkeypatch.setattr(replacement, "audit_generation_eligibility", eligibility)
    resolved = replacement.resolve_replacement_panel(
        rows,
        config=config,
        model_label="Qwen3-8B",
        seed_role="discovery",
        policy=policy,
    )
    assert resolved["complete"] is True
    assert resolved["replacement_count"] == 1
    assert resolved["replacement_mapping"] == [
        {
            "schema_version": replacement.MAPPING_SCHEMA_VERSION,
            "model_label": "Qwen3-8B",
            "prompt_mode": "enumeration_index",
            "split": "discovery",
            "gold_count": 10,
            "analysis_slot_seed": 1234,
            "original_seed": 1234,
            "original_request_id": "original/1234/10",
            "original_failure_reasons": ["fresh_v6_strict_parser_failure"],
            "replacement_seed": 1265,
            "replacement_request_id": "replacement/1265/10",
            "replacement_candidate_rank": 2,
            "selection_basis": "lowest_reserved_seed_with_fresh_strict_parser_PASS",
            "intervention_outcomes_read": False,
        }
    ]
    registry = tmp_path / "selected_cells.jsonl"
    registry.write_text(
        "".join(json.dumps(row) + "\n" for row in resolved["selected_cells"]),
        encoding="utf-8",
    )
    materialized = replacement.resolved_generation_records(
        rows,
        config,
        registry_path=registry,
        model_label="Qwen3-8B",
    )
    replacement_row = next(
        row for row in materialized if int(row["v6_analysis_slot_seed"]) == 1234
        and int(row["gold_count"]) == 10
    )
    assert replacement_row["seed"] == 1265
    assert replacement_row["v6_analysis_slot_seed"] == 1234
    assert replacement_row["v6_replacement_applied"] is True
    assert len(materialized) == 200


def test_count_stream_uses_slot_only_for_panel_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_realistic_niah_v6_count_stream as wrapper

    rows = [
        {
            "request_id": "original/1234/1",
            "seed": 1234,
            "gold_count": 1,
            "v6_analysis_slot_seed": 1234,
            "v6_replacement_applied": False,
        },
        {
            "request_id": "replacement/1264/2",
            "seed": 1264,
            "gold_count": 2,
            "v6_analysis_slot_seed": 1234,
            "v6_replacement_applied": True,
        },
        {
            "request_id": "replacement/1265/1",
            "seed": 1265,
            "gold_count": 1,
            "v6_analysis_slot_seed": 1244,
            "v6_replacement_applied": True,
        },
    ]

    class Mechanism:
        formal_inference_eligible = True

        @staticmethod
        def seed_role(seed: int) -> str | None:
            return "development" if seed in {1234, 1244} else None

        @staticmethod
        def broad_phase(seed: int) -> str:
            return (
                "ranking_discovery" if seed == 1234 else "k_selection_discovery"
            )

        @staticmethod
        def broad_counts_for_seed(seed: int, *, phase: str):
            del seed, phase
            return (1, 2)

    legacy = SimpleNamespace(
        _validate_seed_contract=lambda config, mechanism: None,
        _cohort_exclusion_reason=lambda row, cohort: None,
    )
    monkeypatch.setattr(wrapper.V6Config, "load", lambda path: object())
    monkeypatch.setattr(wrapper, "read_jsonl", lambda path: [])
    monkeypatch.setattr(wrapper, "_registered_adapter", lambda *args, **kwargs: rows)
    args = SimpleNamespace(
        v5_config=Path("unused.json"),
        generations=Path("unused.jsonl"),
        model="Qwen3-8B",
        seed_role="development",
        cohort="one_to_one",
        row_panel="broad_ranking",
        limit=None,
    )
    selected = wrapper._v6_registered_rows(
        args, Mechanism(), legacy=legacy
    )
    assert [row["seed"] for row in selected] == [1234, 1264]
    assert {row["v6_panel_membership_seed"] for row in selected} == {1234}
    assert args.cohort_audit["replacement_rows"] == 1
    assert args.cohort_audit["statistical_identity"] == "true_source_seed"
    assert args.cohort_audit["seed_aliasing"] is False


def test_trace_terminal_analysis_routes_to_registered_answer_outcome() -> None:
    import numpy as np
    import pandas as pd

    from scripts import run_realistic_niah_v6_count_stream as wrapper

    trials = pd.DataFrame(
        [
            {
                "experiment_id": "trace_intermediate_state_patching",
                "terminal_panel_answer_only": False,
                "local_next_city_outcome_registered": True,
                "final_answer_outcome_registered": True,
                "donor_vs_receiver_city_log_odds": 0.5,
                "correct_count_margin": 1.0,
            },
            {
                "experiment_id": "trace_terminal_state_patching",
                "terminal_panel_answer_only": True,
                "local_next_city_outcome_registered": False,
                "final_answer_outcome_registered": True,
                "donor_vs_receiver_city_log_odds": float("nan"),
                "correct_count_margin": 2.0,
            },
        ]
    )
    legacy = SimpleNamespace(pd=pd, np=np)
    local_outcome, local_audit = wrapper._v6_analysis_outcome_for_experiment(
        trials,
        experiment_id="trace_intermediate_state_patching",
        requested_outcome="donor_vs_receiver_city_log_odds",
        legacy=legacy,
    )
    terminal_outcome, terminal_audit = (
        wrapper._v6_analysis_outcome_for_experiment(
            trials,
            experiment_id="trace_terminal_state_patching",
            requested_outcome="donor_vs_receiver_city_log_odds",
            legacy=legacy,
        )
    )
    assert local_outcome == "donor_vs_receiver_city_log_odds"
    assert local_audit is None
    assert terminal_outcome == "correct_count_margin"
    assert terminal_audit is not None
    assert terminal_audit["status"] == (
        "PASS_REGISTERED_TERMINAL_ANSWER_OUTCOME_ROUTING"
    )
    assert terminal_audit["local_panel_outcome_changed"] is False
    assert terminal_audit["model_trials_recomputed"] is False
    assert terminal_audit["seed_replacement_triggered"] is False


def test_coherent_broad_replacement_never_splices_a_seed_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from realistic_niah_v6 import replacement

    config = V6Config.load(
        ROOT / "configs" / "realistic_niah_v6_enumeration_index.json"
    )
    class Mechanism:
        broad_ranking_seeds = tuple(range(1234, 1244))
        broad_k_selection_seeds = tuple(range(1244, 1254))
        confirmation_seeds = tuple(range(1254, 1264))

        @staticmethod
        def broad_counts_for_seed(seed: int, *, phase: str) -> tuple[int, ...]:
            if phase == "ranking_discovery":
                return tuple(range(1, 11))
            seeds = (
                Mechanism.broad_k_selection_seeds
                if phase == "k_selection_discovery"
                else Mechanism.confirmation_seeds
            )
            parity = 1 if seeds.index(seed) % 2 == 0 else 0
            return tuple(count for count in range(1, 11) if count % 2 == parity)

    mechanism = Mechanism()
    policy = replacement.load_replacement_policy(
        ROOT / "configs" / "realistic_niah_v6_replacement_policy.json",
        config,
    )

    def generation(seed: int, count: int, *, strict: bool = True) -> dict:
        return {
            "request_id": f"request/{seed}/{count}",
            "stimulus_id": f"stimulus/{seed}/{count}",
            "model_label": "Qwen3-8B",
            "prompt_mode": config.prompt_mode,
            "seed": seed,
            "gold_count": count,
            "strict_fixture": strict,
        }

    rows = [
        generation(seed, count, strict=not (seed == 1244 and count == 1))
        for seed in config.discovery_seeds
        for count in config.counts
    ]
    rows.extend(
        generation(1264, count, strict=count != 3)
        for count in (1, 3, 5, 7, 9)
    )
    base_registry = []
    for count in config.counts:
        for slot_seed in config.discovery_seeds:
            source_seed = 1264 if (slot_seed, count) == (1244, 1) else slot_seed
            source = next(
                row
                for row in rows
                if row["seed"] == source_seed and row["gold_count"] == count
            )
            base_registry.append(
                {
                    "schema_version": replacement.SELECTED_CELL_SCHEMA_VERSION,
                    "model_label": "Qwen3-8B",
                    "prompt_mode": config.prompt_mode,
                    "split": "discovery",
                    "gold_count": count,
                    "analysis_slot_seed": slot_seed,
                    "source_seed": source_seed,
                    "source_request_id": source["request_id"],
                    "source_stimulus_id": source["stimulus_id"],
                    "replacement_applied": source_seed != slot_seed,
                    "original_failure_reasons": [],
                    "replacement_candidate_rank": (
                        1 if source_seed != slot_seed else None
                    ),
                    "eligibility_rule": (
                        "fresh_v6_parse.strict_causal_eligible_is_true"
                    ),
                    "intervention_outcomes_read": False,
                }
            )

    monkeypatch.setattr(
        replacement,
        "audit_generation_eligibility",
        lambda row: {
            "eligible": bool(row["strict_fixture"]),
            "failure_reasons": (
                [] if row["strict_fixture"] else ["fixture_strict_failure"]
            ),
            "trace_parse": {},
        },
    )
    pending = replacement.resolve_coherent_broad_panel(
        rows,
        config=config,
        model_label="Qwen3-8B",
        seed_role="discovery",
        policy=policy,
        mechanism=mechanism,
        phase="k_selection_discovery",
        base_registry=base_registry,
    )
    assert pending["complete"] is False
    assert pending["affected_slots"] == [1244]
    assert pending["next_candidates"] == [
        {"seed": 1265, "gold_count": count} for count in (1, 3, 5, 7, 9)
    ]

    rows.extend(generation(1265, count) for count in (1, 3, 5, 7, 9))
    resolved = replacement.resolve_coherent_broad_panel(
        rows,
        config=config,
        model_label="Qwen3-8B",
        seed_role="discovery",
        policy=policy,
        mechanism=mechanism,
        phase="k_selection_discovery",
        base_registry=base_registry,
    )
    assert resolved["complete"] is True
    assert resolved["accepted_replacement_seed_by_slot"] == {"1244": 1265}
    panel = [
        row
        for row in resolved["selected_cells"]
        if row["analysis_slot_seed"] == 1244
        and row["gold_count"] in {1, 3, 5, 7, 9}
    ]
    assert len(panel) == 5
    assert {row["source_seed"] for row in panel} == {1265}
    assert {row["source_request_id"] for row in panel} == {
        f"request/1265/{count}" for count in (1, 3, 5, 7, 9)
    }
    assert resolved["coherent_mapping"][0][
        "successful_original_cells_replaced_for_seed_coherence"
    ] == [3, 5, 7, 9]
    assert len(
        {row["source_request_id"] for row in resolved["selected_cells"]}
    ) == len(resolved["selected_cells"])


def test_coherent_native_loop_replaces_complete_count_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from realistic_niah_v6 import replacement

    config = V6Config.load(
        ROOT / "configs" / "realistic_niah_v6_enumeration_index.json"
    )
    policy = replacement.load_replacement_policy(
        ROOT / "configs" / "realistic_niah_v6_replacement_policy.json",
        config,
    )

    def generation(seed: int, count: int, *, strict: bool = True) -> dict:
        return {
            "request_id": f"request/{seed}/{count}",
            "stimulus_id": f"stimulus/{seed}/{count}",
            "model_label": "Qwen3-8B",
            "prompt_mode": config.prompt_mode,
            "seed": seed,
            "gold_count": count,
            "strict_fixture": strict,
        }

    rows = [
        generation(seed, count, strict=not (seed == 1234 and count == 9))
        for seed in config.discovery_seeds
        for count in config.counts
    ]
    rows.extend(generation(1264, count) for count in range(2, 11))
    by_key = {(row["seed"], row["gold_count"]): row for row in rows}
    base_registry = []
    for count in config.counts:
        for slot in config.discovery_seeds:
            source = 1264 if (slot, count) == (1234, 9) else slot
            source_row = by_key[(source, count)]
            base_registry.append(
                {
                    "schema_version": replacement.SELECTED_CELL_SCHEMA_VERSION,
                    "model_label": "Qwen3-8B",
                    "prompt_mode": config.prompt_mode,
                    "split": "discovery",
                    "gold_count": count,
                    "analysis_slot_seed": slot,
                    "source_seed": source,
                    "source_request_id": source_row["request_id"],
                    "source_stimulus_id": source_row["stimulus_id"],
                    "replacement_applied": source != slot,
                    "original_failure_reasons": [],
                    "replacement_candidate_rank": 1 if source != slot else None,
                    "eligibility_rule": "fresh_v6_parse.strict_causal_eligible_is_true",
                    "intervention_outcomes_read": False,
                }
            )
    monkeypatch.setattr(
        replacement,
        "audit_generation_eligibility",
        lambda row: {
            "eligible": bool(row["strict_fixture"]),
            "failure_reasons": (
                [] if row["strict_fixture"] else ["fixture_strict_failure"]
            ),
            "trace_parse": {},
        },
    )
    resolved = replacement.resolve_coherent_native_loop_panel(
        rows,
        config=config,
        model_label="Qwen3-8B",
        seed_role="discovery",
        policy=policy,
        base_registry=base_registry,
    )
    assert resolved["complete"] is True
    assert resolved["phase"] == "native_loop_discovery"
    active = [
        row
        for row in resolved["selected_cells"]
        if int(row["analysis_slot_seed"]) == 1234
        and 2 <= int(row["gold_count"]) <= 10
    ]
    assert len(active) == 9
    assert {int(row["source_seed"]) for row in active} == {1264}
    assert resolved["coherent_mapping"][0][
        "successful_original_cells_replaced_for_seed_coherence"
    ] == [2, 3, 4, 5, 6, 7, 8, 10]


def test_native_loop_planner_uses_one_true_source_seed_per_slot() -> None:
    from types import SimpleNamespace
    from scripts.run_realistic_niah_v6_count_stream import (
        _v6_native_loop_plan_for_rows,
    )

    config = V6Config.load(
        ROOT / "configs" / "realistic_niah_v6_enumeration_index.json"
    )
    rows = []
    for slot in config.discovery_seeds:
        source = 1264 if int(slot) == 1234 else int(slot)
        for count in config.counts:
            rows.append(
                {
                    "seed": source,
                    "gold_count": int(count),
                    "v6_analysis_slot_seed": int(slot),
                }
            )
    captured = {}

    def build(plan_rows: list[dict], **kwargs: object) -> dict:
        captured.update(kwargs)
        captured["row_count"] = len(plan_rows)
        return captured

    legacy = SimpleNamespace(build_fixed_native_loop_plan=build)
    args = SimpleNamespace(
        seed_role="development",
        model="Qwen3-8B",
        donor_offsets=(-3, -2, -1, 1, 2, 3),
        random_seed=20260821,
        allow_incomplete_offsets=False,
        no_boundaries=False,
    )
    mechanism = SimpleNamespace(experiment_id="fixture")
    audit = {}
    result = _v6_native_loop_plan_for_rows(
        args,
        mechanism,
        rows,
        legacy=legacy,
        config=config,
        adapter_audit=audit,
    )
    assert result["seeds"][0] == 1264
    assert len(set(result["seeds"])) == 20
    assert result["candidate_counts"] == tuple(range(2, 11))
    assert audit["native_loop_seed_identity"]["seed_aliasing"] is False


def test_specialized_kernel_installs_slot_membership_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import run_realistic_niah_v5_count_stream as legacy_count
    from scripts import run_realistic_niah_v6_count_stream as v6_count
    from scripts import run_realistic_niah_v6_kernel as kernel

    rows = [
        {
            "request_id": "replacement/1264/10",
            "seed": 1264,
            "gold_count": 10,
            "v6_analysis_slot_seed": 1234,
            "v6_replacement_applied": True,
        }
    ]

    def registered(*_args, **_kwargs):
        return rows

    registered.include_nonstrict = False
    registered.config_sha256 = None
    registered.cohort_registry = None
    monkeypatch.setattr(v6_count, "_registered_adapter", registered)
    monkeypatch.setattr(v6_count, "read_jsonl", lambda _path: [])
    monkeypatch.setattr(legacy_count, "_validate_seed_contract", lambda *a: None)
    monkeypatch.setattr(
        legacy_count, "_cohort_exclusion_reason", lambda _row, _cohort: None
    )

    class Mechanism:
        formal_inference_eligible = True

        @staticmethod
        def seed_role(seed: int) -> str | None:
            return "development" if seed == 1234 else None

    config_path = ROOT / "configs" / "realistic_niah_v6_enumeration_index.json"
    original = {
        "V5Config": legacy_count.V5Config,
        "registered_records": legacy_count.registered_records,
        "_registered_rows": legacy_count._registered_rows,
    }
    try:
        audit = kernel._install_count_stream_registered_rows_adapter(
            config_path=config_path,
            include_nonstrict=False,
            cohort_registry=Path("selected_cells.jsonl"),
        )
        args = SimpleNamespace(
            v5_config=config_path,
            generations=Path("unused.jsonl"),
            model="Qwen3-8B",
            seed_role="development",
            cohort="one_to_one",
            row_panel="all",
            limit=None,
        )
        selected = legacy_count._registered_rows(args, Mechanism())
    finally:
        for name, value in original.items():
            setattr(legacy_count, name, value)
    assert [row["seed"] for row in selected] == [1264]
    assert selected[0]["v6_panel_membership_seed"] == 1234
    assert audit["statistical_identity"] == "true_source_seed"
    assert audit["seed_aliasing"] is False


def test_specialized_role_adapter_changes_only_discovery_spelling(
    tmp_path: Path,
) -> None:
    from scripts import run_realistic_niah_v6_kernel as kernel

    config = V6Config(prompt_mode="enumeration_index")
    config_path = ROOT / "configs" / "realistic_niah_v6_enumeration_index.json"
    panel = tmp_path / "mode_panel.jsonl"
    rows = []
    for slot_seed in DISCOVERY_SEEDS:
        source_seed = 1264 if slot_seed == DISCOVERY_SEEDS[0] else slot_seed
        rows.append(
            {
                "request_id": f"request/{slot_seed}/{source_seed}/10",
                "seed": source_seed,
                "source_seed": source_seed,
                "analysis_slot_seed": slot_seed,
                "replacement_applied": source_seed != slot_seed,
                "gold_count": 10,
                "stratified_ncc_seed_role": "discovery",
                "intervention_sentinel": slot_seed / 1000,
            }
        )
    panel.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    arguments = [
        "--model",
        "Qwen3-8B",
        "--seed-role",
        "development",
        "--panel",
        str(panel),
    ]
    adapted_arguments, view, audit = (
        kernel._materialize_specialized_seed_role_view(
            arguments,
            config=config,
            config_path=config_path,
            target="stratified-targeted-counter-ncc",
        )
    )
    assert view is not None
    assert adapted_arguments[adapted_arguments.index("--panel") + 1] == str(view)
    adapted = [json.loads(line) for line in view.read_text().splitlines()]
    assert {row["stratified_ncc_seed_role"] for row in adapted} == {"development"}
    assert [
        {key: value for key, value in row.items() if key != "stratified_ncc_seed_role"}
        for row in adapted
    ] == [
        {key: value for key, value in row.items() if key != "stratified_ncc_seed_role"}
        for row in rows
    ]
    assert audit["status"] == "APPLIED_V6_TO_LEGACY_ROLE_SPELLING"
    assert audit["analysis_slot_count"] == 20
    assert audit["true_source_seed_count"] == 20
    assert audit["seed_aliasing"] is False
    assert audit["intervention_outcomes_read"] is False


def test_specialized_slot_audit_records_true_replacement_identity(
    tmp_path: Path,
) -> None:
    from scripts import run_realistic_niah_v6_kernel as kernel

    config = V6Config(prompt_mode="enumeration_index")
    generations = tmp_path / "resolved.jsonl"
    registry = tmp_path / "mode_panel.jsonl"
    plan = tmp_path / "bank.csv"
    generation_rows = []
    registry_rows = []
    source_seeds = []
    for slot_seed in DISCOVERY_SEEDS:
        source_seed = 1264 if slot_seed == DISCOVERY_SEEDS[0] else slot_seed
        source_seeds.append(source_seed)
        request_id = f"request/{slot_seed}/{source_seed}/10"
        generation_rows.append(
            {
                "request_id": request_id,
                "model_label": "Qwen3-8B",
                "split": "discovery",
                "seed": source_seed,
                "gold_count": 10,
                "v6_analysis_slot_seed": slot_seed,
                "v6_source_seed": source_seed,
                "v6_replacement_applied": source_seed != slot_seed,
            }
        )
        registry_rows.append(
            {
                "request_id": request_id,
                "seed": source_seed,
                "source_seed": source_seed,
                "analysis_slot_seed": slot_seed,
                "replacement_applied": source_seed != slot_seed,
                "gold_count": 10,
            }
        )
    generations.write_text(
        "".join(json.dumps(row) + "\n" for row in generation_rows),
        encoding="utf-8",
    )
    registry.write_text(
        "".join(json.dumps(row) + "\n" for row in registry_rows),
        encoding="utf-8",
    )
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model_label",
                "condition",
                "bank_size",
                "validation_seeds",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "model_label": "Qwen3-8B",
                "condition": "selected_bank",
                "bank_size": 64,
                "validation_seeds": json.dumps(source_seeds),
            }
        )
    arguments = [
        "--mode",
        "answer",
        "--model",
        "Qwen3-8B",
        "--generations",
        str(generations),
        "--split",
        "discovery",
        "--anchor-registry",
        str(registry),
        "--bank-plan",
        str(plan),
        "--bank-size",
        "64",
    ]
    audit = kernel._audit_specialized_slot_identity(
        arguments,
        config=config,
        target="token-level-ablation",
    )
    assert audit["status"] == "PASS_FIXED_SLOT_TRUE_SOURCE_IDENTITY"
    assert audit["replacement_count"] == 1
    assert audit["analysis_slot_count"] == 20
    assert audit["true_source_seed_count"] == 20
    assert audit["slot_to_true_source_mapping"][0] == {
        "analysis_slot_seed": 1234,
        "true_source_seed": 1264,
        "request_id": "request/1234/1264/10",
        "replacement_applied": True,
    }
    assert audit["bank_validation_membership"]["status"] == (
        "PASS_TRUE_SOURCE_MEMBERSHIP"
    )
    assert audit["seed_aliasing"] is False

    broken = list(registry_rows)
    broken[0] = {**broken[0], "analysis_slot_seed": 9999}
    registry.write_text(
        "".join(json.dumps(row) + "\n" for row in broken),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="slot disagrees"):
        kernel._audit_specialized_slot_identity(
            arguments,
            config=config,
            target="token-level-ablation",
        )


def test_discovery_freeze_builder_locks_all_registered_choices(
    tmp_path: Path,
) -> None:
    from scripts.freeze_realistic_niah_v6_confirmation import build_freeze
    from realistic_niah_v6.suite import validate_confirmation_freeze

    model_root = tmp_path / "model"

    def write(relative: str, value: object = None) -> Path:
        path = model_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if value is None:
            path.write_text("artifact\n", encoding="utf-8")
        elif isinstance(value, str):
            path.write_text(value, encoding="utf-8")
        else:
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return path

    for relative in (
        "discovery-foundation-resolved.COMPLETE",
        "replacement/discovery/discovery.COMPLETE",
        "replacement/discovery_broad_k/k_selection_discovery.COMPLETE",
        "causal/targeted_retrieval/discovery_formal/all.COMPLETE",
        "count_stream/discovery_formal/stage1.COMPLETE",
        "causal/specialized/discovery_formal/discovery.COMPLETE",
        "causal/report_tail/discovery_formal/discovery.COMPLETE",
    ):
        write(relative, "PASS\n")
    write("replacement/discovery/selected_cells.jsonl")
    write("replacement/discovery_broad_k/selected_cells.jsonl")
    write("replacement/discovery_native_loop/selected_cells.jsonl")
    write("replacement/discovery_native_loop/manifest.json", {"status": "PASS"})
    write("representation/formal/v6_adapter_manifest.json", {"status": "PASS"})

    targeted = "causal/targeted_retrieval/discovery_formal"
    plan = write(f"{targeted}/plans/k64/retrieval_anchor_bank_plan.csv")
    write(
        f"{targeted}/analysis/selection.json",
        {
            "status": "DISCOVERY_FROZEN_CHOICE",
            "model_label": "Qwen3-8B",
            "prompt_mode": "enumeration_index",
            "selected_k": 64,
            "selection_rule": "registered_test_rule",
            "frozen_plan_sha256": sha256_file(plan),
        },
    )

    count = "count_stream/discovery_formal"
    write(
        f"{count}/stage1_complete.json",
        {
            "status": "DISCOVERY_COMPLETE",
            "source_layer": 18,
            "readout_layers": [19],
        },
    )
    write(f"{count}/running_basis.npz")
    write(f"{count}/running_basis.json", {"status": "PASS"})
    write(f"{count}/trace_patch_analysis/manifest.json", {"status": "PASS"})

    specialized = "causal/specialized/discovery_formal"
    write(f"{specialized}/specialized_discovery_complete.json", {"status": "PASS"})
    write(f"{specialized}/targeted_counter_write/manifest.json", {"source_layer": 19})
    write(f"{specialized}/token_ablation_answer/worker_00_manifest.json", {})
    write(f"{specialized}/terminal_state_bridge/manifest.json", {})
    write(f"{specialized}/stratified_ncc/manifest.json", {})
    write(f"{specialized}/direct_count_logit_margin/manifest.json", {})
    write(f"{specialized}/count_geometry_ncc/manifest.json", {})

    report = "causal/report_tail/discovery_formal"
    write(
        f"{report}/natural_layer_sweep/layer_sweep_analysis.json",
        {
            "selection_split": "discovery",
            "scopes": [
                {
                    "scope": "event_tail_w4",
                    "status": "FROZEN",
                    "selected_layer": 19,
                    "negative_result_retained": False,
                },
                {
                    "scope": "item_span",
                    "status": "NEGATIVE_FROZEN",
                    "selected_layer": None,
                    "negative_result_retained": True,
                },
            ],
        },
    )
    write(f"{report}/native_loop/contract/manifest.json", {})
    write(f"{report}/native_loop/analysis/claim_gates.json", {})
    write(f"{report}/restoration/analysis/manifest.json", {})

    config_path = ROOT / "configs/realistic_niah_v6_enumeration_index.json"
    mechanism_path = (
        ROOT
        / "configs/realistic_niah_v6_enumeration_index_count_stream_dev.json"
    )
    ledger, freeze, mechanism, audit = build_freeze(
        config_path=config_path,
        mechanism_config=mechanism_path,
        model_label="Qwen3-8B",
        model_root=model_root,
    )
    assert ledger["status"] == "DISCOVERY_FROZEN"
    assert len(ledger["experiments"]) == 14
    assert all(
        cell["status"] == "FROZEN" for cell in ledger["experiments"].values()
    )
    assert all(
        cell["negative_result_retained"] is True
        for cell in ledger["experiments"].values()
    )
    assert ledger["seed_identity_contract"]["seed_aliasing"] is False
    assert mechanism["status"] == "frozen_confirmation"
    assert audit["status"] == "PASS_DISCOVERY_LOCKED_BEFORE_CONFIRMATION"
    assert audit["experiment_count"] == 14
    validate_confirmation_freeze(
        freeze,
        prompt_mode="enumeration_index",
        model_label="Qwen3-8B",
    )


def test_specialized_confirmation_bank_routes_replacement_source_seed(
    tmp_path: Path,
) -> None:
    from scripts import run_realistic_niah_v6_kernel as kernel

    config_path = ROOT / "configs/realistic_niah_v6_enumeration_index.json"
    config = V6Config.load(config_path)
    registry = tmp_path / "confirmation_cells.jsonl"
    rows = []
    for slot_seed in CONFIRMATION_SEEDS:
        source_seed = 1364 if slot_seed == CONFIRMATION_SEEDS[0] else slot_seed
        for count in config.counts:
            rows.append(
                {
                    "split": "confirmation",
                    "gold_count": count,
                    "analysis_slot_seed": slot_seed,
                    "source_seed": source_seed,
                    "replacement_applied": source_seed != slot_seed,
                }
            )
    registry.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    plan = tmp_path / "bank.csv"
    fields = (
        "model_label",
        "fold",
        "condition",
        "bank_size",
        "validation_seeds",
        "heads",
    )
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "model_label": "Qwen3-8B",
                "fold": 0,
                "condition": "selected_bank",
                "bank_size": 1,
                "validation_seeds": json.dumps(list(CONFIRMATION_SEEDS)),
                "heads": json.dumps([[1, 2]]),
            }
        )
    arguments, view, audit = kernel._materialize_confirmation_bank_membership(
        ["--bank-plan", str(plan)],
        config=config,
        config_path=config_path,
        model_label="Qwen3-8B",
        phase="confirmation",
        target="token-level-ablation",
        cohort_registry=registry,
    )
    assert view is not None
    assert arguments == ["--bank-plan", str(view)]
    with view.open("r", encoding="utf-8", newline="") as handle:
        routed = next(csv.DictReader(handle))
    assert 1364 in json.loads(routed["validation_seeds"])
    assert routed["heads"] == json.dumps([[1, 2]])
    assert audit["status"] == "APPLIED_CONFIRMATION_REPLACEMENT_ROUTING"
    assert audit["selected_heads_or_scores_changed"] is False
    assert audit["replacement_slot_to_source_seed"] == [
        {"analysis_slot_seed": 1254, "source_seed": 1364}
    ]
    assert audit["seed_aliasing"] is False


def test_causal_membership_adapter_adds_true_seed_without_aliasing(
    tmp_path: Path,
) -> None:
    from scripts import run_realistic_niah_v6_causal as wrapper
    from realistic_niah_v6.replacement import SELECTED_CELL_SCHEMA_VERSION

    config = V6Config.load(
        ROOT / "configs" / "realistic_niah_v6_enumeration_index.json"
    )
    registry_rows = []
    for count in config.counts:
        for slot_seed in config.discovery_seeds:
            source_seed = 1264 if (count, slot_seed) == (10, 1234) else slot_seed
            registry_rows.append(
                {
                    "schema_version": SELECTED_CELL_SCHEMA_VERSION,
                    "model_label": "Qwen3-8B",
                    "prompt_mode": config.prompt_mode,
                    "split": "discovery",
                    "gold_count": count,
                    "analysis_slot_seed": slot_seed,
                    "source_seed": source_seed,
                    "source_request_id": f"request/{count}/{source_seed}/{slot_seed}",
                    "replacement_applied": source_seed != slot_seed,
                }
            )
    registry = tmp_path / "selected_cells.jsonl"
    registry.write_text(
        "".join(json.dumps(row) + "\n" for row in registry_rows),
        encoding="utf-8",
    )

    adapted, audit = wrapper._causal_seed_membership_config(
        config,
        cohort_registry=registry,
        model_label="Qwen3-8B",
    )

    assert adapted.discovery_seeds == config.discovery_seeds == DISCOVERY_SEEDS
    assert adapted.confirmation_seeds == config.confirmation_seeds
    assert config.causal_development_seeds == DISCOVERY_SEEDS
    assert 1264 in adapted.causal_development_seeds
    assert 1264 not in config.causal_development_seeds
    assert adapted.causal_confirmation_seeds == config.causal_confirmation_seeds
    assert audit["added_true_source_seeds"] == [1264]
    assert audit["panel_membership_identity"] == "analysis_slot_seed"
    assert audit["statistical_identity"] == "true_source_seed"
    assert audit["seed_aliasing"] is False
    assert adapted.to_dict()["v6_causal_seed_membership_adapter"][
        "status"
    ] == "APPLIED_TRUE_SOURCE_SEED_MEMBERSHIP"

    confirmation_rows = []
    for count in config.counts:
        for slot_seed in config.confirmation_seeds:
            source_seed = 1364 if (count, slot_seed) == (10, 1254) else slot_seed
            confirmation_rows.append(
                {
                    "schema_version": SELECTED_CELL_SCHEMA_VERSION,
                    "model_label": "Qwen3-8B",
                    "prompt_mode": config.prompt_mode,
                    "split": "confirmation",
                    "gold_count": count,
                    "analysis_slot_seed": slot_seed,
                    "source_seed": source_seed,
                    "source_request_id": (
                        f"confirmation/{count}/{source_seed}/{slot_seed}"
                    ),
                    "replacement_applied": source_seed != slot_seed,
                }
            )
    confirmation_registry = tmp_path / "confirmation_selected_cells.jsonl"
    confirmation_registry.write_text(
        "".join(json.dumps(row) + "\n" for row in confirmation_rows),
        encoding="utf-8",
    )
    combined, combined_audit = wrapper._causal_seed_membership_config(
        config,
        cohort_registry=confirmation_registry,
        additional_cohort_registries=[registry],
        model_label="Qwen3-8B",
    )
    assert 1264 in combined.causal_development_seeds
    assert 1364 in combined.causal_confirmation_seeds
    assert combined_audit["registry_roles"] == ["confirmation", "discovery"]
    assert combined_audit["added_true_source_seeds_by_role"] == {
        "discovery": [1264],
        "confirmation": [1364],
    }
    assert combined_audit["seed_aliasing"] is False

    routed, routed_audit = wrapper._route_legacy_causal_seed_role(
        combined,
        combined_audit,
        command="causal-source-writes",
        phase="confirmation",
    )
    assert routed.causal_development_seeds == combined.causal_confirmation_seeds
    assert routed.causal_confirmation_seeds == combined.causal_development_seeds
    assert 1364 in routed.causal_development_seeds
    assert 1264 in routed.causal_confirmation_seeds
    assert routed_audit["legacy_command_seed_role_routing"]["status"] == (
        "APPLIED_CONFIRMATION_TO_LEGACY_DEVELOPMENT_FILTER"
    )
    assert routed_audit["legacy_command_seed_role_routing"][
        "source_seed_identities_changed"
    ] is False
    routed_heads, routed_heads_audit = wrapper._route_legacy_causal_seed_role(
        combined,
        combined_audit,
        command="causal-heads",
        phase="confirmation",
    )
    assert routed_heads.causal_development_seeds == (
        combined.causal_confirmation_seeds
    )
    assert 1364 in routed_heads.causal_development_seeds
    assert routed_heads_audit["legacy_command_seed_role_routing"]["status"] == (
        "APPLIED_CONFIRMATION_TO_LEGACY_DEVELOPMENT_FILTER"
    )
    untouched, untouched_audit = wrapper._route_legacy_causal_seed_role(
        combined,
        combined_audit,
        command="causal-heads-behavior",
        phase="confirmation",
    )
    assert untouched is combined
    assert untouched_audit["legacy_command_seed_role_routing"]["status"] == (
        "NOT_REQUIRED"
    )

    plan = tmp_path / "frozen_plan.csv"
    plan_fields = [
        "model_label",
        "mechanism",
        "fold",
        "condition",
        "validation_seeds",
        "heads",
    ]
    with plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=plan_fields)
        writer.writeheader()
        writer.writerow(
            {
                "model_label": "Qwen3-8B",
                "mechanism": "retrieval_anchor_localization",
                "fold": "0",
                "condition": "selected_bank",
                "validation_seeds": json.dumps(list(CONFIRMATION_SEEDS)),
                "heads": json.dumps([[1, 2], [3, 4]]),
            }
        )
    arguments, plan_view, plan_audit = (
        wrapper._materialize_confirmation_plan_membership(
            ["causal-heads-behavior", "--plan", str(plan)],
            config=config,
            config_path=(
                ROOT / "configs" / "realistic_niah_v6_enumeration_index.json"
            ),
            model_label="Qwen3-8B",
            command="causal-heads-behavior",
            phase="confirmation",
            registry_sources=[confirmation_registry, registry],
        )
    )
    assert plan_view is not None
    assert arguments[2] == str(plan_view)
    with plan_view.open("r", encoding="utf-8", newline="") as handle:
        routed = next(csv.DictReader(handle))
    assert 1364 in json.loads(routed["validation_seeds"])
    assert routed["heads"] == json.dumps([[1, 2], [3, 4]])
    assert plan_audit["status"] == "APPLIED_CONFIRMATION_REPLACEMENT_ROUTING"
    assert plan_audit["selected_heads_or_scores_changed"] is False
    assert plan_audit["seed_aliasing"] is False


def test_analysis_rows_combines_resolved_discovery_and_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_realistic_niah_v6 as runner

    config_path = ROOT / "configs" / "realistic_niah_v6_enumeration_index.json"
    config = V6Config.load(config_path)
    generations = tmp_path / "generations.jsonl"
    generations.write_text("{}\n", encoding="utf-8")
    discovery = (tmp_path / "discovery.jsonl").resolve()
    confirmation = (tmp_path / "confirmation.jsonl").resolve()
    discovery.write_text("{}\n", encoding="utf-8")
    confirmation.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "validate_generation_contracts", lambda *a, **k: None)

    def resolve(_rows, _config, *, registry_path, model_label):
        assert model_label == "Qwen3-8B"
        if Path(registry_path).resolve() == discovery:
            return [
                {
                    "request_id": "discovery/replacement",
                    "split": "discovery",
                    "seed": 1264,
                    "gold_count": 10,
                    "v6_analysis_slot_seed": 1234,
                }
            ]
        assert Path(registry_path).resolve() == confirmation
        return [
            {
                "request_id": "confirmation/replacement",
                "split": "confirmation",
                "seed": 1364,
                "gold_count": 10,
                "v6_analysis_slot_seed": 1254,
            }
        ]

    monkeypatch.setattr(runner, "resolved_generation_records", resolve)
    args = SimpleNamespace(
        generations=generations,
        model="Qwen3-8B",
        config=config_path,
        cohort_registry=discovery,
        additional_cohort_registry=[confirmation],
        include_nonstrict=False,
        seed_role="all",
        limit=None,
    )
    rows = runner._analysis_rows(args, config)
    assert [row["seed"] for row in rows] == [1264, 1364]
    assert [row["v6_analysis_slot_seed"] for row in rows] == [1234, 1254]


def test_final_transition_panel_uses_fixed_n10_resolved_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import build_realistic_niah_v6_final_transition_panel as panel_builder

    config_path = ROOT / "configs" / "realistic_niah_v6_enumeration_index.json"
    config_sha = sha256_file(config_path)
    generations = []
    source = tmp_path / "source"
    shards = source / "shards"
    shards.mkdir(parents=True)
    (source / "manifest.json").write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
    shard_index = 0
    resolved_rows = []
    for slot_seed in DISCOVERY_SEEDS:
        source_seed = 1264 if slot_seed == 1234 else slot_seed
        count = 10
        row = _generation_row("enumeration_index")
        cities = [(f"City{index}", index) for index in range(1, 11)]
        raw = "\n".join(
            f"{index}. {city}: {score}"
            for index, (city, score) in enumerate(cities, start=1)
        ) + "\nTotal: 10"
        request_id = f"Qwen3-8B/enumeration_index/v6/N10_seed{source_seed}"
        row.update(
            {
                "request_id": request_id,
                "stimulus_id": request_id,
                "seed": source_seed,
                "gold_count": count,
                "gold_records": [
                    {"city": city, "score": score} for city, score in cities
                ],
                "raw_output_text": raw,
                "output_token_ids": [ord(value) for value in raw],
                "v6_config_sha256": config_sha,
            }
        )
        generations.append(row)
        resolved_rows.append(
            {
                **row,
                "v6_analysis_slot_seed": slot_seed,
                "v6_source_seed": source_seed,
                "v6_replacement_applied": slot_seed != source_seed,
            }
        )
        query = 100 + slot_seed % 7
        source_row = {
            "request_id": request_id,
            "model_label": "Qwen3-8B",
            "split": "discovery",
            "seed": source_seed,
            "gold_count": count,
            "from_occurrence": 9,
            "to_occurrence": 10,
            "anchor_role": "post_marker",
            "anchor_roles": ["post_marker"],
            "anchor_equivalence_id": f"9->10@q{query}",
            "query_output_token_index": query,
            "grammar_pair": (
                "structural_explicit_rank_before_city -> "
                "structural_explicit_rank_before_city"
            ),
            "target_retrieval_surface_variant": "rank_before_city_compact",
            "capture_complete": True,
        }
        (shards / f"trial_{shard_index:04d}.jsonl").write_text(
            json.dumps(source_row) + "\n", encoding="utf-8"
        )
        shard_index += 1
    generations_path = tmp_path / "generations.jsonl"
    generations_path.write_text(
        "".join(json.dumps(row) + "\n" for row in generations),
        encoding="utf-8",
    )

    cohort_registry = tmp_path / "selected_cells.jsonl"
    cohort_registry.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        panel_builder,
        "resolved_generation_records",
        lambda *args, **kwargs: resolved_rows,
    )
    behavior, targeted, panel, manifest = panel_builder.build_panel(
        config_path=config_path,
        model_label="Qwen3-8B",
        generations_path=generations_path,
        cohort_registry=cohort_registry,
        source_writes=source,
        seed_role="discovery",
    )
    assert len(behavior) == len(DISCOVERY_SEEDS) == 20
    assert {row["gold_count"] for row in behavior} == {10}
    assert {row["analysis_slot_seed"] for row in behavior} == set(DISCOVERY_SEEDS)
    assert 1264 in {row["seed"] for row in behavior}
    assert 1234 not in {row["seed"] for row in behavior}
    assert all("@route-q" in row["anchor_equivalence_id"] for row in targeted)
    assert all(row["mode_timing_stratum"] == "rank_before_city" for row in panel)
    assert set(manifest["selected_count_by_analysis_slot"].values()) == {10}
    assert manifest["intervention_outcomes_read"] is False
    assert manifest["source_write_values_used_for_selection"] is False
    assert manifest["selection_rank_used"] is False


def test_bullet_specialized_geometry_is_not_mislabeled_rank_after_city() -> None:
    from realistic_niah_v6.kernel import install_v6_specialized_geometry
    from realistic_niah_v5 import single_seed_walkthrough as walkthrough
    from realistic_niah_v5 import stratified_targeted_counter_ncc as stratified
    from realistic_niah_v5 import targeted_counter_logit_margin as logit_margin
    from realistic_niah_v5 import targeted_counter_ncc as counter_ncc
    from realistic_niah_v5 import terminal_token_state as terminal

    audit = install_v6_specialized_geometry("enumeration_bullet")
    assert audit["status"] == "STRUCTURAL_ITEM_END_INSTALLED"
    assert audit["mode_timing_stratum"] == "structural_item_end"
    assert "rank_after_city" not in audit["marker_interpretation"]

    class Registry:
        trace_items = ((0, 10),)
        query_position = 12

        @staticmethod
        def positions(name: str):
            if name == "terminal_trace_item":
                return tuple(range(10))
            raise KeyError(name)

    def span(start: int, end: int) -> dict[str, object]:
        return {
            "status": "ok",
            "full_sequence_token_start": start,
            "full_sequence_token_end": end,
        }

    event = {
        "grammar_class": "structural_invariant_bullet",
        "sites": {
            "rank_evidence_core_span": span(0, 1),
            "city_target_span": span(2, 5),
            "post_update_commit_state": span(8, 9),
        },
    }
    geometries, geometry_audit = terminal._grammar_timed_geometry_positions(
        Registry(), event
    )
    assert geometry_audit["grammar_timing_stratum"] == "structural_item_end"
    assert geometries["grammar_terminal_update"] == tuple(range(2, 9))
    carrier, component, timing = counter_ncc.transition_carrier_positions(
        Registry(), event, occurrence=1
    )
    assert carrier == tuple(range(2, 9))
    assert component == "city_to_commit_tail"
    assert timing == "structural_item_end"
    walkthrough_geometry, walkthrough_audit = (
        walkthrough.occurrence_counter_geometry(Registry(), event, 1)
    )
    assert walkthrough_geometry["full_item"] == tuple(range(10))
    assert walkthrough_geometry["counter_carrier"] == tuple(range(2, 9))
    assert walkthrough_audit["grammar_timing_stratum"] == "structural_item_end"
    assert walkthrough_audit["counter_carrier_component"] == "city_to_commit_tail"
    assert walkthrough_audit["invariant_bullet_not_interpreted_as_numeric_rank"] is True
    assert stratified.grammar_timing(event) == "structural_item_end"
    assert stratified.STRATIFIED_NCC_ENDPOINTS["structural_item_end"] == (
        "city_to_commit",
    )
    assert logit_margin.LOGIT_MARGIN_ENDPOINTS["structural_item_end"] == (
        "final_answer_sequence_margin",
    )


def test_completion_evidence_registry_exactly_covers_all_twenty_frames() -> None:
    by_id = {experiment.experiment_id: experiment for experiment in EXPERIMENTS}
    assert set(FRAME_EVIDENCE) == set(by_id)
    assert sorted(experiment.report_frame for experiment in EXPERIMENTS) == list(
        range(1, 21)
    )
    for experiment_id, (discovery, confirmation) in FRAME_EVIDENCE.items():
        assert discovery or confirmation
        if by_id[experiment_id].confirmation_required:
            assert confirmation


def test_completion_resolves_legacy_panel_specific_policy_lineage(tmp_path: Path) -> None:
    policy = tmp_path / "coherent.json"
    lineage_hash = "a" * 64
    resolved_path, resolved_hash = _resolve_coherent_policy_lineage(
        {
            "coherent_broad_policy": str(policy),
            "coherent_broad_policy_sha256": lineage_hash,
        },
        panel_kind="broad",
        manifest_path=tmp_path / "manifest.json",
    )
    assert resolved_path == policy
    assert resolved_hash == lineage_hash


def test_completion_rejects_disagreeing_policy_lineage_aliases(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="policy paths disagree"):
        _resolve_coherent_policy_lineage(
            {
                "coherent_policy": str(tmp_path / "generic.json"),
                "coherent_policy_sha256": "a" * 64,
                "coherent_native_loop_policy": str(tmp_path / "specific.json"),
                "coherent_native_loop_policy_sha256": "a" * 64,
            },
            panel_kind="native_loop",
            manifest_path=tmp_path / "manifest.json",
        )


def test_completion_accepts_only_resolved_foundation_supersession(tmp_path: Path) -> None:
    model_root = tmp_path / "enumeration_bullet" / "Qwen3-8B"
    recovery = model_root / "foundation_marker_recovery_audit.json"
    recovery.parent.mkdir(parents=True)
    recovery.write_text("{}\n", encoding="utf-8")
    marker = model_root / "discovery-foundation-resolved.COMPLETE"
    marker.write_text("PASS\n", encoding="utf-8")
    registry = model_root / "replacement/discovery/selected_cells.jsonl"
    registry.parent.mkdir(parents=True)
    registry.write_text('{"seed":1}\n{"seed":2}\n', encoding="utf-8")
    manifest = model_root / "attention/discovery_answer_query_formal.manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "model_label": "Qwen3-8B",
                "prompt_mode": "enumeration_bullet",
                "seed_role": "discovery",
                "formal_cohort": True,
                "requests": 2,
                "cohort_registry": str(registry),
                "cohort_registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    states = _audit_foundation_recovery_evidence(
        model_root,
        prompt_mode="enumeration_bullet",
        model_label="Qwen3-8B",
        recovery_path=recovery,
        validated={
            "formal_answer_query": {"path": str(manifest), "sha256": "0" * 64}
        },
    )
    assert states["formal_answer_query"]["status"] == (
        "PASS_SUPERSEDED_BY_RESOLVED_FOUNDATION"
    )

    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["cohort_registry_sha256"] = "f" * 64
    manifest.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lineage changed"):
        _audit_foundation_recovery_evidence(
            model_root,
            prompt_mode="enumeration_bullet",
            model_label="Qwen3-8B",
            recovery_path=recovery,
            validated={
                "formal_answer_query": {
                    "path": str(manifest),
                    "sha256": "0" * 64,
                }
            },
        )


def test_report_result_registry_and_scalar_projection_cover_twenty_frames() -> None:
    assert set(RESULT_SOURCES) == {
        experiment.experiment_id for experiment in EXPERIMENTS
    }
    targeted_confirmation = (
        "causal/specialized/confirmation_analysis/targeted_counter_write/"
        "confirmation/claim_gates.json"
    )
    assert RESULT_SOURCES["targeted_query_to_carrier"] == (targeted_confirmation,)
    assert RESULT_SOURCES["carrier_to_commit_restore"] == (targeted_confirmation,)
    values = interesting_scalars(
        {
            "status": "PASS",
            "nested": {
                "effect_estimate": 0.125,
                "confirmation_used_for_selection": False,
                "artifact_path": "/must/not/appear",
            },
        }
    )
    assert ("status", "PASS") in values
    assert ("nested.effect_estimate", "0.125") in values
    assert (
        "nested.confirmation_used_for_selection",
        "false",
    ) in values
    assert not any("artifact_path" in key for key, _ in values)


def test_report_validator_requires_exact_twenty_by_four_structure(
    tmp_path: Path,
) -> None:
    sections = "".join(
        f'<section class="experiment-frame" id="frame-{frame}" data-frame="{frame}">'
        + "".join(
            f'<article class="cell" id="cell-{frame}-{cell}"></article>'
            for cell in range(4)
        )
        + "</section>"
        for frame in range(1, 21)
    )
    report = tmp_path / "report.html"
    report.write_text(
        "<!doctype html><html><body>"
        + sections
        + '<script type="application/json" id="v6-report-summary">{}</script>'
        + "</body></html>",
        encoding="utf-8",
    )
    result = validate_report_html(report)
    assert result["status"] == "PASS"
    assert result["report_frame_count"] == 20
    assert result["model_mode_frame_cell_count"] == 80

    report.write_text(
        report.read_text(encoding="utf-8").replace(
            'id="cell-20-3"', 'id="cell-20-2"'
        ),
        encoding="utf-8",
    )
    failed = validate_report_html(report)
    assert failed["status"] == "FAIL"
    assert any("duplicate ids" in error for error in failed["errors"])


def test_failed_reserve_audit_accepts_canonical_and_legacy_labels() -> None:
    rows = [
        {"candidate_kind": "original", "eligible": False, "seed": 1234},
        {"candidate_kind": "replacement", "eligible": False, "seed": 1264},
        {"candidate_kind": "reserve", "eligible": False, "seed": 1265},
        {"candidate_kind": "replacement", "eligible": True, "seed": 1266},
    ]
    failed = _ordinary_failed_reserve_attempts(rows)
    assert [row["seed"] for row in failed] == [1264, 1265]


def test_causal_plan_directory_adapter_routes_complete_shard_bank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from realistic_niah_v5 import causal
    from scripts.run_realistic_niah_v6_causal import (
        _install_causal_plan_directory_adapter,
    )

    original = causal.build_causal_plan
    calls: dict[str, object] = {}

    def fake_source_planner(source: Path, output: Path, **kwargs: object):
        calls.update(source=source, output=output, kwargs=kwargs)
        return {"plan": output / "plan.csv"}

    monkeypatch.setattr(causal, "_build_source_write_causal_plan", fake_source_planner)
    source = tmp_path / "source_writes"
    shards = source / "shards"
    shards.mkdir(parents=True)
    (shards / "trial.jsonl").write_text(
        json.dumps(
            {
                "source_specific_ov_write_norm": 0.25,
                "local_anchor_eligible": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "plan"
    try:
        audit = _install_causal_plan_directory_adapter()
        result = causal.build_causal_plan(
            source,
            output,
            config=object(),
            bank_size=32,
        )
    finally:
        causal.build_causal_plan = original

    assert audit["status"] == "INSTALLED"
    assert calls["source"] == source
    assert calls["output"] == output
    assert isinstance(calls["kwargs"], dict)
    assert calls["kwargs"]["bank_size"] == 32  # type: ignore[index]
    assert "config" in calls["kwargs"]  # type: ignore[operator]
    assert result["plan"] == output / "plan.csv"


def test_completed_source_write_resume_audit_is_quota_and_hash_locked(
    tmp_path: Path,
) -> None:
    from scripts.audit_realistic_niah_v6_completed_source_write_resume import (
        audit_completed_source_write_resume,
    )

    source = tmp_path / "source"
    shards = source / "shards"
    shards.mkdir(parents=True)
    for index in range(2):
        (shards / f"trial_{index}.jsonl").write_text("{}\n", encoding="utf-8")
    cohort = tmp_path / "cohort.jsonl"
    cohort.write_text("{}\n", encoding="utf-8")
    generation_view = str(tmp_path / "materialized.jsonl")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "command": "causal-source-writes",
                "model_label": "Qwen3-8B",
                "anchor_role": "post_marker",
                "completed_shards": 2,
                "eligible_anchor_tasks": 2,
                "generations": generation_view,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "v6_adapter_manifest.json").write_text(
        json.dumps(
            {
                "status": "INSTALLED",
                "run_status": "COMPLETE",
                "command": "causal-source-writes",
                "model_label": "Qwen3-8B",
                "prompt_mode": "enumeration_index",
                "phase": "discovery",
                "materialized_generation_view": generation_view,
                "cohort_registry": str(cohort),
                "cohort_registry_sha256": hashlib.sha256(
                    cohort.read_bytes()
                ).hexdigest(),
                "causal_seed_membership_adapter": {"seed_aliasing": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = audit_completed_source_write_resume(
        source=source,
        model_label="Qwen3-8B",
        prompt_mode="enumeration_index",
        anchor_role="post_marker",
    )
    assert result["status"] == (
        "PASS_COMPLETED_SOURCE_WRITES_REUSED_WITHOUT_RECOMPUTATION"
    )
    assert result["sample_failure"] is False
    assert result["model_outputs_recomputed"] is False

    adapter_path = source / "v6_adapter_manifest.json"
    diagnostic_adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    diagnostic_adapter["phase"] = "diagnostic"
    adapter_path.write_text(
        json.dumps(diagnostic_adapter) + "\n", encoding="utf-8"
    )
    diagnostic_result = audit_completed_source_write_resume(
        source=source,
        model_label="Qwen3-8B",
        prompt_mode="enumeration_index",
        anchor_role="post_marker",
        expected_phase="diagnostic",
    )
    assert diagnostic_result["expected_phase"] == "diagnostic"
    diagnostic_adapter["phase"] = "discovery"
    adapter_path.write_text(
        json.dumps(diagnostic_adapter) + "\n", encoding="utf-8"
    )

    (shards / "trial_1.jsonl").unlink()
    with pytest.raises(ValueError, match="completed_shards"):
        audit_completed_source_write_resume(
            source=source,
            model_label="Qwen3-8B",
            prompt_mode="enumeration_index",
            anchor_role="post_marker",
        )


def test_incompatible_source_write_recovery_is_recoverable_and_fail_closed(
    tmp_path: Path,
) -> None:
    from scripts.recover_realistic_niah_v6_incompatible_source_writes import (
        quarantine_incompatible_source_writes,
    )

    model_root = tmp_path / "enumeration_index" / "Qwen3-8B"
    source = model_root / "causal" / "source_writes" / "post_marker"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "generations": "/adapter/pre_resolved.jsonl",
                "config_sha256": "old-config",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "v6_adapter_manifest.json").write_text(
        json.dumps(
            {
                "status": "INSTALLED",
                "run_status": "DISPATCHED",
                "materialized_generation_view": "/adapter/resolved.jsonl",
                "v6_config_sha256": "new-config",
                "cohort_registry_sha256": "registry",
                "seed_aliasing": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "shard.json").write_text("{}\n", encoding="utf-8")

    result = quarantine_incompatible_source_writes(
        model_root=model_root,
        source=source,
        label="pre_resolved_source_writes_20260828",
    )
    destination = Path(result["recoverable_destination"])
    assert result["status"] == "PASS_PRE_RESOLVED_INCOMPATIBLE_SHARDS_QUARANTINED"
    assert result["sample_failure"] is False
    assert result["scientific_artifacts_reused"] is False
    assert result["deletion_performed"] is False
    assert not source.exists()
    assert (destination / "shard.json").is_file()
    assert Path(result["audit_path"]).is_file()

    compatible = model_root / "compatible"
    compatible.mkdir()
    (compatible / "manifest.json").write_text(
        json.dumps({"generations": "/adapter/same.jsonl"}) + "\n",
        encoding="utf-8",
    )
    (compatible / "v6_adapter_manifest.json").write_text(
        json.dumps(
            {
                "status": "INSTALLED",
                "run_status": "DISPATCHED",
                "materialized_generation_view": "/adapter/same.jsonl",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="compatible"):
        quarantine_incompatible_source_writes(
            model_root=model_root,
            source=compatible,
            label="must_not_move",
        )


def test_empty_confirmation_source_write_recovery_requires_zero_shards(
    tmp_path: Path,
) -> None:
    from scripts.recover_realistic_niah_v6_empty_confirmation_source_writes import (
        FAILURE_MARKER,
        quarantine_empty_confirmation_source_writes,
    )

    model_root = tmp_path / "enumeration_index" / "Qwen3-8B"
    source = (
        model_root
        / "causal"
        / "targeted_retrieval"
        / "confirmation_formal"
        / "source_writes"
        / "post_marker"
    )
    source.mkdir(parents=True)
    (source / "manifest.json").write_text(
        json.dumps({"completed_shards": 0}) + "\n", encoding="utf-8"
    )
    (source / "v6_adapter_manifest.json").write_text(
        json.dumps(
            {
                "status": "INSTALLED",
                "run_status": "DISPATCHED",
                "command": "causal-source-writes",
                "phase": "confirmation",
                "causal_seed_membership_adapter": {"seed_aliasing": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    failure_log = model_root / "causal" / "targeted_retrieval" / "source.log"
    failure_log.parent.mkdir(parents=True, exist_ok=True)
    failure_log.write_text(FAILURE_MARKER + "\n", encoding="utf-8")

    result = quarantine_empty_confirmation_source_writes(
        model_root=model_root,
        source=source,
        failure_log=failure_log,
        label="confirmation_source_role_empty_attempt_20260830",
    )
    destination = Path(result["recoverable_destination"])
    assert result["status"] == (
        "PASS_ZERO_SHARD_CONFIRMATION_ROLE_DISPATCH_QUARANTINED"
    )
    assert result["preexisting_completed_shards"] == 0
    assert result["completed_model_trials_recomputed"] is False
    assert result["deletion_performed"] is False
    assert not source.exists()
    assert (destination / "manifest.json").is_file()

    guarded = model_root / "guarded"
    guarded.mkdir()
    (guarded / "manifest.json").write_text(
        json.dumps({"completed_shards": 1}) + "\n", encoding="utf-8"
    )
    (guarded / "v6_adapter_manifest.json").write_text(
        (destination / "v6_adapter_manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="completed source-write shard"):
        quarantine_empty_confirmation_source_writes(
            model_root=model_root,
            source=guarded,
            failure_log=failure_log,
            label="must_not_move_completed",
        )


def test_confirmation_representation_uses_shared_native_aligned_path() -> None:
    enumeration = (
        ROOT / "scripts" / "supervise_realistic_niah_v6_enumeration.sh"
    ).read_text(encoding="utf-8")
    foundation = (
        ROOT / "scripts" / "supervise_realistic_niah_v6_confirmation_foundation.sh"
    ).read_text(encoding="utf-8")
    final_audit = (
        ROOT / "scripts" / "queue_realistic_niah_v6_final_audit.sh"
    ).read_text(encoding="utf-8")
    confirmation_queue = (
        ROOT / "scripts" / "queue_realistic_niah_v6_confirmation.sh"
    ).read_text(encoding="utf-8")
    report_tail = (
        ROOT / "scripts" / "supervise_realistic_niah_v6_report_tail_confirmation.sh"
    ).read_text(encoding="utf-8")
    assert "discovery_representation_formal_resolved" in enumeration
    assert '--output "$RUN_ROOT/representation/formal"' in enumeration
    assert enumeration.count('--output "$RUN_ROOT/representation/formal"') == 1
    assert "CONFIRMATION_FORMAL_REPRESENTATION" not in enumeration
    assert "CONFIRMATION_ALL_REPRESENTATION" not in enumeration
    assert "\n  run_logged representation_formal_resolved " not in enumeration
    assert "representation_all_sample" not in enumeration
    assert enumeration.count('--output "$FORMAL_CAPTURE"') == 2
    assert enumeration.count('--output "$CONFIRMATION_FORMAL_CAPTURE"') == 2
    assert "capture/confirmation_formal/v6_adapter_manifest.json" in foundation
    assert "representation/confirmation_formal/v6_adapter_manifest.json" not in foundation
    assert "DEFERRED_UNTIL_ALL_FOUR_ORIGINAL_CAPTURES_EXIST" in foundation
    assert "analyze_realistic_niah_v6_native_aligned_representation.py" in final_audit
    assert 'OPENBLAS_NUM_THREADS="$CPU_THREADS"' in final_audit
    assert "realistic_niah_v6_native_analysis_alignment_v1.json" in final_audit
    assert "reuse_or_run" in confirmation_queue
    assert "V6_REPLACEMENT_POLICY" in enumeration
    assert "V6_REPLACEMENT_POLICY" in foundation
    assert "V6_REPLACEMENT_POLICY" in confirmation_queue
    assert "V6_REPLACEMENT_POLICY" in report_tail
    assert "V6_COHERENT_NATIVE_LOOP_POLICY" in confirmation_queue
    assert "V6_COHERENT_NATIVE_LOOP_POLICY" in report_tail
    assert (
        "NATIVE_POLICY=${V6_COHERENT_NATIVE_LOOP_POLICY:-$ROOT/configs/"
        "realistic_niah_v6_coherent_native_loop_replacement_policy.json}"
        in report_tail
    )
    assert 'start|native-loop|walkthrough' in report_tail
    assert "PASS_REUSE_COMPLETED_NATURAL_CONFIRMATION" in report_tail
    assert '"completed_model_trials_recomputed": False' in report_tail
    assert '"seed_selection_changed": False' in report_tail
    assert '"frozen_k_changed": False' in report_tail
    assert "V6_CONFIRMATION_FOUNDATION_RESUME_FROM" in foundation
    assert "V6_CONFIRMATION_FOUNDATION_RESUME_FROM" in confirmation_queue
    assert 'start|coherent-broad' in foundation
    assert '--replacement-policy "$REPLACEMENT_POLICY"' in foundation
    assert '--replacement-policy "$REPLACEMENT_POLICY"' in report_tail
    assert (
        '--replacement-policy "$ROOT/configs/realistic_niah_v6_replacement_policy.json"'
        not in report_tail
    )
    assert '"$freeze_root/confirmation-foundation.COMPLETE"' in confirmation_queue
    assert "causal/targeted_retrieval/confirmation_formal/confirmation.COMPLETE" in confirmation_queue
    assert "count_stream/confirmation_formal/confirmation.COMPLETE" in confirmation_queue
    assert "causal/specialized/confirmation_formal/confirmation.COMPLETE" in confirmation_queue
    assert "causal/report_tail/confirmation_formal/confirmation.COMPLETE" in confirmation_queue
    assert FRAME_EVIDENCE["layerwise_representation"] == (
        ("representation/formal/v6_adapter_manifest.json",),
        ("representation/native_aligned/cell_manifest.json",),
    )
    assert foundation.count("validate-confirmation-freeze") == 2


def test_confirmation_trace_patch_has_outcome_blind_resume_adapter() -> None:
    wrapper = (
        ROOT / "scripts" / "run_realistic_niah_v6_count_stream.py"
    ).read_text(encoding="utf-8")
    supervisor = (
        ROOT / "scripts" / "supervise_realistic_niah_v6_count_stream_confirmation.sh"
    ).read_text(encoding="utf-8")
    queue = (
        ROOT / "scripts" / "queue_realistic_niah_v6_confirmation.sh"
    ).read_text(encoding="utf-8")

    assert "def _v6_command_plan_trace_patch_confirmation" in wrapper
    assert 'raw[0] == "plan-trace-patch"' in wrapper
    assert 'seed_role == "confirmation"' in wrapper
    assert '"confirmation_outcomes_used_for_selection": False' in wrapper
    assert '"panel_membership_identity": "v6_analysis_slot_seed"' in wrapper
    assert '"statistical_identity": "true_source_seed"' in wrapper
    assert "reusable_broad_confirmation" in supervisor
    assert "trials_manifest_sha256" in supervisor
    assert "PASS_CONFIRMATION_TRACE_PLAN_ROLE_ADAPTER_RECOVERY" in supervisor
    assert "completed_trace_patch_shards_before_recovery\": 0" in supervisor
    assert "PASS_IMPLICIT_DEFAULT_RESUME_FLAG_RECOVERY" in supervisor
    assert "--no-resume" not in supervisor
    assert "--output \"$TRACE_PATCH\"" in supervisor
    assert "REUSE validated $MODEL $mode discovery freeze" in queue
    assert "validation_before_queue_resume.json" in queue
