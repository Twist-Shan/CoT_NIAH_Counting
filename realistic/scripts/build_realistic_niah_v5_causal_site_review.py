#!/usr/bin/env python3
"""Compile and audit grammar-aware native-thinking causal token sites.

This is a review compiler, not an intervention runner.  It consumes frozen
generation JSONL files, re-runs the current trace parser, requires exact output
re-tokenization, and emits event/transition registries plus a human-readable
HTML review surface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tokenizers import Tokenizer

from realistic_niah_v5.causal_sites import (
    causal_site_rules,
    compile_causal_site_plan,
    flatten_anchor_rows,
    flatten_event_rows,
    flatten_transition_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "realistic_niah_v5_causal_site_review.json"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    empty_fields: Sequence[str] = (),
) -> None:
    preferred = [
        "level",
        "kind",
        "model_label",
        "request_id",
        "seed",
        "split",
        "sequence_source",
        "trace_category",
        "causal_cohort",
        "grammar_signature",
        "grammar_class",
        "association",
        "surface_order",
        "marker_kind",
        "transition_kind",
        "reason",
        "count",
    ]
    discovered = {str(key) for row in rows for key in row}
    fields = [key for key in preferred if key in discovered]
    fields.extend(sorted(discovered - set(fields)))
    if not fields:
        fields = list(empty_fields)
    buffer = io.StringIO(newline="")
    if fields:
        writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def _trajectory_coverage(plans: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "model_label",
        "sequence_source",
        "marker_kind",
        "trace_category",
        "causal_cohort",
        "grammar_signature",
    )
    grouped: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    for plan in plans:
        key = tuple(plan.get(field) for field in fields)
        grouped[key]["count"] += 1
        grouped[key]["primary_complete"] += int(
            bool(plan.get("primary_full_chain_site_complete"))
        )
        grouped[key]["with_site_exclusion"] += int(
            bool(plan.get("plan_exclusion_reasons"))
        )
    output = []
    for key, counts in sorted(grouped.items(), key=lambda value: tuple(map(str, value[0]))):
        output.append(
            {
                "level": "trajectory",
                **dict(zip(fields, key)),
                "count": counts["count"],
                "eligible_count": counts["primary_complete"],
                "with_exclusion_count": counts["with_site_exclusion"],
            }
        )
    return output


def _event_coverage(event_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "model_label",
        "causal_cohort",
        "grammar_class",
        "association",
        "surface_order",
        "marker_semantics",
    )
    grouped: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    for event in event_rows:
        key = tuple(event.get(field) for field in fields)
        grouped[key]["count"] += 1
        grouped[key]["retrieval"] += int(bool(event.get("retrieval_eligible")))
        grouped[key]["marker"] += int(bool(event.get("marker_control_eligible")))
        grouped[key]["format_shell"] += int(
            bool(event.get("format_shell_control_eligible"))
        )
        grouped[key]["invariant_surface"] += int(
            bool(event.get("invariant_marker_surface_control_eligible"))
        )
        grouped[key]["progress"] += int(bool(event.get("progress_commit_eligible")))
        grouped[key]["city_zero_spill"] += int(
            int(event.get("city_left_spill_chars") or 0) == 0
            and int(event.get("city_right_spill_chars") or 0) == 0
        )
        has_rank = event.get("rank_output_token_start") not in (None, "")
        grouped[key]["rank_span_count"] += int(has_rank)
        grouped[key]["rank_zero_spill"] += int(
            has_rank
            and int(event.get("rank_left_spill_chars") or 0) == 0
            and int(event.get("rank_right_spill_chars") or 0) == 0
        )
    output = []
    for key, counts in sorted(grouped.items(), key=lambda value: tuple(map(str, value[0]))):
        output.append(
            {
                "level": "event",
                **dict(zip(fields, key)),
                "count": counts["count"],
                "retrieval_eligible_count": counts["retrieval"],
                "marker_control_eligible_count": counts["marker"],
                "format_shell_control_eligible_count": counts["format_shell"],
                "invariant_surface_control_eligible_count": counts[
                    "invariant_surface"
                ],
                "progress_commit_eligible_count": counts["progress"],
                "city_zero_spill_count": counts["city_zero_spill"],
                "rank_span_count": counts["rank_span_count"],
                "rank_zero_spill_count": counts["rank_zero_spill"],
            }
        )
    return output


def _transition_coverage(
    transition_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "model_label",
        "causal_cohort",
        "transition_kind",
        "grammar_pair",
    )
    grouped: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    for transition in transition_rows:
        key = tuple(transition.get(field) for field in fields)
        grouped[key]["count"] += 1
        grouped[key]["local"] += int(
            bool(transition.get("local_or_observed_stop_eligible"))
        )
        grouped[key]["primary"] += int(bool(transition.get("primary_eligible")))
        grouped[key]["tokens"] += int(transition.get("continuation_token_count") or 0)
        grouped[key]["max_tokens"] = max(
            grouped[key]["max_tokens"],
            int(transition.get("continuation_token_count") or 0),
        )
        grouped[key]["interstitial"] += int(
            transition.get("interstitial_token_count") or 0
        )
        grouped[key]["max_interstitial"] = max(
            grouped[key]["max_interstitial"],
            int(transition.get("interstitial_token_count") or 0),
        )
    output = []
    for key, counts in sorted(grouped.items(), key=lambda value: tuple(map(str, value[0]))):
        count = counts["count"]
        output.append(
            {
                "level": "transition",
                **dict(zip(fields, key)),
                "count": count,
                "local_eligible_count": counts["local"],
                "primary_eligible_count": counts["primary"],
                "mean_continuation_tokens": round(counts["tokens"] / count, 3),
                "max_continuation_tokens": counts["max_tokens"],
                "mean_interstitial_tokens": round(
                    counts["interstitial"] / count, 3
                ),
                "max_interstitial_tokens": counts["max_interstitial"],
            }
        )
    return output


def _exclusion_rows(
    plans: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for failure in failures:
        rows.append({"kind": "compile_failure", **failure})
    for plan in plans:
        base = {
            "model_label": plan.get("model_label"),
            "request_id": plan.get("request_id"),
            "sequence_source": plan.get("sequence_source"),
            "trace_category": plan.get("trace_category"),
            "causal_cohort": plan.get("causal_cohort"),
            "grammar_signature": plan.get("grammar_signature"),
        }
        if plan.get("causal_cohort") != "primary_rank_resolved_full_chain":
            rows.append(
                {
                    "kind": "cohort_scope",
                    **base,
                    "reason": f"not_primary:{plan.get('causal_cohort')}",
                }
            )
        for reason in plan.get("plan_exclusion_reasons", []):
            rows.append({"kind": "site_exclusion", **base, "reason": reason})
    return rows


def _audit(
    *,
    config: Mapping[str, Any],
    rows_seen: Mapping[str, int],
    plans: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    plans_by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for plan in plans:
        plans_by_model[str(plan.get("model_label"))].append(plan)
    gates: list[dict[str, Any]] = []

    def gate(name: str, passed: bool, detail: Any) -> None:
        gates.append({"gate": name, "passed": bool(passed), "detail": detail})

    expected = {
        model: int(spec["expected_rows"])
        for model, spec in config["models"].items()
    }
    gate("expected_input_rows", dict(rows_seen) == expected, {"expected": expected, "seen": dict(rows_seen)})
    compiled = Counter(str(plan.get("model_label")) for plan in plans)
    gate("all_rows_compiled", not failures and dict(compiled) == expected, {"compiled": dict(compiled), "failure_count": len(failures)})
    gate("exact_output_retokenization", all(bool(plan.get("token_reencode_exact")) for plan in plans), {"checked": len(plans)})
    gate("prompt_source_city_registry_exact", all(bool(plan.get("prompt_source_registry_exact")) for plan in plans), {"checked": len(plans)})

    primary = [
        plan
        for plan in plans
        if plan.get("causal_cohort") == "primary_rank_resolved_full_chain"
    ]
    primary_by_model = Counter(str(plan.get("model_label")) for plan in primary)
    gate("primary_cohort_nonempty_per_model", all(primary_by_model[model] > 0 for model in expected), dict(primary_by_model))
    incomplete_primary = [
        str(plan.get("request_id"))
        for plan in primary
        if not plan.get("primary_full_chain_site_complete")
    ]
    gate("all_primary_plans_site_complete", not incomplete_primary, {"primary_count": len(primary), "incomplete": incomplete_primary[:20]})

    bad_retrieval: list[str] = []
    bad_rank_core_spill: list[str] = []
    bad_transition: list[str] = []
    bad_anchor_registry: list[str] = []
    bad_terminal: list[str] = []
    scope_violations: list[str] = []
    for plan in plans:
        cohort = str(plan.get("causal_cohort"))
        for event in plan.get("events", []):
            rank = event["sites"].get("rank_evidence_core_span")
            if rank is not None and int(rank.get("right_spill_chars", 0)) > 0:
                bad_rank_core_spill.append(
                    f"{plan.get('request_id')}#{event.get('occurrence')}"
                )
            if event["eligibility"]["retrieval"]:
                query = event["sites"]["retrieve_query_state"].get("output_token_index")
                target = event["sites"]["city_target_span"].get("output_token_start")
                if query is None or target is None or int(query) >= int(target):
                    bad_retrieval.append(f"{plan.get('request_id')}#{event.get('occurrence')}")
        for transition in plan.get("transitions", []):
            if transition.get("local_transition_eligible"):
                query = transition.get("query_output_token_index")
                start = transition.get("full_continuation_output_token_start")
                end = transition.get("full_continuation_output_token_end")
                city_end = transition.get("next_city_output_token_end")
                if None in (query, start, end, city_end) or not int(query) < int(start) < int(end) or int(end) != int(city_end):
                    bad_transition.append(
                        f"{plan.get('request_id')}#{transition.get('from_occurrence')}"
                    )
                anchors = transition.get("anchors", [])
                city_start = transition.get("next_city_output_token_start")
                queries = [anchor.get("output_token_index") for anchor in anchors]
                roles = {
                    role
                    for anchor in anchors
                    for role in anchor.get("anchor_roles", [])
                }
                if (
                    not anchors
                    or len(queries) != len(set(queries))
                    or city_start is None
                    or any(value is None or int(value) >= int(city_start) for value in queries)
                    or "p0_item_end" not in roles
                    or "city_pre_d1" not in roles
                    or any(
                        anchor.get("primary_anchor_eligible")
                        and not anchor.get("event_specific")
                        for anchor in anchors
                    )
                ):
                    bad_anchor_registry.append(
                        f"{plan.get('request_id')}#{transition.get('from_occurrence')}"
                    )
        terminal = plan.get("terminal_transition") or {}
        if terminal.get("observed_stop_eligible"):
            query = terminal.get("query_output_token_index")
            start = terminal.get("full_continuation_output_token_start")
            end = terminal.get("full_continuation_output_token_end")
            if None in (query, start, end) or not int(query) < int(start) < int(end):
                bad_terminal.append(str(plan.get("request_id")))
        if cohort == "occurrence_retrieval_only_duplicates":
            if any(
                event["eligibility"]["progress_commit"]
                or event["eligibility"]["marker_control"]
                or event["eligibility"]["format_shell_control"]
                or event["eligibility"]["invariant_marker_surface_control"]
                for event in plan.get("events", [])
            ) or any(
                transition.get("local_transition_eligible")
                for transition in plan.get("transitions", [])
            ) or terminal.get("observed_stop_eligible"):
                scope_violations.append(str(plan.get("request_id")))
        if cohort == "secondary_local_partial_unique" and terminal.get(
            "observed_stop_eligible"
        ):
            scope_violations.append(str(plan.get("request_id")))
        if cohort in {
            "secondary_evidence_sequence_exploratory",
            "secondary_evidence_sequence_partial_exploratory",
        }:
            full_sequence = cohort == "secondary_evidence_sequence_exploratory"
            if any(
                event.get("rank_basis") != "compiler_occurrence_index_only"
                or event["sites"].get("rank_evidence_core_span") is not None
                or not event["eligibility"].get("retrieval", False)
                or not event["eligibility"].get("progress_commit", False)
                or event["eligibility"].get("marker_control", False)
                or event["eligibility"].get("format_shell_control", False)
                or event["eligibility"].get(
                    "invariant_marker_surface_control", False
                )
                for event in plan.get("events", [])
            ) or any(
                not transition.get("local_transition_eligible")
                for transition in plan.get("transitions", [])
            ) or bool(terminal.get("observed_stop_eligible")) != full_sequence:
                scope_violations.append(str(plan.get("request_id")))
        if cohort == "audit_only_unresolved":
            if any(
                any(
                    event["eligibility"].get(role, False)
                    for role in (
                        "retrieval",
                        "marker_control",
                        "format_shell_control",
                        "invariant_marker_surface_control",
                        "progress_commit",
                    )
                )
                for event in plan.get("events", [])
            ) or any(
                transition.get("local_transition_eligible")
                for transition in plan.get("transitions", [])
            ) or terminal.get("observed_stop_eligible"):
                scope_violations.append(str(plan.get("request_id")))
    gate("all_retrieval_queries_precede_city", not bad_retrieval, bad_retrieval[:20])
    gate(
        "all_rank_evidence_cores_have_zero_right_spill",
        not bad_rank_core_spill,
        bad_rank_core_spill[:20],
    )
    gate("all_continue_targets_follow_commit", not bad_transition, bad_transition[:20])
    gate(
        "all_local_transitions_have_deduplicated_p0_and_city_anchors",
        not bad_anchor_registry,
        bad_anchor_registry[:20],
    )
    gate("all_stop_targets_follow_final_commit", not bad_terminal, bad_terminal[:20])
    gate(
        "cohort_estimand_scope_enforced",
        not scope_violations,
        scope_violations[:20],
    )
    missing_answer = [
        str(plan.get("request_id"))
        for plan in primary
        if not plan.get("answer_query_v3_span")
    ]
    gate("all_primary_have_answer_query_v3", not missing_answer, missing_answer[:20])
    unresolved = [
        str(plan.get("request_id"))
        for plan in plans
        if plan.get("causal_cohort") == "audit_only_unresolved"
    ]
    gate("no_unresolved_cohort", not unresolved, unresolved[:20])
    return {
        "schema_version": "realistic_niah_v5_causal_site_audit_v1",
        "status": "pass" if all(row["passed"] for row in gates) else "fail",
        "trajectory_count": len(plans),
        "primary_trajectory_count": len(primary),
        "compile_failure_count": len(failures),
        "gates": gates,
    }


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _short(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    head = "".join(f"<th>{_escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _select_samples(
    plans: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    policy = config.get("review_sampling", {})
    per_group = int(policy.get("examples_per_model_grammar_cohort", 2))
    max_exclusions = int(policy.get("max_exclusion_examples", 100))
    selected: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()
    counts: Counter[tuple[str, str, str]] = Counter()
    ordered = sorted(plans, key=lambda plan: (str(plan.get("model_label")), str(plan.get("request_id"))))
    for plan in ordered:
        key = (
            str(plan.get("model_label")),
            str(plan.get("causal_cohort")),
            str(plan.get("grammar_signature")),
        )
        if counts[key] < per_group:
            selected.append(plan)
            selected_ids.add(str(plan.get("request_id")))
            counts[key] += 1
    extras = 0
    for plan in ordered:
        request_id = str(plan.get("request_id"))
        if request_id in selected_ids or not plan.get("plan_exclusion_reasons"):
            continue
        if extras >= max_exclusions:
            break
        selected.append(plan)
        selected_ids.add(request_id)
        extras += 1
    return selected


def _sample_html(plan: Mapping[str, Any]) -> str:
    event_rows = []
    for event in plan.get("events", []):
        sites = event["sites"]
        source = event.get("prompt_source_record") or {}
        query = sites["retrieve_query_state"]
        city = sites["city_target_span"]
        rank = sites.get("rank_evidence_core_span") or {}
        rank_surface = sites.get("rank_evidence_surface_span") or {}
        rank_shell = sites.get("rank_visible_format_shell_tokens") or {}
        commit = sites["post_update_commit_state"]
        event_rows.append(
            (
                event.get("occurrence"),
                event.get("rank"),
                event.get("rank_basis"),
                event.get("city"),
                event.get("grammar_class"),
                event.get("marker_semantics"),
                f"prompt[{source.get('prompt_token_start')}:{source.get('prompt_token_end')}]",
                f"q={query.get('output_token_index')} id={query.get('token_id')} text={query.get('token_text')!r}",
                f"city[{city.get('output_token_start')}:{city.get('output_token_end')}] ids={city.get('token_ids')}",
                (
                    "—"
                    if not rank
                    else f"core[{rank.get('output_token_start')}:{rank.get('output_token_end')}] ids={rank.get('token_ids')} text={rank.get('token_text')!r}; surface[{rank_surface.get('output_token_start')}:{rank_surface.get('output_token_end')}] text={rank_surface.get('token_text')!r}; shell idx={rank_shell.get('output_token_indices')} ids={rank_shell.get('token_ids')} text={rank_shell.get('token_text')!r}"
                ),
                f"commit={commit.get('output_token_index')} id={commit.get('token_id')} text={commit.get('token_text')!r}",
                _short(sites["semantic_item_span"].get("char_text")),
                ", ".join(event.get("exclusion_reasons", [])) or "—",
            )
        )
    transition_rows = []
    for transition in [*plan.get("transitions", []), plan.get("terminal_transition")]:
        if not transition:
            continue
        token_ids = transition.get("full_continuation_token_ids", [])
        is_terminal = transition.get("transition_kind") == "stop_to_answer_query_v3"
        target_start = transition.get(
            "answer_query_output_token_start"
            if is_terminal
            else "next_city_output_token_start"
        )
        target_end = transition.get(
            "answer_query_output_token_end"
            if is_terminal
            else "next_city_output_token_end"
        )
        target_ids = transition.get(
            "answer_query_token_ids" if is_terminal else "next_city_token_ids", []
        )
        target_text = transition.get(
            "answer_query_token_text" if is_terminal else "next_city_token_text", ""
        )
        transition_rows.append(
            (
                transition.get("transition_kind"),
                transition.get("from_occurrence"),
                transition.get("to_occurrence", "—"),
                transition.get("query_output_token_index"),
                f"[{transition.get('full_continuation_output_token_start')}:{transition.get('full_continuation_output_token_end')}]",
                len(token_ids),
                f"[{target_start}:{target_end}] ids={target_ids} text={target_text!r}",
                max(0, len(token_ids) - len(target_ids)),
                _short(transition.get("full_continuation_token_text")),
                transition.get(
                    "local_transition_eligible",
                    transition.get("observed_stop_eligible"),
                ),
                transition.get("primary_transition_eligible", transition.get("primary_terminal_eligible")),
                ", ".join(transition.get("exclusion_reasons", [])) or "—",
            )
        )
    anchor_rows = []
    for transition in plan.get("transitions", []):
        resolved = {
            role: anchor
            for anchor in transition.get("anchors", [])
            for role in anchor.get("anchor_roles", [])
        }
        for candidate in transition.get("anchor_candidates", []):
            role = str(candidate.get("anchor_role"))
            anchor = resolved.get(role)
            anchor_rows.append(
                (
                    f"{transition.get('from_occurrence')}→{transition.get('to_occurrence')}",
                    transition.get("target_city"),
                    role,
                    " | ".join(anchor.get("anchor_roles", [])) if anchor else "—",
                    candidate.get("timing_stage"),
                    candidate.get("status"),
                    candidate.get("output_token_index"),
                    repr(candidate.get("token_text", "")),
                    candidate.get("event_specific"),
                    False if anchor is None else anchor.get("primary_anchor_eligible"),
                    candidate.get("not_applicable_reason") or "—",
                    (
                        f"[{transition.get('next_city_output_token_start')}:"
                        f"{transition.get('next_city_output_token_end')}] "
                        f"{transition.get('next_city_token_text')!r}"
                    ),
                )
            )
    title = f"{plan.get('model_label')} · {plan.get('request_id')}"
    meta = (
        f"cohort={plan.get('causal_cohort')} · source={plan.get('sequence_source')} · "
        f"trace={plan.get('trace_category')} · grammar={plan.get('grammar_signature')} · "
        f"allowed estimands={', '.join(plan.get('allowed_estimands', [])) or 'none'}"
    )
    return (
        f"<details><summary>{_escape(title)}</summary><p class='meta'>{_escape(meta)}</p>"
        + _table(
            ["occ", "rank", "rank basis", "city", "grammar", "marker semantics", "prompt source", "retrieve query", "city target", "rank core / parser surface", "commit", "semantic item", "exclusions"],
            event_rows,
        )
        + _table(
            ["transition", "from", "to", "query", "full target span", "full tokens", "semantic target only", "interstitial tokens", "full target text", "cohort eligible", "primary", "exclusions"],
            transition_rows,
        )
        + _table(
            ["transition", "target city", "semantic role", "deduplicated aliases", "stage", "status", "query token", "query text", "event-specific", "primary", "N/A reason", "fixed scored city span"],
            anchor_rows,
        )
        + "</details>"
    )


def _review_html(
    *,
    config: Mapping[str, Any],
    plans: Sequence[Mapping[str, Any]],
    coverage: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    exclusions: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
) -> str:
    cohort_counts = Counter(
        (str(plan.get("model_label")), str(plan.get("causal_cohort"))) for plan in plans
    )
    cohort_table = _table(
        ["model", "causal cohort", "trajectories"],
        ((model, cohort, count) for (model, cohort), count in sorted(cohort_counts.items())),
    )
    gates = _table(
        ["gate", "pass", "detail"],
        (
            (row["gate"], "PASS" if row["passed"] else "FAIL", _short(json.dumps(row["detail"], ensure_ascii=False), 360))
            for row in audit["gates"]
        ),
    )
    trajectory_coverage = [row for row in coverage if row.get("level") == "trajectory"]
    event_coverage = [row for row in coverage if row.get("level") == "event"]
    transition_coverage = [
        row for row in coverage if row.get("level") == "transition"
    ]
    coverage_table = _table(
        ["model", "source", "marker", "trace", "cohort", "grammar", "n", "primary complete", "site exclusions"],
        (
            (
                row.get("model_label"), row.get("sequence_source"), row.get("marker_kind"),
                row.get("trace_category"), row.get("causal_cohort"), row.get("grammar_signature"),
                row.get("count"), row.get("eligible_count"), row.get("with_exclusion_count"),
            )
            for row in trajectory_coverage
        ),
    )
    exclusion_counts = Counter((str(row.get("model_label")), str(row.get("kind")), str(row.get("reason"))) for row in exclusions)
    exclusion_table = _table(
        ["model", "kind", "reason", "n"],
        ((model, kind, reason, count) for (model, kind, reason), count in sorted(exclusion_counts.items())),
    )
    role_rows = config.get("semantic_site_policy", {}).items()
    role_table = _table(["semantic role", "frozen rule"], role_rows)
    cohort_rows = config.get("cohort_policy", {}).items()
    policy_table = _table(["cohort", "allowed estimand"], cohort_rows)
    samples = "".join(_sample_html(plan) for plan in _select_samples(plans, config))
    event_table = _table(
        ["model", "cohort", "grammar", "association", "surface order", "marker semantics", "events", "retrieval", "rank-core control", "format-shell control", "invariant-surface control", "progress commit", "city zero-spill", "rank zero-spill / spans"],
        (
            (
                row.get("model_label"), row.get("causal_cohort"), row.get("grammar_class"),
                row.get("association"), row.get("surface_order"), row.get("marker_semantics"),
                row.get("count"), row.get("retrieval_eligible_count"),
                row.get("marker_control_eligible_count"),
                row.get("format_shell_control_eligible_count"),
                row.get("invariant_surface_control_eligible_count"),
                row.get("progress_commit_eligible_count"),
                row.get("city_zero_spill_count"),
                f"{row.get('rank_zero_spill_count')} / {row.get('rank_span_count')}",
            )
            for row in event_coverage
        ),
    )
    transition_table = _table(
        ["model", "cohort", "transition", "grammar pair", "n", "local eligible", "primary eligible", "mean full tokens", "max full tokens", "mean interstitial", "max interstitial"],
        (
            (
                row.get("model_label"), row.get("causal_cohort"), row.get("transition_kind"),
                row.get("grammar_pair"), row.get("count"), row.get("local_eligible_count"),
                row.get("primary_eligible_count"), row.get("mean_continuation_tokens"),
                row.get("max_continuation_tokens"), row.get("mean_interstitial_tokens"),
                row.get("max_interstitial_tokens"),
            )
            for row in sorted(
                transition_coverage,
                key=lambda value: int(value.get("max_continuation_tokens") or 0),
                reverse=True,
            )
        ),
    )
    longest_rows = sorted(
        transition_rows,
        key=lambda value: int(value.get("interstitial_token_count") or 0),
        reverse=True,
    )[:30]
    longest_table = _table(
        ["model", "request", "cohort", "transition", "from", "grammar pair", "full tokens", "semantic target tokens", "interstitial", "semantic target"],
        (
            (
                row.get("model_label"), row.get("request_id"), row.get("causal_cohort"),
                row.get("transition_kind"), row.get("from_occurrence"), row.get("grammar_pair"),
                row.get("continuation_token_count"), row.get("target_only_token_count"),
                row.get("interstitial_token_count"), _short(row.get("target_only_token_text"), 80),
            )
            for row in longest_rows
        ),
    )
    status = str(audit["status"]).upper()
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Native-thinking causal site review</title>
<style>
:root{{--ink:#17202a;--muted:#5c6773;--line:#d8dee6;--panel:#f6f8fa;--accent:#0b6e69;--warn:#9a6700}}
*{{box-sizing:border-box}} body{{margin:0;font:14px/1.48 Inter,Segoe UI,Arial,sans-serif;color:var(--ink);background:#fff}}
main{{max-width:1520px;margin:0 auto;padding:36px 28px 80px}} h1{{font-size:30px;margin:0 0 8px}} h2{{margin-top:34px;border-bottom:1px solid var(--line);padding-bottom:8px}}
.lead,.meta{{color:var(--muted)}} .badge{{display:inline-block;padding:4px 10px;border-radius:999px;background:#dafbe1;color:#116329;font-weight:700}} .badge.fail{{background:#ffebe9;color:#cf222e}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 16px;min-width:180px}} .card strong{{display:block;font-size:24px}}
.table-wrap{{overflow:auto;margin:12px 0 22px}} table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top;white-space:nowrap}} th{{background:var(--panel);position:sticky;top:0}} td:nth-last-child(-n+2){{white-space:normal;max-width:420px}}
details{{border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:12px 0}} summary{{cursor:pointer;font-weight:650}} code{{background:var(--panel);padding:1px 4px;border-radius:4px}}
</style></head><body><main>
<h1>Native-thinking causal site review</h1>
<p class="lead">Frozen parser/cohort/token-site decisions. This artifact does not run or report causal interventions.</p>
<span class="badge {'fail' if status == 'FAIL' else ''}">AUDIT {status}</span>
<div class="cards"><div class="card"><strong>{len(plans)}</strong>trajectories</div><div class="card"><strong>{sum(len(plan.get('events', [])) for plan in plans)}</strong>events</div><div class="card"><strong>{audit.get('primary_trajectory_count')}</strong>primary trajectories</div><div class="card"><strong>{len(exclusions)}</strong>scope/exclusion rows</div></div>
<h2>Decision hierarchy</h2><p><code>surface grammar</code> selects token roles; <code>trajectory cohort</code> selects the estimand. Cohort membership never moves a site.</p>
{role_table}{policy_table}
<h2>Audit gates</h2>{gates}
<h2>Cohort totals</h2>{cohort_table}
<h2>Trajectory grammar coverage</h2>{coverage_table}
<h2>Event-level grammar and eligibility</h2>{event_table}
<h2>Continuation diagnostics</h2><p class="lead">The full continuation is the teacher-forced path from the commit through the semantic target. The city or <code>Total: </code> target-only span is retained separately; interstitial length is reported rather than silently thresholded.</p>{transition_table}
<h2>Longest interstitial paths</h2>{longest_table}
<h2>Scope restrictions and site exclusions</h2>{exclusion_table}
<h2>Stratified examples</h2><p class="lead">Deterministic samples: up to {config.get('review_sampling', {}).get('examples_per_model_grammar_cohort', 2)} per model × cohort × grammar signature, plus technical-exclusion examples.</p>{samples}
</main></body></html>"""


def build(config_path: Path) -> tuple[Path, dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plans: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    rows_seen: dict[str, int] = {}
    input_paths: dict[str, Path] = {}
    tokenizer_paths: dict[str, Path] = {}

    for model, spec in config["models"].items():
        input_path = _resolve(spec["input_jsonl"])
        tokenizer_path = _resolve(spec["tokenizer_json"])
        input_paths[model] = input_path
        tokenizer_paths[model] = tokenizer_path
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        seen = 0
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                seen += 1
                row: dict[str, Any] = {}
                try:
                    row = json.loads(line)
                    if str(row.get("model_label")) != model:
                        raise ValueError(
                            f"row model_label={row.get('model_label')!r}, config model={model!r}"
                        )
                    plans.append(compile_causal_site_plan(row, tokenizer))
                except Exception as error:  # audit artifact must retain every failure
                    failures.append(
                        {
                            "model_label": model,
                            "request_id": str(row.get("request_id", f"line:{line_number}")),
                            "line_number": line_number,
                            "reason": f"{type(error).__name__}: {error}",
                        }
                    )
        rows_seen[model] = seen

    event_rows = [row for plan in plans for row in flatten_event_rows(plan)]
    transition_rows = [row for plan in plans for row in flatten_transition_rows(plan)]
    anchor_rows = [row for plan in plans for row in flatten_anchor_rows(plan)]
    coverage = [
        *_trajectory_coverage(plans),
        *_event_coverage(event_rows),
        *_transition_coverage(transition_rows),
    ]
    exclusions = _exclusion_rows(plans, failures)
    audit = _audit(
        config=config,
        rows_seen=rows_seen,
        plans=plans,
        failures=failures,
    )

    outputs = {
        "trajectory_registry": output_dir / "trajectory_registry.jsonl",
        "event_registry": output_dir / "event_registry.csv",
        "transition_registry": output_dir / "transition_registry.csv",
        "anchor_registry": output_dir / "anchor_registry.csv",
        "coverage": output_dir / "coverage.csv",
        "exclusions": output_dir / "exclusions.csv",
        "compile_failures": output_dir / "compile_failures.csv",
        "audit": output_dir / "audit.json",
        "review": output_dir / "causal_site_review.html",
    }
    _write_jsonl(outputs["trajectory_registry"], plans)
    _write_csv(outputs["event_registry"], event_rows)
    _write_csv(outputs["transition_registry"], transition_rows)
    _write_csv(outputs["anchor_registry"], anchor_rows)
    _write_csv(outputs["coverage"], coverage)
    _write_csv(outputs["exclusions"], exclusions)
    _write_csv(
        outputs["compile_failures"],
        failures,
        empty_fields=("model_label", "request_id", "line_number", "reason"),
    )
    _write_json(outputs["audit"], audit)
    _atomic_text(
        outputs["review"],
        _review_html(
            config=config,
            plans=plans,
            coverage=coverage,
            transition_rows=transition_rows,
            exclusions=exclusions,
            audit=audit,
        ),
    )

    provenance_paths = {
        "config": config_path,
        "causal_site_compiler": REPO_ROOT / "src" / "realistic_niah_v5" / "causal_sites.py",
        "trace_parser": REPO_ROOT / "src" / "realistic_niah_v5" / "hybrid_trace_parser.py",
        "parsing_adapter": REPO_ROOT / "src" / "realistic_niah_v5" / "parsing.py",
        **{f"input:{model}": path for model, path in input_paths.items()},
        **{f"tokenizer:{model}": path for model, path in tokenizer_paths.items()},
    }
    manifest = {
        "schema_version": "realistic_niah_v5_causal_site_review_manifest_v1",
        "audit_status": audit["status"],
        "rules": causal_site_rules(),
        "row_counts": rows_seen,
        "plan_count": len(plans),
        "event_count": len(event_rows),
        "transition_count": len(transition_rows),
        "anchor_candidate_count": len(anchor_rows),
        "provenance": {
            name: {"path": _relative(path), "sha256": _sha256(path)}
            for name, path in provenance_paths.items()
        },
        "outputs": {
            name: {"path": _relative(path), "sha256": _sha256(path)}
            for name, path in outputs.items()
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    output_dir, audit = build(config_path)
    print(json.dumps({"output_dir": str(output_dir), **audit}, ensure_ascii=False))
    return 0 if audit["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
