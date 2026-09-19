from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    precision_recall_fscore_support,
)

INDEX_RE = re.compile(r"(?m)^\s*(\d+)[.)]\s+\S+")
BULLET_RE = re.compile(r"(?m)^\s*[-*•]\s+\S+")
WORD_COUNTER_RE = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
    re.IGNORECASE,
)
PROSE_PAIR_RE = re.compile(r"\b[A-Z][A-Za-z .'-]{1,40}\s*:\s*-?\d+\b")
TALLY_RE = re.compile(
    r"\b(?:count|total|so far|now)\s*(?:=|:|is|becomes)?\s*\d+\b",
    re.IGNORECASE,
)
ARITHMETIC_RE = re.compile(r"\b\d+\s*\+\s*\d+(?:\s*\+\s*\d+)*\s*=\s*\d+\b")
SCAN_RE = re.compile(
    r"\b(?:scan(?:ning|ned)?|search(?:ing|ed)?|find(?:ing|s|found)?|"
    r"match(?:ing|ed|es)?|retriev(?:e|ed|ing)|look(?:ing)?\s+for)\b",
    re.IGNORECASE,
)
RESTART_RE = re.compile(
    r"\b(?:restart|start(?:ing)? over|recount|count again|let me redo)\b",
    re.IGNORECASE,
)
SELF_CORRECTION_RE = re.compile(
    r"\b(?:correction|actually|rather|I was wrong|should be|not \d+ but \d+)\b",
    re.IGNORECASE,
)
TEMPLATE_LEAK_RE = re.compile(
    r"<\s*(?:integer|city|score|passage)\s*>|\{\s*(?:PASSAGE|integer|city|score)\s*\}",
    re.IGNORECASE,
)

CORE_STYLES = (
    "index_enumeration",
    "bullet_enumeration",
    "word_enumeration",
    "running_tally",
    "arithmetic_grouping",
    "scan_or_retrieval_summary",
)
PROCESS_FLAGS = (
    "repetition",
    "restart",
    "self_correction",
    "template_leakage",
    "truncated",
    "empty_reasoning",
)
HUMAN_DOMINANT_COLUMN = "human_dominant_style"
HUMAN_CORE_COLUMNS = tuple(f"human_{style}" for style in CORE_STYLES)


@dataclass(frozen=True)
class StyleResult:
    observability: str
    dominant_style: str
    index_enumeration: bool
    bullet_enumeration: bool
    word_enumeration: bool
    running_tally: bool
    arithmetic_grouping: bool
    scan_or_retrieval_summary: bool
    answer_only: bool
    mixed: bool
    other_unclassifiable: bool
    repetition: bool
    restart: bool
    self_correction: bool
    template_leakage: bool
    truncated: bool
    empty_reasoning: bool
    detected_core_style_count: int
    classifier_confidence: float


def _normalized_nonempty_lines(text: str) -> list[str]:
    return [
        " ".join(line.lower().split()) for line in text.splitlines() if line.strip()
    ]


def _has_repetition(text: str) -> bool:
    lines = _normalized_nonempty_lines(text)
    return len(lines) != len(set(lines))


def _has_index_restart(text: str) -> bool:
    indices = [int(value) for value in INDEX_RE.findall(text)]
    return any(current <= previous for previous, current in zip(indices, indices[1:]))


def classify_counting_style(
    *,
    reasoning_text: str | None,
    final_text: str | None = None,
    raw_output_text: str | None = None,
    separate_reasoning_text: str | None = None,
    truncated: bool = False,
) -> StyleResult:
    separate = (separate_reasoning_text or "").strip()
    reasoning = (reasoning_text or "").strip()
    final = (final_text or "").strip()
    raw = (raw_output_text or "").strip()
    if separate:
        source = separate
        observability = "separate_reasoning"
    elif reasoning:
        source = reasoning
        observability = "inline_reasoning"
    elif final or raw:
        source = ""
        observability = "final_only"
    else:
        source = ""
        observability = "missing_completion"

    index_hits = len(INDEX_RE.findall(source))
    bullet_hits = len(BULLET_RE.findall(source))
    word_hits = len(WORD_COUNTER_RE.findall(source))
    prose_pairs = len(PROSE_PAIR_RE.findall(source))
    tally_hits = len(TALLY_RE.findall(source))
    flags = {
        "index_enumeration": index_hits >= 2,
        "bullet_enumeration": bullet_hits >= 2,
        "word_enumeration": (
            word_hits >= 2 or (prose_pairs >= 2 and index_hits < 2 and bullet_hits < 2)
        ),
        "running_tally": tally_hits >= 2,
        "arithmetic_grouping": ARITHMETIC_RE.search(source) is not None,
        "scan_or_retrieval_summary": SCAN_RE.search(source) is not None,
    }
    active = [name for name in CORE_STYLES if flags[name]]
    empty = not bool(source.strip())
    if not active:
        dominant = "answer_only" if empty else "other_unclassifiable"
    elif len(active) == 1:
        dominant = active[0]
    else:
        structural = [
            name
            for name in ("index_enumeration", "bullet_enumeration", "word_enumeration")
            if flags[name]
        ]
        dominant = structural[0] if len(structural) == 1 else "mixed"
    mixed = dominant == "mixed"
    answer_only = dominant == "answer_only"
    other = dominant == "other_unclassifiable"
    confidence = 1.0 if len(active) <= 1 else (0.75 if not mixed else 0.5)
    return StyleResult(
        observability=observability,
        dominant_style=dominant,
        **flags,
        answer_only=answer_only,
        mixed=mixed,
        other_unclassifiable=other,
        repetition=_has_repetition(source),
        restart=bool(RESTART_RE.search(source)) or _has_index_restart(source),
        self_correction=SELF_CORRECTION_RE.search(source) is not None,
        template_leakage=TEMPLATE_LEAK_RE.search(source) is not None,
        truncated=bool(truncated),
        empty_reasoning=empty,
        detected_core_style_count=len(active),
        classifier_confidence=confidence,
    )


def classify_request_table(requests: pd.DataFrame) -> pd.DataFrame:
    required = {
        "request_id",
        "reasoning_text",
        "final_text",
        "raw_output_text",
        "truncated",
    }
    missing = required.difference(requests.columns)
    if missing:
        raise ValueError(f"Style classification missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for row in requests.itertuples(index=False):
        separate = getattr(row, "separate_reasoning_text", None)
        result = classify_counting_style(
            reasoning_text=getattr(row, "reasoning_text"),
            final_text=getattr(row, "final_text"),
            raw_output_text=getattr(row, "raw_output_text"),
            separate_reasoning_text=separate,
            truncated=bool(getattr(row, "truncated")),
        )
        rows.append({"request_id": str(getattr(row, "request_id")), **asdict(result)})
    return pd.DataFrame(rows)


def _model_family(label: str) -> str:
    if label.startswith("Qwen"):
        return "Qwen"
    if label.startswith("Gemma"):
        return "Gemma"
    if "Nemotron" in label:
        return "Nemotron"
    if label.startswith("GLM"):
        return "GLM"
    if label.startswith("Ministral"):
        return "Ministral"
    return "Other"


def build_blinded_annotation_samples(
    requests: pd.DataFrame,
    styles: pd.DataFrame,
    *,
    random_size: int = 600,
    challenge_size: int = 200,
    random_seed: int = 20_260_807,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = requests.merge(
        styles,
        on="request_id",
        validate="one_to_one",
        suffixes=("", "_style"),
    )
    merged["model_family"] = merged["model_label"].map(_model_family)
    rng = np.random.default_rng(random_seed)
    strata = ["prompt_mode", "model_family"]
    groups = list(merged.groupby(strata, sort=True, dropna=False))
    base = random_size // len(groups)
    remainder = random_size % len(groups)
    sampled: list[pd.DataFrame] = []
    for index, (_, group) in enumerate(groups):
        take = min(len(group), base + (1 if index < remainder else 0))
        chosen = rng.choice(group.index.to_numpy(), size=take, replace=False)
        part = group.loc[chosen].copy()
        part["sampling_probability"] = take / len(group)
        sampled.append(part)
    random_sample = pd.concat(sampled, ignore_index=True)
    if len(random_sample) < min(random_size, len(merged)):
        remaining = merged.loc[~merged["request_id"].isin(random_sample["request_id"])]
        take = min(random_size - len(random_sample), len(remaining))
        chosen = rng.choice(remaining.index.to_numpy(), size=take, replace=False)
        extra = remaining.loc[chosen].copy()
        extra["sampling_probability"] = take / len(remaining)
        random_sample = pd.concat([random_sample, extra], ignore_index=True)

    challenge_score = (
        (1.0 - merged["classifier_confidence"].astype(float))
        + merged["mixed"].astype(float)
        + merged["repetition"].astype(float)
        + merged["restart"].astype(float)
        + merged.get("truncated_style", merged["truncated"]).astype(float)
        + (~merged["parse_success"].astype(bool)).astype(float)
        + np.log1p(merged["reasoning_text"].fillna("").str.len()) / 20.0
    )
    challenge_pool = merged.loc[
        ~merged["request_id"].isin(random_sample["request_id"])
    ].copy()
    challenge_pool["challenge_score"] = challenge_score.loc[challenge_pool.index]
    challenge = challenge_pool.sort_values(
        ["challenge_score", "request_id"], ascending=[False, True]
    ).head(challenge_size)

    blinded_columns = [
        "request_id",
        "prompt_mode",
        "observability",
        "reasoning_text",
        "final_text",
        "raw_output_text",
    ]
    random_output = random_sample[blinded_columns + ["sampling_probability"]].copy()
    random_output["analysis_weight"] = 1.0 / random_output["sampling_probability"]
    challenge_output = challenge[blinded_columns + ["challenge_score"]].copy()
    for column in (HUMAN_DOMINANT_COLUMN, *HUMAN_CORE_COLUMNS):
        random_output[column] = ""
        challenge_output[column] = ""
    return random_output, challenge_output


def _annotation_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid annotation boolean: {value!r}")


def evaluate_style_annotations(
    annotations: pd.DataFrame,
    automated_styles: pd.DataFrame,
    *,
    weight_column: str | None = "analysis_weight",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    required_annotations = {"request_id", HUMAN_DOMINANT_COLUMN, *HUMAN_CORE_COLUMNS}
    missing = required_annotations.difference(annotations.columns)
    if missing:
        raise ValueError(f"Annotation table missing columns: {sorted(missing)}")
    required_automated = {"request_id", "dominant_style", *CORE_STYLES}
    missing = required_automated.difference(automated_styles.columns)
    if missing:
        raise ValueError(f"Automated style table missing columns: {sorted(missing)}")
    merged = annotations.merge(
        automated_styles[list(required_automated)],
        on="request_id",
        validate="one_to_one",
    )
    if merged[HUMAN_DOMINANT_COLUMN].astype(str).str.strip().eq("").any():
        raise ValueError("Dominant-style human annotations are incomplete")
    weights = (
        merged[weight_column].to_numpy(dtype=float)
        if weight_column is not None and weight_column in merged
        else np.ones(len(merged), dtype=float)
    )
    kappa = float(
        cohen_kappa_score(
            merged[HUMAN_DOMINANT_COLUMN].astype(str),
            merged["dominant_style"].astype(str),
            sample_weight=weights,
        )
    )
    per_label: list[dict[str, Any]] = []
    f1_values: list[float] = []
    for style, human_column in zip(CORE_STYLES, HUMAN_CORE_COLUMNS):
        human = merged[human_column].map(_annotation_boolean).to_numpy(dtype=bool)
        automated = merged[style].astype(bool).to_numpy()
        precision, recall, f1, support = precision_recall_fscore_support(
            human,
            automated,
            labels=[True],
            average=None,
            sample_weight=weights,
            zero_division=0,
        )
        per_label.append(
            {
                "style": style,
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "weighted_support": float(support[0]),
            }
        )
        if float(support[0]) > 0 or bool(automated.any()):
            f1_values.append(float(f1[0]))
    macro_f1 = float(np.mean(f1_values)) if f1_values else math.nan
    confusion = pd.crosstab(
        merged[HUMAN_DOMINANT_COLUMN].astype(str),
        merged["dominant_style"].astype(str),
        values=weights,
        aggfunc="sum",
        dropna=False,
    ).fillna(0.0)
    confusion.index.name = "human_dominant_style"
    confusion = confusion.reset_index()
    summary = {
        "annotated_requests": len(merged),
        "dominant_style_weighted_cohen_kappa": kappa,
        "dominant_style_threshold": 0.75,
        "dominant_style_threshold_passed": kappa >= 0.75,
        "multilabel_weighted_macro_f1": macro_f1,
        "multilabel_labels_with_human_or_automated_positive_support": len(f1_values),
        "multilabel_threshold": 0.80,
        "multilabel_threshold_passed": bool(macro_f1 >= 0.80),
        "confirmatory_automated_reporting_allowed": (
            kappa >= 0.75 and macro_f1 >= 0.80
        ),
    }
    return summary, pd.DataFrame(per_label), confusion
