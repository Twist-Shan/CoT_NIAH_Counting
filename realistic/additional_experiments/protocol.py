"""Task definitions and audit helpers; standard library only."""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODES = ("nonthinking", "native_thinking")
TASKS = ("count_all", "kth_needle", "topic_count")
TOPICS = {
    "astronomy": [
        "The project measured the orbital period of a distant planet.",
        "The project observed a supernova through an optical telescope.",
        "The project mapped the motion of stars near the galactic center.",
        "The project compared the spectra of several distant galaxies.",
        "The project tracked changes in the tail of a passing comet.",
        "The project measured the brightness of an eclipsing binary star.",
        "The project estimated the mass of a black hole from stellar motion.",
        "The project recorded radio pulses from a rotating neutron star.",
        "The project studied the distribution of craters on a lunar surface.",
        "The project detected a planet crossing in front of its host star.",
    ],
    "botany": [
        "The project measured the germination rate of dormant seeds.",
        "The project observed root growth in plants under water stress.",
        "The project mapped the movement of pollen between flowering plants.",
        "The project compared photosynthesis rates in shaded leaves.",
        "The project tracked changes in the stems of climbing vines.",
        "The project measured the response of seedlings to soil nutrients.",
        "The project estimated the growth of moss under varying humidity.",
        "The project recorded the opening times of flowers during spring.",
        "The project studied the distribution of stomata on leaf surfaces.",
        "The project detected a mutation affecting the color of flower petals.",
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def validate_config(config: dict) -> None:
    d, c = config["discovery_seeds"], config["confirmation_seeds"]
    if not d or not c or len(set(d + c)) != len(d + c):
        raise ValueError("Discovery and confirmation seeds must be nonempty, unique and disjoint")
    levels = config["levels"]
    if len(set(levels)) != len(levels) or len(levels) < 2:
        raise ValueError("Need at least two unique count/k levels")
    if any(not 1 <= n <= config["total_records_new_tasks"] for n in levels):
        raise ValueError("Level outside the available record range")
    if set(config["tasks"]) != set(TASKS) or set(config["modes"]) != set(MODES):
        raise ValueError("Unregistered task/mode")


def source_records(source: dict) -> list[dict]:
    records = sorted(source["active_needle_spans"], key=lambda x: x["char_start"])
    if len(records) != source["gold_count"]:
        raise ValueError("Source active spans/count mismatch")
    for r in records:
        if source["passage"][r["char_start"]:r["char_end"]] != r["text"]:
            raise ValueError("Source character offsets do not match text")
    if len({r["city"] for r in records}) != len(records):
        raise ValueError("This pilot requires unique record city identifiers")
    return records


def make_case(source: dict, task: str, level: int, split: str) -> dict:
    records = source_records(source)
    if task not in TASKS or not 1 <= level <= len(records):
        raise ValueError("Invalid task/level")
    if task == "count_all" and level != len(records):
        raise ValueError("Reference count must equal the source count")
    rng = random.Random(20260905 + int(source["seed"]))
    order = list(range(len(records)))
    rng.shuffle(order)
    target_slots = set(order[:level])
    target_topic = ("astronomy" if int(source["seed"]) % 2 == 0 else "botany")
    other_topic = "botany" if target_topic == "astronomy" else "astronomy"
    fragments, new_records, cursor, size = [], [], 0, 0
    for i, r in enumerate(records):
        before = source["passage"][cursor:r["char_start"]]
        fragments.append(before)
        size += len(before)
        record_text = r["text"]
        topic = None
        if task == "topic_count":
            topic = target_topic if i in target_slots else other_topic
            description = TOPICS[topic][(i + int(source["seed"])) % len(TOPICS[topic])]
            # Keep the original city/score sentence and excerpt boundaries intact.
            if record_text.count("End excerpt.") != 1:
                raise ValueError("Source record lacks a unique excerpt terminator")
            record_text = record_text.replace("End excerpt.", description + "\nEnd excerpt.")
        new_records.append({
            "city": r["city"], "score": int(r["score"]), "ordinal": i + 1,
            "char_start": size, "char_end": size + len(record_text),
            "text": record_text, "topic": topic,
            "is_target": topic == target_topic if task == "topic_count" else True,
        })
        fragments.append(record_text)
        size += len(record_text)
        cursor = r["char_end"]
    fragments.append(source["passage"][cursor:])
    passage = "".join(fragments)
    target = new_records[level - 1]
    gold = f"{target['city']}|{target['score']}" if task == "kth_needle" else str(level)
    case = {
        "case_id": f"{task}_seed{source['seed']}_level{level}",
        "task": task, "seed": int(source["seed"]), "split": split, "level": level,
        "source_stimulus_id": source["stimulus_id"],
        "source_passage_sha256": text_hash(source["passage"]),
        "source_canonical_passage_tokens": source["canonical_passage_tokens"],
        "passage": passage, "passage_sha256": text_hash(passage),
        "records": new_records, "total_records": len(records),
        "target_topic": target_topic if task == "topic_count" else None,
        "gold": gold, "answer_prefix": "Needle:" if task == "kth_needle" else "Total:",
    }
    audit_case(case)
    return case


def audit_case(case: dict) -> None:
    records, passage = case["records"], case["passage"]
    if len(records) != case["total_records"] or text_hash(passage) != case["passage_sha256"]:
        raise ValueError("Passage/count audit failed")
    last = -1
    for i, r in enumerate(records, 1):
        if r["ordinal"] != i or r["char_start"] < last:
            raise ValueError("Record ordering/overlap audit failed")
        if passage[r["char_start"]:r["char_end"]] != r["text"]:
            raise ValueError("Mutated record offsets failed")
        last = r["char_end"]
    if case["task"] == "topic_count":
        if sum(r["is_target"] for r in records) != int(case["gold"]):
            raise ValueError("Topic target count audit failed")
    if case["task"] == "kth_needle":
        r = records[case["level"] - 1]
        if case["gold"] != f"{r['city']}|{r['score']}":
            raise ValueError("k-th indexing audit failed")


def user_prompt(case: dict, mode: str) -> str:
    if mode not in MODES:
        raise ValueError("Unknown mode")
    definition = "A city-score audit record names one city and gives that city's numeric score."
    if case["task"] == "count_all":
        cue = "You will need to count all city-score audit records in the passage below."
        question = "How many city-score audit records are in the passage?"
        fmt = "Total: <integer>" if mode == "native_thinking" else "Total:<integer>"
        finish = "determine the count"
    elif case["task"] == "kth_needle":
        cue = "You will need to find a specified city-score audit record in passage order."
        question = (
            f"Which city-score audit record is number {case['level']} in passage order?\n"
            "Count occurrences from the beginning of the passage, starting at 1.\n"
            "Return the actual city name and numeric score of that record."
        )
        fmt, finish = "Needle:<city>|<integer>", "identify that record"
    else:
        cue = f"You will need to count city-score audit records about {case['target_topic']} in the passage below."
        definition += (
            "\nEach record also describes a research project. Determine its topic from the project description."
        )
        question = (
            f"How many city-score audit records describe a project about {case['target_topic']}?\n"
            "Count only matching records. Ignore records about other topics and ordinary passage text."
        )
        fmt, finish = "Total:<integer>", "determine the count"
    if mode == "native_thinking":
        instruction = (
            "Reason concisely without repeating or restarting.\n"
            f"Stop as soon as you {finish}, then output exactly one line:\n{fmt}"
        )
    else:
        numeric = (
            "Write the count using ordinary decimal digits, with no space after the colon.\n"
            if case["task"] != "kth_needle" else "Use no spaces around the colon or vertical bar.\n"
        )
        instruction = (
            "Do not explain, reason aloud, quote, or list any records.\n"
            + numeric + "Your entire response must be exactly one line:\n" + fmt
        )
    return f"{cue}\n{definition}\n\n<passage>\n{case['passage']}\n</passage>\n\n{question}\n{instruction}"


def parse_answer(text: str, case: dict) -> dict:
    # Reasoning must already have been removed with the registered channel parser.
    if case["task"] == "kth_needle":
        m = re.fullmatch(r"\s*Needle:\s*([^|\r\n]+)\|\s*(\d+)\s*", text)
        value = f"{m[1].strip()}|{int(m[2])}" if m else None
    else:
        m = re.fullmatch(r"\s*Total:\s*(\d+)\s*", text)
        value = str(int(m[1])) if m else None
    strict = value is not None
    return {"prediction": value, "parse_ok": strict, "correct": strict and value == case["gold"]}


@dataclass(frozen=True)
class Encoding:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    query_position: int

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


def encode_ids(ids: list[int] | tuple[int, ...]) -> Encoding:
    if not ids:
        raise ValueError("Empty encoding")
    return Encoding(tuple(ids), (1,) * len(ids), len(ids) - 1)


def token_end(offsets: list, start: int, end: int) -> int:
    positions = [i for i, (s, e) in enumerate(offsets) if e > s and s < end and e > start]
    if not positions:
        raise ValueError("No token overlaps the requested character span")
    return positions[-1]


def matched_random_heads(heads: list, heads_per_layer: list | tuple, seed: int) -> list:
    rng = random.Random(seed)
    layers: dict[int, int] = {}
    if len(set(map(tuple, heads))) != len(heads):
        raise ValueError("Duplicate selected heads")
    for layer, head in heads:
        if not 0 <= layer < len(heads_per_layer) or not 0 <= head < heads_per_layer[layer]:
            raise ValueError("Head index outside model architecture")
        layers[layer] = layers.get(layer, 0) + 1
    # Same sampling population as the historical protocol; overlap is allowed.
    return [[l, h] for l, n in sorted(layers.items()) for h in rng.sample(range(heads_per_layer[l]), n)]
