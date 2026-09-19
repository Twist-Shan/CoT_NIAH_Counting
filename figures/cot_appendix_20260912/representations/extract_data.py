"""Export existing CoT appendix plotting data without fitting or GPU work.

Run from any directory with Python 3.10+ (standard library only):
    python figures/cot_appendix_20260912/representations/extract_data.py

All source coordinates and metrics are copied at their saved precision.
Assertions cross-check HTML, CSV, audit hashes, populations, and layer indices.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R = ROOT / "realistic"
OUT = HERE / "data"
MODELS = ("Qwen3-8B", "Gemma4-E4B")
SOURCES: dict[str, dict] = {}
OUTPUTS: dict[str, dict] = {}
FIELDS = ["model", "endpoint", "condition", "domain", "split", "seed",
          "count_or_index", "gold_count", "layer_zero_based", "layer_display_one_based",
          "pc1", "pc2", "pc3", "correct"]
SPLITS = {"discovery": list(range(1234, 1254)), "confirmation": list(range(1254, 1264))}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path: Path) -> Path:
    assert path.is_file(), path
    SOURCES[path.relative_to(ROOT).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return path


def read_json(path: Path):
    return json.loads(source(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with source(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def embedded(path: Path, name: str):
    text = source(path).read_text(encoding="utf-8")
    matches = list(re.finditer(r"\bconst\s+" + re.escape(name) + r"\s*=", text))
    assert len(matches) == 1, (path, name, len(matches))
    return json.JSONDecoder().raw_decode(text[matches[0].end():])[0]


def write_json(name: str, data):
    path = OUT / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    OUTPUTS[name] = {"sha256": sha(path), "bytes": path.stat().st_size}


def write_csv(name: str, rows: list[dict], fields: list[str] | None = None):
    path = OUT / name
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUTS[name] = {"sha256": sha(path), "bytes": path.stat().st_size, "rows": len(rows)}


def typed(row: dict) -> dict:
    result = {}
    for key, value in row.items():
        try:
            result[key] = int(value) if re.fullmatch(r"-?\d+", value) else float(value)
        except (ValueError, TypeError):
            result[key] = value
    return result


def flat_point(model, endpoint, layer, seed, label, gold, xyz, *,
               split="confirmation", condition="CoT-reasoning", domain="city", correct=""):
    assert all(math.isfinite(v) for v in xyz)
    return dict(zip(FIELDS, [model, endpoint, condition, domain, split, seed,
                           label, gold, layer, layer + 1, *xyz, correct]))


def literal_string(path: Path, variable: str) -> str:
    tree = ast.parse(source(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(variable)


def export_canonical() -> dict:
    dual = embedded(R / "reports/NiaH_Geometry_Comparison.html", "DUAL")
    points, curves, evrs = [], [], []
    meta = {"source_constant": "DUAL", "split_seeds": SPLITS,
            "gold_counts": list(range(1, 11)), "registered_trajectories_per_model": {"discovery": 200, "confirmation": 100},
            "pca_fit": "Separate StandardScaler then unwhitened PCA3 per model/endpoint/layer, fitted on discovery states only; randomized SVD, random_state=0.",
            "pca_precision": "Saved HTML coordinates have five decimal places; EVR has six. No refit or additional rounding in this export.",
            "readout_fit": "StandardScaler then whitened PCA16 refitted in each of five seed-grouped discovery folds; class-weighted logistic regression (max_iter=5000, default C=1) and Euclidean nearest centroid without shrinkage. Confirmation projection/probes fit on all discovery states. random_state=0.",
            "metric": "Multiclass balanced accuracy over labels 1..10; chance 0.1. Discovery curves are grouped-CV out-of-fold scores; confirmation curves are held-out diagnostics.",
            "selection_rule": "Maximize mean discovery-CV logistic and NCC balanced accuracy; tie-break NCC, logistic, earlier source layer. Existing frozen all-trace winners, not the main figure's NCC-first selected-format analysis.",
            "limitations": ["Running labels are parser-observed occurrence k, not necessarily unique retrieved-item count when traces repeat.", "Ragged traces contribute observed items only; no padding, missing-event reconstruction, or correctness filtering.", "This is the original 300-trajectory panel per model, not the main figure's separately selected 30 strict-format N=10 trajectories.", "The PCA3 display and whitened PCA16 decoder spaces differ; independent PCA panels do not share distances or axes."],
            "models": {}}
    all_coordinates = {"point_columns": ["split", "seed", "count_or_index", "pc1", "pc2", "pc3", "gold_count"], "models": {}}
    for model in MODELS:
        base = R / f"reports/v5_dual_endpoint_geometry_full300/{model}/pca16_whiten"
        audit = read_json(base / "dual_endpoint_geometry_audit.json")
        assert audit["random_state"] == 0 and audit["pca_whiten"] is True
        meta["models"][model] = {}
        all_coordinates["models"][model] = {}
        for panel_name, endpoint in (("running_native", "running_index"), ("final_native", "final_count")):
            panel = dual[model]["panels"][panel_name]
            chosen = [typed(r) for r in read_csv(base / f"{endpoint}_selected.csv") if r["mode"] == "native_thinking"]
            assert len(chosen) == 1
            selected = chosen[0]
            layer = panel["default_layer"]
            assert layer == selected["layer"] and panel["token_site"] == selected["token_site"]
            candidates = [typed(r) for r in read_csv(base / f"{endpoint}_candidate_metrics.csv")
                          if r["mode"] == "native_thinking" and r["analysis_group"] == selected["analysis_group"]
                          and r["selector"] == selected["selector"] and r["token_site"] == selected["token_site"]]
            assert len(candidates) == len(panel["layers"])
            for row in candidates:
                z = row["layer"]
                metric = panel["metrics"][str(z)]
                mapping = {"discovery_oof_logistic_balanced_accuracy": "discovery_logistic",
                           "discovery_oof_ncc_balanced_accuracy": "discovery_ncc",
                           "discovery_selection_score": "discovery_score",
                           "confirmation_logistic_balanced_accuracy": "confirmation_logistic",
                           "confirmation_ncc_balanced_accuracy": "confirmation_ncc"}
                for key, key2 in mapping.items():
                    assert math.isclose(row[key], metric[key2], abs_tol=1e-12), (model, endpoint, z, key)
                curves.append({"model": model, "endpoint": endpoint, "token_site": panel["token_site"],
                               "layer_zero_based": z, "layer_display_one_based": z + 1,
                               "is_selected_layer": int(z == layer),
                               "discovery_cv_ncc": metric["discovery_ncc"], "discovery_cv_logistic": metric["discovery_logistic"],
                               "discovery_selection_score": metric["discovery_score"],
                               "confirmation_ncc": metric["confirmation_ncc"], "confirmation_logistic": metric["confirmation_logistic"],
                               "discovery_states": row["discovery_oof_rows"], "confirmation_states": row["confirmation_rows"],
                               "confirmation_seed_count": row["confirmation_seed_count"], "chance_balanced_accuracy": row["chance_balanced_accuracy"]})
                coordinate = panel["coordinates"][str(z)]
                evrs.append({"model": model, "endpoint": endpoint, "layer_zero_based": z,
                             "layer_display_one_based": z + 1, "evr_pc1": coordinate["evr"][0],
                             "evr_pc2": coordinate["evr"][1], "evr_pc3": coordinate["evr"][2]})
            coordinate = panel["coordinates"][str(layer)]
            counts = dict(Counter(p[0] for p in coordinate["points"]))
            assert counts == {"discovery": selected["discovery_oof_rows"], "confirmation": selected["confirmation_rows"]}
            for split, expected in SPLITS.items():
                assert sorted({p[1] for p in coordinate["points"] if p[0] == split}) == expected
                assert sorted({p[2] for p in coordinate["points"] if p[0] == split}) == list(range(1, 11))
            for split, seed, label, x, y, z, gold in coordinate["points"]:
                points.append(flat_point(model, endpoint, layer, seed, label, gold, [x, y, z], split=split))
            meta["models"][model][endpoint] = {"source_key": f"DUAL[{model!r}]['panels'][{panel_name!r}]",
                "selected_layer_zero_based": layer, "selected_layer_display_one_based": layer + 1,
                "default_layer_zero_based": layer, "available_layers_zero_based": panel["layers"],
                "token_site": panel["token_site"], "label_semantics": "parser-observed occurrence k" if endpoint == "running_index" else "gold final count N",
                "states_by_split": counts, "evr": coordinate["evr"], "selected_metrics_source_row": selected}
            all_coordinates["models"][model][endpoint] = panel["coordinates"]
    write_csv("canonical_selected_pca.csv", points, FIELDS)
    write_csv("canonical_layerwise_readouts.csv", curves)
    write_csv("canonical_layerwise_evr.csv", evrs)
    write_json("canonical_all_layer_coordinates.json", all_coordinates)
    return meta


def export_domain(canonical: dict) -> dict:
    base = R / "reports/v5_domain_endpoint_comparison"
    payload_path = base / "geometry_payload.json"
    metric_path = base / "domain_endpoint_metrics.csv"
    payload, audit = read_json(payload_path), read_json(base / "audit.json")
    for path in (payload_path, metric_path):
        expected = next(value for key, value in audit["outputs"].items() if Path(key).name == path.name)
        assert sha(path) == expected, f"Domain audit output mismatch: {path}"
    source_payload = read_json(R / "work/domain_transfer_geometry/analysis/report_payload.json")
    for recorded_path, expected in audit["inputs"].items():
        recorded = Path(recorded_path)
        if recorded.name in ("report_payload.json", "running_index_selected.csv"):
            assert sha(source(recorded)) == expected, f"Domain layer-selection input changed: {recorded}"
    meta = {"source": str(payload_path.relative_to(ROOT).as_posix()), "source_schema": payload["schema_version"],
            "design": source_payload["design"], "audit_selection": audit["selection"],
            "pca_fit": "Each model/endpoint independently fits StandardScaler then unwhitened PCA3 on city discovery only (random_state=0); city, flower, and animal confirmation share that frozen basis.",
            "readout_fit": "The endpoint-comparison script refits StandardScaler then whitened PCA16 on city discovery only; balanced logistic regression (max_iter=4000, solver=lbfgs, default C=1, random_state=0) and Euclidean NCC without shrinkage. This is the endpoint-comparison analysis, not the older report_payload.json probe fit.",
            "layer_selection": {"running_index": "Reuse canonical all-trace discovery winner at fixed item_end.", "answer_token": "Reuse original domain-transfer city-only five-fold seed-grouped mean logistic/NCC winner, ties earlier layer. The selected layer is inherited, while projections and probes are refit by the endpoint-comparison script."},
            "limitations": ["Running k is parser-observed occurrence; ragged/duplicate items are retained and state counts differ across domains.", "Answer readout predicts gold N; it is not generated answer accuracy.", "Answer layers L26/L42 differ from canonical final-count layers L27/L35. Do not splice metrics or coordinates between these analyses.", "The running and answer panels each use a separate PCA basis; only domains within one model/endpoint share coordinates."], "models": {}}
    points, readouts = [], []
    csv_rows = [typed(r) for r in read_csv(metric_path) if r["mode"] == "native_thinking"]
    for model in MODELS:
        meta["models"][model] = {}
        for endpoint in ("running_index", "answer_token"):
            panel = payload["models"][model]["native_thinking"][endpoint]
            layer = panel["layer"]
            assert layer == (canonical["models"][model]["running_index"]["selected_layer_zero_based"] if endpoint == "running_index" else source_payload["models"][model]["native_thinking"]["selected_layer"])
            meta["models"][model][endpoint] = {key: value for key, value in panel.items() if key != "points"}
            meta["models"][model][endpoint].update({"selected_layer_zero_based": layer, "selected_layer_display_one_based": layer + 1,
                "evr": panel["pca3_explained_variance_ratio"],
                "display_split": "confirmation", "fit_states": canonical["models"][model]["running_index" if endpoint == "running_index" else "final_count"]["states_by_split"]["discovery"]})
            for domain, metric in panel["metrics"].items():
                subset = [p for p in panel["points"] if p["domain"] == domain]
                assert len(subset) == metric["states"]
                assert len({p["trajectory_id"] for p in subset}) == metric["trajectories"] == 100
                assert sorted({p["seed"] for p in subset}) == SPLITS["confirmation"]
                assert {str(k): v for k, v in sorted(Counter(p["count"] for p in subset).items())} == metric["support"]
                matching = [r for r in csv_rows if r["model_label"] == model and r["endpoint"] == endpoint and r["entity_domain"] == domain]
                assert len(matching) == 1
                row = matching[0]
                for key in ("states", "trajectories", "logistic_balanced_accuracy", "ncc_balanced_accuracy"):
                    assert math.isclose(row[key], metric[key], abs_tol=1e-12)
                readouts.append({"model": model, "endpoint": endpoint, "domain": domain,
                    "layer_zero_based": layer, "layer_display_one_based": layer + 1, "split": "confirmation",
                    "states": metric["states"], "trajectories": metric["trajectories"],
                    "ncc_balanced_accuracy": metric["ncc_balanced_accuracy"], "logistic_balanced_accuracy": metric["logistic_balanced_accuracy"],
                    "chance_balanced_accuracy": 0.1})
            for p in panel["points"]:
                points.append(flat_point(model, endpoint, layer, p["seed"], p["count"], p["gold_count"],
                    [p["x"], p["y"], p["z"]], condition=p["domain"], domain=p["domain"]))
    write_csv("domain_selected_pca.csv", points, FIELDS)
    write_csv("domain_readouts.csv", readouts)
    return meta


def export_cue() -> dict:
    path = R / "reports/v4_non-thinking_causal/v4_4_2/realistic_niah_v4_4_2_mode_geometry_attention_report.html"
    payload = embedded(path, "NATIVE_GEOM")
    meta = {"source_constant": "NATIVE_GEOM", "source_site": "answer_query",
            "state_definition": "Last captured token whose query role is answer_query (the registered answer-query interval before numeric answer); not trace_last, trace_mean, or an item-end running state.",
            "label_semantics": "Gold final N=1..10", "seeds": list(range(1234, 1244)),
            "pairs_per_model": 100, "states_per_condition_per_model": 100,
            "conditions": payload["conditions"],
            "pca_fit": "Unstandardized six-component PCA fitted jointly to the 100 cue-present and 100 cue-absent answer-query vectors per model/layer; randomized SVD, random_state=0. Export displays the first three raw pooled PCs. No cue-centering is applied.",
            "selection_rule": "Reuse existing NATIVE_GEOM.landmarks[model].display (source layers 29 and 37); these are saved descriptive display landmarks, not newly selected readout winners.",
            "prompt_manipulation": "Enable model thinking in both branches. Remove only COMMON_COUNTING_CUE opening text; passage and V4_NUMERIC_QUERY_BLOCK remain identical. No assistant-side Total: prefill is supplied in thinking mode.",
            "retained_final_query": literal_string(R / "src/realistic_niah_v4/prompts.py", "V4_NUMERIC_QUERY_BLOCK"),
            "removed_opening_cue": literal_string(R / "src/realistic_niah/prompts.py", "COMMON_COUNTING_CUE"),
            "limitations": ["Auxiliary numeric-output prompt explicitly says not to explain/reason aloud/list; it differs from the main CoT prompt despite the thinking template being enabled.", "Both conditions contribute to the descriptive PCA fit; this is not a held-out projection.", "Each condition generates its own continuation; the manipulation does not hold the trace text or token positions fixed.", "No event-aligned cue-removal running-index PCA is available in this source. Final-N answer states must not be relabelled as running-index states.", "Do not interpret saved ridge R-squared or CKA as classification accuracy or causal necessity."],
            "saved_inference_description": payload["inference"], "models": {}}
    selected_points, all_points, evrs = [], [], []
    for model in MODELS:
        layer = payload["landmarks"][model]["display"]
        key = f"{model}|answer_query|{layer}"
        dataset = payload["datasets"][key]
        meta["models"][model] = {"source_key": f"NATIVE_GEOM['datasets'][{key!r}]",
            "selected_layer_zero_based": layer, "selected_layer_display_one_based": layer + 1,
            "default_layer_zero_based": layer, "evr": dataset["evr_raw"][:3], "evr_all_six": dataset["evr_raw"],
            "saved_statistics": payload["statistics"][key]}
        for key, dataset in payload["datasets"].items():
            if dataset["model"] != model or dataset["site"] != "answer_query":
                continue
            z = dataset["layer"]
            rows = dataset["rows"]
            assert len(rows) == 100 and all(len(row) == 28 for row in rows)
            assert sorted({row[0] for row in rows}) == meta["seeds"]
            assert Counter(row[1] for row in rows) == Counter({k: 10 for k in range(1, 11)})
            assert len({(row[0], row[1]) for row in rows}) == 100
            evrs.append({"model": model, "endpoint": "answer_query", "layer_zero_based": z,
                         "layer_display_one_based": z + 1, **{f"evr_pc{i+1}": v for i, v in enumerate(dataset["evr_raw"])}})
            for row in rows:
                for condition, start, correct_i in (("cue_present", 4, 2), ("cue_absent", 10, 3)):
                    point = flat_point(model, "answer_query", z, row[0], row[1], row[1], row[start:start+3],
                        split="paired_descriptive", condition=condition, correct=row[correct_i])
                    all_points.append(point)
                    if z == layer:
                        selected_points.append(point)
    write_csv("cue_answer_query_selected_pca.csv", selected_points, FIELDS)
    write_csv("cue_answer_query_all_layer_pca.csv", all_points, FIELDS)
    write_csv("cue_answer_query_layerwise_evr.csv", evrs)
    return meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for rel in ["scripts/build_niah_geometry_comparison_report.py", "src/realistic_niah_v5/dual_endpoint_geometry.py",
                "src/realistic_niah_v5/trace_stratified_geometry.py", "scripts/analyze_niah_domain_endpoint_comparison.py",
                "scripts/analyze_niah_domain_transfer_geometry.py", "src/realistic_niah_v5/domain_transfer_geometry.py",
                "scripts/analyze_realistic_niah_v4_4_2_counter_geometry.py", "src/realistic_niah_v4_4_2/prompts.py",
                "src/realistic_niah_v4_4_2/capture.py"]:
        source(R / rel)
    canonical = export_canonical()
    domain = export_domain(canonical)
    cue = export_cue()
    write_json("metadata.json", {"schema_version": "cot_appendix_representations_export_v1", "public_mode_name": "CoT-reasoning",
        "source_mode_key": "native_thinking", "layer_convention": "source arrays and layer keys are zero-based; all display layer numbers add one",
        "plot_csv_columns": FIELDS, "default_plot_filter": "Canonical PCA: split=confirmation; domain: all points are confirmation; cue: all points are descriptive paired conditions.",
        "canonical": canonical, "domain": domain, "cue": cue})
    manifest = {"schema_version": "cot_appendix_representations_manifest_v1", "extraction_script": {
        "path": Path(__file__).resolve().relative_to(ROOT).as_posix(), "sha256": sha(Path(__file__))},
        "runtime": "Python standard library; no model execution, hidden-state tensor reads, re-fitting, or coordinate editing",
        "validation": ["DUAL selected layer equals selected CSV layer for every model/endpoint.",
            "All-layer HTML readouts match source CSV values to 1e-12.", "Original split seed sets, class labels, and selected-layer state counts match source rows.",
            "Domain payload and metric CSV match their audit output SHA-256 hashes; their populations and readout values cross-check.",
            "Domain running layers equal canonical winners; answer layers equal city-discovery source selection.",
            "Cue source site is answer_query with exactly 100 unique seed/count pairs at each available layer, ten seeds, N=1..10.",
            "No data from selected-format INDEXED_NUMERIC_N10 or forced-immediate-answer experiments are included."],
        "sources": SOURCES, "outputs": OUTPUTS}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_directory": str(OUT), "files": {name: info.get("rows", "JSON") for name, info in OUTPUTS.items()}, "validation": "passed"}, indent=2))


if __name__ == "__main__":
    main()
