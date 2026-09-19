# Source coverage and release scope

This is a code release. Original realistic corpus/stimulus bytes, pretrained
weights, generations, checkpoints, activation caches, and historical results
are intentionally not distributed. Their absence is different from a missing
implementation. The intended workflow is to construct inputs and produce new
measurements locally, with explicit provenance for the new run.

## Follow-up source review

| Review finding | Resolution / scope |
|---|---|
| Missing Enumeration launchers | Restored `launch_enumeration_n10_update.py` and `launch_enumeration_qwen_layer_diagnostic.py`; the previously omitted diagnostic tests now run in the selected suite. |
| Final plotting code outside the original experiment folders | Added the curated `figures/` sources and editable diagram template. The final Figure 2 builder now accepts explicit input/output paths. See the figure map and input catalog. |
| Qwen YaRN-off experiment stored under an output directory | Restored `realistic/scripts/qwen_yarn_off/full_run.py`, `summarize.py`, and a configuration without deployment/account settings. Preparation, four workers and complete-result merging are exposed in the registry. |
| Additional-task protocols and entry points | Restored the three technical protocol documents used by the Appendix H task-package constructors. The registry exposes kth-record and category-counting input/generation stages; the early transfer pilot is excluded. |
| Report paths tied to historical output folders | Parameterized the synthetic report and the two earlier empirical report/figure builders. Final Figure 2 is a separate builder; the earlier fitted-law figures are not its replacement. |
| Original-input hashes versus fresh generation | Historical hash checks remain unchanged. V3.1 preparation offers an explicit `--fresh-dataset` route that reruns the registered grid/tokenizer audit and records new hashes and a fresh-replication identity. |
| Environment validation | A new isolated CPU environment was installed from the declared requirements; installation, dependency consistency, selected tests and actual report/figure generation were checked. See `VALIDATION.md`. |
| Minor legacy directory label and stale hosting wording | Removed the label and clarified the difference between the prepared ZIP and hosting-service downloads. |

An additional read-only comparison against the source repository's current
default branch found the same committed baseline as the local original
repository. All 1,307 upstream Python/shell/JSON files were present locally,
and all 224 upstream `src/` modules were represented in the prepared release.
Excluded files include account/cluster launch wrappers and the non-paper
workflows below. This comparison did not identify additional remote-only
paper code; it is a source-coverage check, not a GPU reproduction claim.

## Auxiliary workflows excluded after manuscript review

The supplied 72-page manuscript and the final figure input paths were checked
before excluding these two earlier workflows. They are outside the paper
release scope and are no longer presented as missing paper deliverables.

| Excluded workflow | Manuscript and source evidence | Retained paper implementation |
|---|---|---|
| Integrated V4.4 HTML report consuming `correct_state_route_analysis.json` | The old report conditions on both models being correct at counts 1, 2, 3. Appendix D.3, p. 35, Figure 19 instead uses correct-input transfers with offsets +/-1, +/-3, +/-5. Its builder reads `v4_4_causal_v2/correct_patching_aggregate.csv`, not the old route-analysis artifact. | `scripts/run_realistic_niah_v4_4_correct_interventions.py`, its parallel runner, `scripts/build_realistic_niah_v4_4_causal_v2_report.py`, and `figures/nonthinking_appendix_effects_20260913/build_figures.py` remain. The old integrated report renderer and its bundle audit are removed. |
| Early additional-task PCA transfer pilot using `item_end_discovery_basis.json` / `.npz` | The pilot uses four discovery and four confirmation seeds and a count-representation transfer analysis. Appendix H, pp. 65-72, reports 20/10-seed task-specific natural behavior, head rankings and ablations (Table 11; Figures 42-44), without this PCA transfer assay. | The kth-record/category input builders, natural generation, task-local head selection, ablation and figure stages remain. The pilot freezer, analyzer, configuration, preflight and dedicated runner stages are removed. |

`realistic/additional_experiments/run.py` retains only the shared `render` and
`summarize_generation` functions imported by the Appendix H stages. Their
function bodies are unchanged. The removed pilot's dedicated tests and the
old integrated-report assertions were removed with those workflows; tests for
retained parsing, scoring and report code remain. Other PCA analyses and
correct-input controls actually used in the paper are retained.

## What has and has not been established

The release now contains the recovered experiment and plotting code, with
explicit commands and stage connections in [WORKFLOWS.md](WORKFLOWS.md) and
[the figure map](../figures/README.md). Source presence and successful CPU
checks do not establish a complete rerun of every GPU experiment. Many
historical figure builders still expect their documented relative artifact
layout; they are source-recovered, not all independently executed during this
review. Hand-edited vector diagrams can also require a diagrams.net export.

Generated outputs can contain local paths and hardware metadata. Audit them
separately before publishing. The source/ZIP identity scan does not certify
future artifacts or anonymous hosting-service metadata.
