# Execution order and artifact contracts

All commands start at the repository root unless a `cd` is shown. `run.py`
changes into the selected component. Use absolute paths for external inputs.
GPU commands below describe the retained protocol; they were not executed as
full experiments during this release review. Generated data and results remain
local. See [COMPLETENESS.md](COMPLETENESS.md) for explicit historical gaps.

## Behavioral grid: generation, inference, merge, analysis

```bash
python run.py haystack --output-dir data/haystacks/paul_graham_full
python run.py behavior-freeze --output-dir runs/paper/behavior/dataset --haystack-dir data/haystacks/paul_graham_full --haystack-corpus-manifest data/haystacks/paul_graham_full/corpus_manifest.json
python run.py realistic scripts/prepare_realistic_niah_v3_1.py --run-root runs/paper/behavior --fresh-dataset
```

The downloader writes a corpus manifest; use its reported filename if the
output layout is changed. The freezer writes `stimuli.jsonl`, `manifest.json`,
`SHA256SUMS`, and `audit_report.json`. The registered grid is 14 counts x
8 lengths x 30 seeds = 3,360 stimuli. Preparation requires a clean committed
checkout and the actual registered tokenizer; it writes `orchestration/`
plans and `prepare_audit.json`. `--fresh-dataset` records new input identity;
omitting it preserves the historical byte-for-byte validation route.

Run each model in `orchestration/formal_bundles.json`, using the appropriate
GPU count and batch settings. For example:

```bash
python run.py realistic scripts/run_realistic_niah_v3_1_model_bundle.py --stimuli runs/paper/behavior/dataset/stimuli.jsonl --run-root runs/paper/behavior --model Qwen3-8B --require-clean-git
```

There are 14 physical model bundles, 48 logical model/mode shards and 161,280
requests. The bundled runner writes each mode under `shards/<task_id>/main/`.
Only after **all** registered bundles are complete, run:

```bash
python run.py realistic scripts/merge_realistic_niah_v3_1_shards.py --run-root runs/paper/behavior
```

The merger checks the plans, dataset audit, commits, IDs, counts, model revisions
and shard manifests. It produces `orchestration/final_shard_audit.json` and
the merged request collections. `behavior-analysis` and the V3.2 empirical-law
analyses consume these completed tables. Do not fabricate a final audit for a
partial model run; use that run's own request file for a separately labeled
partial analysis.

## Extended lengths and the final Figure 2

The long-context freezer/worker and configuration are the `long-context-freeze`
and `long-context` registry entries. The registered long worker requires exactly
two visible H100 GPUs. It is not a generic CPU or arbitrary-GPU command.
The short and long results feed
`analyze_realistic_niah_v3_3_regression_scan.py`, whose inputs include the
request-level tables and the short-range coefficient/freeze files. The final
observed-accuracy plot does not use a fitted empirical-law curve.

The Qwen unscaled-RoPE supplement reuses the matched short/long stimuli and
checks their IDs, passage hashes and prompt hashes against both model runs.
The `--backup-root` layout is:

```text
realistic_niah_v3_1/20260819_formal/
  dataset/stimuli.jsonl
  shards/<model>__<mode>/main/requests.jsonl
realistic_niah_v3_3_long_context/20260906_holdout/
  dataset/stimuli.jsonl
  formal/<model>/worker-<0|1>/main/requests.jsonl
```

Here `<model>` is Qwen3-32B or Gemma4-31B and `<mode>` is direct or
native_thinking. These directories can contain newly generated outputs with
matching protocol/identity; the labels define the archival layout.

```bash
python run.py qwen-yarn-off --root runs/paper/qwen_yarn_off --prepare --backup-root /absolute/path/to/behavior_archive --cache-dir /absolute/path/to/model_cache
CUDA_VISIBLE_DEVICES=0,1 python run.py qwen-yarn-off --root runs/paper/qwen_yarn_off --worker 0
```

Run worker IDs 0, 1, 2 and 3 on suitable visible device pairs (they can be
scheduled separately), then:

```bash
python run.py qwen-yarn-off-summary --root runs/paper/qwen_yarn_off --final
python run.py paper-figure2 --short-tables /absolute/path/to/short_tables --request-tables /absolute/path/to/request_tables --saved-cells-dir /absolute/path/to/observed_cells --qwen-results /absolute/path/to/qwen_yarn_off --output-dir /absolute/path/to/figure2
```

| Figure 2 input role | Required files / producer |
|---|---|
| Short tables | `cell_outcomes.csv.gz` from the V3.2 empirical-law analysis; 2,688 direct/native cells, 30 seeds per cell |
| Request tables | `v3_1_two_model_request_level.csv.gz`, `v3_3_two_model_request_level.csv.gz` from the V3.3 regression input exports; 28,560 archived request outcomes |
| Saved observed cells | `figure2_observed_cells.csv` from `build_niah_empirical_paper_figures.py`; checked against request aggregates, not used as a replacement for them |
| Qwen supplement | `config.json`, `requests.jsonl`, `cell_summary.csv`, four `worker-*/run_manifest.json`; 14,280 requests, all truncations treated as errors |
| Output | `empirical_accuracy_1x4.pdf/.svg/.png`, a full-range companion, seven numerical CSVs, layout audits and `build_manifest.json` |

The two earlier builders now accept relocated input directories:
`build_niah_all_n_length_comparison.py` takes `--linear-dir`, `--log-dir`,
`--holdout-metrics`, `--short-range-metrics`, `--output-dir`;
`build_niah_empirical_paper_figures.py` adds `--short-tables`,
`--length-comparison-dir`, and optional `--preview`. Run the comparison first.
Their fitted-law layouts are earlier report figures, not final Figure 2.

## Non-thinking mechanisms: Figures 3–4 / Appendix D

1. `nonthinking-freeze` creates the controlled V4 `stimuli.jsonl` and audit.
2. `nonthinking` produces baseline generations, capture indexes and attention
   measurements. V4.4-specific experiments use
   `scripts/run_realistic_niah_v4_causal_v2.py`, with stages in this order:
   `prompt-alignment`, `baseline`, `head-rankings`, then the intended `ablation`,
   `prompt-patching`, `answer-patching` or steering stages. The runner requires
   `--run-root`, `--stimuli`, `--model` and the retained causal configuration.
3. For screen/confirmation experiments, run the screen, freeze its selection
   with `--stage select`, then the confirmation using `--selection-json`.
   `confirmation-stats` combines the specified screen and confirmation tables.
4. The full-span head membership producer is executable as:

```bash
python run.py realistic scripts/analyze_realistic_niah_v4_4_full_span_topk.py --run-root /absolute/path/to/causal_v2_run --output-dir /absolute/path/to/full_span_topk
```

It writes `full_span_topk_membership.csv`, seed effects, primary statistics and
`full_span_topk_analysis.json`. The stage-specific root is
`<run-root>/<model>/numeric/causal_v2/`. The later V4.4.2–V4.4.5 interventions
retain distinct configurations and producers; do not relabel a generic V4
ablation as every Appendix D panel. The final figure source map identifies
their consuming builders and literal input paths.

## Native Thinking: Figures 5–6 / Appendix E

For each registered model, an explicit foundation sequence is:

```bash
python run.py thinking generate --model Qwen3-8B --stimuli /absolute/path/to/stimuli.jsonl --output runs/paper/thinking/Qwen3-8B/generations.jsonl
python run.py thinking parse --input runs/paper/thinking/Qwen3-8B/generations.jsonl --output runs/paper/thinking/Qwen3-8B/parsed.jsonl
python run.py thinking capture --model Qwen3-8B --generations runs/paper/thinking/Qwen3-8B/parsed.jsonl --output runs/paper/thinking/Qwen3-8B/capture
python run.py thinking attention --model Qwen3-8B --generations runs/paper/thinking/Qwen3-8B/parsed.jsonl --output runs/paper/thinking/Qwen3-8B/attention.csv
```

Repeat for Gemma4-E4B. The capture stage's saved index feeds `representation
--capture-index ... --output ...`. Final-paper geometry uses the retained
dual-endpoint/cue/domain analyses, and causal panels use their own frozen
plans, site definitions and discovery-selected banks. The final NCC display
selection is implemented in `figures/paper_ncc_unification_20260913/prepare.py`.
The current update-scope display is
`figures/thinking_update_concision_20260915/build_scope_figure.py`.

The Figure 5 single-query capture implementations are also included:
`figures/aurora_attention_pca_concept/extract_seed_attention_remote.py` and
`figures/cot-reasoning/attention/capture/capture_remote.py`. Despite the legacy
names they contain model computations, not remote account/deployment setup.
They consume saved token IDs and registered query sites; generated traces are
not reconstructed from figure coordinates.

## Structured Enumeration: Appendix F

The retained Bash supervisors encode generation, replacement/cohort resolution,
discovery, frozen selection and confirmation. Set `V6_ROOT`, `V6_PYTHON`,
`V6_STIMULI`, `V6_CACHE` and `V6_RUN_ROOT` to current locations. For example,
from a suitable Bash/CUDA environment:

```bash
bash realistic/scripts/supervise_realistic_niah_v6_enumeration.sh bullet Qwen3-8B preflight
bash realistic/scripts/supervise_realistic_niah_v6_enumeration.sh bullet Qwen3-8B discovery-generate
bash realistic/scripts/supervise_realistic_niah_v6_enumeration.sh bullet Qwen3-8B discovery-foundation
```

Subsequent confirmation supervisors require their frozen discovery artifacts;
the default-seed fresh supplementation has its own `prepare_enumeration_*`,
`launch_enumeration_default_seed.py` and restored N10/layer-diagnostic launchers.
These versions are not interchangeable. Final bullet rendering uses
`enumeration_default_seed_20260918`, `enumeration_thinking_style_20260917` and
`enumeration_bullet_only_20260917` under `figures/`; see the per-asset map.

## Synthetic: Figure 7 / Appendix G

The full training sequence is in [REPRODUCING.md](REPRODUCING.md). Use the main
preset, seed 1234, 10,000 steps and the same run directory for every stage.
An example supplemental sequence after training and baseline analyses is:

```bash
python run.py synthetic-alignment --run-dir runs/paper/synthetic_v58 --output runs/paper/synthetic_v58/analysis/v58_alignment_supplement_20260905
python run.py synthetic scripts/run_v58_commit_query.py --run-dir runs/paper/synthetic_v58 --output runs/paper/synthetic_v58/analysis/v58_commit_query_20260905
python run.py synthetic-continuation --run-dir runs/paper/synthetic_v58 --output runs/paper/synthetic_v58/analysis/v58_native_continuation_20260905
```

Alignment freezes `input_registry.csv` (200 discovery / 100 confirmation
prompts) and saves per-mode trials and a cross-mode multiset audit. Preserve
its selected sites and controls for the downstream continuation; it is not a
new training-seed replication. The final fixed-layer NCC training-dynamics
producer is `figures/paper_ncc_unification_20260913/run_synthetic_dynamics.py`,
using the selection exported by `prepare.py`.

The report command now follows the chosen run directory:

```bash
python run.py synthetic-report --run-root runs/paper/synthetic_v58 --output reports/synthetic_v58.html
```

This historical integrated report also requires its clean-NCC, Top-K,
high-power factorial, sufficiency and dynamics supplements. Generic training
outputs alone do not satisfy that full input contract. The final Figure 7
source is `figures/synthetic_training_dynamics_font_revision_20260911/build_figure.py`,
which uses the preceding `synthetic_training_dynamics_three-panel` exports.

## Additional tasks: Appendix H

The final path is task-local selection followed by two explicitly distinct
Broad-scope amendments. It is not the older `additional-freeze` pilot.

| Order | Preparation → model stages → analysis |
|---|---|
| Natural task inputs/outputs | `protocol.py`, `kth_retrieval.py`, `category_full.py`, `category_trial.py`; preserve the task-specific prompt, gold, tokens and endpoint registry |
| Task-local disjoint-first selection | `deployment/prepare_task_local.py` → staged `run_task_local.py --model <model> --stage discover`, then `canary`, `full` → `analyze_task_local.py --root <package>` |
| Native full generated-record spans | `deployment/prepare_native_broad_full_span.py` → `run_native_broad_full_span.py --root <package> --model <model> --stage geometry`, `discovery`, `canary`, `full` → corresponding analyzer |
| Non-thinking target-category Broad | `deployment/prepare_category_target_broad.py` → `run_category_target_broad.py --root <package> --model <model> --stage discovery`, `canary`, `full` → corresponding analyzer |
| Final figures | `figures/additional_tasks_aurora/build_figures.py`; use the per-asset map for later one-based label staging |

These constructors consume the preceding audited plans, generations and
rankings and create isolated code/config packages. Their historical directory
names are input contracts. Read the restored technical protocol documents for
the 20/10 discovery/confirmation seeds, model-specific K grids, minimum-overlap
random controls and exploratory peak-K qualification. Do not manufacture
missing audits or replace frozen token sequences with retokenized text.

For the **historical pilot only**, explicit prerequisite paths are now possible:

```bash
python run.py additional-freeze --source /absolute/path/to/stimuli.jsonl --config additional_experiments/configs/pilot_v1.json --head-membership /absolute/path/to/full_span_topk_membership.csv --native-basis-root /absolute/path/to/native_bases --output runs/pilot/frozen
```

The basis root must contain `<model>/item_end_discovery_basis.json` and `.npz`.
Its historical export source remains unresolved; see `COMPLETENESS.md`.
