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
`--length-comparison-dir`. Run the comparison first.
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

The final Figure 7
source is `figures/synthetic_training_dynamics_font_revision_20260911/build_figure.py`,
which uses the preceding `synthetic_training_dynamics_three-panel` exports.

## Additional tasks: Appendix H

The fresh route below starts from V4.4 count-10 stimuli for seeds 1234-1263.
Commands here invoke scripts directly, so relative paths are relative to the
repository root. The two `for` loops use Bash syntax; in PowerShell, run their
inner commands once per listed model/stage or use equivalent `foreach` loops.

```bash
python realistic/additional_experiments/kth_retrieval.py --freeze-source /absolute/path/to/stimuli.jsonl --frozen runs/paper/additional/kth/frozen
python realistic/additional_experiments/category_full.py --freeze-source /absolute/path/to/stimuli.jsonl --frozen runs/paper/additional/category/frozen
for model in Qwen3-8B Gemma4-E4B; do
  python realistic/additional_experiments/kth_retrieval.py --frozen runs/paper/additional/kth/frozen --output runs/paper/additional/kth/generations --model "$model" --cache-dir /absolute/path/to/hf_cache --natural-only
  python realistic/additional_experiments/category_full.py --frozen runs/paper/additional/category/frozen --output runs/paper/additional/category/generations --model "$model" --cache-dir /absolute/path/to/hf_cache
done
python realistic/additional_experiments/deployment/prepare_fresh_task_local.py --kth-frozen runs/paper/additional/kth/frozen --kth-generations runs/paper/additional/kth/generations --category-frozen runs/paper/additional/category/frozen --category-generations runs/paper/additional/category/generations --cache-dir /absolute/path/to/hf_cache --output runs/paper/additional/package
for stage in discover canary full; do
  for model in Qwen3-8B Gemma4-E4B; do
    python runs/paper/additional/package/run_task_local.py --model "$model" --stage "$stage" --cache-dir /absolute/path/to/hf_cache
  done
done
python runs/paper/additional/package/analyze_task_local.py --root runs/paper/additional/package
python figures/additional_tasks_aurora/build_figures.py --task-local-root runs/paper/additional/package --output-dir runs/paper/additional/figures
```

| Producer | Required output / next consumer |
|---|---|
| Task freezers | 300 cases and 600 frozen user prompts per task, plus file hashes |
| Natural generation | 600 captures per task/model under `<generations>/<model>/captures/<mode>/<case>/`; each saves the exact prompt and original generated token IDs |
| `prepare_fresh_task_local.py` (CPU; cached registered tokenizers) | 2,400 audited trajectories, four 600-row plan files, original-token registry, natural accuracy CSVs, `input_audit.json`, hashed `protocol.json`, portable input copies and stage scripts |
| `discover` (GPU) | Fresh Broad attention and Targeted attention on discovery seeds only; frozen global Top-K banks and three layer-matched minimum-overlap random controls |
| `canary`, then `full` (GPU) | Both model canaries must pass before full intervention; unavailable sites are recorded, with the same eligible panel at every K |
| `analyze_task_local.py` | Rechecks hashes, banks, controls and scores; exports accuracy/coverage/diagnostic/statistical CSVs and `analysis/audit.json` |
| `additional_tasks_aurora/build_figures.py` (CPU) | Figures 42-44 as PDF/SVG/PNG, numerical plot tables, layout audits and input hashes; head/layer labels are already one-based |

The preparation route verifies the full task/seed grid, natural-run completion,
frozen prompt identity, registered model revisions, chat templates, original
token decoding and input hashes. It chooses the retained middle eligible
transition independently of correctness, using seeds 1234-1253 for discovery
and 1254-1263 for confirmation. It does not read any historical Top-K package,
attention files, head banks or intervention results. Missing sites remain
explicit; inadequate discovery support stops selection rather than inventing
a bank. Expected result counts follow the new plans instead of requiring the
historical 6,739 points. Reduced seed support is reported as descriptive in the
statistical output. Match the discovery Python version during analysis, or use
the retained `--discovery-sum python310-sequential` option for a Python 3.10 bank.

The fresh Broad score uses the epsilon convention stated in Appendix H and
implemented in `kth_retrieval.broad`: epsilon and the zero-mass threshold are
1e-12. This is recorded in the protocol. The earlier aligned-transfer runner
uses exact normalization for nonzero mass; its historical source and saved
bank checks remain separate. Fresh runs do not claim historical byte identity.

The package copies natural prompts/generations so its plans are relocatable.
It can be large and belongs in the ignored run directory. Its `launch.sh` uses
`python` from the active environment, overridable with `--python` at preparation
or `PYTHON` at launch. After moving it to a GPU machine, the direct stage CLI's
`--cache-dir` overrides the recorded model cache. Preparation requires a new
output directory; stage restarts verify already completed file hashes.

`prepare_task_local.py` remains the separate historical-package constructor,
with explicit previous-package, alignment-audit, output and Python options.
`prepare_native_broad_full_span.py` and `prepare_category_target_broad.py` retain
historical supplementary scope variants. They are not prerequisites for the
three Appendix H figures: those show Non-thinking Broad over all ten source
records and Thinking Targeted next-entity ablation. The optional category-scope
figure requires `--category-amendment-root`; it is not generated by default.
The earlier PCA transfer pilot remains excluded; see [COMPLETENESS.md](COMPLETENESS.md).


## Geometry data without HTML reports

The full-panel geometry runner retains its capture audits, dual-endpoint analysis
and band analysis. Set `--output reports/NiaH_Geometry_Comparison.json` and an
explicit `--manifest` path; the JSON contains the `DUAL` plotting payload using
the original discovery-fitted coordinate calculation. Report-only optional
inputs are no longer used for this export.

For the cue-removal appendix, run
`realistic/scripts/analyze_realistic_niah_v4_4_2_counter_geometry.py` with the
original captured vectors and `--output-dir realistic/outputs/v4_4_2_counter_geometry`
from the repository root. The appendix extractor reads its
`counter_geometry_data.json` directly. Existing archived HTML inputs remain
readable for compatibility. These are numerical export changes; they do not
replace the required GPU captures or alter the registered cohorts.


The final Non-thinking PCA16 builder accepts `--cue-data` with the original
`prompt_counter_geometry_data.json` emitted by
`realistic/scripts/analyze_realistic_niah_v4_4_2_prompt_counter_geometry.py`.
`--cue-report` remains an alias for archived HTML inputs. The native main-figure
data preparer hashes and checks its measurement inputs directly; an HTML report
is no longer a prerequisite. Historical presentation helpers outside the final
figure workflow can still consume archived HTML, but their narrative generators
are not part of this source release.
