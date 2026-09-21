# Non-thinking PCA16 update

This is the current Non-thinking running-index and final-count readout workflow. It changes the former 32-dimensional readout PCA to 16 dimensions while preserving the original input states, seed folds, preprocessing order, whitening, classifiers and layer-selection rule. Earlier dated scripts and frozen configurations describe historical runs and remain unchanged.

Running-index preprocessing is unwhitened randomized PCA (seed 20260806), then coordinate standardization; NCC has no shrinkage. Final-count preprocessing is standardization, then unwhitened PCA (seed 442); NCC shrinkage is 0.1. Five folds group inputs by seed. Layers maximize pooled out-of-fold NCC accuracy, breaking ties by logistic accuracy and earlier depth. The held-out seeds remain separate from fitting and selection.

The resulting selected running-index layers are Qwen L16 and Gemma L14, with held-out NCC accuracy 43% and 38%. Final-count CV selects Qwen L25 (67.5%) and Gemma L36 (57%). These are two different evaluation sets and metrics. Reusing the original held-out inputs does not make this an independent replication.

## Inputs and execution

Install the verified numerical environment with `python -m pip install -r requirements/nonthinking-pca16.txt`. Run the following from the repository root. The input variables denote external, completed-run artifacts; cached activations and generated reports are not bundled.

`CAPTURES` must contain:

- `runs/run_20260731_v4_numeric_presentation_v3/<model>/numeric/representation/capture/`: original running-index states and `capture_index.jsonl`.
- `runs/v4_4_counter_channel_20260806/packed/layers/<model>__answer_query__Lxx.npz`: original final-count classifier inputs, with `states`, `count` and `seed` arrays.
- `runs/v4_4_geometry_comparison_20260816/<model>/numeric/representation/answer_query_all_layers_v1/`: unchanged relative-noise inputs. This later capture must not replace the original packed classifier inputs.

Models are `Qwen3-8B` and `Gemma4-E4B`. Historical cache keys still call the fitting and held-out splits `discovery` and `confirmation`; the loader validates those keys without changing the data.

```bash
python figures/nonthinking_pca16_only_20260921/analyze.py --source "$CAPTURES" --folds figures/nonthinking_pca16_only_20260921/folds.json --output runs/nonthinking_pca16_running
python figures/nonthinking_pca16_only_20260921/update_answer.py --source "$CAPTURES" --previous runs/nonthinking_pca16_running --folds figures/nonthinking_pca16_only_20260921/folds.json --expected figures/nonthinking_pca16_only_20260921/expected32.json --output runs/nonthinking_pca16
python figures/nonthinking_pca16_only_20260921/verify.py --results runs/nonthinking_pca16
python figures/nonthinking_pca16_only_20260921/build.py --results runs/nonthinking_pca16 --legacy-figures "$LEGACY_FIGURES" --cue-report "$CUE_REPORT" --output runs/nonthinking_pca16_figures
```

`LEGACY_FIGURES/nonthinking_ncc_selection_20260913/` must contain `selection.json`, `pca_plot_data.csv`, `geometry_manifest.json` and `domain_report_payload.json`. `CUE_REPORT` is the original all-layer cue report containing the `const PROMPT_GEOM=` payload. These are the existing inputs to the cue/domain figure; no domain measurement is recomputed or omitted. City, flower and animal remain in the bottom row. The existing renderer, palette and camera are reused.

The first two stages deliberately use separate output directories. The answer stage accepts only a new output directory and copies the running-index selections. Baseline NCC checks in `expected32.json` guard against accidentally substituting the later capture; logistic baseline differences are recorded because the original iteration limit can produce small platform-dependent differences. `verify.py` recomputes selected-layer CV and running-index held-out scores.

The builder exports the readout scan, cue/domain geometry and relative-noise figures. Relative noise uses the new final-count selected layers, retains its original measurement and bootstrap procedure, and checks 20 literal bootstrap resamples against the vectorized implementation. Display PCA3/PCA6, domain-transfer PCA16, Thinking results, Figure 1C and the separate historical PCA32-ridge steering intervention are unchanged.

## Validation

The released selected-layer analysis was executed on the archived inputs and reproduced the preceding PCA16 results. All three released figure builders were executed on the actual inputs; their numerical exports were compared with the manuscript update. The historical PCA32 NCC/logistic values were reproduced locally before the dimension change. No model inference was rerun. Use the output audits for source hashes, seed identities, selection and measured scores.
