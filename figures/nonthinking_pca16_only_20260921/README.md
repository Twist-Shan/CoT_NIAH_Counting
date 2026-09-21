# Non-thinking count readouts

This workflow measures running-index and final-count readability in Non-thinking mode using 16-dimensional PCA.

Running-index preprocessing is unwhitened randomized PCA (seed 20260806), then coordinate standardization; NCC has no shrinkage. Final-count preprocessing is standardization, then unwhitened PCA (seed 442); NCC shrinkage is 0.1. Five folds group inputs by seed. Layers maximize pooled out-of-fold NCC accuracy, breaking ties by logistic accuracy and earlier depth. The held-out seeds remain separate from fitting and selection.

The resulting selected running-index layers are Qwen L16 and Gemma L14, with held-out NCC accuracy 43% and 38%. Final-count CV selects Qwen L25 (67.5%) and Gemma L36 (57%). These are two different evaluation sets and metrics.

## Inputs and execution

Install the verified numerical environment with `python -m pip install -r requirements/nonthinking-pca16.txt`. Run the following from the repository root. The input variables denote external, completed-run artifacts; cached activations and generated reports are not bundled.

`CAPTURES` must contain:

- `runs/run_20260731_v4_numeric_presentation_v3/<model>/numeric/representation/capture/`: running-index states and `capture_index.jsonl`.
- `runs/v4_4_counter_channel_20260806/packed/layers/<model>__answer_query__Lxx.npz`: final-count classifier inputs, with `states`, `count` and `seed` arrays.
- `runs/v4_4_geometry_comparison_20260816/<model>/numeric/representation/answer_query_all_layers_v1/`: relative-noise inputs. Classification and relative noise use their respective input sets.

Models are `Qwen3-8B` and `Gemma4-E4B`. Cache keys call the fitting and held-out splits `discovery` and `confirmation`; the loader validates those keys without changing the data.

```bash
python figures/nonthinking_pca16_only_20260921/analyze.py --source "$CAPTURES" --folds figures/nonthinking_pca16_only_20260921/folds.json --output runs/nonthinking_pca16_running
python figures/nonthinking_pca16_only_20260921/update_answer.py --source "$CAPTURES" --previous runs/nonthinking_pca16_running --folds figures/nonthinking_pca16_only_20260921/folds.json --expected figures/nonthinking_pca16_only_20260921/expected32.json --output runs/nonthinking_pca16
python figures/nonthinking_pca16_only_20260921/verify.py --results runs/nonthinking_pca16
python figures/nonthinking_pca16_only_20260921/build.py --results runs/nonthinking_pca16 --legacy-figures "$LEGACY_FIGURES" --cue-report "$CUE_REPORT" --output runs/nonthinking_pca16_figures
```

`LEGACY_FIGURES/nonthinking_ncc_selection_20260913/` must contain `selection.json`, `pca_plot_data.csv`, `geometry_manifest.json` and `domain_report_payload.json`. `CUE_REPORT` is the all-layer cue report containing the `const PROMPT_GEOM=` payload. The cue/domain figure shows city, flower and animal in the bottom row, with shared projections and display settings within each panel.

The first two stages deliberately use separate output directories. The answer stage accepts only a new output directory and copies the running-index selections. Reference NCC checks in `expected32.json` validate the classifier inputs; logistic differences are recorded because the iteration limit can produce small platform-dependent differences. `verify.py` recomputes selected-layer CV and running-index held-out scores.

The builder exports the readout scan, cue/domain geometry and relative-noise figures. Relative noise uses the final-count selected layers and checks 20 literal bootstrap resamples against the vectorized implementation.

## Validation

Selected-layer CV and held-out scores were verified from the input states. All three figure builders were executed, and their numerical exports match the manuscript results. Output audits record source hashes, seed identities, layer selections and measured scores.
