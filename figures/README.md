# Final figure sources

This directory restores numerical analyses, measurement capture helpers and plotting code that were stored separately from the two experiment components. Generated tables, cached tensors, finished figures and manuscript text are not included.

The existing subdirectory names preserve cross-imports and identify historical variants. Paths to the experiment components now use `realistic/` and `synthetic/`; figure cross-imports use `figures/`. Manuscript-copy destinations are redirected to the ignored `runs/paper_figures/` tree. These directory substitutions do not change the numerical procedures.

Use [WORKFLOWS.md](../docs/WORKFLOWS.md) for stage order and [COMPLETENESS.md](../docs/COMPLETENESS.md) for release scope and validation limits. The [machine-readable asset map](paper_assets.json) identifies 45 locally referenced manuscript PDF assets, their builder and the strength of the matching evidence. The [input path catalog](input_path_catalog.json) helps locate the tables, caches and selections each builder consumes. It is a static index, not an automatically verified complete producer graph.

## Portable final Figure 2

`python run.py paper-figure2` accepts explicit short-table, request-table, observed-cell, Qwen-supplement and output directories. This builder was actually executed on the available local archived measurements in a newly installed CPU environment. All seven numerical CSV exports matched the original exports within 1e-12. This checks rendering/data plumbing; it does not rerun model inference.

## Source map

Other rows below are recovered sources, not claims of independently rerun results. Some builders run code at import time, so inspect their input contract before importing them. Keep the frozen cohort, parser, discovery selection and control definitions. A dated output directory is a required preceding-stage artifact, not a bundled dataset.

| Manuscript asset | Producing source / final display stage |
|---|---|
| `figures/empirical_accuracy_1x4.pdf` | [empirical_law_1x4/build_figure.py](empirical_law_1x4/build_figure.py) |
| `figures/nonthinking_broad_retrieval.pdf` | [nonthinking_retrieval_horizontal_20260910/build_figure.py](nonthinking_retrieval_horizontal_20260910/build_figure.py) |
| `figures/nonthinking_form_retrieve_consolidate.pdf` | [one_based_layer_head_indices_20260911/build_main_figures.py](one_based_layer_head_indices_20260911/build_main_figures.py) nonthinking_main |
| `figures/native_targeted_retrieval.pdf` | [cot-reasoning/attention/build_figure.py](cot-reasoning/attention/build_figure.py) |
| `figures/native_retrieve_encode_count_loop.pdf` | [cot-reasoning/build_figure.py](cot-reasoning/build_figure.py) |
| `figures/synthetic_section.pdf` | [synthetic_training_dynamics_font_revision_20260911/build_figure.py](synthetic_training_dynamics_font_revision_20260911/build_figure.py) |
| `figures/additional_tasks/01_behavior.pdf` | [additional_tasks_aurora/build_figures.py](additional_tasks_aurora/build_figures.py) |
| `figures/additional_tasks/02_head_scores.pdf` | [one_based_layer_head_indices_20260911/build_figures.py](one_based_layer_head_indices_20260911/build_figures.py) additional |
| `figures/additional_tasks/03_ablation.pdf` | [additional_tasks_aurora/build_figures.py](additional_tasks_aurora/build_figures.py) |
| `figures/empirical_appendix/empirical_relative_noise.pdf` | [nonthinking_relative_noise_20260919/build.py](nonthinking_relative_noise_20260919/build.py) |
| `figures/empirical_appendix/empirical_median_fits.pdf` | [empirical_appendix_revision_20260915/build_figures.py](empirical_appendix_revision_20260915/build_figures.py) |
| `figures/empirical_appendix/empirical_accuracy_12models.pdf` | [empirical_law_12models_4x3_20260919/build_figure.py](empirical_law_12models_4x3_20260919/build_figure.py) |
| `figures/empirical_appendix/empirical_qwen_all_counts.pdf` | [empirical_appendix_revision_20260915/build_figures.py](empirical_appendix_revision_20260915/build_figures.py) |
| `figures/empirical_appendix/empirical_gemma_all_counts.pdf` | [empirical_appendix_revision_20260915/build_figures.py](empirical_appendix_revision_20260915/build_figures.py) |
| `figures/empirical_appendix/empirical_length_candidates.pdf` | [empirical_appendix_revision_20260915/build_figures.py](empirical_appendix_revision_20260915/build_figures.py) |
| `figures/cot_appendix/cot_full_head_scores.pdf` | [enumeration_thinking_style_20260917/build_figures.py](enumeration_thinking_style_20260917/build_figures.py) |
| `figures/cot_appendix/cot_count_readouts.pdf` | [thinking_appendix_style_20260913/build_figures.py](thinking_appendix_style_20260913/build_figures.py) |
| `figures/cot_appendix/cot_count_pca.pdf` | [thinking_appendix_style_20260913/build_figures.py](thinking_appendix_style_20260913/build_figures.py) |
| `figures/cot_appendix/cot_domain_pca.pdf` | [thinking_appendix_style_20260913/build_figures.py](thinking_appendix_style_20260913/build_figures.py) |
| `figures/cot_appendix/cot_current_bank_dose.pdf` | [thinking_appendix_style_20260913/build_figures.py](thinking_appendix_style_20260913/build_figures.py) |
| `figures/cot_appendix/cot_progress_controls.pdf` | [thinking_update_concision_20260915/build_scope_figure.py](thinking_update_concision_20260915/build_scope_figure.py) |
| `figures/cot_appendix/cot_answer_readout_controls.pdf` | [thinking_appendix_style_20260913/build_figures.py](thinking_appendix_style_20260913/build_figures.py) |
| `figures/nonthinking_appendix/nonthinking_full_head_scores.pdf` | [nonthinking_headmap_thinking_style_20260917/build_figure.py](nonthinking_headmap_thinking_style_20260917/build_figure.py) |
| `figures/nonthinking_appendix/nonthinking_count_readouts.pdf` | [nonthinking_ncc_selection_20260913/build_readouts.py](nonthinking_ncc_selection_20260913/build_readouts.py) |
| `figures/nonthinking_appendix/nonthinking_cue_domain_pca.pdf` | [nonthinking_appendix_revision_20260913/build_geometry.py](nonthinking_appendix_revision_20260913/build_geometry.py) |
| `figures/nonthinking_appendix/nonthinking_head_ablation.pdf` | [nonthinking_appendix_effects_20260913/build_figures.py](nonthinking_appendix_effects_20260913/build_figures.py) |
| `figures/nonthinking_appendix/nonthinking_answer_patching.pdf` | [nonthinking_appendix_layout_20260913/build_figures.py](nonthinking_appendix_layout_20260913/build_figures.py) |
| `figures/nonthinking_appendix/nonthinking_answer_function.pdf` | [nonthinking_appendix_effects_20260913/build_figures.py](nonthinking_appendix_effects_20260913/build_figures.py) |
| `figures/nonthinking_appendix/nonthinking_serial_mediation.pdf` | [nonthinking_appendix_effects_20260913/build_figures.py](nonthinking_appendix_effects_20260913/build_figures.py) |
| `figures/nonthinking_appendix/nonthinking_qwen_routing.pdf` | [nonthinking_appendix_effects_20260913/build_figures.py](nonthinking_appendix_effects_20260913/build_figures.py) |
| `figures/nonthinking_appendix/nonthinking_gemma_residual.pdf` | [nonthinking_appendix_effects_20260913/build_figures.py](nonthinking_appendix_effects_20260913/build_figures.py) |
| `figures/enumeration_bullet/enumeration_head_scores.pdf` | [enumeration_default_seed_20260918/build_figures.py](enumeration_default_seed_20260918/build_figures.py) |
| `figures/enumeration_bullet/enumeration_representations.pdf` | [enumeration_thinking_style_20260917/build_figures.py](enumeration_thinking_style_20260917/build_figures.py) |
| `figures/enumeration_bullet/enumeration_pca_bullet.pdf` | [enumeration_bullet_only_20260917/build_figures.py](enumeration_bullet_only_20260917/build_figures.py) |
| `figures/enumeration_bullet/enumeration_retrieve.pdf` | [enumeration_default_seed_20260918/build_figures.py](enumeration_default_seed_20260918/build_figures.py) |
| `figures/enumeration_bullet/enumeration_update.pdf` | [enumeration_default_seed_20260918/build_figures.py](enumeration_default_seed_20260918/build_figures.py) |
| `figures/enumeration_bullet/enumeration_readout.pdf` | [enumeration_default_seed_20260918/build_figures.py](enumeration_default_seed_20260918/build_figures.py) |
| `figures/synthetic_appendix/01_training_and_behavior.pdf` | [synthetic_appendix_evidence_revision_20260908/build_figures.py](synthetic_appendix_evidence_revision_20260908/build_figures.py) |
| `figures/synthetic_appendix/02_retrieval_head_scores.pdf` | [one_based_layer_head_indices_20260911/build_figures.py](one_based_layer_head_indices_20260911/build_figures.py) head_scores |
| `figures/synthetic_appendix/02_retrieval_ablation.pdf` | [synthetic_appendix_evidence_revision_20260908/build_figures.py](synthetic_appendix_evidence_revision_20260908/build_figures.py) |
| `figures/synthetic_appendix/03_count_geometry_3d.pdf` | [paper_ncc_unification_20260913/build_synthetic_geometry.py](paper_ncc_unification_20260913/build_synthetic_geometry.py) |
| `figures/synthetic_appendix/04_count_readability.pdf` | [synthetic_appendix_label_revision_20260908/build_readability.py](synthetic_appendix_label_revision_20260908/build_readability.py) |
| `figures/synthetic_appendix/05_progress_sources.pdf` | [synthetic_appendix_label_revision_20260908/build_sources.py](synthetic_appendix_label_revision_20260908/build_sources.py) |
| `figures/synthetic_appendix/06_counter_interventions.pdf` | [synthetic_appendix_label_revision_20260908/build_counter_triptych.py](synthetic_appendix_label_revision_20260908/build_counter_triptych.py) |
| `figures/synthetic_appendix/08_training_dynamics.pdf` | [paper_ncc_unification_20260913/build_synthetic_dynamics.py](paper_ncc_unification_20260913/build_synthetic_dynamics.py) |

## Shared stages and vector artwork

- Run `empirical_section3_refit/analyze.py` before the empirical appendix renderers; `empirical_appendix_revision_20260915/analyze_length.py` prepares the length-fit comparisons.
- `paper_ncc_unification_20260913/prepare.py` freezes NCC display selections and generates shared coordinate/readout exports. `run_synthetic_dynamics.py` regenerates the fixed-layer dynamics from checkpoints; set the synthetic component on `PYTHONPATH` as described in its module.
- `synthetic_training_dynamics_three-panel/plot_synthetic_section.py` creates the numerical exports checked by the final Figure 7 typography revision.
- `non-thinking/transformer_mechanism_polished.drawio` is an editable template. The Non-thinking main builder creates diagram/panel sources; the outlined composite used by `compose_selectable_pdf.py` requires a diagrams.net PDF export. The binary export is intentionally not bundled.
- Some historical builders assume Times New Roman or Windows font locations and optional PDF composition libraries. The final Figure 2 and earlier empirical report builders were checked here; portability of every historical font/composition branch has not been established.
