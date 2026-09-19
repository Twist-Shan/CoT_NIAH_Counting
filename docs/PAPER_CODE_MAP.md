# Paper-to-code map

This map was prepared against the supplied 72-page manuscript. Section labels
refer to that manuscript. Paths are relative to the repository root.

| Paper content | Code and entry points | Configuration / input contract |
|---|---|---|
| Section 2; Figure 2A/B; Table 1; Appendix C.1 | `realistic/src/realistic_niah_v3_1/`; `behavior-freeze`, `behavior`, `behavior-analysis` | `realistic/configs/realistic_niah_v3_1.json`; 14 counts × 8 lengths × 30 paired seeds; model registry in `realistic/src/realistic_niah_v3/spec.py` |
| Figure 2C/D; Appendix C.4–C.5 | `realistic/src/realistic_niah_v3_3_long_context/`; `long-context-freeze`, `long-context`; `realistic/scripts/analyze_realistic_niah_v3_3_regression_scan.py` | `realistic/configs/realistic_niah_v3_3_long_context.json`; original worker/shard interfaces retained |
| Appendix C.2–C.3, count/length fits | `realistic/scripts/analyze_nonthinking_noise_by_count.py`; `analyze_nonthinking_noise_alternatives.py`; `analyze_realistic_niah_v3_2_empirical_laws.py`; `analyze_realistic_niah_v3_2_n_fixed.py` | Existing aggregate/request tables; fit and freeze JSON files in `realistic/configs/` |
| Section 3; Figures 3–4; Appendix D | `realistic/src/realistic_niah_v4/`, `realistic_niah_v4_4_2/`, `realistic_niah_v4_4_3/`, `realistic_niah_v4_4_4/`, `realistic_niah_v4_4_5/`; `nonthinking`; associated `run_`, `analyze_`, `select_` scripts | `realistic/configs/realistic_niah_v4.json` and `realistic_niah_v4_4_*.json`; Qwen3-8B/Gemma4-E4B; discovery/confirmation seeds preserved |
| Section 4; Figures 5–6; Appendix E | `realistic/src/realistic_niah_v5/`; `thinking`; `realistic/scripts/analyze_realistic_niah_v5_*.py`, `run_realistic_niah_v5_*.py` | `realistic/configs/realistic_niah_v5*.json`; accepted trace sites, cohort rules, frozen head selections, and causal controls |
| Appendix F, structured enumeration | `realistic/src/realistic_niah_v6/`; `enumeration`; `realistic/scripts/run_enumeration_*.py` / `analyze_enumeration_*.py` | `realistic/configs/realistic_niah_v6_enumeration_index.json`, `realistic_niah_v6_enumeration_bullet.json`, and `enumeration_*.json` supplements |
| Section 5; Figure 7; Appendix G.1–G.2 | `synthetic/src/synthetic_counting_v58/`; shared kernels in `synthetic/src/synthetic_counting_v20/`; `synthetic-train` | `synthetic/configs/paper_v58_saved_config.json`; four layers, eight heads, width 512; 256 characters, counts 1–10, seed 1234, 10,000 steps |
| Appendix G.3–G.4, geometry/causal/dynamics | `synthetic/scripts/run_v58_alignment_supplement.py`, `run_v58_commit_query.py`, `run_v58_native_continuation.py`, `run_v58_top1to8_aligned.py`, `export_v58_geometry_cloud.py`; `synthetic/src/synthetic_counting_v20/` analysis modules | Trained checkpoints, saved manifests, and outputs from preceding stages; confirmation and control policies retained in v58 modules/scripts |
| Appendix H, kth retrieval and category count | `realistic/additional_experiments/protocol.py`, `kth_retrieval.py`, `category_full.py`, `category_target_broad.py`, `native_broad_full_span.py`, `local_selection.py`, `task_scoring.py` | Task-specific frozen cases, model outputs, head banks, and source-basis inputs |
| Final Figure 2; Qwen YaRN-off supplement | `realistic/scripts/qwen_yarn_off/full_run.py`, `summarize.py`; `figures/empirical_law_1x4/build_figure.py`; registry `qwen-yarn-off`, `qwen-yarn-off-summary`, `paper-figure2` | Explicit short/long request tables and audited YaRN-off workers; observed statistics, not fitted empirical-law curves |
| Final figure sources | `figures/`; [45-asset source map](../figures/README.md), `figures/paper_assets.json`, `figures/input_path_catalog.json` | Recovered final display sources and shared helpers; only the final Figure 2 pipeline was executed during this review |
| Earlier report/figure preparation | `realistic/scripts/build_niah_empirical_front_figures.py`, `build_niah_empirical_paper_figures.py`, `build_v5_native_thinking_report_final.py`, `build_realistic_niah_v6_enumeration_report.py`; `synthetic/scripts/build_v58_synthetic_report.py` | Historical report variants consume completed-run tables/caches; their inclusion is not evidence of final manuscript regeneration |

The paper compares broad retrieval against successive targeted retrieval. The
score implementations, model revisions, parser algorithms, seeds, and statistical
controls are retained in the source. A module's presence does not establish that
every exploratory analysis it contains was used in the paper.

## Why earlier versions remain

The versioned packages share code. For example, v58 delegates training and
analysis to v20, uses v35/v44 functionality, and its original tests compare
against v57. Supplemental scripts also reuse earlier packages. These packages
remain in their original import locations. Remote launchers, notebooks with
account setup, caches, and Git history are excluded from the review workflow.

The file checksum manifest identifies the exact anonymous source snapshot.
The private preparation audit, kept outside this repository, records which
source files were copied and which identifying metadata was changed.

See [WORKFLOWS.md](WORKFLOWS.md) for execution order and
[COMPLETENESS.md](COMPLETENESS.md) for the bounded source-coverage assessment.
