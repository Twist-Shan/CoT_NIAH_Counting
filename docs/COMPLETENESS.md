# Source coverage and remaining historical gaps

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
| Additional pilot input paths | Added `--head-membership` and `--native-basis-root`; missing prerequisites are detected before creating the output directory. Restored the three technical protocol documents used by later task-package constructors. |
| Report paths tied to historical output folders | Parameterized the synthetic report and the two earlier empirical report/figure builders. Final Figure 2 is a separate builder; the earlier fitted-law figures are not its replacement. |
| Original-input hashes versus fresh generation | Historical hash checks remain unchanged. V3.1 preparation offers an explicit `--fresh-dataset` route that reruns the registered grid/tokenizer audit and records new hashes and a fresh-replication identity. |
| Environment validation | A new isolated CPU environment was installed from the declared requirements; installation, dependency consistency, selected tests and actual report/figure generation were checked. See `VALIDATION.md`. |
| Minor legacy directory label and stale hosting wording | Removed the label and clarified the difference between the prepared ZIP and hosting-service downloads. |

## Historical dependencies that are not claimed to be recovered

1. **Correct-only route report component.** The integrated V4.4 HTML report
   consumes `correct_state_route_analysis.json` and related summaries. Its
   dedicated historical producer was not located. The retained audit still
   records this gap. The archived estimand uses pairs where both models are
   initially correct at counts 1, 2, 3; it must not be substituted for the
   distinct main-paper screen or later correct-input transfer experiment.
   No direct reference to this artifact was found in the restored final
   figure builders. This is a qualified source-tracing observation, not proof
   that every indirect dependency has been ruled out.
2. **Legacy pilot PCA export.** The original additional-task pilot expects
   `item_end_discovery_basis.json` and `.npz`. Their exact export routine was
   not located. Generic V5 `representation` does not produce those filenames.
   The pilot is labeled historical in the registry. The later Appendix H
   task-local selection, target-category Broad and full-span Native Broad
   implementations have their own retained preparation/run/analysis code;
   do not describe the pilot as that final pathway.

These limitations are not resolved merely by omitting the corresponding
artifacts. They are recorded so that archived HTML report variants are not
mistaken for a verified end-to-end paper workflow.

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
