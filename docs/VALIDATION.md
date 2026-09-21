# Validation

These checks establish software behavior and the explicitly stated report
regeneration results. They are not a full GPU rerun or an independent
confirmation of the paper's measurements.

| Check | Observed result |
|---|---|
| Clean CPU installation | Passed in a newly created Windows AMD64 / Python 3.12.7 virtual environment without system packages; CPU PyTorch followed by `requirements/cpu.txt`; `pip check` reported no broken requirements |
| CPU smoke | Passed: parser/corpus hashes, saved/main-preset agreement, trace serialization, SDPA/explicit-attention agreement, causal mask, forward/backward and optimizer step in both synthetic modes |
| Selected realistic tests | **196 passed, 2 skipped** |
| Selected synthetic tests | **39 passed** |
| Retained V4.4 report tests | **5 passed** after removing assertions for the excluded integrated report |
| Registered interfaces | Original 20 entries checked; the two new Appendix H preparation/figure interfaces also pass help checks |
| Final Figure 2 | Rebuilt PDF/SVG/PNG and seven CSVs from explicit external input paths; all seven numerical exports match the original exports with relative/absolute tolerance 1e-12; built-in layout checks passed and PNG inspected |
| Fresh-result Figure 2 | A changed-outcome fixture with negative Thinking gains renders using DejaVu Serif; explicit reference verification rejects it; inconsistent accuracy/count fields are still rejected |
| Appendix H figures | Three paper figures rendered from archived inputs with explicit paths; all three plot CSVs match the original exports at 1e-12; built-in layout checks pass |
| Appendix H fresh preparation | CPU character-tokenizer fixture: 2,400 trajectories produce four 600-row plans, natural tables and a portable hashed package; all 2,400 anchor plans match the original preparation algorithm on that fixture; no model inference |
| Figure dependencies | Static inventory: 46 mapped assets, 104 figure Python files, 12 explicit helper/artwork/config files and available imported packages; not a complete dynamic dependency graph |
| Recovered helpers / PDF tools | Three consuming synthetic builders really import and their PCA helpers render a fixture; pypdf composes two PDF pages; restored overview XML parses |
| Earlier empirical report figures | Both length-comparison and fitted-law figure builders executed successfully using relocated CSV/JSON inputs |
| Synthetic integrated report | Rebuilt successfully using an external `--run-root`, output and asset directory |
| Local Enumeration Bash orchestration | Eight scripts passed `bash -n`; no GPU execution |
| Source checks | Python/JSON parsing, registered entry files, reader-facing links and credential/private-path scans |

The two skips remain explicit: a CUDA-only test and the check requiring the
original frozen realistic stimulus JSONL. Neither is counted as passing.
The restored Qwen layer-diagnostic tests are included, together with provenance
guards for fresh V3.1 preparation and the retained additional-task checks.
The latter mock the expensive grid auditor to test routing/rejection behavior;
they do not constitute a newly generated full realistic dataset validation.
The current total is 240 passing tests and two skips, including the five
retained report tests. Six pilot-only tests from the previous selection were
removed with their corresponding workflow.

The 20 added selected tests exercise task-local discovery/control rules,
original-token alignment and the fresh input bridge. The new complete-package
fixture uses synthetic text and a character tokenizer. It tests construction
and provenance, not registered model-tokenizer compatibility or GPU attention.
The fresh runner's Broad score follows Appendix H's declared epsilon convention;
the historical aligned-transfer normalization variant remains explicit in
[WORKFLOWS.md](WORKFLOWS.md).

```bash
python tools/smoke_test.py
python tools/check_tests.py
python tools/check_figure_dependencies.py
python -m pytest -q -p no:cacheprovider realistic/tests/test_realistic_niah_v4_4_report.py
python tools/validate_repo.py --check-manifest
```

`tools/check_tests.py` lists the exact selected test files and runs the two
components in separate processes. Artifact-dependent and old infrastructure
tests elsewhere are not included in the stated pass counts.

## Observed clean CPU environment

| Component | Version |
|---|---|
| Python | 3.12.7 |
| PyTorch | 2.14.0+cpu |
| NumPy | 1.26.4 |
| pandas | 3.0.6 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.1 |
| statsmodels | 0.15.0 |
| Matplotlib | 3.11.2 |
| Transformers | 5.17.0 |
| Plotly | 7.1.0 |
| pytest | 9.1.1 |

The full observed package set is in
[`cpu-windows-py312.lock.txt`](../requirements/cpu-windows-py312.lock.txt).
The dependency installation used the declared ranges, then exported this
resolved lock. It is a tested CPU environment record, not a Linux/CUDA lockfile
or a claim about GPU model-loading compatibility. The registered inference
and mechanistic requirement files remain separate.

## Scope of changes and limits

Recovered sources include two Enumeration launchers, the Qwen YaRN-off
experiment, three technical protocols required by package construction,
104 figure/analysis/capture/helper modules, editable overview artwork/settings,
and the Non-thinking diagram template.
Paths and provenance labels were sanitized; paper measurements, selection
rules and registered model/experiment settings were not replaced with new
results. Portable CLI options were added for the affected input contracts.

Figure 2, Appendix H figures and report checks consumed local archived inputs solely for validation;
those inputs and generated exports are not distributed. Other restored figure
families were not all executed. Model downloads, real GPU inference, training,
all interventions and full end-to-end paper reproduction remain unverified
in this packaging environment. The two excluded auxiliary workflows and the
retained paper evidence are documented in [COMPLETENESS.md](COMPLETENESS.md).

The anonymous preparation includes a separate scan for known manuscript-author
and account identifiers, credentials and private paths in the release and its
Git objects. Legitimate third-party attribution is retained. The checksum
manifest and deterministic ZIP describe the final delivered source snapshot;
future result uploads and hosting-service metadata require their own review.
## Non-thinking PCA16 validation

The current source is `figures/nonthinking_pca16_only_20260921/`. Its selected-layer verification reproduced running-index CV/held-out scores and packed final-count CV scores. The released builder executed on the archived inputs: five numerical CSV exports and the relative-noise summary matched the manuscript results exactly. All three registry entries passed `--help` in the verified numerical environment.

Source syntax, input links and release-content checks passed. A broader figure dependency inventory in this numerical environment reports missing optional packages for unrelated GPU/historical PDF scripts (`torch`, `tokenizers`, `pypdf`, `reportlab`, `pypdfium2`); it is not a full-repository execution pass. The PCA16 pipeline itself ran successfully. Use an isolated environment with `requirements/nonthinking-pca16.txt`; the machine's unrelated user-site NumPy 2 installation is incompatible with the verified NumPy 1 environment.
