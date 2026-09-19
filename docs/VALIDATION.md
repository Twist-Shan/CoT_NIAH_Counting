# Validation

These checks establish software behavior and the explicitly stated report
regeneration results. They are not a full GPU rerun or an independent
confirmation of the paper's measurements.

| Check | Observed result |
|---|---|
| Clean CPU installation | Passed in a newly created Windows AMD64 / Python 3.12.7 virtual environment without system packages; CPU PyTorch followed by `requirements/cpu.txt`; `pip check` reported no broken requirements |
| CPU smoke | Passed: parser/corpus hashes, saved/main-preset agreement, trace serialization, SDPA/explicit-attention agreement, causal mask, forward/backward and optimizer step in both synthetic modes |
| Selected realistic tests | **182 passed, 2 skipped** |
| Selected synthetic tests | **39 passed** |
| Registered interfaces | All 20 entries checked: 19 help commands plus the included-corpus verification |
| Final Figure 2 | Rebuilt PDF/SVG/PNG and seven CSVs from explicit external input paths; all seven numerical exports match the original exports with relative/absolute tolerance 1e-12; built-in layout checks passed and PNG inspected |
| Earlier empirical report figures | Both length-comparison and fitted-law figure builders executed successfully using relocated CSV/JSON inputs |
| Synthetic integrated report | Rebuilt successfully using an external `--run-root`, output and asset directory |
| Local Enumeration Bash orchestration | Eight scripts passed `bash -n`; no GPU execution |
| Source checks | Python/JSON parsing, registered entry files, reader-facing links and credential/private-path scans |

The two skips remain explicit: a CUDA-only test and the check requiring the
original frozen realistic stimulus JSONL. Neither is counted as passing.
The restored Qwen layer-diagnostic tests are now included; so are missing-input
checks for the additional pilot and provenance guards for fresh V3.1 preparation.
The latter mock the expensive grid auditor to test routing/rejection behavior;
they do not constitute a newly generated full realistic dataset validation.

```bash
python tools/smoke_test.py
python tools/check_tests.py
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
100 figure/analysis/capture helper files, and an editable diagram template.
Paths and provenance labels were sanitized; paper measurements, selection
rules and registered model/experiment settings were not replaced with new
results. Portable CLI options were added for the affected input contracts.

Figure 2 and report checks consumed local archived inputs solely for validation;
those inputs and generated exports are not distributed. Other restored figure
families were not all executed. Model downloads, real GPU inference, training,
all interventions and full end-to-end paper reproduction remain unverified
in this packaging environment. Known historical producer gaps are recorded in
[COMPLETENESS.md](COMPLETENESS.md).

The anonymous preparation includes a separate scan for known manuscript-author
and account identifiers, credentials and private paths in the release and its
Git objects. Legitimate third-party attribution is retained. The checksum
manifest and deterministic ZIP describe the final delivered source snapshot;
future result uploads and hosting-service metadata require their own review.
