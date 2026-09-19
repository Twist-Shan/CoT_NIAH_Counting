# Validation

The following checks were performed on the anonymous source snapshot. These
results establish the stated software checks, not a rerun of the paper's GPU
experiments or a confirmation of its numerical results.

| Check | Result |
|---|---|
| CPU smoke test | Passed: count parsing, frozen parser and corpus checksums, exact saved/main-preset agreement, trace serialization, SDPA/explicit-attention agreement, causal masking, forward/backward and optimizer step in both modes |
| Selected realistic regression suite | **166 passed, 2 skipped** |
| Selected synthetic regression suite | **39 passed** |
| Registered CLI entry points | All 15 experiment help commands passed; the corpus command verified the included file; Enumeration `print-suite` and the registry listing passed |
| Local Enumeration Bash orchestration | Eight scripts passed `bash -n`; no GPU execution was performed |
| Source audit | Python/JSON syntax, registered entry-point files, reader-facing document links, credential/private-path patterns passed |
| Preparation-time identity scan | No remaining matches for the supplied manuscript's author names, project-associated account names, or identified private login addresses |
| Original source integrity | All initially inventoried source files retained their original byte hashes in the two source directories |

The two skips are explicit: one CUDA-only test and the integration check for the
original frozen realistic stimulus JSONL, which is not included. That integration
check still verifies its original checksum and cohort sizes when the artifact
is restored. It is not counted as passing.

Run the selected original suites with:

```bash
python tools/check_tests.py
```

This command lists the exact test files in its source and isolates the two
components in separate processes. The complete historical test directories also
contain artifact-dependent, report-dependent, and old infrastructure tests; the
counts above apply only to the selected suites.

## Reference CPU environment

The checks used an existing Python environment on Windows AMD64, not a newly
installed Linux/CUDA environment:

| Dependency | Observed version |
|---|---|
| Python | 3.12.7 |
| PyTorch | 2.13.0+cpu |
| NumPy | 1.26.4 |
| pandas | 3.0.3 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.0 |
| Transformers | 5.13.1 |
| Plotly | 6.6.0 |
| pytest | 9.1.1 |

The original registered GPU requirement pins remain in separate files and were
not replaced with these CPU versions. A clean dependency installation, real
checkpoint loading, vLLM inference, full training, intervention runs, and final
manuscript figure regeneration remain outside this verification.

## Packaging changes and limits

The scientific source layout, model revisions, seeds, parser algorithms, and
experiment settings were preserved. Snapshot edits remove identifying metadata
and map private absolute artifact locations to relative locations. The CPU test
for V5 source immutability now compares file hashes directly, so it also works
in a downloaded archive without Git history.

Notebook/account bootstrap code, remote deployment state, cached tensors,
weights, original logs and Git history were excluded. Local Enumeration
orchestration required by the source tests was retained and syntax-checked.
The exact anonymous files are recorded in `MANIFEST.sha256`.
