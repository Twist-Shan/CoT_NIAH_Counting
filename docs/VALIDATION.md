# Validation after source cleanup

The 2026-09-23 checks below cover the cleaned source and CPU behavior. They do
not establish a complete GPU rerun or independent confirmation of the paper's
measurements. Generated experiment inputs and results are not bundled.

| Check | Observed result |
|---|---|
| Selected realistic tests | 309 passed, 4 skipped |
| Selected synthetic tests | 38 passed |
| All test modules collect | 1,296 realistic tests and 47 synthetic tests load without collection errors |
| CPU smoke | Passed: frozen parser/corpus hashes, saved/main preset agreement, trace serialization, attention implementation agreement, causal mask, forward/backward and optimizer step |
| Figure source inventory | 46 mapped manuscript assets, 109 figure Python modules and 12 explicitly declared helper/artwork/config files; no missing sources or imported packages |
| Changed command interfaces | Geometry export, evidence export, answer/trace summary, empirical figure builder, Non-thinking PCA16 builder and root entry-point list all load |
| Numeric preservation | Eight retained geometry, empirical prediction and evidence-validation function definitions match their pre-cleanup ASTs exactly |
| JSON figure-data bridge | CPU fixture verifies PCA coordinates survive serialization, the input/output hash ledger agrees, and the appendix extractor reads JSON without an HTML file |
| Source audit | Python/JSON parsing, entry points, reader-facing links, common credential/private-path patterns, direct and escaped Han characters |
| Metadata audit | Four editable diagram files and the STIX notice remain byte-identical to the inspected versions; embedded font attribution is retained |

There are 347 passing tests and four explicit skips in `tools/check_tests.py`.
The skips require CUDA, the original frozen stimulus JSONL, or either of two
historical machine-specific deployment queues that were already excluded from
the release. A skipped test is not a passing test. Artifact-dependent tests
outside the selected set were collected, not all executed.

```bash
python tools/smoke_test.py
python tools/check_tests.py
python tools/check_figure_dependencies.py
python tools/validate_repo.py --check-manifest
```

Tests ran in the existing dedicated Windows/Python 3.12.7 CPU environment:
NumPy 1.26.4, SciPy 1.17.1, PyTorch 2.14.0+cpu and Transformers 5.17.0.
The previously recorded package sets are in
[`cpu-windows-py312.lock.txt`](../requirements/cpu-windows-py312.lock.txt) and
[`figures-windows-py312.lock.txt`](../requirements/figures-windows-py312.lock.txt).
They are CPU environment records, not Linux/CUDA lockfiles. The first attempt
using the system Python encountered a NumPy/SciPy binary incompatibility;
the isolated environment above was used for the reported results.

Narrative-only report generators and rendering assertions were removed.
Shared numerical helpers retain their existing import names. HTML dependencies
in the affected final-figure path were replaced by direct measurement or JSON
inputs; archived HTML remains supported where stated in
[WORKFLOWS.md](WORKFLOWS.md). The original figure asset hashes in
`figures/paper_assets.json` identify reference figures; they are not claims that
every figure was regenerated during this cleanup.

Full model downloads, GPU inference, synthetic training, all interventions and
end-to-end regeneration of the paper remain outside this validation. Future
result uploads and review-hosting metadata need their own anonymity inspection;
see [ANONYMITY.md](ANONYMITY.md).
