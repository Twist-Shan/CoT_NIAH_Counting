# CoT NIAH Counting

Code repository for the paper **Targeted Retrieval, Compact Representations: How CoT Reasoning Improves Long-Context Counting**.

This repository brings together the synthetic training experiments and realistic language-model experiments for studying counting in long contexts. It includes paired Thinking/Non-thinking evaluations, retrieval-head analysis, count-representation analysis, causal interventions, frozen experimental configurations, tests, and code for generating numerical results and paper figures. Tiny Shakespeare and small input specifications are included; pretrained weights and generated results are not tracked.

**Start here:** [Installation and quick start](#installation-and-quick-start) · [Execution workflows](docs/WORKFLOWS.md) · [Paper figure sources](figures/README.md) · [Dataset specification](docs/DATA.md) · [Validation record](docs/VALIDATION.md)

This is a code release: data construction, experiment execution, analysis and figure sources are provided, while large generated inputs and results remain local. The [source coverage record](docs/COMPLETENESS.md) documents recovered paper code, excluded auxiliary workflows and the limits of the completed validation.

## Research problem

The task asks a model to count target records dispersed through a long passage. A Non-thinking model answers directly; a Thinking model generates an enumeration trace before giving the final count. The experiments ask how these output formats change retrieval, internal count representations, and the model's use of intermediate states.

In the controlled synthetic task, the input is a character sequence $x=(x_1,\ldots,x_L)$ and a set $S$ of three target characters. The required count is

$$N=\sum_{t=1}^{L}\mathbf{1}\{x_t\in S\}.$$

Each occurrence is counted separately, and matching is case-sensitive. The Thinking trace lists matching characters in source order, separated by a fixed marker, without explicit numbering. The realistic task instead counts inserted city-score records and uses the same final-count objective.

The central comparison is between **broad retrieval**, which attends to multiple records at the answer query, and **targeted retrieval**, which retrieves individual records during enumeration. Attention and representation measurements describe these mechanisms; controlled interventions test their influence on subsequent retrieval and final-count predictions.

## Repository structure

```text
CoT_NIAH_Counting/
├── README.md                        Paper and code overview
├── run.py                           Unified experiment launcher
├── experiments.json                 Paper entry-point registry
├── requirements/                    CPU, figure, synthetic, and GPU environments
├── docs/                            Reproduction, data, paper map, validation
├── tools/                           Smoke tests, source audit, ZIP packaging
├── figures/                         Final figure sources and paper-asset map
├── synthetic/
│   ├── src/synthetic_counting_v58/   Reported synthetic experiment
│   ├── src/synthetic_counting_v20/   Shared data, model, training, analysis
│   ├── src/                         Earlier packages used by supporting code
│   ├── configs/                     Saved paper-run configuration
│   ├── scripts/                     Aligned analyses and numerical exports
│   ├── tests/                       Synthetic and intervention tests
│   └── requirements.txt
└── realistic/
    ├── src/                         Data, models, parsers, geometry, interventions
    ├── configs/                     Frozen designs and protocol amendments
    ├── scripts/                     Experiment stages and numerical exports
    ├── additional_experiments/      kth retrieval and category counting
    ├── data/                        Entity lists, prompt templates, corpus URLs
    ├── tests/                       Data, parser, analysis, and protocol tests
    └── requirements*.txt
```

The two subprojects retain their original import layout and execution directories. There is no root Python package: `run.py` selects the component directory and sets its import path. Version numbers remain unchanged because later experiments reuse earlier implementations and configuration schemas.

## Installation and quick start

Use Python **3.12** for the verified CPU setup. A CPU is sufficient for the smoke test, selected regression tests, and command-line checks. Full synthetic training and pretrained-model experiments require an appropriate GPU environment; see [environment setup](docs/ENVIRONMENT.md).

Download or clone this repository, then open a terminal in its root directory. Create and activate a virtual environment:

```sh
python -m venv .venv
```

On Linux/macOS, activate it with `source .venv/bin/activate`. In PowerShell, use `.\.venv\Scripts\Activate.ps1`.

Install dependencies and validate the local source:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/cpu.txt
python -m pip check
python tools/smoke_test.py
python tools/check_tests.py
python tools/validate_repo.py
```

For figure production, also install `requirements/figures.txt`; the [figure guide](figures/README.md) covers optional PDF assembly, fonts and vector-artwork export.

Then inspect the experiment entry points without downloading model weights:

```sh
python run.py --list
python run.py synthetic-train --help
python run.py behavior --help
python run.py thinking --help
python run.py enumeration --help
```

The smoke test checks both output formats with a small randomly initialized model, including forward/backward execution and agreement between attention implementations. It is not the paper's training protocol. Relative input/output arguments passed through `run.py` are interpreted inside the selected subproject; use absolute paths for externally stored inputs.

## Experiments

### Synthetic experiments

| Study | Configuration or entry point in `synthetic/` | Setting |
|---|---|---|
| Learning two counting mechanisms | `configs/paper_v58_saved_config.json`; `synthetic_counting_v58.run_v58` | Two independent models; four layers, eight heads, width 512; seed 1234 |
| Training dynamics | Shared v20 training and phase-analysis modules | 10,000 steps, batch 128; loss switches after step 1,500 |
| Retrieval and representation alignment | `scripts/run_v58_alignment_supplement.py` | Discovery/confirmation analyses with matched endpoints |
| Counter-state and answer-state interventions | `scripts/run_v58_commit_query.py`; `scripts/run_v58_native_continuation.py` | State transfer, continuation, and matched controls |
| Head ablation and geometry | `scripts/run_v58_top1to8_aligned.py`; `scripts/export_v58_geometry_cloud.py` | Selected-head comparisons and count-state readout |

The main task uses 256-character permuted Tiny Shakespeare windows, three-character target sets, and counts 1–10. Both modes share the input distribution and training budget. The Thinking trace uses separators without explicit indices. The main preset is checked against the saved paper configuration.

Run training, then analysis, from the repository root:

```sh
python run.py synthetic-train --preset main --stage prepare,train --device cuda --seed 1234 --out-root runs/paper --run-name synthetic_v58
python run.py synthetic-train --preset main --stage phase,causal,extended,attention,state,plots --device cuda --seed 1234 --out-root runs/paper --run-name synthetic_v58
```

See the [synthetic overview](synthetic/README.md) and [reproduction guide](docs/REPRODUCING.md) for supplemental stages, checkpoint inputs, and interpretation. Configuration settings describe the supplied experiment; see the validation record for checks performed.

### Realistic language-model experiments

| Study | Main implementation | Setting |
|---|---|---|
| Thinking/Non-thinking behavior | `realistic_niah_v3_1` | Twelve model comparison groups; counts 1–20; contexts 1k–20k; 30 paired seeds |
| Long-context extension | `realistic_niah_v3_3_long_context` | Qwen3-32B and Gemma-4-31B; contexts 25k–100k |
| Non-thinking mechanisms | `realistic_niah_v4` and V4.4 extensions | Broad retrieval, count geometry, span/head/state interventions |
| Native Thinking mechanisms | `realistic_niah_v5` | Trace parsing, targeted retrieval, counter-state and answer-state tests |
| Structured enumeration | `realistic_niah_v6` and Enumeration supplements | Index/bullet output with Thinking disabled |
| Additional retrieval tasks | `additional_experiments/` | kth-record retrieval and category-specific counting |

The core mechanistic studies use Qwen3-8B and Gemma-4-E4B with approximately 10k-token prompts. Model/tokenizer revisions, discovery/confirmation splits, parsing rules, and control definitions remain in the configurations and source. Frozen amendments are retained separately from the original designs.

See the [realistic overview](realistic/README.md), [dataset specification](docs/DATA.md), and [paper-to-code map](docs/PAPER_CODE_MAP.md) for input preparation and the corresponding manuscript sections.

## Methods and interpretation

| Method or component | Role and qualification |
|---|---|
| Parsed exact-count accuracy | Behavioral endpoint; truncation and unparseable outputs are not successful answers. |
| Broad and targeted attention scores | Describe where a head retrieves information; attention concentration alone does not establish causal use. |
| PCA, nearest-centroid classification, and probes | Describe count-state geometry and readability; predictive accuracy is not proof of a causal counter. |
| Head ablation and activation/state patching | Test the contribution of specified heads, positions, and states under the retained controls. |
| Discovery/confirmation separation | Keeps selection and evaluation distinct where specified; historical and exploratory analyses retain their own qualifications. |
| Additional-task analyses | Test transfer to new tasks under task-specific scoring, head selection, and input requirements. |

The independent unit for many realistic comparisons is the seed. Preserve source-seed versus analysis-slot identity when using replacement cohorts. Do not substitute results from different trace formats, endpoint definitions, model revisions, or attention backends without documenting the change.

## Data and generated artifacts

The repository contains source, configurations, tests, documentation, a public corpus URL list, entity/template inputs, and Tiny Shakespeare with its source checksum. Pretrained weights, essay bodies, raw generations, checkpoints, activation caches, analysis tables, result figures, reports, virtual environments, and run logs are excluded.

Run the supplied stages to generate results locally; see [REPRODUCING.md](docs/REPRODUCING.md). Figure builders require completed-run measurements; narrative-only report builders are omitted. Some model providers require access approval and local authentication before weight downloads.

The supplied manuscript is excluded because it contains identifying information. The source ZIP omits Git metadata. A separate anonymous Git copy must start from the clean source files, without copying the development history or its remote; see [ANONYMITY.md](docs/ANONYMITY.md) for the checked scope.

## Citation and licensing

If you use this code, please cite the accompanying paper, **Targeted Retrieval, Compact Representations: How CoT Reasoning Improves Long-Context Counting**, and identify the repository snapshot used. Author-identifying citation metadata is omitted during anonymous review.

No project-wide software license was present in the source directories, and this preparation does not introduce a new license grant. Corpus, font, model, and library use remains subject to the respective terms; required third-party notices are retained. See [LICENSE_STATUS.md](LICENSE_STATUS.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
