# Reproducing the experiments

Commands below start in the repository root. `run.py` executes each command
inside its own component. A relative output such as `runs/paper` is therefore
inside `realistic/` or `synthetic/`. Use absolute paths for externally stored
inputs. `python run.py --dry-run ...` prints the resolved command without running it.

Use [WORKFLOWS.md](WORKFLOWS.md) for ordered producer/consumer commands,
the fresh-input preparation route, the Qwen YaRN-off supplement and final
figure inputs. [The figure map](../figures/README.md) identifies the actual
final plotting sources. Missing result files are expected in this code release;
the separately documented [historical source gaps](COMPLETENESS.md) are not
claimed to be resolved by that packaging choice.

## Synthetic experiment (Section 5 / Appendix G)

```bash
python tools/smoke_test.py
python run.py synthetic-train --preset main --stage prepare,train --device cuda --seed 1234 --out-root runs/paper --run-name synthetic_v58
python run.py synthetic-train --preset main --stage phase,causal,extended,attention,state,plots --device cuda --seed 1234 --out-root runs/paper --run-name synthetic_v58
```

The v58 main preset matches `synthetic/configs/paper_v58_saved_config.json`.
It trains independent Thinking/Non-thinking models for 10,000 steps, batch
size 128, with four layers and eight attention heads. Both receive 256-character
permuted contexts. The loss changes after step 1,500 to task-output loss;
count/trace/structure weights are 8/8/16. The trace has separators and no explicit
indices. The debug preset still inherits substantial v58 settings; use the
dedicated smoke script for a small CPU check.

Additional aligned analyses have their own input contracts:

```bash
python run.py synthetic-alignment --help
python run.py synthetic-continuation --help
python run.py synthetic scripts/run_v58_commit_query.py --help
python run.py synthetic scripts/run_v58_top1to8_aligned.py --help
```

Do not treat generic analysis-stage outputs as replacements for a paper-specific
confirmation experiment. The supplemental scripts preserve selection splits,
matched controls, and input-manifest checks.

## Behavior comparison (Section 2 / Appendix C)

First prepare the corpus as described in [DATA.md](DATA.md), then provide its
generated manifest to the freezer:

```bash
python run.py behavior-freeze --help
python run.py behavior --help
python run.py behavior-analysis --help
python run.py long-context-freeze --help
python run.py long-context --help
```

`behavior-freeze` requires `--output-dir`, `--haystack-dir`, and
`--haystack-corpus-manifest`. It constructs and audits the registered 3,360-item
grid. `behavior` requires `--stimuli`, `--output-dir`, and `--model`; model labels
must come from the retained registry. The long-context workflow uses separate
shards/workers. Follow its CLI and `realistic_niah_v3_3_long_context.json`,
including the Qwen YaRN setting, instead of extending the short grid informally.

For all-count/length fits and report plots, use the analysis scripts identified
in [PAPER_CODE_MAP.md](PAPER_CODE_MAP.md). They consume completed request tables.

## Non-thinking mechanisms (Section 3 / Appendix D)

```bash
python run.py nonthinking-freeze --config configs/realistic_niah_v4.json --output-dir runs/paper/nonthinking/dataset --haystack-dir data/haystacks/paul_graham_full
python run.py nonthinking --help
```

The generic V4 runner exposes preflight, behavior, state capture, attention,
ablation, patching, geometric steering, and analysis stages. Paper subexperiments
also use the retained V4.4-specific runners and frozen selection files. Select
the intended design variant and stage explicitly; running a generic V4 stage
does not reproduce every Appendix D intervention.

## Native Thinking and structured enumeration (Sections 4 / Appendices E–F)

```bash
python run.py thinking generate --help
python run.py thinking parse --help
python run.py thinking capture --help
python run.py thinking attention --help
python run.py thinking representation --help
python run.py enumeration generate --help
python run.py enumeration print-suite
```

The native runner reads frozen V4 stimuli. Structured enumeration uses its index
or bullet configuration and has its own strict parsing/cohort policies. Supply
the `--stimuli` location explicitly when regenerating inputs. Original defaults
that name completed `work/` artifacts are retained as historical input contracts.

Preserve discovery versus confirmation, seed versus analysis-slot identity, and
the actual token boundary being intervened on. The code retains replacement
policies and amendments; these are not interchangeable with the original panel.
Fresh Enumeration supplements are in `realistic/scripts/run_enumeration_*.py`
and matching configuration files.

The retained local Bash supervisors encode the ordered discovery/confirmation
stages. They accept `V6_ROOT`, `V6_PYTHON`, `V6_STIMULI`, `V6_CACHE`, and run-root
environment variables. Set them for the current checkout before execution;
the repository does not include a remote account or cluster submission setup.

## Additional tasks (Appendix H)

The task-input builders, task-local selection and Broad-scope amendments are
listed in [WORKFLOWS.md](WORKFLOWS.md#additional-tasks-appendix-h). Start with
the task-specific input and generation entries:

```bash
python run.py additional-kth --help
python run.py additional-category --help
```

Each input builder takes `--freeze-source` for the controlled stimuli and
writes a new `--frozen` directory. Task-specific retrieval, category-counting,
head selection, trace parsing and scoring implementations live in
`additional_experiments/`. Subsequent selection and ablation stages consume
the audited generations, plans and rankings described in the workflow guide.

## What constitutes a successful reproduction

Check the input hashes, saved configuration, model/tokenizer revisions, cohort
sizes, parsing/truncation outcomes, and matched control definitions before
comparing metrics. Record hardware and dependency versions with a new run.
CPU tests establish software behavior only. The paper's accuracy, geometry,
causal effects, and training dynamics require full experimental reruns.
