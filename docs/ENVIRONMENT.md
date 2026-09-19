# Environment setup

Use separate virtual environments for behavioral inference and mechanistic
analysis. vLLM and PyTorch/CUDA must be mutually compatible; installing the
three requirement files together is not a supported setup.

## CPU checks and synthetic work

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tools/smoke_test.py
python tools/check_tests.py
```

For a dedicated synthetic training environment, use
`requirements/synthetic.txt`. Select a PyTorch wheel appropriate for the
machine's CUDA version. The original source requirement files use version
ranges; they are not a complete lockfile. The CPU verification environment is
reported in [VALIDATION.md](VALIDATION.md).

## Pretrained-model mechanistic experiments

```bash
python -m pip install -r requirements/realistic-mechanisms.txt
python run.py nonthinking --help
python run.py thinking --help
```

The source's registered mechanism requirements specify `torch==2.7.0`,
`torchvision==0.22.0`, and `transformers==5.14.1`, with CUDA 12.8 wheels described
in the original requirements. These pins were preserved, not newly benchmarked
on a GPU during packaging. Tokenizer/model revisions are separate from package
versions and must also remain fixed.

## Behavioral inference

```bash
python -m pip install -r requirements/realistic-inference.txt
python run.py behavior --help
```

The registered inference source pins `transformers==5.14.1` and `vllm==0.25.1`.
This workflow requires an appropriate Linux/CUDA system. Exact memory needs
depend on model size, context length, batch size, and captured activations.
The repository does not claim that full 100k-context runs fit a particular GPU.

Some model repositories require access approval and local Hugging Face
authentication. Keep credentials in the local environment or credential store.
They are not repository inputs. Inspect the original CLI's `--cache-dir`,
`--device-map`, `--torch-dtype`, and attention-backend options for each stage.

`run.py` uses the interpreter that invoked it. It does not activate an
environment, install packages, launch remote jobs, or download model weights
merely to display help.
