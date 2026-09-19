from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


_WORKER_REQUESTS: pd.DataFrame | None = None
_WORKER_LAWS: Any = None


def _load_frozen_analysis_script(source_root: Path) -> Any:
    """Load the exact frozen analysis entry point, not the dirty worktree copy."""

    source_root = source_root.resolve()
    sys.path.insert(0, str(source_root / "src"))
    sys.path.insert(0, str(source_root / "scripts"))
    path = source_root / "scripts" / "analyze_realistic_niah_v3_1.py"
    spec = importlib.util.spec_from_file_location("frozen_v31_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load frozen analysis script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _process_initializer(
    request_cache: str,
    source_root: str,
    fit_backend: str,
    analysis_device: str,
) -> None:
    """Load one immutable request table per worker process."""

    global _WORKER_REQUESTS, _WORKER_LAWS
    _load_frozen_analysis_script(Path(source_root))
    import realistic_niah_v3_1.laws as laws

    laws.configure_fit_backend(backend=fit_backend, device=analysis_device)
    _WORKER_LAWS = laws
    _WORKER_REQUESTS = pd.read_pickle(request_cache)


def _process_candidate_task(
    payload: tuple[str, str, int],
) -> tuple[
    tuple[str, str],
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
]:
    if _WORKER_REQUESTS is None or _WORKER_LAWS is None:
        raise RuntimeError("Candidate worker was not initialized")
    outcome, mode, replicates = payload
    mode_rows = _WORKER_REQUESTS.loc[_WORKER_REQUESTS["prompt_mode"] == mode]
    result = _WORKER_LAWS.fit_candidate_grid(
        mode_rows,
        outcome_models=(outcome,),
        interaction_bootstrap_replicates=replicates,
    )
    return (outcome, mode), result


def _process_validation_task(
    payload: tuple[str, str, str, int],
) -> tuple[tuple[str, str], pd.DataFrame]:
    if _WORKER_REQUESTS is None or _WORKER_LAWS is None:
        raise RuntimeError("Validation worker was not initialized")
    kind, outcome, mode, replicates = payload
    mode_rows = _WORKER_REQUESTS.loc[_WORKER_REQUESTS["prompt_mode"] == mode]
    if kind == "seed":
        result = _WORKER_LAWS.nested_seed_validation(
            mode_rows,
            outcome_models=(outcome,),
            interaction_bootstrap_replicates=replicates,
        )
    elif kind in {"N", "L"}:
        result = _WORKER_LAWS.nested_held_axis_validation(
            mode_rows,
            axis=kind,
            outcome_models=(outcome,),
            interaction_bootstrap_replicates=replicates,
        )
    else:
        raise ValueError(f"Unknown validation kind: {kind}")
    return (outcome, mode), result


def _progress(label: str, key: tuple[str, str], completed: int, total: int) -> None:
    print(
        json.dumps(
            {
                "stage": label,
                "completed_group": {
                    "outcome_model": key[0],
                    "prompt_mode": key[1],
                },
                "completed_groups": completed,
                "total_groups": total,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _parallel_candidate_processes(
    *,
    request_cache: Path,
    source_root: Path,
    ordered_keys: list[tuple[str, str]],
    workers: int,
    replicates: int,
    fit_backend: str,
    analysis_device: str,
) -> dict[
    tuple[str, str],
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
]:
    results: dict[
        tuple[str, str],
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_process_initializer,
        initargs=(
            str(request_cache),
            str(source_root),
            fit_backend,
            analysis_device,
        ),
    ) as pool:
        futures = {
            pool.submit(_process_candidate_task, (outcome, mode, replicates)): (
                outcome,
                mode,
            )
            for outcome, mode in ordered_keys
        }
        for future in as_completed(futures):
            key, value = future.result()
            results[key] = value
            _progress("candidate_selection", key, len(results), len(ordered_keys))
    return results


def _parallel_validation_processes(
    *,
    request_cache: Path,
    source_root: Path,
    ordered_keys: list[tuple[str, str]],
    workers: int,
    replicates: int,
    kind: str,
    label: str,
    fit_backend: str,
    analysis_device: str,
) -> pd.DataFrame:
    results: dict[tuple[str, str], pd.DataFrame] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_process_initializer,
        initargs=(
            str(request_cache),
            str(source_root),
            fit_backend,
            analysis_device,
        ),
    ) as pool:
        futures = {
            pool.submit(
                _process_validation_task,
                (kind, outcome, mode, replicates),
            ): (outcome, mode)
            for outcome, mode in ordered_keys
        }
        for future in as_completed(futures):
            key, value = future.result()
            results[key] = value
            _progress(label, key, len(results), len(ordered_keys))
    return pd.concat([results[key] for key in ordered_keys], ignore_index=True)


def _parallel_grouped(
    *,
    requests: pd.DataFrame,
    outcome_models: tuple[str, ...],
    modes: tuple[str, ...],
    workers: int,
    label: str,
    task: Callable[[pd.DataFrame, str], pd.DataFrame],
) -> pd.DataFrame:
    """Run independent outcome-by-mode validation blocks and restore registry order."""

    ordered_keys = [(outcome, mode) for outcome in outcome_models for mode in modes]
    results: dict[tuple[str, str], pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=label) as pool:
        futures = {
            pool.submit(
                task,
                requests.loc[requests["prompt_mode"] == mode],
                outcome,
            ): (outcome, mode)
            for outcome, mode in ordered_keys
        }
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()
            _progress(label, key, len(results), len(ordered_keys))
    return pd.concat([results[key] for key in ordered_keys], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continue the frozen V3.1 analysis after candidate selection, "
            "parallelizing only independent outcome-by-mode validation loops."
        )
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--frozen-source-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--executor", choices=("process", "thread"), default="process")
    parser.add_argument("--interaction-bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--validation-bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--coefficient-bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--fit-backend", choices=("scipy", "torch"), default="scipy")
    parser.add_argument("--analysis-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be positive")

    frozen = _load_frozen_analysis_script(Path(args.frozen_source_root))
    from realistic_niah_v3_1.laws import (  # imported from the frozen source root
        OUTCOME_MODELS,
        bootstrap_selected_coefficients,
        configure_fit_backend,
        fit_candidate_grid,
        fit_backend_metadata,
        nested_held_axis_validation,
        nested_seed_validation,
    )

    run_root = Path(args.run_root).resolve()
    output = Path(args.analysis_dir).resolve()
    tables = output / "tables"
    state_path = output / "analysis_state.json"
    # The request table is reconstructed and audited from the immutable run root.
    # A copied request_level.csv.gz checkpoint is therefore optional when this
    # continuation is deployed to another host.
    required = [state_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Candidate-selection checkpoint is incomplete: {missing}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    allowed_start_stages = {"cot_style_complete", "candidate_selection_complete"}
    if state.get("stage") not in allowed_start_stages:
        raise RuntimeError(
            "Refusing to continue unless analysis_state is cot_style_complete or "
            "candidate_selection_complete; "
            f"observed {state.get('stage')!r}"
        )

    configure_fit_backend(
        backend=args.fit_backend,
        device=args.analysis_device,
    )
    requests, sources = frozen.load_request_table(run_root)
    if len(requests) != frozen.EXPECTED_REQUESTS:
        raise RuntimeError(
            f"Expected {frozen.EXPECTED_REQUESTS:,} requests, found {len(requests):,}"
        )
    modes = tuple(sorted(requests["prompt_mode"].astype(str).unique()))
    outcome_models = tuple(OUTCOME_MODELS)
    workers = min(args.workers, len(outcome_models) * len(modes))
    ordered_keys = [(outcome, mode) for outcome in outcome_models for mode in modes]
    source_root = Path(args.frozen_source_root).resolve()
    request_cache = output / ".analysis_cache" / "request_level.pkl"
    if args.executor == "process" and not request_cache.is_file():
        request_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary_cache = request_cache.with_suffix(".pkl.tmp")
        requests.to_pickle(temporary_cache, protocol=5)
        temporary_cache.replace(request_cache)

    if state.get("stage") == "cot_style_complete":
        def candidate_task(
            mode_rows: pd.DataFrame, outcome: str
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            return fit_candidate_grid(
                mode_rows,
                outcome_models=(outcome,),
                interaction_bootstrap_replicates=args.interaction_bootstrap_replicates,
            )

        if args.executor == "process":
            candidate_results = _parallel_candidate_processes(
                request_cache=request_cache,
                source_root=source_root,
                ordered_keys=ordered_keys,
                workers=workers,
                replicates=args.interaction_bootstrap_replicates,
                fit_backend=args.fit_backend,
                analysis_device=args.analysis_device,
            )
        else:
            candidate_results = {}
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="candidate_selection"
            ) as pool:
                futures = {
                    pool.submit(
                        candidate_task,
                        requests.loc[requests["prompt_mode"] == mode],
                        outcome,
                    ): (outcome, mode)
                    for outcome, mode in ordered_keys
                }
                for future in as_completed(futures):
                    key = futures[future]
                    candidate_results[key] = future.result()
                    _progress(
                        "candidate_selection",
                        key,
                        len(candidate_results),
                        len(ordered_keys),
                    )
        comparison = pd.concat(
            [candidate_results[key][0] for key in ordered_keys], ignore_index=True
        )
        selected = pd.concat(
            [candidate_results[key][1] for key in ordered_keys], ignore_index=True
        )
        coefficients = pd.concat(
            [candidate_results[key][2] for key in ordered_keys], ignore_index=True
        )
        interaction_parts = [
            candidate_results[key][3]
            for key in ordered_keys
            if not candidate_results[key][3].empty
        ]
        interactions = (
            pd.concat(interaction_parts, ignore_index=True)
            if interaction_parts
            else pd.DataFrame()
        )
        frozen._write_csv(tables / "candidate_comparison.csv", comparison)
        frozen._write_csv(tables / "selected_laws.csv", selected)
        frozen._write_csv(tables / "coefficients.csv", coefficients)
        frozen._write_csv(tables / "interaction_tests.csv", interactions)
        frozen._write_state(
            state_path,
            "candidate_selection_complete",
            attempted_laws=len(comparison),
            selected_laws=len(selected),
            continuation="parallel_outcome_mode_candidate_selection",
            parallel_candidate_workers=workers,
        )
    else:
        checkpoint_files = [
            tables / "candidate_comparison.csv",
            tables / "selected_laws.csv",
            tables / "coefficients.csv",
            tables / "interaction_tests.csv",
        ]
        checkpoint_missing = [
            str(path) for path in checkpoint_files if not path.is_file()
        ]
        if checkpoint_missing:
            raise RuntimeError(
                f"Candidate-selection checkpoint is incomplete: {checkpoint_missing}"
            )
        selected = pd.read_csv(tables / "selected_laws.csv")

    if args.fit_backend == "torch":
        backend_parity = frozen._backend_parity_audit(
            requests,
            selected,
            torch_device=args.analysis_device,
        )
        frozen._write_csv(tables / "fit_backend_parity.csv", backend_parity)
        if not backend_parity["passed"].all():
            raise RuntimeError(
                "Torch fit backend failed the preregistered SciPy parity gate"
            )

    frozen._write_state(
        state_path,
        "parallel_validation_started",
        workers=workers,
        interaction_bootstrap_replicates=args.interaction_bootstrap_replicates,
        validation_bootstrap_replicates=args.validation_bootstrap_replicates,
    )

    def seed_task(mode_rows: pd.DataFrame, outcome: str) -> pd.DataFrame:
        return nested_seed_validation(
            mode_rows,
            outcome_models=(outcome,),
            interaction_bootstrap_replicates=args.validation_bootstrap_replicates,
        )

    nested_seed = (
        _parallel_validation_processes(
            request_cache=request_cache,
            source_root=source_root,
            ordered_keys=ordered_keys,
            workers=workers,
            replicates=args.validation_bootstrap_replicates,
            kind="seed",
            label="nested_held_seed",
            fit_backend=args.fit_backend,
            analysis_device=args.analysis_device,
        )
        if args.executor == "process"
        else _parallel_grouped(
            requests=requests,
            outcome_models=outcome_models,
            modes=modes,
            workers=workers,
            label="nested_held_seed",
            task=seed_task,
        )
    )
    frozen._write_csv(tables / "nested_held_seed_validation.csv", nested_seed)
    frozen._write_state(
        state_path,
        "nested_held_seed_complete",
        rows=len(nested_seed),
        workers=workers,
    )

    def held_n_task(mode_rows: pd.DataFrame, outcome: str) -> pd.DataFrame:
        return nested_held_axis_validation(
            mode_rows,
            axis="N",
            outcome_models=(outcome,),
            interaction_bootstrap_replicates=args.validation_bootstrap_replicates,
        )

    held_n = (
        _parallel_validation_processes(
            request_cache=request_cache,
            source_root=source_root,
            ordered_keys=ordered_keys,
            workers=workers,
            replicates=args.validation_bootstrap_replicates,
            kind="N",
            label="nested_held_N",
            fit_backend=args.fit_backend,
            analysis_device=args.analysis_device,
        )
        if args.executor == "process"
        else _parallel_grouped(
            requests=requests,
            outcome_models=outcome_models,
            modes=modes,
            workers=workers,
            label="nested_held_N",
            task=held_n_task,
        )
    )
    frozen._write_csv(tables / "held_N_validation.csv", held_n)
    frozen._write_state(
        state_path,
        "nested_held_N_complete",
        rows=len(held_n),
        workers=workers,
    )

    def held_l_task(mode_rows: pd.DataFrame, outcome: str) -> pd.DataFrame:
        return nested_held_axis_validation(
            mode_rows,
            axis="L",
            outcome_models=(outcome,),
            interaction_bootstrap_replicates=args.validation_bootstrap_replicates,
        )

    held_l = (
        _parallel_validation_processes(
            request_cache=request_cache,
            source_root=source_root,
            ordered_keys=ordered_keys,
            workers=workers,
            replicates=args.validation_bootstrap_replicates,
            kind="L",
            label="nested_held_L",
            fit_backend=args.fit_backend,
            analysis_device=args.analysis_device,
        )
        if args.executor == "process"
        else _parallel_grouped(
            requests=requests,
            outcome_models=outcome_models,
            modes=modes,
            workers=workers,
            label="nested_held_L",
            task=held_l_task,
        )
    )
    frozen._write_csv(tables / "held_L_validation.csv", held_l)
    frozen._write_state(
        state_path,
        "nested_held_L_complete",
        rows=len(held_l),
        workers=workers,
    )

    # Keep this serial: the preregistered function advances one RNG stream across
    # selected rows, so splitting rows would alter bootstrap draws.
    coefficient_bootstrap = bootstrap_selected_coefficients(
        requests,
        selected,
        replicates=args.coefficient_bootstrap_replicates,
    )
    frozen._write_csv(
        tables / "selected_coefficient_bootstrap.csv", coefficient_bootstrap
    )

    distribution_detail, distribution_calibration, distribution_summaries = (
        frozen._distribution_outputs(requests, selected)
    )
    frozen._write_csv(
        tables / "accuracy_distribution_predictions.csv", distribution_detail
    )
    frozen._write_csv(
        tables / "accuracy_distribution_calibration.csv", distribution_calibration
    )
    frozen._atomic_json(
        tables / "accuracy_distribution_diagnostics.json", distribution_summaries
    )

    frozen._write_state(
        state_path,
        "complete",
        lomo_run=False,
        bootstrap_reselection_run=False,
        output_tables=len(list(tables.glob("*"))),
        continuation="parallel_outcome_mode_validation",
        parallel_validation_workers=workers,
    )
    output_files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "schema_version": "realistic_niah_v3_1_analysis_manifest_v1",
        "protocol_version": frozen.PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "request_rows": len(requests),
        "sources": sources,
        "settings": {
            "run_root": str(run_root),
            "output_dir": str(output),
            "interaction_bootstrap_replicates": args.interaction_bootstrap_replicates,
            "coefficient_bootstrap_replicates": args.coefficient_bootstrap_replicates,
            "validation_bootstrap_replicates": args.validation_bootstrap_replicates,
            "run_lomo": False,
            "lomo_bootstrap_replicates": 2_000,
            "run_bootstrap_reselection": False,
            "reselection_bootstrap_replicates": 2_000,
            "annotation_random_size": 600,
            "annotation_challenge_size": 200,
            "fit_backend": args.fit_backend,
            "analysis_device": args.analysis_device,
            "parallel_executor": args.executor,
        },
        "fit_backend": fit_backend_metadata(),
        "lomo_run": False,
        "bootstrap_reselection_run": False,
        "continuation": {
            "kind": "parallel_outcome_mode_candidate_and_validation",
            "workers": workers,
            "executor": args.executor,
            "invariance": (
                "Only independent outcome-by-prompt-mode candidate and outer-validation "
                "loops were run concurrently; each frozen function, seed, bootstrap "
                "count, and within-group output order is unchanged."
            ),
        },
        "files": [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": frozen._sha256(path),
            }
            for path in output_files
            if path.name != "analysis_manifest.json" and ".analysis_cache" not in path.parts
        ],
    }
    frozen._atomic_json(output / "analysis_manifest.json", manifest)
    print(
        json.dumps(
            {
                "passed": True,
                "requests": len(requests),
                "selected_laws": len(selected),
                "output": str(output),
                "parallel_validation_workers": workers,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
