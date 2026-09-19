from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from realistic_niah_v3_3_long_context.integrity import validate_frozen_dataset
from realistic_niah_v3_3_long_context.spec import PROTOCOL_VERSION


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and record a formal V3.3 long-context run root."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-dataset-seal-sha256", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    repo = Path(args.repo_root).resolve()
    if "runs/realistic_niah_v3_3_long_context/" not in run_root.as_posix():
        raise ValueError("Unexpected V3.3 long-context run root")
    if len(args.expected_commit) != 40:
        raise ValueError("Expected commit must be a full 40-character SHA")
    actual_commit = _git(repo, "rev-parse", "HEAD")
    if actual_commit != args.expected_commit:
        raise RuntimeError(
            f"Git commit mismatch: {actual_commit} != {args.expected_commit}"
        )
    if _git(repo, "status", "--short"):
        raise RuntimeError("Formal preparation requires a clean Git worktree")
    dataset = validate_frozen_dataset(
        run_root / "dataset",
        expected_seal_sha256=args.expected_dataset_seal_sha256,
    )
    prompt_audit_path = run_root / "orchestration" / "prompt_audit.json"
    if not prompt_audit_path.is_file():
        raise FileNotFoundError(f"Missing prompt audit: {prompt_audit_path}")
    prompt_audit = json.loads(prompt_audit_path.read_text(encoding="utf-8"))
    if (
        prompt_audit.get("schema_version")
        != "realistic_niah_prompt_audit_v3_3_long_context"
        or prompt_audit.get("protocol_version") != PROTOCOL_VERSION
        or prompt_audit.get("passed") is not True
        or prompt_audit.get("dataset_seal_sha256") != dataset["seal_sha256"]
        or int(prompt_audit.get("stimuli", -1)) != 3_780
        or int(prompt_audit.get("requests", -1)) != 15_120
        or int(prompt_audit.get("unique_request_ids", -1)) != 15_120
    ):
        raise RuntimeError("Formal preparation requires the exact prompt audit")
    render_audits = prompt_audit.get("render_audits")
    if (
        not isinstance(render_audits, list)
        or len(render_audits) != 4
        or any(item.get("passed") is not True for item in render_audits)
        or any(
            int(item.get("maximum_total_budget", 10**9)) > 131_072
            for item in render_audits
        )
    ):
        raise RuntimeError("Maximum-context rendered-prompt audit is incomplete")
    config_path = repo / "configs" / "realistic_niah_v3_3_long_context.json"
    config_bytes = config_path.read_bytes()
    payload = {
        "schema_version": "realistic_niah_preparation_v3_3_long_context",
        "protocol_version": PROTOCOL_VERSION,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "git_commit": actual_commit,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "dataset": dataset,
        "prompt_audit_path": str(prompt_audit_path),
        "prompt_audit_sha256": hashlib.sha256(
            prompt_audit_path.read_bytes()
        ).hexdigest(),
    }
    _atomic_json(run_root / "orchestration" / "preparation.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
