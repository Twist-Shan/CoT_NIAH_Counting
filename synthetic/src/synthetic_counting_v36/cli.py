from __future__ import annotations

import sys

from synthetic_counting_v20.cli import main as _shared_main


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(value == flag or value.startswith(f"{flag}=") for value in argv)


def _preset(argv: list[str]) -> str:
    for index, value in enumerate(argv):
        if value == "--preset" and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith("--preset="):
            return value.split("=", 1)[1]
    return "debug"


def main(argv: list[str] | None = None) -> None:
    values = list(sys.argv[1:] if argv is None else argv)
    if _preset(values) == "debug":
        for flag, value in {"--seq-len": "256", "--n-positions": "384"}.items():
            if not _has_flag(values, flag):
                values.extend([flag, value])
    _shared_main(values, version="v36")


__all__ = ["main"]
