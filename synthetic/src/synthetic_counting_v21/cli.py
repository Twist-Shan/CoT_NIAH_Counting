from __future__ import annotations

from synthetic_counting_v20.cli import main as _shared_main


def main(argv: list[str] | None = None) -> None:
    _shared_main(argv, version="v21")


__all__ = ["main"]
