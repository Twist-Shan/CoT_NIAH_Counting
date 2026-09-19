"""Portable fonts shared by the paper's figure and composition stages."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
from matplotlib import font_manager


def paper_font(font_dir: str | Path | None = None, family: str | None = None) -> str:
    """Use an installed/custom paper font, with Matplotlib's bundled fallback.

    COUNTING_FONT_DIR may contain user-licensed .ttf/.otf files. Set
    COUNTING_FONT_FAMILY to select a particular installed font explicitly.
    No operating-system-specific font path is required.
    """
    directory = font_dir or os.environ.get("COUNTING_FONT_DIR")
    if directory:
        directory = Path(directory).expanduser()
        if not directory.is_dir():
            raise FileNotFoundError(f"Font directory does not exist: {directory}")
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in {".ttf", ".otf"}:
                font_manager.fontManager.addfont(str(path))
    requested = family or os.environ.get("COUNTING_FONT_FAMILY")
    candidates = [requested] if requested else ["Times New Roman", "DejaVu Serif"]
    for name in candidates:
        try:
            font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            return name
        except ValueError:
            pass
    raise ValueError(f"Requested font is not available: {requested}")


def stix_font_path() -> Path:
    """Resolve the redistributable STIX font from the Matplotlib installation."""
    path = Path(matplotlib.get_data_path()) / "fonts/ttf/STIXGeneral.ttf"
    if not path.is_file():
        raise FileNotFoundError("Matplotlib's STIXGeneral.ttf is required for vector math labels")
    return path


def preview_font(size: int):
    from PIL import ImageFont
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
    return ImageFont.truetype(path, size)
