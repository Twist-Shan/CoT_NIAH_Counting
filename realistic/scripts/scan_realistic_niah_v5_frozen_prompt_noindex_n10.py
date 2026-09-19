#!/usr/bin/env python3
"""Scan fixed-N=10 natural no-index traces after the fixed-N=3 scan."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import scan_realistic_niah_v5_frozen_prompt_noindex_n3 as core


def main() -> None:
    core.SCHEMA = "realistic_niah_v5_frozen_prompt_noindex_n10_scan_v4"
    core.FIXED_COUNT = 10
    core.AUDIT_KEY = "noindex_n10_format_audit"
    core.COHORT_KEY = "noindex_n10_cohort"
    core.LOG_LABEL = "noindex-n10-scan"
    core.PROSPECTIVE_ORIGIN = "prospective_fixed_n10_scan"
    core.ORIGIN_KEY = "noindex_n10_generation_origin"
    core.__doc__ = __doc__
    core.main()


if __name__ == "__main__":
    main()
