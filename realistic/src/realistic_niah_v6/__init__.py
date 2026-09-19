"""V6 structured-enumeration mechanism replication.

V6 is intentionally isolated from the frozen V5 Native-thinking code.  It
reuses audited numerical kernels through explicit adapters, while prompts,
schemas, configs, selection contracts, and output roots are V6-owned.
"""

from .spec import V6Config

__all__ = ["V6Config"]
