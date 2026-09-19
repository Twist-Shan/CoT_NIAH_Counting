"""Three-character-set Tiny Shakespeare counting experiments (v20)."""

import sys
import types

# Pandas optional accelerators are not used here.  A minimal pyarrow stub avoids
# both an incompatible system wheel and pandas 3's unsafe dereference of a
# ``None`` sentinel during ordinary DataFrame construction.
if "pyarrow" not in sys.modules:
    _pyarrow_stub = types.ModuleType("pyarrow")
    _pyarrow_stub.__version__ = "0.0.0"
    _pyarrow_stub.Array = type("Array", (), {})
    _pyarrow_stub.ChunkedArray = type("ChunkedArray", (), {})
    # Narwhals and newer scikit-learn validation helpers inspect these core
    # PyArrow container types even when PyArrow is only an unavailable
    # optional dependency.  Keep the stub structurally complete for their
    # isinstance guards without importing or requiring PyArrow.
    _pyarrow_stub.Table = type("Table", (), {})
    _pyarrow_stub.RecordBatch = type("RecordBatch", (), {})
    sys.modules["pyarrow"] = _pyarrow_stub
for _optional in ("numexpr", "bottleneck"):
    sys.modules.setdefault(_optional, None)

from .config import V20Config, config_from_dict, preset_config  # noqa: E402

__all__ = ["V20Config", "config_from_dict", "preset_config"]
