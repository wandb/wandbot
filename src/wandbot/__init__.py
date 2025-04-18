"""
wandbot package.

Light‑weight symbols are available immediately.
Heavy symbols are imported lazily the first time they are requested.
"""

# --- cheap re‑exports ---------------------------------------------------
from .wandbot_tool  import WANDBOT_TOOL_DESCRIPTION, public_function

# --- lazy loading for heavy stuff --------------------------------------
import importlib, sys as _sys


_HEAVY_ATTRS = {"LargeModel", "train"}        # names defined in _heavy.py

def __getattr__(name):
    if name in _HEAVY_ATTRS:
        heavy = importlib.import_module("wandbot._heavy")
        value = getattr(heavy, name)
        setattr(_sys.modules[__name__], name, value)  # cache
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + list(_HEAVY_ATTRS))