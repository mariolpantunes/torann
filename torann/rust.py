"""The Rust acceleration: ``torann._native`` (PyO3 + rayon).

The native class implements the same behaviour as ``lsh.py`` — byte-identical
tables, equivalent results — at 25–75× the speed
(ANALYSIS.md). It is a native ``pyclass``, so it registers as a virtual
subclass of :class:`torann.base.BaseIndex` instead of inheriting.
"""

from __future__ import annotations

from .base import BaseIndex

try:
    from ._native import RustLshIndex
    BaseIndex.register(RustLshIndex)
    AVAILABLE = True
except ImportError:  # pure-Python install: wrapper falls back to lsh.py
    RustLshIndex = None
    AVAILABLE = False

__all__ = ["RustLshIndex", "AVAILABLE"]
