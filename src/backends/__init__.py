"""ToroidalLSH backend registry.

A *backend* owns the hash tables and the entire LSH path — build, selective
update, promote-merge, multi-probe k-NN with prefix relaxation, and range
queries. The facade (``src.ann.ToroidalNN``) owns validation, tuning,
parameter drawing and the exact brute-force path, and is backend-agnostic.

Backends implement the contract in ``CONTRACT.md``:

* ``python`` — ``src.backends.python.PythonLshIndex`` (always available,
  the reference implementation)
* ``rust``   — ``ann_backend_rust.RustLshIndex`` (PyO3 + maturin; the
  chosen native backend — the C contender lives at tag
  ``archive/backend-c``, see ANALYSIS.md)
"""

from __future__ import annotations

__all__ = ["get_backend", "resolve_backend", "available_backends"]

_AUTO_ORDER = ("rust", "python")


def get_backend(name: str):
    """Return the backend class for ``name``; ImportError if not installed."""
    if name == "python":
        from .python import PythonLshIndex
        return PythonLshIndex
    if name == "rust":
        from ann_backend_rust import RustLshIndex
        return RustLshIndex
    raise ValueError(f"unknown backend {name!r}; expected one of "
                     f"'auto', {', '.join(map(repr, _AUTO_ORDER))}")


def resolve_backend(name: str = "auto"):
    """Return ``(name, backend_class)``, resolving ``'auto'`` to the first
    installed backend in preference order (native first)."""
    if name != "auto":
        return name, get_backend(name)
    for cand in _AUTO_ORDER:
        try:
            return cand, get_backend(cand)
        except ImportError:
            continue
    raise ImportError("no LSH backend available (not even 'python')")


def available_backends() -> list[str]:
    """Names of the backends importable in this environment."""
    out = []
    for name in _AUTO_ORDER:
        try:
            get_backend(name)
        except ImportError:
            continue
        out.append(name)
    return out
