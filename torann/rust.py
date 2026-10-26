"""The Rust acceleration: ``torann._native`` (PyO3 + rayon).

The native class implements the same behaviour as ``lsh.py`` — byte-identical
tables, equivalent results — at 25–75× the speed. It is a native
``pyclass``, so it registers as a virtual subclass of
:class:`torann.base.BaseIndex` instead of inheriting.

**Published x86-64 wheels require AVX2 and FMA.** That is not a portability
oversight, it is worth 17%: the distance kernel is a plain scalar loop that
LLVM auto-vectorizes, and without AVX there is no 256-bit register for it to
lower to — the query goes from 340 ms to 408 ms on the reference shape.
AVX2 is Haswell (2013) and Excavator (2015) onward. AVX-512 is *not*
required and not wanted: it is a further 4% at d=32 but nearly 2x worse at
d=8, where a 512-bit vector is the wrong shape for a one-vector distance.
`_cpu_supports_avx2` gates the import so a pre-AVX2 machine falls back to
the pure-Python backend instead of taking SIGILL on the first packed
instruction — a slow index beats a crash with no traceback.

From ``d = 17`` (to 4096) the batched k-NN join also runs behind an exact
8-bit prefilter, for blocks of at least 64 queries (smaller blocks cannot repay
its setup and keep the plain kernel): every coordinate is kept a second time as
``floor(256 x)`` (one byte per dimension, padded to 16 or 32; about +3% peak
memory) and a candidate whose circular cell lower bound already exceeds the
current k-th best, widened for float32 rounding, is never scored in float32.
Results are bit-identical with and without it (checked on millions of random
rows). It cuts the all-pairs self-join by 7-47% at ``d = 17-64`` (2-6% of
pairs survive) and is off below ``d = 17``, where it would cost more than it
saves.

``STATS`` says whether the loaded extension carries the query
instrumentation behind ``RustLshIndex.stats()``. Debug builds and builds with
``--features stats`` do; release wheels do not, so they ship no counters and
no introspection of the index's internals.

The floor is a property of how the wheel is *built*, not of the source: the
kernel names no vector type and no intrinsic, so a source build on a
pre-AVX2 machine (`RUSTFLAGS="-C target-cpu=x86-64"`) is correct and still
~13% faster than the hand-vectorized kernel this replaced.
"""

from __future__ import annotations

import logging
import platform
from collections.abc import Callable
from typing import Any

from .base import BaseIndex

logger = logging.getLogger(__name__)

# Declared before they are bound, because `._native` ships no stub: a type
# checker sees only the `= None` assignments in the fallback branches and
# concludes the names *are* None, so every call through them is an error and
# every unpacking of a result is "Never is not iterable". The annotation says
# what the try branch actually binds.
RustLshIndex: Any
STATS: bool
brute_knn: Callable[..., Any] | None
brute_radius: Callable[..., Any] | None
weighted_directions: Callable[..., Any] | None

_X86 = frozenset({"x86_64", "amd64", "i386", "i686", "x86"})


def _cpu_supports_avx2() -> bool:
    """Whether this CPU can run the published x86-64 build.

    Returns True on non-x86 (NEON is baseline on aarch64, so there is no
    floor to check) and True when the answer cannot be determined. The
    unknown case defaults to *yes* deliberately: the fallback costs 25–75×,
    so guessing "no" on a machine that was fine is far more damaging than
    the residual crash risk on genuinely pre-2013 hardware running an OS we
    could not probe.
    """
    if platform.machine().lower() not in _X86:
        return True

    try:
        system = platform.system()
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="ascii", errors="ignore") as fh:
                for line in fh:
                    if line.startswith(("flags", "Features")):
                        return " avx2 " in f" {line.split(':', 1)[1].strip()} "
            return True  # no flags line: unreadable format, assume yes

        if system == "Darwin":
            # Imported on the branch that needs it: this runs at import time
            # and most installs never reach here.
            import subprocess

            out = subprocess.run(
                ["sysctl", "-n", "hw.optional.avx2_0"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return out.stdout.strip() != "0"

        if system == "Windows":
            import ctypes

            # PF_AVX2_INSTRUCTIONS_AVAILABLE. Absent before Windows 10, where
            # the call returns 0 for an unknown feature -- indistinguishable
            # from "unsupported", so a false negative is possible on an old
            # Windows with a new CPU. It costs speed, never correctness.
            # `windll` exists only on Windows, so it is not on the
            # cross-platform stub; this branch cannot run elsewhere.
            windll = ctypes.windll  # type: ignore[attr-defined]
            return bool(windll.kernel32.IsProcessorFeaturePresent(40))
    except Exception:  # detection must never be able to break the import
        logger.debug("AVX2 detection failed; assuming supported", exc_info=True)

    return True


if _cpu_supports_avx2():
    try:
        # Compiled by maturin; absent from a source checkout, which is
        # exactly what the ImportError below handles.
        from ._native import (  # type: ignore[reportMissingImports]
            STATS,
            RustLshIndex,
            brute_knn,
            brute_radius,
            weighted_directions,
        )

        BaseIndex.register(RustLshIndex)
        AVAILABLE = True
    except ImportError:  # pure-Python install: wrapper falls back to lsh.py
        STATS = False
        RustLshIndex = None
        brute_knn = None
        brute_radius = None
        weighted_directions = None
        AVAILABLE = False
else:
    logger.warning(
        "torann: CPU lacks AVX2, so the published wheel cannot run here and "
        "the pure-Python backend will be used instead (25-75x slower). This "
        "is a last resort, not the only option: a baseline build is still "
        "30-41x faster than the NumPy path, because the win is structural "
        "(no intermediate matrix, one fused pass, every core) and AVX2 only "
        "adds 1.6-1.8x on top. Get one with\n"
        "    RUSTFLAGS='-C target-cpu=x86-64' "
        "pip install --no-binary torann torann"
    )
    STATS = False
    RustLshIndex = None
    brute_knn = None
    brute_radius = None
    weighted_directions = None
    AVAILABLE = False

__all__ = ["AVAILABLE", "STATS", "RustLshIndex", "brute_knn", "brute_radius", "weighted_directions"]
