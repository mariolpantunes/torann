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

The floor is a property of how the wheel is *built*, not of the source: the
kernel names no vector type and no intrinsic, so a source build on a
pre-AVX2 machine (`RUSTFLAGS="-C target-cpu=x86-64"`) is correct and still
~13% faster than the hand-vectorized kernel this replaced.
"""

from __future__ import annotations

import logging
import platform

from .base import BaseIndex

logger = logging.getLogger(__name__)

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
                capture_output=True, text=True, timeout=5, check=False,
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
            RustLshIndex,
            brute_knn,
            brute_radius,
        )
        BaseIndex.register(RustLshIndex)
        AVAILABLE = True
    except ImportError:  # pure-Python install: wrapper falls back to lsh.py
        RustLshIndex = None
        brute_knn = None
        brute_radius = None
        AVAILABLE = False
else:
    logger.warning(
        "torann: CPU lacks AVX2; the compiled backend is not loaded and the "
        "pure-Python implementation will be used instead (25-75x slower). "
        "Build from the sdist to get a native module tuned for this CPU."
    )
    RustLshIndex = None
    brute_knn = None
    brute_radius = None
    AVAILABLE = False

__all__ = ["AVAILABLE", "RustLshIndex", "brute_knn", "brute_radius"]
