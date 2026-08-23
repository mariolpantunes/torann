"""Every example still imports.

`examples/` is not imported by the package, is not on the lint gate's path and
has no tests, so it rots silently: five scripts here were dead for weeks after
`ess.utils` dropped a metric, and nothing failed until someone ran them by
hand. Each had completed all of its work and died in the reporting.

This is the cheap half of the guard. Importing a module executes its imports,
its module-level constants and its function *definitions*, which catches a
renamed symbol, a moved dependency and a syntax error — the failure modes that
arrive from outside. It does not catch a bad dict key inside a report
function; only running the script does that, which is what
`bench_ess_loop.py` and the ESS-side smoke test are for.

Scripts that need an optional dependency (`ess`, `faiss`, `matplotlib`) skip
rather than fail: this must stay green on a bare checkout.
"""

import importlib
import os
import pathlib
import sys
import unittest

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"

# Import-time optional dependencies, by module name. A script listed here is
# skipped when its dependency is missing rather than reported as broken.
OPTIONAL = {
    "bench_ess_loop": ("ess",),
    "bench_ess_quality": ("ess",),
    "bench_ess_suite": ("ess",),
    "bench_recall_ablation": ("ess",),
    "bench_refine_rounds": ("ess",),
    "benchmark": ("matplotlib",),
    "compare_faiss": ("faiss",),
    "compare_faiss_flat": ("faiss",),
    "ess_sim": ("ess", "faiss"),
    "figures": ("matplotlib",),
    "plot_benchmarks": ("matplotlib",),
    "report_recall_ablation": ("matplotlib",),
    "report_session": ("matplotlib",),
}


def _missing(name):
    """Optional dependencies of `name` that this environment lacks."""
    out = []
    for dep in OPTIONAL.get(name, ()):
        try:
            importlib.import_module(dep)
        except ImportError:
            out.append(dep)
    return out


class TestExamplesImport(unittest.TestCase):
    """One subtest per script, so one breakage does not mask the rest."""

    @classmethod
    def setUpClass(cls):
        # The scripts import each other by bare name (`from benchmark import
        # quality`), which works when run as `python examples/x.py` because
        # the script's directory leads sys.path. Reproduce that.
        cls._path = str(EXAMPLES)
        if cls._path not in sys.path:
            sys.path.insert(0, cls._path)

    def test_every_example_imports(self):
        scripts = sorted(p.stem for p in EXAMPLES.glob("*.py")
                         if not p.stem.startswith("_"))
        self.assertTrue(scripts, "no examples found — is the layout still right?")
        for name in scripts:
            with self.subTest(script=name):
                missing = _missing(name)
                if missing:
                    self.skipTest(f"needs {', '.join(missing)}")
                # Several scripts chdir or write into examples/out on import;
                # none should, but importing from the repo root keeps any that
                # do from scattering files.
                cwd = os.getcwd()
                try:
                    importlib.import_module(name)
                finally:
                    os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
