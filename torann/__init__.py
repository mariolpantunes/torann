"""torann — TORoidal Approximate Nearest Neighbours.

Exact + approximate k-NN and range search under **toroidal L1** on the unit
torus $[0, 1)^d$, built for ESS-style epoch workloads: static anchors, a
moving candidate tier, selective updates, and batch promotion.

Quick start::

    import numpy as np
    from torann import ToroidalNN

    d = 16
    nn = ToroidalNN(seed=42)
    nn.fit(np.random.rand(15_000, d),      # anchors: never move
           np.random.rand(3_000, d),       # candidates: move each epoch
           k=2 * d)
    idx, dist = nn.query()                 # each candidate vs everything

Modules:
    * `torann.wrapper` — `ToroidalNN`, the public facade (validation,
      tuning, implementation selection).
    * `torann.base` — the index contract every implementation satisfies.
    * `torann.brute` — exact blocked-NumPy search (small n, ground truth).
    * `torann.lsh` — the pure-NumPy LSH reference implementation.
    * `torann.rust` — the compiled core (``torann._native``, PyO3 + rayon).

The hash, table layout and cross-implementation guarantees are defined by
``torann.lsh``, the reference implementation. Benchmarks are regenerated
with ``python examples/benchmark.py``; the maintainers' measurement notes
are kept out of the distribution because they are rewritten constantly.
"""

from .wrapper import ToroidalNN, available_backends

__all__ = ["ToroidalNN", "available_backends"]
