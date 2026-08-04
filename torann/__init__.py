"""torann — TORoidal Approximate Nearest Neighbours.

Exact and approximate k-NN and range search under **toroidal L1** on the unit
torus $[0, 1)^d$: opposite faces are identified, so the domain has no
boundary and a pair straddling an edge is as near as the same gap in the
interior. Built for epoch workloads — static anchors, a moving candidate
tier, selective updates, batch promotion.

**The approximation is in candidate selection only.** The hash decides which
points are compared; every distance returned is exact, and the index is
exact after every update.

Quick start
-----------
::

    import numpy as np
    from torann import ToroidalNN

    d = 16
    nn = ToroidalNN(seed=42)
    nn.fit(np.random.rand(15_000, d),      # anchors: never move
           np.random.rand(3_000, d),       # candidates: move each epoch
           k=2 * d)
    idx, dist = nn.query()                 # each candidate vs everything

    nn.update(new_candidate_positions)     # only re-places what moved
    nn.promote(next_batch)                 # candidates become anchors

The API
-------
One class does everything; the rest of the package is the machinery it
selects between. Grouped by what you are doing:

**Build and rebuild**
    `ToroidalNN.fit` draws the hash functions, tunes $B$, $K$ and $L$ from
    the workload, and builds both tiers. `ToroidalNN.update` re-places only
    the candidates whose hash cell changed. `ToroidalNN.promote` folds the
    candidate tier into the static tier as a linear merge, never a re-sort.

**Search**
    `ToroidalNN.query` for k-NN — with no arguments it is the candidate
    self-join, the ESS inner loop. `ToroidalNN.query_radius` for a metric
    ball, exactly or approximately.

**Inspect**
    `ToroidalNN.is_approximate` and `ToroidalNN.backend_name` for which path
    a query will take; `ToroidalNN.n_static`, `ToroidalNN.n_candidates`,
    `ToroidalNN.candidates` and `ToroidalNN.dimensions` for the contents; and
    `available_backends` for which implementations this install can reach.

Implementations
---------------
Three interchangeable backends behind one contract
(`torann.base.BaseIndex`), chosen by size and availability rather than by
the caller:

* `torann.brute` — exact blocked-NumPy search. Used below the measured
  crossover, and the ground truth the LSH paths are validated against.
* `torann.lsh` — the pure-NumPy LSH reference. **This is the
  specification**: the hash, the table layout and the tie rules are defined
  here, and the compiled core is required to reproduce its tables
  byte-for-byte.
* `torann.rust` — the compiled core (`torann._native`, PyO3 + rayon), 25-75x
  faster and byte-identical. Published x86-64 wheels need AVX2; a CPU
  without it falls back to `torann.lsh` rather than crashing.

Notes
-----
Coordinates are reduced mod 1 on the way in, so any real input is accepted
and the $[0, 1)$ domain is the facade's responsibility rather than the
caller's. Benchmarks are regenerated with ``python examples/benchmark.py``;
the maintainers' measurement notes are deliberately kept out of the
distribution, because they are rewritten constantly.
"""

from .wrapper import ToroidalNN, available_backends

__all__ = ["ToroidalNN", "available_backends"]
