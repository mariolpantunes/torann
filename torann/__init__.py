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

**Measure a point set**
    `toroidal_separation` — the shortest toroidal-L1 distance from a point
    to any other, i.e. the maximin criterion. See the scope note below for
    why a *definition* lives here and a *methodology* does not.

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

What this library is responsible for
------------------------------------
torann answers *geometric* questions about points on the torus: which points
are near which, and how far apart they are. That is the index, and it is also
`torann.metrics` — a metric like `toroidal_separation` is one exact k-NN scan,
so it belongs beside the scan rather than in whichever caller needed it first.

**torann does not decide what a "good" point set is.** Ranking one design
against another is a question about the *purpose* the points serve — a
design of experiments, a space-filling sample, an optimiser's population —
and the answer depends on that purpose, not on the geometry. That judgement
belongs to the caller. `torann.metrics` documents which competing metrics
were measured and how each behaves, so a caller can choose with numbers in
front of it; it does not choose.

The distinction is not pedantic, it is the fix for a real failure. When the
definition of a metric lived in one project and the *choice* of metric lived
in another, a rename in one silently outlived the other: benchmark scripts
kept asking for keys that no longer existed and died in their reporting
after completing every run, and one kept printing a lower-is-better number
under a higher-is-better heading for weeks without ever failing. Definitions
here, choices in the caller, and neither guessing about the other.

The immediate consumer is `ess` (Empty Space Search), which generates
space-filling designs on the torus and uses this library both as its
neighbour engine and as the source of that definition. It ranks its designs
on discrepancies rather than on any point metric — deliberately, because its
own objective is a toroidal-L1 repulsion and grading an optimiser with its
own loss proves nothing. That reasoning is `ess`'s to make, and it lives
there.

Notes
-----
Coordinates are reduced mod 1 on the way in, so any real input is accepted
and the $[0, 1)$ domain is the facade's responsibility rather than the
caller's. Benchmarks are regenerated with ``python examples/benchmark.py``;
the maintainers' measurement notes are deliberately kept out of the
distribution, because they are rewritten constantly.
"""

import importlib.metadata

__author__ = "Mário Antunes"
__license__ = "MIT"
__email__ = "mario.antunes@ua.pt"
__url__ = "https://github.com/mariolpantunes/torann"
__status__ = "Development"

# Read from the installed distribution rather than a literal here: a hand-kept
# copy drifts from pyproject.toml without failing anything, which is how
# pyBlindOpt shipped 0.3.0 reporting 0.2.0. Source checkouts that were never
# installed have no metadata, hence the fallback.
try:
    __version__ = importlib.metadata.version("torann")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0.dev0"

from .base import DistArray, IdArray, Neighbours
from .metrics import toroidal_separation
from .wrapper import ToroidalNN, available_backends

__all__ = ["DistArray", "IdArray", "Neighbours", "ToroidalNN",
           "available_backends", "toroidal_separation"]
