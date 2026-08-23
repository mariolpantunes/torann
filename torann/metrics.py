r"""Design-quality metrics on the unit torus — the shared definition.

One metric lives here, because one metric survived. Designs on $[0, 1)^d$ get
scored by several familiar quantities and **most of them stop discriminating
before $d = 64$**, which is inside the range this library's callers run. What
follows is the measured case for the one that does not, so that nobody
re-adopts a broken one by picking whichever name looks familiar.

What the metric has to score
----------------------------
The workload is *refinement*, not sampling from scratch: given static points
already placed — a design of experiments so far, an optimiser's population, a
history before a restart — put a new batch where nothing has been looked at
yet. A space-filling sampler cannot answer that, because it ignores the
existing points and covers the whole domain, so some of its batch lands where
you have already been.

So the quantity is the **maximin restricted to pairs touching the new batch**:
the shortest toroidal-$L_1$ distance from any new point to any other point,
anchor or new. It is one number, higher is better, and it penalises both
failure modes at once — a point placed on top of an anchor, and a batch that
finds one void and piles into it. With no anchors it degenerates to the plain
maximin separation of the design, so the same function covers the from-scratch
case rather than needing a second definition.

Measured on the refinement task (500 anchors, 200 new), against a Latin
hypercube batch, which is the "ignore the anchors" baseline:

| $d$ | ESS | LHS batch | ratio |
|---|---|---|---|
| 2 | 0.0178 | 0.0013 | **13.7x** |
| 8 | 1.0254 | 0.4165 | **2.51x** |
| 32 | 6.4861 | 4.4098 | **1.47x** |
| 64 | 14.1989 | 10.8270 | **1.31x** |

Note LHS is not a rival here — it is a reasonable initialiser for the batch,
and the point of the row is that ignoring the anchors costs you a factor even
when the sampler itself is good.

What was rejected, and why
--------------------------
Each was measured on the same designs. None is a naming quibble: each either
loses its resolution in high dimension or answers a different question.

* **Toroidal Clark-Evans** — the mean nearest-neighbour distance over its
  uniform expectation. Calibrated and bounded, and only trustworthy to about
  $d = 16$: above that, concentration of measure flattens the distance
  *distribution* and every design scores alike. That is a property of the
  mean, not of nearest-neighbour distances as such — a minimum over the same
  distances survives, which is why separation does.

* **Coverage radius** — the largest empty ball, the natural dual of
  separation, and it flattens too: 5.967 against 6.040 at $d = 32$, and
  13.183 against **13.180** at $d = 64$, where the better design is
  fractionally *behind*. A 0.02% gap cannot rank two methods.

* **Wrap-around $L_2$ discrepancy** over all $d$ coordinates measures the
  right thing and loses resolution on the same schedule: 0.798 against 0.905
  at $d = 32$ is usable, 0.987 against 0.996 at $d = 64$ is a 0.9% gap.

* **Projection discrepancy**, averaged over 1-D and 2-D projections, does
  hold a fixed scale at every $d$ — and answers a different question. It
  scores *marginal* coverage, which is what a Latin hypercube is built to
  optimize, so a plain LHS wins it at every dimensionality by construction
  (0.002-0.008 against 0.044-0.144). It is a real criterion for a real
  purpose, and it is not a ranking of how well the voids were filled;
  selecting on it inverts the conclusion. Worth *reporting* beside
  separation when marginal coverage matters.

* **Euclidean separation** ignores the wrap. On four points with one pair
  straddling the seam it reports 0.633 where the toroidal version returns
  0.020. On a torus it is the wrong geometry, not a stricter one.
"""

from __future__ import annotations

import numpy as np

from .brute import exact_knn

__all__ = ["toroidal_separation"]


def toroidal_separation(
    points: np.ndarray, anchors: np.ndarray | None = None
) -> float:
    r"""Maximin quality of a design, or of a batch added to one.

    With `anchors`, the shortest toroidal-$L_1$ distance from any point of
    `points` to any *other* point of `anchors + points`:

    $$ \min_{p \in P} \; \min_{q \in (A \cup P) \setminus \{p\}}
       d_{L_1}^{tor}(p, q) $$

    That is the refinement objective — how far the new batch stayed from
    everything already placed, and from itself. Anchor-anchor pairs are
    deliberately excluded: they are given, and including them would floor the
    score at the anchors' own spacing, so a perfect batch could not improve
    it.

    Without `anchors`, the plain maximin separation of `points`.

    Coordinates are reduced modulo 1 on the way in, so any real input is
    accepted, matching the rest of the package.

    Args:
        points: Design, or the newly placed batch, shape ``(n, d)``.
        anchors: Optional ``(m, d)`` points already placed. An empty array
            behaves like ``None``.

    Returns:
        The minimum distance, in toroidal L1; higher is better. ``0.0`` when
        no qualifying pair exists — fewer than two points and no anchors.

    Example:
        >>> import numpy as np
        >>> from torann.metrics import toroidal_separation
        >>> # 0.02 apart across the seam, not 0.98 apart the long way
        >>> float(round(toroidal_separation(np.array([[0.99], [0.01]])), 10))
        0.02
        >>> # against an anchor at 0.5, the batch at 0.4 sits 0.1 away
        >>> float(round(toroidal_separation(np.array([[0.4]]),
        ...                                 np.array([[0.5]])), 10))
        0.1
    """
    pts = np.mod(np.ascontiguousarray(points, dtype=np.float64), 1.0)
    if pts.ndim != 2:
        raise ValueError("points must be an (n, d) array")

    if anchors is None or len(anchors) == 0:
        if pts.shape[0] < 2:
            return 0.0
        union = pts
    else:
        anc = np.mod(np.ascontiguousarray(anchors, dtype=np.float64), 1.0)
        if anc.ndim != 2 or anc.shape[1] != pts.shape[1]:
            raise ValueError("anchors must be (m, d) with the same d as points")
        if pts.shape[0] == 0:
            return 0.0
        union = np.vstack([anc, pts])

    # Column 0 of each row is the query point itself at distance 0 — it is in
    # the searched set — so the nearest *other* point is column 1.
    _, dists = exact_knn(union, pts, 2)
    return float(np.min(dists[:, 1]))
