"""torann — TORoidal Approximate Nearest Neighbours.

Exact + approximate k-NN and range search under toroidal L1 on the unit
torus [0, 1)^d, built for ESS-style epoch workloads. See README.md.
"""

from .wrapper import ToroidalNN, available_backends

__all__ = ["ToroidalNN", "available_backends"]
