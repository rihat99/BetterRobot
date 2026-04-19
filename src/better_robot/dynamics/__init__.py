"""``better_robot.dynamics`` — RNEA, ABA, CRBA, centroidal, integrators.

**Skeleton only in v1.** Every function is named, signed, docstringed, and
raises ``NotImplementedError`` with a pointer to the owning milestone in
``docs/06_DYNAMICS.md``. The only exceptions are ``center_of_mass`` and
``integrate_q``, which land numerically in milestone D1.

The point of the skeleton is to lock the shape so that ``residuals/``,
``optim/``, and ``tasks/`` can be written against a stable dynamics API.
"""

from __future__ import annotations

from .aba import aba
from .centroidal import (
    ccrba,
    center_of_mass,
    compute_centroidal_map,
    compute_centroidal_momentum,
)
from .crba import compute_minverse, crba
from .derivatives import (
    compute_aba_derivatives,
    compute_centroidal_dynamics_derivatives,
    compute_crba_derivatives,
    compute_rnea_derivatives,
)
from .integrators import (
    integrate_q,
    rk4,
    semi_implicit_euler,
    symplectic_euler,
)
from .rnea import (
    bias_forces,
    compute_coriolis_matrix,
    compute_generalized_gravity,
    nle,  # deprecated alias for bias_forces — remove in v1.1
    rnea,
)

__all__ = [
    "rnea",
    "bias_forces",
    "nle",  # deprecated alias
    "compute_generalized_gravity",
    "compute_coriolis_matrix",
    "aba",
    "crba",
    "compute_minverse",
    "center_of_mass",
    "compute_centroidal_map",
    "compute_centroidal_momentum",
    "ccrba",
    "compute_rnea_derivatives",
    "compute_aba_derivatives",
    "compute_crba_derivatives",
    "compute_centroidal_dynamics_derivatives",
    "integrate_q",
    "semi_implicit_euler",
    "symplectic_euler",
    "rk4",
]
