"""The joint kinds: final frozen dataclasses under the abstract `JointModel` (record 0013), one file per kind.

Each kind declares ``nq``, ``nv``, its `Manifold` and ``axis``, and implements ``placement``, ``motion_subspace``,
``neutral`` and ``random``. `io.build_model` selects a kind for each joint of a parsed model; a new kind subclasses
`JointModel` (record 0025) and is checked by `better_robot.testing.check_joint_model`.
"""

from __future__ import annotations

from .base import JointModel
from .fixed import JointFixed
from .free_flyer import JointFreeFlyer
from .helical import JointHelical
from .planar import JointPlanar
from .prismatic import JointPrismatic
from .revolute import JointRevolute, JointRevoluteUnbounded
from .spherical import JointSpherical
from .translation import JointTranslation

__all__ = [
    "JointFixed",
    "JointFreeFlyer",
    "JointHelical",
    "JointModel",
    "JointPlanar",
    "JointPrismatic",
    "JointRevolute",
    "JointRevoluteUnbounded",
    "JointSpherical",
    "JointTranslation",
]
