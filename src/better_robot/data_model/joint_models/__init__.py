"""``better_robot.data_model.joint_models`` — per-joint dispatch objects.

One file per joint family. Every class implements the ``JointModel`` protocol
from ``base.py``. ``build_model()`` selects the right class for each
``IRJoint`` based on ``kind`` + ``axis``.

See ``docs/02_DATA_MODEL.md §5``.
"""

from __future__ import annotations

from .base import JointKind, JointModel
from .composite import JointComposite
from .fixed import JointFixed, JointUniverse
from .free_flyer import JointFreeFlyer
from .helical import JointHelical
from .mimic import JointMimic
from .planar import JointPlanar
from .prismatic import (
    JointPrismaticUnaligned,
    JointPX,
    JointPY,
    JointPZ,
)
from .revolute import (
    JointRevoluteUnaligned,
    JointRevoluteUnbounded,
    JointRX,
    JointRY,
    JointRZ,
)
from .spherical import JointSpherical
from .translation import JointTranslation

__all__ = [
    "JointModel",
    "JointKind",
    "JointUniverse",
    "JointFixed",
    "JointRX",
    "JointRY",
    "JointRZ",
    "JointRevoluteUnaligned",
    "JointRevoluteUnbounded",
    "JointPX",
    "JointPY",
    "JointPZ",
    "JointPrismaticUnaligned",
    "JointSpherical",
    "JointFreeFlyer",
    "JointPlanar",
    "JointTranslation",
    "JointHelical",
    "JointComposite",
    "JointMimic",
]
