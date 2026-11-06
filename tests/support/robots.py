"""The two tracked test robots (decision record 0039), loaded by path so that every test runs offline.

Source, licence and changes of each robot: ``tests/fixtures/robots/panda/README.md``,
``tests/fixtures/robots/g1/README.md``.
"""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "robots"


def panda_urdf() -> Path:
    """Path of the Franka Emika Panda URDF: the arm and its two-finger gripper."""
    return FIXTURES / "panda" / "panda.urdf"


def g1_urdf() -> Path:
    """Path of the Unitree G1 URDF with 29 actuated joints; load it with ``free_flyer=True`` for the floating base."""
    return FIXTURES / "g1" / "g1_29dof.urdf"
