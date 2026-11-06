"""The two tracked test robots (decision record 0049), loaded by path so that every test runs offline.

- Panda, ``tests/fixtures/robots/panda/``: ``robots/panda_description`` of example-robot-data at commit
  ``d0d9098``, under the description's own Apache-2.0 licence. The nine collision meshes are tracked and the URDF's
  collision ``filename`` attributes name them relative to the URDF (upstream writes ``package://`` URIs); the visual
  COLLADA meshes (11 MB) are not tracked, so the visual entries resolve to no file.
- G1, ``tests/fixtures/robots/g1/``: ``robots/g1_description/g1_29dof.urdf`` of unitree_ros at commit ``267182b``,
  under its BSD-3-Clause licence, without meshes (24 of them exceed the 1 MiB file limit, principle 9.3).
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
