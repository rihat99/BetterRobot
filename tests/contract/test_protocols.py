"""Contract tests for extension Protocols.

See ``docs/conventions/extension.md`` for the seam inventory.
"""

from __future__ import annotations


import pytest

from better_robot.data_model.joint_models.base import JointModel
from better_robot.data_model.joint_models import (
    JointFixed,
    JointFreeFlyer,
    JointPlanar,
    JointPX,
    JointRX,
    JointRZ,
    JointSpherical,
    JointUniverse,
)


def _is_runtime_checkable(proto: type) -> bool:
    return getattr(proto, "_is_runtime_protocol", False)


@pytest.mark.parametrize(
    "proto",
    [JointModel],
)
def test_protocol_is_runtime_checkable(proto: type) -> None:
    """Every extension-seam Protocol must support ``isinstance``."""
    assert _is_runtime_checkable(proto), (
        f"{proto.__name__} is a Protocol but not runtime_checkable; "
        f"docs/conventions/extension.md requires @runtime_checkable."
    )


# ── Joint models ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "jm",
    [
        JointUniverse(),
        JointFixed(),
        JointRX(),
        JointRZ(),
        JointPX(),
        JointSpherical(),
        JointFreeFlyer(),
        JointPlanar(),
    ],
)
def test_joint_model_instances_satisfy_protocol(jm) -> None:
    assert isinstance(jm, JointModel), type(jm).__name__
