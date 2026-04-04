"""Tests for Robot and forward kinematics (stubs — fail until implemented)."""

import pytest


def test_robot_from_urdf_placeholder() -> None:
    """Placeholder: verify Robot.from_urdf raises NotImplementedError until implemented."""
    from better_robot import Robot
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    urdf = load_robot_description("panda_description")
    with pytest.raises(NotImplementedError):
        Robot.from_urdf(urdf)
