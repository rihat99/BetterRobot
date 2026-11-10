"""Every public function taking ``(model, state)`` checks that the state belongs to the model (contract 06,
C-STATE-6): another structure raises ``ModelError``, another dtype or device the mismatch error,
never wrong numbers or a raw torch error. The functions are found from the packages' ``__all__``."""

from __future__ import annotations

import inspect

import pytest
import torch

from better_robot import collision, dynamics, kinematics
from better_robot.exceptions import DtypeMismatchError, ModelError
from better_robot.io import ModelBuilder, build_model

from ._public_ops import build_case

PACKAGES = (kinematics, dynamics, collision)


def _takes_model_and_state(function: object) -> bool:
    if not inspect.isfunction(function):
        return False
    return list(inspect.signature(function).parameters)[:2] == ["model", "state"]


STATE_FUNCTIONS = sorted(
    name for package in PACKAGES for name in package.__all__ if _takes_model_and_state(getattr(package, name))
)


def _other_arm(axis: tuple[float, float, float]):
    """Two revolute joints like `two_link_arm`, about ``axis``: same sizes, another structure."""
    builder = ModelBuilder("other_arm")
    inertia = 0.01 * torch.eye(3)
    builder.add_body("base", mass=0.5, inertia=inertia)
    builder.add_body("link1", mass=1.0, com=torch.tensor([0.1, 0.0, 0.0]), inertia=inertia)
    builder.add_body("link2", mass=0.7, com=torch.tensor([0.1, 0.0, 0.0]), inertia=inertia)
    origin = torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
    builder.add_revolute("j1", axis=axis, parent="base", child="link1", origin=origin, lower=-1.0, upper=1.0)
    builder.add_revolute("j2", axis=axis, parent="link1", child="link2", origin=origin, lower=-1.0, upper=1.0)
    return build_model(builder.finalize(), dtype=torch.float64)


def test_the_cases_cover_every_state_function() -> None:
    assert STATE_FUNCTIONS == [
        "center_of_mass",
        "centroidal_map",
        "centroidal_momentum",
        "forward_dynamics",
        "frame_jacobian",
        "frame_jacobian_time_variation",
        "inverse_dynamics",
        "joint_jacobian",
        "joint_jacobian_time_variation",
        "mass_matrix",
        "point_jacobian",
    ]


@pytest.mark.parametrize("name", STATE_FUNCTIONS)
def test_state_of_another_model_is_rejected(name: str) -> None:
    function, (model, state), kwargs = build_case(name, torch.float64, torch.device("cpu"))
    function(model.with_values(armature=model.armature + 0.1), state, **kwargs)  # same robot, new values: fine
    with pytest.raises(ModelError, match="whose structure differs"):
        function(_other_arm((1.0, 0.0, 0.0)), state, **kwargs)
    with pytest.raises(DtypeMismatchError, match=r"state\.dtype=torch\.float64 != model\.dtype=torch\.float32"):
        function(model.to(dtype=torch.float32), state, **kwargs)
    with pytest.raises(TypeError, match="state must be a KinematicState"):
        function(model, state.q, **kwargs)
