"""Every public function taking ``(model, state)`` checks that the state belongs to the model (contract 06,
C-STATE-6): another structure raises ``ModelError``, another dtype or device the mismatch error,
never wrong numbers or a raw torch error. The functions are found from the packages' ``__all__``."""

from __future__ import annotations

import inspect

import pytest
import torch

from better_robot import collision, dynamics, kinematics
from better_robot.exceptions import DtypeMismatchError, ModelError
from tests.support.robots import two_link_arm

from ._public_ops import build_case

PACKAGES = (kinematics, dynamics, collision)


def _takes_model_and_state(function: object) -> bool:
    if not inspect.isfunction(function):
        return False
    return list(inspect.signature(function).parameters)[:2] == ["model", "state"]


STATE_FUNCTIONS = sorted(
    name for package in PACKAGES for name in package.__all__ if _takes_model_and_state(getattr(package, name))
)


@pytest.mark.parametrize("name", STATE_FUNCTIONS)
def test_state_of_another_model_is_rejected(name: str) -> None:
    function, (model, state), kwargs = build_case(name, torch.float64, torch.device("cpu"))
    function(model.with_values(armature=model.armature + 0.1), state, **kwargs)  # same robot, new values: fine
    with pytest.raises(ModelError, match="whose structure differs"):
        function(two_link_arm(torch.float64, axes=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))), state, **kwargs)
    with pytest.raises(DtypeMismatchError, match=r"state\.dtype=torch\.float64 != model\.dtype=torch\.float32"):
        function(model.to(dtype=torch.float32), state, **kwargs)
    with pytest.raises(TypeError, match="state must be a KinematicState"):
        function(model, state.q, **kwargs)
