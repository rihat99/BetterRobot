"""Public passes validate each tensor input once, at their boundary, and trust the attached model (record 0021)."""

from __future__ import annotations

import pytest
import torch

from better_robot import dynamics, kinematics
from better_robot.model import _checks

from ._public_ops import CASE_NAMES, build_case

_CPU = torch.device("cpu")

#: ``check_tensor`` calls per public call; a name absent here makes none (Jacobians read the state, lie/spatial
#: take plain tensors). ``forward_dynamics`` checks ``v``, the zero acceleration of its bias call and ``tau``.
EXPECTED_CHECKS = {
    "forward_kinematics_state": 1,
    "frame_jacobian_time_variation": 1,
    "joint_jacobian_time_variation": 1,
    "inverse_dynamics": 2,
    "forward_dynamics": 3,
    "centroidal_momentum": 1,
}


@pytest.fixture
def checked_names(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Names passed to ``check_tensor`` by the kinematics and dynamics boundaries during the test."""
    names: list[str] = []
    original = _checks.check_tensor

    def counted(name: str, value: object, **kwargs: object) -> torch.Tensor:
        names.append(name)
        return original(name, value, **kwargs)

    for module in (kinematics.state, dynamics.state):
        monkeypatch.setattr(module, "check_tensor", counted)
    return names


@pytest.mark.parametrize("name", CASE_NAMES)
def test_each_public_call_validates_its_inputs_once(name: str, checked_names: list[str]) -> None:
    fn, args, kwargs = build_case(name, torch.float64, _CPU)
    checked_names.clear()  # building the case ran forward kinematics once
    fn(*args, **kwargs)
    assert len(checked_names) == EXPECTED_CHECKS.get(name, 0), checked_names


def test_dynamics_on_a_fresh_state_validates_q_once_and_its_own_inputs(checked_names: list[str]) -> None:
    _, (model, _), kwargs = build_case("inverse_dynamics", torch.float64, _CPU)
    q = torch.tensor([0.3, -0.4], dtype=torch.float64)
    checked_names.clear()
    dynamics.inverse_dynamics(model, kinematics.forward_kinematics_state(model, q), **kwargs)
    assert checked_names == ["q", "v", "a"]
