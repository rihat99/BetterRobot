"""Regression net for the public differentiable robotics blocks."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import better_robot as br
from better_robot.dynamics.state import center_of_mass


@pytest.fixture(scope="module")
def panda():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description  # noqa: PLC0415

    return br.load(panda_description.URDF_PATH, dtype=torch.float64)


def _configuration(model) -> torch.Tensor:
    q = model.q_neutral.clone()
    q[3] = -1.0  # Panda joint 4's interval excludes the nominal zero.
    q[-1] = 0.02
    return q


def _hand_frame(model) -> int:
    for name in ("body_panda_hand", "body_panda_link8", "body_panda_link7"):
        if name in model.frame_name_to_id:
            return model.frame_id(name)
    raise AssertionError("Panda hand frame is missing")


def _flatten(*outputs: torch.Tensor) -> torch.Tensor:
    return torch.cat(tuple(output.reshape(-1) for output in outputs))


def _assert_gradients(output: torch.Tensor, *inputs: torch.Tensor) -> None:
    weights = torch.linspace(0.5, 1.5, output.numel(), dtype=output.dtype, device=output.device)
    gradients = torch.autograd.grad((output.reshape(-1) * weights).sum(), inputs)
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0.0


def _placement_model(model):
    delta = torch.zeros_like(model.values.joint_placements, requires_grad=True)
    return model.with_values(joint_placements=model.values.joint_placements + delta), delta


def _inertia_model(model):
    delta = torch.zeros_like(model.values.body_inertias, requires_grad=True)
    return model.with_values(body_inertias=model.values.body_inertias + delta), delta


def test_workspace_fk_and_public_jacobians_reach_q_and_model_values(panda) -> None:
    q = _configuration(panda).requires_grad_()
    model, placements = _placement_model(panda)
    state = br.forward_kinematics_state(model, q)
    frame = br.frame_jacobian(model, state, frame=_hand_frame(model))
    joint = br.joint_jacobian(model, state, joint=model.njoints - 1, reference="world")
    _assert_gradients(
        _flatten(
            state.joint_pose,
            state.frame_pose,
            frame,
            joint,
        ),
        q,
        placements,
    )


def test_frame_jacobian_reaches_q_and_model_values(panda) -> None:
    q = _configuration(panda).requires_grad_()
    model, joint_placements = _placement_model(panda)
    frame_placements = torch.zeros_like(model.values.frame_placements, requires_grad=True)
    model = model.with_values(frame_placements=model.values.frame_placements + frame_placements)
    jacobian = br.frame_jacobian(model, br.forward_kinematics_state(model, q), frame=_hand_frame(model))
    _assert_gradients(jacobian, q, joint_placements, frame_placements)


def _dynamics_inputs(model) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    velocity = torch.linspace(0.03, 0.15, model.nv, dtype=model.values.q_neutral.dtype)
    acceleration = torch.linspace(-0.07, 0.11, model.nv, dtype=velocity.dtype)
    torque = torch.linspace(0.1, 0.3, model.nv, dtype=velocity.dtype)
    return velocity, acceleration, torque


def _rnea(model, q, velocity, acceleration, _torque) -> torch.Tensor:
    return br.inverse_dynamics(model, br.forward_kinematics_state(model, q), v=velocity, a=acceleration)


def _aba(model, q, velocity, _acceleration, torque) -> torch.Tensor:
    return br.forward_dynamics(model, br.forward_kinematics_state(model, q), v=velocity, tau=torque)


def _crba(model, q, _velocity, _acceleration, _torque) -> torch.Tensor:
    return br.mass_matrix(model, br.forward_kinematics_state(model, q))


def _centroidal_map(model, q, _velocity, _acceleration, _torque) -> torch.Tensor:
    return br.centroidal_map(model, br.forward_kinematics_state(model, q))


def _centroidal_momentum(model, q, velocity, _acceleration, _torque) -> torch.Tensor:
    return br.centroidal_momentum(model, br.forward_kinematics_state(model, q), v=velocity)


def _center_of_mass(model, q, _velocity, _acceleration, _torque) -> torch.Tensor:
    return center_of_mass(model, br.forward_kinematics_state(model, q))


@pytest.mark.parametrize(
    "entry",
    (_rnea, _aba, _crba, _centroidal_map, _centroidal_momentum, _center_of_mass),
)
def test_public_dynamics_reaches_q_and_body_inertias(
    panda,
    entry: Callable,
) -> None:
    q = _configuration(panda).requires_grad_()
    model, inertias = _inertia_model(panda)
    velocity, acceleration, torque = _dynamics_inputs(model)
    _assert_gradients(entry(model, q, velocity, acceleration, torque), q, inertias)


def test_integrate_and_difference_reach_q(panda) -> None:
    step = torch.linspace(0.01, 0.04, panda.nv, dtype=torch.float64)

    q_integrate = _configuration(panda).requires_grad_()
    _assert_gradients(panda.integrate(q_integrate, step), q_integrate)

    target = panda.integrate(_configuration(panda), step)
    q_difference = _configuration(panda).requires_grad_()
    _assert_gradients(panda.difference(q_difference, target), q_difference)
