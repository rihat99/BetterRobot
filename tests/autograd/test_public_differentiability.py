"""Regression net for the public differentiable robotics blocks."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import better_robot as br
from better_robot.dynamics import aba_raw, ccrba_raw, crba_raw, rnea_raw
from better_robot.kinematics import (
    compute_joint_jacobians,
    forward_kinematics_raw,
    frame_placements_raw,
    get_frame_jacobian,
    get_joint_jacobian,
    joint_jacobians_raw,
)
from better_robot.optim import RobotVariable
from better_robot.residuals import PoseResidual


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


def test_fk_and_frame_raw_blocks_reach_q_and_model_values(panda) -> None:
    q = _configuration(panda).requires_grad_()
    model, placements = _placement_model(panda)
    fk = forward_kinematics_raw(model.structure, model.values, q)
    frames = frame_placements_raw(model.structure, model.values, fk.joint_pose_world)
    _assert_gradients(
        _flatten(fk.joint_pose_world, fk.joint_pose_local, frames.frame_pose_world),
        q,
        placements,
    )


def test_workspace_fk_and_public_jacobians_reach_q_and_model_values(panda) -> None:
    q = _configuration(panda).requires_grad_()
    model, placements = _placement_model(panda)
    data = br.forward_kinematics(model, q, compute_frames=True)
    compute_joint_jacobians(model, data)
    frame = get_frame_jacobian(model, data, _hand_frame(model))
    joint = get_joint_jacobian(model, data, model.njoints - 1)
    _assert_gradients(
        _flatten(
            data.joint_pose_world,
            data.frame_pose_world,
            data.joint_jacobians,
            frame,
            joint,
        ),
        q,
        placements,
    )


def test_joint_jacobians_raw_reaches_q_and_model_values(panda) -> None:
    q = _configuration(panda).requires_grad_()
    model, placements = _placement_model(panda)
    fk = forward_kinematics_raw(model.structure, model.values, q)
    result = joint_jacobians_raw(model.structure, q, fk.joint_pose_world)
    _assert_gradients(result.joint_jacobians, q, placements)


def _dynamics_inputs(model) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    velocity = torch.linspace(0.03, 0.15, model.nv, dtype=model.values.q_neutral.dtype)
    acceleration = torch.linspace(-0.07, 0.11, model.nv, dtype=velocity.dtype)
    torque = torch.linspace(0.1, 0.3, model.nv, dtype=velocity.dtype)
    return velocity, acceleration, torque


def _rnea(model, q, velocity, acceleration, _torque) -> torch.Tensor:
    return br.rnea(model, q, velocity, acceleration)


def _aba(model, q, velocity, _acceleration, torque) -> torch.Tensor:
    return br.aba(model, q, velocity, torque)


def _crba(model, q, _velocity, _acceleration, _torque) -> torch.Tensor:
    return br.crba(model, q)


def _ccrba(model, q, velocity, _acceleration, _torque) -> torch.Tensor:
    centroidal_map, momentum = br.dynamics.ccrba(model, q, velocity)
    return _flatten(centroidal_map, momentum)


def _centroidal_map(model, q, _velocity, _acceleration, _torque) -> torch.Tensor:
    return br.compute_centroidal_map(model, q)


def _centroidal_momentum(model, q, velocity, _acceleration, _torque) -> torch.Tensor:
    return br.dynamics.compute_centroidal_momentum(model, q, velocity)


def _center_of_mass(model, q, velocity, _acceleration, _torque) -> torch.Tensor:
    return br.center_of_mass(model, q, velocity)


@pytest.mark.parametrize(
    "entry",
    (_rnea, _aba, _crba, _ccrba, _centroidal_map, _centroidal_momentum, _center_of_mass),
)
def test_public_dynamics_reaches_q_and_body_inertias(
    panda,
    entry: Callable,
) -> None:
    q = _configuration(panda).requires_grad_()
    model, inertias = _inertia_model(panda)
    velocity, acceleration, torque = _dynamics_inputs(model)
    _assert_gradients(entry(model, q, velocity, acceleration, torque), q, inertias)


def _rnea_raw(model, q, velocity, acceleration, _torque) -> torch.Tensor:
    return rnea_raw(model.structure, model.values, q, velocity, acceleration).tau


def _aba_raw(model, q, velocity, _acceleration, torque) -> torch.Tensor:
    return aba_raw(model.structure, model.values, q, velocity, torque).ddq


def _crba_raw(model, q, _velocity, _acceleration, _torque) -> torch.Tensor:
    return crba_raw(model.structure, model.values, q).mass_matrix


def _ccrba_raw(model, q, velocity, _acceleration, _torque) -> torch.Tensor:
    result = ccrba_raw(model.structure, model.values, q, velocity)
    assert result.momentum is not None
    return _flatten(result.centroidal_map, result.momentum)


@pytest.mark.parametrize("entry", (_rnea_raw, _aba_raw, _crba_raw, _ccrba_raw))
def test_raw_dynamics_reaches_q_and_body_inertias(panda, entry: Callable) -> None:
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


def test_pose_residual_reaches_q_and_model_values(panda) -> None:
    q = _configuration(panda).requires_grad_()
    model, placements = _placement_model(panda)
    target_q = model.integrate(_configuration(model), torch.full((model.nv,), 0.02, dtype=q.dtype))
    target = br.forward_kinematics(model, target_q, compute_frames=True).frame_pose_world[_hand_frame(model)].detach()
    q_variable = RobotVariable(model, q, name="q")
    residual = PoseResidual(q_variable, frame_id=_hand_frame(model), target=target)
    _assert_gradients(residual.error(), q, placements)
