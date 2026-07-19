"""Inverse contact-force task coverage."""

from __future__ import annotations

import pytest
import torch

from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics.forward import forward_kinematics
from better_robot.lie import so3
from better_robot.optim import Variable
from better_robot.residuals import Node
from better_robot.tasks.contact_forces import (
    ContactForceWeights,
    _ContactDynamicsNode,
    _ForceSmoothResidual,
    _TorqueSmoothResidual,
    _trajectory_derivatives,
    solve_contact_forces,
)


class _FixedDynamics(Node):
    def __init__(self, generalized_force: torch.Tensor) -> None:
        self.generalized_force = generalized_force
        super().__init__()

    def compute(self) -> dict[str, torch.Tensor]:
        return {"generalized_force": self.generalized_force}


def _floating_body(*, dtype: torch.dtype = torch.float64):
    builder = ModelBuilder("contact_force_body")
    builder.add_body(
        "base",
        mass=2.0,
        inertia=torch.diag(torch.tensor([0.2, 0.25, 0.3])),
    )
    builder.add_free_flyer_root(
        "floating",
        child="base",
        origin=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )
    return build_model(builder.finalize(), dtype=dtype)


def _clip(model, time: int = 3) -> torch.Tensor:
    return model.q_neutral.expand(time, -1).clone()


def test_contact_force_solve_reduces_floating_base_wrench() -> None:
    model = _floating_body()
    q = _clip(model)
    result = solve_contact_forces(
        model,
        q,
        [1],
        torch.ones(3, 1, dtype=torch.bool),
        dt=0.05,
        weights=ContactForceWeights(
            base_wrench=1.0,
            force_magnitude=1e-6,
            force_smooth=1e-3,
        ),
        max_iter=30,
        tolerance=1e-9,
    )

    gravity_only = model.values.gravity[:3].norm() * 2.0
    assert result.generalized_force[..., :6].norm(dim=-1).max() < gravity_only * 1e-3
    assert result.forces_world.shape == (3, 1, 3)
    assert result.fext_local.shape == (3, model.njoints, 6)


def test_contact_force_node_is_differentiable_through_fext() -> None:
    model = _floating_body()
    q = _clip(model)
    data = forward_kinematics(model, q)
    velocity, acceleration = _trajectory_derivatives(model, data.q, 0.1)
    ids = torch.tensor([1])
    pose = data.joint_pose_world.index_select(-2, ids)
    forces = torch.randn(3, 1, 3, dtype=torch.float64, requires_grad=True)
    force_variable = Variable(forces, name="forces")
    node = _ContactDynamicsNode(
        force_variable,
        model=model,
        q=data.q,
        velocity=velocity,
        acceleration=acceleration,
        world_to_local=so3.to_matrix(pose[..., 3:]).mT,
        active=torch.ones(3, 1, dtype=torch.float64),
        contact_to_joint=torch.nn.functional.one_hot(ids, model.njoints).to(torch.float64),
        values=model.values,
    )

    def generalized_force(value: torch.Tensor) -> torch.Tensor:
        force_variable.tensor = value
        return node.compute()["generalized_force"][..., :6]

    assert torch.autograd.gradcheck(
        generalized_force,
        (forces,),
        atol=2e-5,
        rtol=2e-4,
    )


def test_contact_force_result_preserves_available_output_graphs() -> None:
    model = _floating_body()
    q = _clip(model, time=1)
    active = torch.ones(1, 1, dtype=q.dtype, requires_grad=True)
    gravity = model.values.gravity[:3].detach().clone().requires_grad_()

    result = solve_contact_forces(
        model,
        q,
        [1],
        active,
        dt=0.1,
        gravity=gravity,
        weights=ContactForceWeights(base_wrench=1.0, force_magnitude=1e-6),
        max_iter=30,
        tolerance=1e-9,
    )
    active_gradient = torch.autograd.grad(result.fext_local.sum(), active, retain_graph=True)[0]
    gravity_gradient = torch.autograd.grad(result.generalized_force.sum(), gravity)[0]

    assert not result.forces_world.requires_grad
    assert result.fext_local.requires_grad
    assert result.generalized_force.requires_grad
    assert torch.isfinite(active_gradient).all() and active_gradient.abs().max() > 0.0
    assert torch.isfinite(gravity_gradient).all() and gravity_gradient.abs().max() > 0.0


def test_force_and_torque_smooth_terms_match_hand_differences() -> None:
    forces = torch.tensor([[[1.0, 0.0, 0.0]], [[3.0, 1.0, 0.0]], [[2.0, 4.0, 1.0]]])
    force_rows = _ForceSmoothResidual(Variable(forces), 3, 1).error()
    expected_force = torch.tensor([2.0, 1.0, 0.0, -1.0, 3.0, 1.0])
    torch.testing.assert_close(force_rows, expected_force)

    tau = torch.zeros(3, 8)
    tau[:, 6:] = torch.tensor([[1.0, 2.0], [4.0, 3.0], [2.0, 8.0]])
    torque_rows = _TorqueSmoothResidual(_FixedDynamics(tau), 3, 2).error()
    torch.testing.assert_close(torque_rows, torch.tensor([3.0, 1.0, -2.0, 5.0]))


def test_batched_contact_solve_matches_sequential() -> None:
    model = _floating_body()
    q = _clip(model)
    q_batch = q.expand(2, -1, -1).clone()
    active = torch.ones(2, 3, 1, dtype=torch.bool)
    cfg = ContactForceWeights(base_wrench=1.0, force_magnitude=1e-5)

    batched = solve_contact_forces(model, q_batch, [1], active, dt=0.1, weights=cfg, max_iter=20)
    sequential = [
        solve_contact_forces(model, q_batch[index], [1], active[index], dt=0.1, weights=cfg, max_iter=20)
        for index in range(2)
    ]
    expected = torch.stack([item.forces_world for item in sequential])
    torch.testing.assert_close(batched.forces_world, expected, rtol=2e-5, atol=2e-6)


def test_contact_inputs_reject_lossy_ids_and_invalid_float_masks() -> None:
    model = _floating_body()
    q = _clip(model)

    with pytest.raises(TypeError, match="contain integers"):
        solve_contact_forces(model, q, [1.5], torch.ones(3, 1), dt=0.1)
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        solve_contact_forces(model, q, [1], torch.full((3, 1), 1.5), dt=0.1)


def test_contact_force_gravity_accepts_per_clip_values() -> None:
    model = _floating_body()
    q = _clip(model).expand(2, -1, -1).clone()
    gravity = torch.stack(
        (
            torch.zeros(3, dtype=q.dtype),
            model.values.gravity[:3],
        )
    )
    result = solve_contact_forces(
        model,
        q,
        [1],
        torch.ones(2, 3, 1, dtype=torch.bool),
        dt=0.1,
        gravity=gravity,
        weights=ContactForceWeights(base_wrench=1.0, force_magnitude=1e-6),
        max_iter=30,
        tolerance=1e-9,
    )

    assert result.forces_world[0].abs().max() < 1e-10
    assert result.generalized_force[..., :6].norm(dim=-1).max() < 1e-4
