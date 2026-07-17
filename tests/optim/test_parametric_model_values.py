"""Synthetic shape-parameter flow through batched FK, IK cost, and dynamics."""

from __future__ import annotations

import torch

from better_robot.dynamics import rnea
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics.forward import forward_kinematics
from better_robot.optim import (
    Problem,
    ResidualItem,
    RobotConfig,
    RobotStateProvider,
    VarSpec,
)
from better_robot.residuals import PoseResidual


def _pose(x: float = 0.0) -> torch.Tensor:
    return torch.tensor([x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def _model():
    builder = ModelBuilder("synthetic_betas")
    builder.add_body("base", mass=1.0, inertia=torch.eye(3) * 0.1)
    builder.add_body("link", mass=1.5, inertia=torch.eye(3) * 0.2)
    builder.add_revolute_y(
        "joint",
        parent="base",
        child="link",
        origin=_pose(0.4),
        lower=-1.5,
        upper=1.5,
    )
    builder.add_frame("tip", parent_body="link", placement=_pose(0.3))
    return build_model(builder.finalize(), dtype=torch.float64)


def test_fake_betas_reach_batched_fk_ik_objective_and_rnea() -> None:
    model = _model()
    batch = 3
    betas = torch.tensor(
        [[0.1, -0.2], [0.0, 0.15], [-0.1, 0.25]],
        dtype=torch.float64,
        requires_grad=True,
    )

    placement_basis = torch.zeros(2, model.njoints, 7, dtype=torch.float64)
    placement_basis[0, 2, 0] = 0.2
    placement_basis[1, 2, 2] = -0.1
    inertia_basis = torch.zeros(2, model.nbodies, 10, dtype=torch.float64)
    inertia_basis[0, 2, 0] = 0.3
    inertia_basis[1, 2, 4:7] = torch.tensor([0.05, 0.04, 0.03])
    placements = model.values.joint_placements + torch.einsum(
        "bk,knd->bnd",
        betas,
        placement_basis,
    )
    inertias = model.values.body_inertias + torch.einsum(
        "bk,knd->bnd",
        betas,
        inertia_basis,
    )
    shaped = model.with_values(
        joint_placements=placements,
        body_inertias=inertias,
    )
    q = model.q_neutral.expand(batch, -1).clone()
    q[:, 0] = torch.tensor([0.1, -0.2, 0.3], dtype=q.dtype)

    fk = forward_kinematics(shaped, q, compute_frames=True)
    assert fk.frame_pose_world is not None
    target_q = q.detach().clone()
    target_q[:, 0] += 0.15
    target = (
        forward_kinematics(
            shaped,
            target_q,
            compute_frames=True,
        )
        .frame_pose_world[..., shaped.frame_id("tip"), :]
        .detach()
    )

    problem = Problem(
        vars=(VarSpec("q", (model.nq,), manifold=RobotConfig(shaped)),),
        residuals=(
            ResidualItem(
                "pose",
                PoseResidual(
                    model=shaped,
                    frame_id=shaped.frame_id("tip"),
                    target=target,
                ),
            ),
        ),
        providers=(RobotStateProvider(shaped),),
    )
    ik_cost = problem.objective({"q": q}).sum()
    v = torch.zeros(batch, model.nv, dtype=q.dtype)
    a = torch.full_like(v, 0.2)
    tau = rnea(shaped, shaped.create_data(), q, v, a)
    loss = ik_cost + 0.01 * tau.square().sum() + 0.01 * fk.joint_pose_world.square().sum()
    loss.backward()

    assert betas.grad is not None
    assert betas.grad.shape == betas.shape
    assert torch.isfinite(betas.grad).all()
    assert betas.grad.abs().sum() > 0
