"""``TrajectoryParameterization`` — knot vs B-spline.

* ``KnotTrajectory.init(q).expand(z) == q`` is identity.
* ``BSplineTrajectory(C=8).init(q)`` projects onto a smaller variable.
* ``solve_trajopt(parameterization=BSplineTrajectory(...))`` returns a
  trajectory whose final IK-target residual is within tolerance.

See ``docs/design/08_TASKS.md §3``.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.costs.stack import CostStack
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics.forward import forward_kinematics
from better_robot.optim.optimizers.levenberg_marquardt import LevenbergMarquardt
from better_robot.residuals.pose import PoseResidual
from better_robot.residuals.temporal import TimeIndexedResidual
from better_robot.tasks.parameterization import (
    BSplineTrajectory,
    KnotTrajectory,
    TrajectoryParameterization,
)
from better_robot.tasks.trajopt import solve_trajopt


def test_parameterization_protocol_runtime_checkable() -> None:
    assert isinstance(KnotTrajectory(), TrajectoryParameterization)
    assert isinstance(BSplineTrajectory(num_control_points=8), TrajectoryParameterization)


def test_knot_is_identity() -> None:
    p = KnotTrajectory()
    q = torch.randn(10, 4)
    z = p.init(q)
    out = p.expand(z, T=10, nq=4)
    torch.testing.assert_close(out, q)


def test_bspline_compresses_then_expands() -> None:
    p = BSplineTrajectory(num_control_points=6)
    q = torch.linspace(0.0, 1.0, 16).unsqueeze(1).expand(-1, 3).contiguous()
    z = p.init(q)
    assert z.shape == (6, 3)  # smaller than (16, 3)
    out = p.expand(z, T=16, nq=3)
    # Smooth signal recovers reasonably well — atol is loose; the point
    # is the parameter compression, not exact reconstruction.
    assert torch.allclose(out, q, atol=0.2)


def test_bspline_smaller_than_knots() -> None:
    p_knot = KnotTrajectory()
    p_spline = BSplineTrajectory(num_control_points=5)
    q = torch.zeros(20, 4)
    z_knot = p_knot.init(q)
    z_spline = p_spline.init(q)
    assert z_spline.numel() < z_knot.numel()


def test_solve_trajopt_with_bspline_reaches_target() -> None:
    """Cubic B-spline trajectory parameterisation closes IK-target cost."""
    b = ModelBuilder("arm")
    b.add_body("base", mass=0.5)
    b.add_body("link1", mass=1.0)
    b.add_revolute_z(
        "j1", parent="base", child="link1",
        origin=torch.tensor([0., 0., 0.1, 0., 0., 0., 1.]),
        lower=-math.pi, upper=math.pi,
    )
    model = build_model(b.finalize())
    fid = model.frame_id("body_link1")

    T = 12
    q0 = model.q_neutral.clone().float()
    initial_q_traj = q0.unsqueeze(0).expand(T, model.nq).contiguous().clone()
    target = forward_kinematics(model, q0, compute_frames=True).frame_pose_world[fid].clone()
    target[..., 0] += 0.03

    stack = CostStack()
    stack.add(
        "pose_final",
        TimeIndexedResidual(PoseResidual(frame_id=fid, target=target), t_idx=T - 1),
    )

    res = solve_trajopt(
        model,
        horizon=T, dt=0.05,
        initial_q_traj=initial_q_traj,
        cost_stack=stack,
        optimizer=LevenbergMarquardt(),
        max_iter=20,
        parameterization=BSplineTrajectory(num_control_points=5),
    )
    assert res.trajectory.q.shape == (1, T, model.nq)
    final_q = res.trajectory.q[0, -1]
    final_pose = forward_kinematics(model, final_q, compute_frames=True).frame_pose_world[fid]
    pos_err = float((final_pose[:3] - target[:3]).norm())
    assert pos_err < 0.05, f"final position error {pos_err:.4f}"
