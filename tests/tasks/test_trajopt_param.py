"""``TrajectoryParameterization`` — knot vs B-spline.

* ``KnotTrajectory.init(q).expand(z) == q`` is identity.
* ``BSplineTrajectory(C=8).init(q)`` projects onto a smaller variable.
* Robot ``solve_trajopt`` supports knots and honestly rejects the numerical
  B-spline basis until manifold-safe trajectory optimisation lands in M5.

See ``docs/concepts/tasks.md §3``.
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
from better_robot.optim.optimizers.lm_then_lbfgs import LMThenLBFGS
from better_robot.residuals.pose import PoseResidual
from better_robot.residuals.temporal import TimeIndexedResidual
from better_robot.tasks.parameterization import (
    BSplineTrajectory,
    KnotTrajectory,
    TrajectoryParameterization,
)
from better_robot.tasks.trajopt import solve_trajopt


def _fixed_arm():
    builder = ModelBuilder("arm")
    builder.add_body("base", mass=0.5)
    builder.add_body("link1", mass=1.0)
    builder.add_revolute_z(
        "j1",
        parent="base",
        child="link1",
        origin=torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]),
        lower=-math.pi,
        upper=math.pi,
    )
    return build_model(builder.finalize())


def _floating_body():
    builder = ModelBuilder("floating")
    builder.add_body("base", mass=1.0)
    builder.add_free_flyer_root("floating_root", child="base")
    return build_model(builder.finalize())


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


def test_solve_trajopt_with_knots_reaches_target() -> None:
    """The supported knot parameterisation still closes an IK-target cost."""
    model = _fixed_arm()
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
        horizon=T,
        dt=0.05,
        initial_q_traj=initial_q_traj,
        cost_stack=stack,
        optimizer=LevenbergMarquardt(),
        max_iter=20,
        parameterization=KnotTrajectory(),
    )
    assert res.trajectory.q.shape == (1, T, model.nq)
    final_q = res.trajectory.q[0, -1]
    final_pose = forward_kinematics(model, final_q, compute_frames=True).frame_pose_world[fid]
    pos_err = float((final_pose[:3] - target[:3]).norm())
    assert pos_err < 0.05, f"final position error {pos_err:.4f}"


def test_solve_trajopt_rejects_floating_base_bspline() -> None:
    model = _floating_body()
    horizon = 8
    q_seed = model.q_neutral.unsqueeze(0).expand(horizon, -1).clone()

    with pytest.raises(NotImplementedError, match="manifold-safe.*M5"):
        solve_trajopt(
            model,
            horizon=horizon,
            dt=0.05,
            initial_q_traj=q_seed,
            cost_stack=CostStack(),
            optimizer=LevenbergMarquardt(),
            parameterization=BSplineTrajectory(num_control_points=4),
        )


def test_solve_trajopt_rejects_bounded_bspline() -> None:
    model = _fixed_arm()
    horizon = 8
    q_seed = model.q_neutral.unsqueeze(0).expand(horizon, -1).clone()

    with pytest.raises(NotImplementedError, match="bound preservation.*M5"):
        solve_trajopt(
            model,
            horizon=horizon,
            dt=0.05,
            initial_q_traj=q_seed,
            cost_stack=CostStack(),
            optimizer=LevenbergMarquardt(),
            lower=torch.full_like(model.q_neutral, -0.25),
            upper=torch.full_like(model.q_neutral, 0.25),
            parameterization=BSplineTrajectory(num_control_points=4),
        )


def test_solve_trajopt_rejects_multistage_bspline() -> None:
    model = _fixed_arm()
    horizon = 8
    q_seed = model.q_neutral.unsqueeze(0).expand(horizon, -1).clone()

    with pytest.raises(NotImplementedError, match="multi-stage replacement.*M5"):
        solve_trajopt(
            model,
            horizon=horizon,
            dt=0.05,
            initial_q_traj=q_seed,
            cost_stack=CostStack(),
            optimizer=LMThenLBFGS(),
            parameterization=BSplineTrajectory(num_control_points=4),
        )
