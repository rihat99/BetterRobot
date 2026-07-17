"""Public named-block acceptance: jointly recover robot q and camera pose."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.kinematics import forward_kinematics
from better_robot.lie import se3
from better_robot.optim import (
    Bounds,
    LevenbergMarquardt,
    Problem,
    ResidualItem,
    RobotConfig,
    RobotStateProvider,
    SE3Manifold,
    VarSpec,
)


class _CameraFrameObservations:
    """Test-local consumer residual coupling FK and an SE(3) camera block."""

    name = "camera_frames"
    reads = ("q", "T_cam", "data", "base_target", "tip_target")
    dim = 12

    def __init__(self, base_frame: int, tip_frame: int) -> None:
        self.base_frame = base_frame
        self.tip_frame = tip_frame

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        frame_poses = ctx["data"].frame_pose_world
        assert frame_poses is not None
        world_from_camera = ctx["T_cam"]
        camera_from_world = se3.inverse(world_from_camera)
        predicted_base = se3.compose(camera_from_world, frame_poses[..., self.base_frame, :])
        predicted_tip = se3.compose(camera_from_world, frame_poses[..., self.tip_frame, :])
        base_error = se3.log(se3.compose(se3.inverse(ctx["base_target"]), predicted_base))
        tip_error = se3.log(se3.compose(se3.inverse(ctx["tip_target"]), predicted_tip))
        return torch.cat((base_error, tip_error), dim=-1)


def _build_one_joint_robot(dtype: torch.dtype):
    builder = ModelBuilder("q_camera_acceptance")
    base = builder.add_body("base")
    tip = builder.add_body("tip", mass=1.0)
    builder.add_revolute_z(
        "joint",
        parent=base,
        child=tip,
        origin=torch.tensor([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=dtype),
        lower=-1.2,
        upper=1.2,
    )
    return build_model(builder.finalize(), dtype=dtype)


def test_joint_q_and_camera_extrinsics_solve_through_public_block_api() -> None:
    dtype = torch.float64
    model = _build_one_joint_robot(dtype)
    base_frame = model.frame_id("body_base")
    tip_frame = model.frame_id("body_tip")

    q_true = torch.tensor([0.38], dtype=dtype)
    camera_true = se3.exp(torch.tensor([0.30, -0.22, 0.18, 0.16, -0.11, 0.07], dtype=dtype))
    true_data = forward_kinematics(model, q_true, compute_frames=True)
    assert true_data.frame_pose_world is not None
    camera_from_world = se3.inverse(camera_true)
    base_target = se3.compose(camera_from_world, true_data.frame_pose_world[base_frame])
    tip_target = se3.compose(camera_from_world, true_data.frame_pose_world[tip_frame])

    residual = _CameraFrameObservations(base_frame, tip_frame)
    problem = Problem(
        vars=(
            VarSpec(
                "q",
                (model.nq,),
                manifold=RobotConfig(model),
                bounds=Bounds(model.lower_pos_limit, model.upper_pos_limit),
            ),
            VarSpec("T_cam", (7,), manifold=SE3Manifold()),
        ),
        residuals=(ResidualItem(residual.name, residual, group_size=6),),
        providers=(RobotStateProvider(model),),
        parameters={"base_target": base_target, "tip_target": tip_target},
    )
    initial = {
        "q": torch.tensor([-0.31], dtype=dtype),
        "T_cam": se3.compose(
            camera_true,
            se3.exp(torch.tensor([-0.20, 0.16, -0.12, -0.13, 0.09, -0.08], dtype=dtype)),
        ),
    }

    gradient = problem.gradient(initial)
    assert gradient["q"].norm() > 1e-3
    assert gradient["T_cam"].norm() > 1e-3

    values, state = LevenbergMarquardt(
        max_iter=40,
        gtol=1e-10,
        xtol=1e-12,
        ftol=1e-12,
    ).run(initial, problem)

    assert bool(state.converged)
    assert state.cost < 1e-16
    torch.testing.assert_close(values["q"], q_true, atol=2e-8, rtol=2e-8)
    camera_error = se3.log(se3.compose(se3.inverse(camera_true), values["T_cam"]))
    torch.testing.assert_close(camera_error, torch.zeros(6, dtype=dtype), atol=2e-8, rtol=0.0)
    torch.testing.assert_close(values["T_cam"][3:].norm(), torch.tensor(1.0, dtype=dtype))
