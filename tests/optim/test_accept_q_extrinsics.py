"""Public v2 acceptance: jointly recover robot q and camera pose."""

from __future__ import annotations

import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.kinematics import forward_kinematics
from better_robot.lie import se3
from better_robot.optim import (
    LevenbergMarquardt,
    Problem,
    Residual,
    RobotVariable,
    SE3Variable,
    Variable,
)
from better_robot.residuals.nodes import RobotState


class _CameraFrameObservations(Residual):
    """Test-local consumer residual coupling FK and an SE(3) camera block."""

    def __init__(
        self,
        state: RobotState,
        camera: SE3Variable,
        base_target: Variable,
        tip_target: Variable,
        *,
        base_frame: int,
        tip_frame: int,
    ) -> None:
        self.state = state
        self.nodes = (state,)
        self.q = state.q
        self.camera = camera
        self.base_target = base_target
        self.tip_target = tip_target
        self.base_frame = base_frame
        self.tip_frame = tip_frame
        super().__init__(
            self.q,
            camera,
            base_target,
            tip_target,
            dim=12,
            group_size=6,
            name="camera_frames",
        )

    def error(self) -> torch.Tensor:
        frame_poses = self.state.value().frame_pose_world
        assert frame_poses is not None
        camera_from_world = se3.inverse(self.camera.tensor)
        predicted_base = se3.compose(camera_from_world, frame_poses[..., self.base_frame, :])
        predicted_tip = se3.compose(camera_from_world, frame_poses[..., self.tip_frame, :])
        base_error = se3.log(se3.compose(se3.inverse(self.base_target.tensor), predicted_base))
        tip_error = se3.log(se3.compose(se3.inverse(self.tip_target.tensor), predicted_tip))
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
    base_target_tensor = se3.compose(camera_from_world, true_data.frame_pose_world[base_frame])
    tip_target_tensor = se3.compose(camera_from_world, true_data.frame_pose_world[tip_frame])

    q = RobotVariable(model, torch.tensor([-0.31], dtype=dtype), name="q", bounds=True)
    camera = SE3Variable(
        se3.compose(
            camera_true,
            se3.exp(torch.tensor([-0.20, 0.16, -0.12, -0.13, 0.09, -0.08], dtype=dtype)),
        ),
        name="T_cam",
    )
    base_target = Variable(base_target_tensor, name="base_target", trainable=False)
    tip_target = Variable(tip_target_tensor, name="tip_target", trainable=False)
    residual = _CameraFrameObservations(
        RobotState(q),
        camera,
        base_target,
        tip_target,
        base_frame=base_frame,
        tip_frame=tip_frame,
    )
    problem = Problem([residual])

    gradient = problem.gradient()
    assert gradient["q"].norm() > 1e-3
    assert gradient["T_cam"].norm() > 1e-3

    optimizer = LevenbergMarquardt(
        problem,
        max_iterations=40,
        tolerance=1e-10,
        step_tolerance=1e-12,
        relative_tolerance=1e-12,
    )
    info = optimizer.optimize()

    assert bool(info.converged)
    assert info.cost < 1e-16
    torch.testing.assert_close(q.tensor, q_true, atol=2e-8, rtol=2e-8)
    camera_error = se3.log(se3.compose(se3.inverse(camera_true), camera.tensor))
    torch.testing.assert_close(camera_error, torch.zeros(6, dtype=dtype), atol=2e-8, rtol=0.0)
    torch.testing.assert_close(camera.tensor[3:].norm(), torch.tensor(1.0, dtype=dtype))
