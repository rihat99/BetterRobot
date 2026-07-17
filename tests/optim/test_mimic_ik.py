"""Named-block IK coverage for a reduced-coordinate mimic gripper."""

from __future__ import annotations

import torch

from better_robot.data_model.joint_models import JointPX
from better_robot.data_model.reduced_coordinates import expand_configuration
from better_robot.io import ModelBuilder, build_model
from better_robot.kinematics import forward_kinematics
from better_robot.tasks.ik import IKCostConfig, OptimizerConfig, solve_ik


def _mimic_gripper():
    identity = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    builder = ModelBuilder("mimic_parallel_gripper")
    base = builder.add_body("base")
    left = builder.add_body("left_finger")
    right = builder.add_body("right_finger")
    builder.add_joint(
        "left_slide",
        kind=JointPX(),
        parent=base,
        child=left,
        lower=0.0,
        upper=0.4,
    )
    builder.add_joint(
        "right_slide",
        kind=JointPX(),
        parent=base,
        child=right,
        lower=-0.4,
        upper=0.0,
        mimic_source="left_slide",
        mimic_multiplier=-1.0,
    )
    builder.add_frame("right_tip", parent_body=right, placement=identity)
    return build_model(builder.finalize(), dtype=torch.float64)


def test_solve_ik_optimizes_a_mimic_gripper_in_reduced_coordinates() -> None:
    """The public facade carries the mimic reduction through named-block LM."""
    model = _mimic_gripper()
    target_q = torch.tensor([0.27], dtype=torch.float64)
    target = forward_kinematics(model, target_q, compute_frames=True).frame_pose_world[model.frame_id("right_tip")]

    result = solve_ik(
        model,
        {"right_tip": target},
        initial_q=torch.tensor([0.04], dtype=torch.float64),
        cost_cfg=IKCostConfig(limit_weight=0.0, rest_weight=0.0),
        optimizer_cfg=OptimizerConfig(max_iter=20, tol=1e-10),
    )

    assert model.has_mimic
    assert (model.nq, model.nq_full) == (1, 2)
    assert result.converged is True
    torch.testing.assert_close(result.q, target_q, rtol=0.0, atol=1e-10)
    torch.testing.assert_close(
        expand_configuration(model.structure, result.q),
        torch.tensor([0.27, -0.27], dtype=torch.float64),
        rtol=0.0,
        atol=1e-10,
    )
    torch.testing.assert_close(result.frame_pose("right_tip"), target, rtol=0.0, atol=1e-10)
