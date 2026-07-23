"""Mimic FK/Jacobian parity against an explicit unconstrained twin."""

from __future__ import annotations

import dataclasses

import torch

from better_robot.data_model.reduced_coordinates import expand_configuration, expand_tangent
from better_robot.io import ModelBuilder, build_model
from better_robot.kinematics import (
    compute_joint_jacobians,
    compute_joint_jacobians_time_variation,
    forward_kinematics,
)


def _model_pair():
    builder = ModelBuilder("mimic_kinematics")
    root = builder.add_body("root")
    first = builder.add_body("first")
    second = builder.add_body("second")
    builder.add_revolute_z(
        "source",
        parent=root,
        child=first,
        origin=torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        lower=-1.0,
        upper=1.0,
    )
    builder.add_revolute_z(
        "target",
        parent=first,
        child=second,
        origin=torch.tensor([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        lower=-2.0,
        upper=2.0,
        mimic_source="source",
        mimic_multiplier=-0.5,
        mimic_offset=0.2,
    )
    constrained_ir = builder.finalize()
    full_ir = dataclasses.replace(
        constrained_ir,
        joints=[dataclasses.replace(joint, mimic_source=None) for joint in constrained_ir.joints],
    )
    return (
        build_model(constrained_ir, dtype=torch.float64),
        build_model(full_ir, dtype=torch.float64),
    )


def test_batched_fk_and_jacobians_equal_full_twin_with_chain_rule() -> None:
    constrained, full = _model_pair()
    q = torch.linspace(-0.3, 0.3, 6, dtype=torch.float64).reshape(2, 3, 1)
    q_full = expand_configuration(constrained.structure, q)

    constrained_data = forward_kinematics(constrained, q, compute_frames=True)
    full_data = forward_kinematics(full, q_full, compute_frames=True)
    torch.testing.assert_close(
        constrained_data.joint_pose_world,
        full_data.joint_pose_world,
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        constrained_data.frame_pose_world,
        full_data.frame_pose_world,
        rtol=1e-12,
        atol=1e-12,
    )

    compute_joint_jacobians(constrained, constrained_data)
    compute_joint_jacobians(full, full_data)
    expected = full_data.joint_jacobians @ constrained.v_expansion
    torch.testing.assert_close(
        constrained_data.joint_jacobians,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_batched_jacobian_time_variation_equals_full_twin_with_chain_rule() -> None:
    constrained, full = _model_pair()
    q = torch.linspace(-0.3, 0.3, 6, dtype=torch.float64).reshape(2, 3, 1)
    q_full = expand_configuration(constrained.structure, q)
    v = torch.linspace(0.4, -0.6, 6, dtype=torch.float64).reshape(2, 3, 1)
    v_full = expand_tangent(constrained.structure, v)

    constrained_data = forward_kinematics(constrained, q)
    constrained_data.v = v
    full_data = forward_kinematics(full, q_full)
    full_data.v = v_full

    compute_joint_jacobians_time_variation(constrained, constrained_data)
    compute_joint_jacobians_time_variation(full, full_data)
    expected = full_data.joint_jacobians_dot @ constrained.v_expansion
    torch.testing.assert_close(
        constrained_data.joint_jacobians_dot,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
