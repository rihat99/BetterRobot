"""Mimic FK/Jacobian parity against an explicit unconstrained twin."""

from __future__ import annotations

import dataclasses

import torch

from better_robot.kinematics.configuration import full_configuration, full_tangent, mimic_plan
from better_robot.io import ModelBuilder, build_model
from better_robot.kinematics import forward_kinematics, joint_jacobian, joint_jacobian_time_variation


def _model_pair():
    builder = ModelBuilder("mimic_kinematics")
    root = builder.add_body("root")
    first = builder.add_body("first")
    second = builder.add_body("second")
    builder.add_revolute(
        "source", axis=(0.0, 0.0, 1.0),
        parent=root,
        child=first,
        origin=torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        lower=-1.0,
        upper=1.0,
    )
    builder.add_revolute(
        "target", axis=(0.0, 0.0, 1.0),
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
    q_full = full_configuration(constrained, q)

    constrained_state = forward_kinematics(constrained, q)
    full_state = forward_kinematics(full, q_full)
    torch.testing.assert_close(
        constrained_state.joint_poses,
        full_state.joint_poses,
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        constrained_state.frame_poses,
        full_state.frame_poses,
        rtol=1e-12,
        atol=1e-12,
    )
    expected = (
        torch.stack([joint_jacobian(full, full_state, joint=j, reference="world") for j in range(full.njoints)], dim=-3)
        @ mimic_plan(constrained, dtype=torch.float64, device=torch.device("cpu")).v_expansion
    )
    torch.testing.assert_close(
        torch.stack(
            [
                joint_jacobian(constrained, constrained_state, joint=j, reference="world")
                for j in range(constrained.njoints)
            ],
            dim=-3,
        ),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_batched_jacobian_time_variation_equals_full_twin_with_chain_rule() -> None:
    constrained, full = _model_pair()
    q = torch.linspace(-0.3, 0.3, 6, dtype=torch.float64).reshape(2, 3, 1)
    q_full = full_configuration(constrained, q)
    v = torch.linspace(0.4, -0.6, 6, dtype=torch.float64).reshape(2, 3, 1)
    v_full = full_tangent(constrained, v)

    constrained_state = forward_kinematics(constrained, q)
    full_state = forward_kinematics(full, q_full)
    expected = (
        torch.stack(
            [
                joint_jacobian_time_variation(full, full_state, v=v_full, joint=j, reference="world")
                for j in range(full.njoints)
            ],
            dim=-3,
        )
        @ mimic_plan(constrained, dtype=torch.float64, device=torch.device("cpu")).v_expansion
    )
    torch.testing.assert_close(
        torch.stack(
            [
                joint_jacobian_time_variation(constrained, constrained_state, v=v, joint=j, reference="world")
                for j in range(constrained.njoints)
            ],
            dim=-3,
        ),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
