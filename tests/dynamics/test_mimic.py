"""Projected mimic dynamics against an explicit unconstrained twin."""

from __future__ import annotations

import dataclasses

import torch

from better_robot.dynamics import (
    centroidal_map,
    centroidal_momentum,
    forward_dynamics,
    inverse_dynamics,
    mass_matrix,
)
from better_robot.kinematics import forward_kinematics
from better_robot.io import ModelBuilder, build_model
from better_robot.model import JointRevolute

# The mimic rule target = -0.5 source + 0.2, written by hand: the twin's configuration is [q, -0.5 q + 0.2] and its
# tangent EXPANSION @ v, so its Jacobians and torques map to the reduced ones through EXPANSION.
EXPANSION = torch.tensor([[1.0], [-0.5]], dtype=torch.float64)  # (nv_full, nv)


def _model_pair():
    inertia = torch.diag(torch.tensor([0.2, 0.3, 0.4]))
    builder = ModelBuilder("mimic_dynamics")
    root = builder.add_body("root", mass=1.0, inertia=inertia)
    first = builder.add_body("first", mass=1.3, com=torch.tensor([0.2, 0.0, 0.0]), inertia=inertia)
    second = builder.add_body("second", mass=0.7, com=torch.tensor([0.25, 0.0, 0.0]), inertia=inertia)
    builder.add_joint(
        "source",
        kind=JointRevolute(axis=(0.0, 0.0, 1.0)),
        parent=root,
        child=first,
        origin=torch.tensor([0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        lower=-1.0,
        upper=1.0,
    )
    builder.add_joint(
        "target",
        kind=JointRevolute(axis=(0.0, 0.0, 1.0)),
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


def test_rnea_crba_centroidal_and_aba_use_one_reduced_map() -> None:
    constrained, full = _model_pair()
    q = torch.tensor([[-0.2], [0.3]], dtype=torch.float64)
    v = torch.tensor([[0.1], [-0.15]], dtype=torch.float64)
    acceleration = torch.tensor([[0.25], [-0.3]], dtype=torch.float64)
    q_full = torch.cat([q, -0.5 * q + 0.2], dim=-1)
    v_full = v @ EXPANSION.mT
    acceleration_full = acceleration @ EXPANSION.mT

    state = forward_kinematics(constrained, q)
    state_full = forward_kinematics(full, q_full)
    tau = inverse_dynamics(constrained, state, v=v, a=acceleration)
    tau_full = inverse_dynamics(full, state_full, v=v_full, a=acceleration_full)
    torch.testing.assert_close(tau, tau_full @ EXPANSION, rtol=1e-11, atol=1e-11)

    mass = mass_matrix(constrained, state)
    mass_full = mass_matrix(full, state_full)
    expected_mass = EXPANSION.mT @ mass_full @ EXPANSION
    torch.testing.assert_close(mass, expected_mass, rtol=1e-11, atol=1e-11)

    centroidal = centroidal_map(constrained, state)
    centroidal_full = centroidal_map(full, state_full)
    torch.testing.assert_close(
        centroidal,
        centroidal_full @ EXPANSION,
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(
        centroidal_momentum(constrained, state, v=v),
        (centroidal @ v.unsqueeze(-1)).squeeze(-1),
    )

    ddq = forward_dynamics(constrained, state, v=v, tau=tau)
    torch.testing.assert_close(ddq, acceleration, rtol=1e-10, atol=1e-10)


def test_projected_aba_matches_reduced_mass_and_bias_solve() -> None:
    constrained, full = _model_pair()
    q = torch.tensor([0.25], dtype=torch.float64)
    v = torch.tensor([-0.12], dtype=torch.float64)
    tau = torch.tensor([0.7], dtype=torch.float64)
    q_full = torch.cat([q, -0.5 * q + 0.2], dim=-1)
    v_full = v @ EXPANSION.mT

    state_full = forward_kinematics(full, q_full)
    mass_full = mass_matrix(full, state_full)
    bias_full = inverse_dynamics(full, state_full, v=v_full, a=torch.zeros_like(v_full))
    reduced_mass = EXPANSION.mT @ mass_full @ EXPANSION
    reduced_bias = bias_full @ EXPANSION
    expected = torch.linalg.solve(reduced_mass, (tau - reduced_bias).unsqueeze(-1)).squeeze(-1)

    actual = forward_dynamics(constrained, forward_kinematics(constrained, q), v=v, tau=tau)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
