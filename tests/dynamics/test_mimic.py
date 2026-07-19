"""Projected mimic dynamics against an explicit unconstrained twin."""

from __future__ import annotations

import dataclasses

import torch

from better_robot.data_model.reduced_coordinates import (
    expand_configuration,
    expand_tangent,
)
from better_robot.dynamics import aba, ccrba, crba, rnea
from better_robot.io import ModelBuilder, build_model


def _model_pair():
    inertia = torch.diag(torch.tensor([0.2, 0.3, 0.4]))
    builder = ModelBuilder("mimic_dynamics")
    root = builder.add_body("root", mass=1.0, inertia=inertia)
    first = builder.add_body("first", mass=1.3, com=torch.tensor([0.2, 0.0, 0.0]), inertia=inertia)
    second = builder.add_body("second", mass=0.7, com=torch.tensor([0.25, 0.0, 0.0]), inertia=inertia)
    builder.add_revolute_z(
        "source",
        parent=root,
        child=first,
        origin=torch.tensor([0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
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


def test_rnea_crba_centroidal_and_aba_use_one_reduced_map() -> None:
    constrained, full = _model_pair()
    q = torch.tensor([[-0.2], [0.3]], dtype=torch.float64)
    v = torch.tensor([[0.1], [-0.15]], dtype=torch.float64)
    acceleration = torch.tensor([[0.25], [-0.3]], dtype=torch.float64)
    q_full = expand_configuration(constrained.structure, q)
    v_full = expand_tangent(constrained.structure, v)
    acceleration_full = expand_tangent(constrained.structure, acceleration)
    expansion = constrained.v_expansion

    tau = rnea(constrained, q, v, acceleration)
    tau_full = rnea(full, q_full, v_full, acceleration_full)
    torch.testing.assert_close(tau, tau_full @ expansion, rtol=1e-11, atol=1e-11)

    mass = crba(constrained, q)
    mass_full = crba(full, q_full)
    expected_mass = expansion.mT @ mass_full @ expansion
    torch.testing.assert_close(mass, expected_mass, rtol=1e-11, atol=1e-11)

    centroidal = ccrba(constrained, q, v)
    centroidal_full = ccrba(full, q_full, v_full)
    torch.testing.assert_close(
        centroidal.centroidal_map,
        centroidal_full.centroidal_map @ expansion,
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(
        centroidal.momentum,
        (centroidal.centroidal_map @ v.unsqueeze(-1)).squeeze(-1),
    )

    ddq = aba(constrained, q, v, tau)
    torch.testing.assert_close(ddq, acceleration, rtol=1e-10, atol=1e-10)


def test_projected_aba_matches_reduced_mass_and_bias_solve() -> None:
    constrained, full = _model_pair()
    q = torch.tensor([0.25], dtype=torch.float64)
    v = torch.tensor([-0.12], dtype=torch.float64)
    tau = torch.tensor([0.7], dtype=torch.float64)
    q_full = expand_configuration(constrained.structure, q)
    v_full = expand_tangent(constrained.structure, v)
    expansion = constrained.v_expansion

    mass_full = crba(full, q_full)
    bias_full = rnea(full, q_full, v_full, torch.zeros_like(v_full))
    reduced_mass = expansion.mT @ mass_full @ expansion
    reduced_bias = bias_full @ expansion
    expected = torch.linalg.solve(reduced_mass, (tau - reduced_bias).unsqueeze(-1)).squeeze(-1)

    actual = aba(constrained, q, v, tau)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
