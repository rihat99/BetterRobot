"""Build-time contracts for scalar mimic reduced coordinates."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from better_robot.data_model.joint_models import (
    JointHelical,
    JointPrismaticUnaligned,
    JointPX,
    JointPY,
    JointPZ,
    JointRevoluteUnaligned,
    JointRX,
    JointRY,
    JointRZ,
    JointSpherical,
)
from better_robot.data_model.reduced_coordinates import (
    expand_configuration,
    expand_tangent,
)
from better_robot.io import IRError, ModelBuilder, build_model
from better_robot.residuals import JointPositionLimit


_IDENTITY = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def _mimic_ir(*, multiplier: float = -2.0, offset: float = 0.25):
    builder = ModelBuilder("reduced_mimic")
    root = builder.add_body("root", mass=1.0, inertia=torch.eye(3))
    source = builder.add_body("source", mass=1.0, inertia=torch.eye(3))
    target = builder.add_body("target", mass=1.0, inertia=torch.eye(3))
    builder.add_revolute_z(
        "source_joint",
        parent=root,
        child=source,
        origin=_IDENTITY,
        lower=-1.0,
        upper=1.0,
        velocity_limit=3.0,
        effort_limit=5.0,
    )
    builder.add_revolute_z(
        "target_joint",
        parent=source,
        child=target,
        origin=torch.tensor([0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        lower=-0.5,
        upper=0.75,
        velocity_limit=4.0,
        effort_limit=7.0,
        mimic_source="source_joint",
        mimic_multiplier=multiplier,
        mimic_offset=offset,
    )
    return builder.finalize()


def test_reduced_and_full_layouts_and_affine_expansion_are_explicit() -> None:
    model = build_model(_mimic_ir(), dtype=torch.float64)
    source = model.joint_id("source_joint")
    target = model.joint_id("target_joint")

    assert model.has_mimic is True
    assert model.nq == model.nv == 1
    assert model.nq_full == model.nv_full == 2
    assert model.nqs[source] == model.nvs[source] == 1
    assert model.nqs[target] == model.nvs[target] == 0
    assert model.nqs_full[target] == model.nvs_full[target] == 1
    torch.testing.assert_close(
        model.q_expansion,
        torch.tensor([[1.0], [-2.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        model.q_offset,
        torch.tensor([0.0, 0.25], dtype=torch.float64),
    )
    torch.testing.assert_close(model.v_expansion, model.q_expansion)

    q = torch.tensor([[[-0.1], [0.2]]], dtype=torch.float64, requires_grad=True)
    q_full = expand_configuration(model.structure, q)
    v_full = expand_tangent(model.structure, torch.ones_like(q))
    assert q_full.shape == (1, 2, 2)
    torch.testing.assert_close(q_full[..., 1], -2.0 * q[..., 0] + 0.25)
    torch.testing.assert_close(v_full[..., 1], -2.0 * torch.ones_like(q[..., 0]))
    q_full.square().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


@pytest.mark.parametrize(
    "factory",
    (
        pytest.param(JointRX, id="revolute_rx"),
        pytest.param(JointRY, id="revolute_ry"),
        pytest.param(JointRZ, id="revolute_rz"),
        pytest.param(
            lambda: JointRevoluteUnaligned(torch.tensor([0.6, 0.8, 0.0])),
            id="revolute_unaligned",
        ),
        pytest.param(JointPX, id="prismatic_px"),
        pytest.param(JointPY, id="prismatic_py"),
        pytest.param(JointPZ, id="prismatic_pz"),
        pytest.param(
            lambda: JointPrismaticUnaligned(torch.tensor([0.6, 0.0, 0.8])),
            id="prismatic_unaligned",
        ),
        pytest.param(
            lambda: JointHelical(torch.tensor([0.0, 0.0, 1.0]), pitch=0.2),
            id="helical",
        ),
    ),
)
def test_every_supported_concrete_scalar_kind_builds_a_reduced_map(factory) -> None:
    builder = ModelBuilder("supported_mimic_kind")
    root = builder.add_body("root")
    source = builder.add_body("source")
    target = builder.add_body("target")
    builder.add_joint(
        "source_joint",
        kind=factory(),
        parent=root,
        child=source,
        lower=-1.0,
        upper=1.0,
    )
    builder.add_joint(
        "target_joint",
        kind=factory(),
        parent=source,
        child=target,
        lower=-1.0,
        upper=1.0,
        mimic_source="source_joint",
        mimic_multiplier=-0.75,
        mimic_offset=0.1,
    )
    model = build_model(builder.finalize(), dtype=torch.float64)
    target_id = model.joint_id("target_joint")

    assert model.nq == model.nv == 1
    assert model.nq_full == model.nv_full == 2
    assert model.nqs[target_id] == model.nvs[target_id] == 0
    expanded = expand_configuration(model.structure, torch.tensor([0.4], dtype=torch.float64))
    torch.testing.assert_close(
        expanded[model.idx_qs_full[target_id]],
        torch.tensor(-0.2, dtype=torch.float64),
    )


def test_sign_aware_limits_and_generalized_capacities_are_reduced() -> None:
    model = build_model(_mimic_ir(), dtype=torch.float64)

    torch.testing.assert_close(model.lower_pos_limit, torch.tensor([-0.25], dtype=torch.float64))
    torch.testing.assert_close(model.upper_pos_limit, torch.tensor([0.375], dtype=torch.float64))
    torch.testing.assert_close(model.velocity_limit, torch.tensor([2.0], dtype=torch.float64))
    torch.testing.assert_close(model.effort_limit, torch.tensor([19.0], dtype=torch.float64))

    residual = JointPositionLimit(model)
    value = {"q": torch.tensor([0.5], dtype=torch.float64), "model": model}
    result = residual(value)
    assert result.shape == (2,)
    torch.testing.assert_close(result, torch.tensor([0.0, 0.125], dtype=torch.float64))


def test_mimic_chains_compose_multiplier_and_offset() -> None:
    ir = _mimic_ir()
    builder = ModelBuilder("chain")
    root = builder.add_body("root")
    a = builder.add_body("a")
    b = builder.add_body("b")
    c = builder.add_body("c")
    builder.add_revolute_z("a_joint", parent=root, child=a, lower=-2.0, upper=2.0)
    builder.add_revolute_z(
        "b_joint",
        parent=a,
        child=b,
        lower=-10.0,
        upper=10.0,
        mimic_source="a_joint",
        mimic_multiplier=-2.0,
        mimic_offset=0.25,
    )
    builder.add_revolute_z(
        "c_joint",
        parent=b,
        child=c,
        lower=-10.0,
        upper=10.0,
        mimic_source="b_joint",
        mimic_multiplier=0.5,
        mimic_offset=-0.1,
    )
    model = build_model(builder.finalize(), dtype=torch.float64)
    del ir

    c_joint = model.joint_id("c_joint")
    c_row = model.idx_qs_full[c_joint]
    assert model.q_expansion[c_row, 0].item() == pytest.approx(-1.0)
    assert model.q_offset[c_row].item() == pytest.approx(0.025)


def test_cycles_and_unsupported_manifold_mimics_fail_at_build() -> None:
    cycle = ModelBuilder("cycle")
    root = cycle.add_body("root")
    a = cycle.add_body("a")
    b = cycle.add_body("b")
    cycle.add_revolute_z("a_joint", parent=root, child=a, mimic_source="b_joint")
    cycle.add_revolute_z("b_joint", parent=root, child=b, mimic_source="a_joint")
    with pytest.raises(IRError, match="cycle"):
        build_model(cycle.finalize())

    ir = _mimic_ir()
    target_index = next(i for i, joint in enumerate(ir.joints) if joint.name == "target_joint")
    ir.joints[target_index] = dataclasses.replace(
        ir.joints[target_index],
        kind="spherical",
        joint_model=JointSpherical(),
    )
    with pytest.raises(IRError, match="scalar Euclidean"):
        build_model(ir)


def test_zero_multiplier_is_constant_only_when_offset_satisfies_target_limits() -> None:
    model = build_model(_mimic_ir(multiplier=0.0, offset=0.3))
    q_full = expand_configuration(model.structure, torch.tensor([0.8]))
    torch.testing.assert_close(q_full, torch.tensor([0.8, 0.3]))

    with pytest.raises(IRError, match="outside its position limits"):
        build_model(_mimic_ir(multiplier=0.0, offset=2.0))


def test_identity_models_take_the_zero_copy_fast_path() -> None:
    ir = _mimic_ir()
    ir.joints = [dataclasses.replace(joint, mimic_source=None) for joint in ir.joints]
    model = build_model(ir)
    q = torch.zeros(model.nq)
    assert expand_configuration(model.structure, q) is q
    assert expand_tangent(model.structure, q) is q


def test_public_manifold_permutation_and_device_views_skip_mimic_targets() -> None:
    model = build_model(_mimic_ir(), dtype=torch.float64)
    q = torch.tensor([0.1], dtype=torch.float64)
    v = torch.tensor([-0.2], dtype=torch.float64)
    torch.testing.assert_close(model.difference(q, model.integrate(q, v)), v)
    assert model.random_configuration().shape == (model.nq,)

    perm_q, perm_v = model.q_permutation(("target_joint", "source_joint"))
    torch.testing.assert_close(perm_q, torch.tensor([0]))
    torch.testing.assert_close(perm_v, torch.tensor([0]))

    moved = model.to(dtype=torch.float32)
    assert moved.q_expansion is moved.structure.q_expansion
    assert moved.v_expansion is moved.structure.v_expansion
    assert moved.q_expansion.dtype == torch.float32
