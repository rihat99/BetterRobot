"""Build-time contracts for scalar mimic reduced coordinates."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from better_robot.exceptions import ModelError
from better_robot.io import ModelBuilder, build_model
from better_robot.kinematics.configuration import difference, full_configuration, full_tangent, integrate, mimic_plan
from better_robot.model.joints import (
    JointHelical,
    JointPrismatic,
    JointRevolute,
    JointSpherical,
)

_IDENTITY = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
_CPU = torch.device("cpu")


def _mimic_ir(*, multiplier: float = -2.0, offset: float = 0.25):
    builder = ModelBuilder("reduced_mimic")
    root = builder.add_body("root", mass=1.0, inertia=torch.eye(3))
    source = builder.add_body("source", mass=1.0, inertia=torch.eye(3))
    target = builder.add_body("target", mass=1.0, inertia=torch.eye(3))
    builder.add_revolute(
        "source_joint",
        axis=(0.0, 0.0, 1.0),
        parent=root,
        child=source,
        origin=_IDENTITY,
        lower=-1.0,
        upper=1.0,
        velocity_limit=3.0,
        effort_limit=5.0,
    )
    builder.add_revolute(
        "target_joint",
        axis=(0.0, 0.0, 1.0),
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

    assert model.nq == model.nv == 1
    assert model.nq_full == model.nv_full == 2
    assert model.nqs[source] == model.nvs[source] == 1
    assert model.nqs[target] == model.nvs[target] == 0
    assert model.nqs_full[target] == model.nvs_full[target] == 1
    assert model.mimic_source_ids[target] == source
    assert model.mimic_multipliers[target] == -2.0
    assert model.mimic_offsets[target] == 0.25
    plan = mimic_plan(model, dtype=torch.float64, device=_CPU)
    torch.testing.assert_close(plan.q_expansion, torch.tensor([[1.0], [-2.0]], dtype=torch.float64))
    torch.testing.assert_close(plan.q_offset, torch.tensor([0.0, 0.25], dtype=torch.float64))
    torch.testing.assert_close(plan.v_expansion, plan.q_expansion)

    q = torch.tensor([[[-0.1], [0.2]]], dtype=torch.float64, requires_grad=True)
    q_full = full_configuration(model, q)
    v_full = full_tangent(model, torch.ones_like(q))
    assert q_full.shape == (1, 2, 2)
    torch.testing.assert_close(q_full[..., 1], -2.0 * q[..., 0] + 0.25)
    torch.testing.assert_close(v_full[..., 1], -2.0 * torch.ones_like(q[..., 0]))
    q_full.square().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


@pytest.mark.parametrize(
    "factory",
    (
        pytest.param(lambda: JointRevolute(axis=(0.0, 0.0, 1.0)), id="revolute_z"),
        pytest.param(lambda: JointRevolute(axis=(0.6, 0.8, 0.0)), id="revolute_unaligned"),
        pytest.param(lambda: JointPrismatic(axis=(1.0, 0.0, 0.0)), id="prismatic_x"),
        pytest.param(lambda: JointPrismatic(axis=(0.6, 0.0, 0.8)), id="prismatic_unaligned"),
        pytest.param(lambda: JointHelical(axis=(0.0, 0.0, 1.0), pitch=0.2), id="helical"),
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
    expanded = full_configuration(model, torch.tensor([0.4], dtype=torch.float64))
    torch.testing.assert_close(
        expanded[model.idx_qs_full[target_id]],
        torch.tensor(-0.2, dtype=torch.float64),
    )


def test_mimic_chains_resolve_to_the_public_root_with_composed_multiplier_and_offset() -> None:
    builder = ModelBuilder("chain")
    root = builder.add_body("root")
    a = builder.add_body("a")
    b = builder.add_body("b")
    c = builder.add_body("c")
    builder.add_revolute("a_joint", axis=(0.0, 0.0, 1.0), parent=root, child=a, lower=-2.0, upper=2.0)
    builder.add_revolute(
        "b_joint",
        axis=(0.0, 0.0, 1.0),
        parent=a,
        child=b,
        lower=-10.0,
        upper=10.0,
        mimic_source="a_joint",
        mimic_multiplier=-2.0,
        mimic_offset=0.25,
    )
    builder.add_revolute(
        "c_joint",
        axis=(0.0, 0.0, 1.0),
        parent=b,
        child=c,
        lower=-10.0,
        upper=10.0,
        mimic_source="b_joint",
        mimic_multiplier=0.5,
        mimic_offset=-0.1,
    )
    model = build_model(builder.finalize(), dtype=torch.float64)

    a_joint, c_joint = model.joint_id("a_joint"), model.joint_id("c_joint")
    assert model.mimic_source_ids[c_joint] == a_joint
    assert model.mimic_multipliers[c_joint] == pytest.approx(-1.0)
    assert model.mimic_offsets[c_joint] == pytest.approx(0.025)
    plan = mimic_plan(model, dtype=torch.float64, device=_CPU)
    c_row = model.idx_qs_full[c_joint]
    assert plan.q_expansion[c_row, 0].item() == pytest.approx(-1.0)
    assert plan.q_offset[c_row].item() == pytest.approx(0.025)


def test_cycles_and_unsupported_manifold_mimics_fail_at_build() -> None:
    cycle = ModelBuilder("cycle")
    root = cycle.add_body("root")
    a = cycle.add_body("a")
    b = cycle.add_body("b")
    cycle.add_revolute("a_joint", axis=(0.0, 0.0, 1.0), parent=root, child=a, mimic_source="b_joint")
    cycle.add_revolute("b_joint", axis=(0.0, 0.0, 1.0), parent=root, child=b, mimic_source="a_joint")
    with pytest.raises(ModelError, match="cycle"):
        build_model(cycle.finalize())

    ir = _mimic_ir()
    target_index = next(i for i, joint in enumerate(ir.joints) if joint.name == "target_joint")
    ir.joints[target_index] = dataclasses.replace(
        ir.joints[target_index],
        joint=JointSpherical(),
    )
    with pytest.raises(ModelError, match="scalar Euclidean"):
        build_model(ir)


def test_zero_multiplier_is_constant_only_when_offset_satisfies_target_limits() -> None:
    model = build_model(_mimic_ir(multiplier=0.0, offset=0.3))
    q_full = full_configuration(model, torch.tensor([0.8]))
    torch.testing.assert_close(q_full, torch.tensor([0.8, 0.3]))

    with pytest.raises(ModelError, match="outside its position limits"):
        build_model(_mimic_ir(multiplier=0.0, offset=2.0))


def test_identity_models_take_the_zero_copy_fast_path() -> None:
    ir = _mimic_ir()
    ir.joints = [dataclasses.replace(joint, mimic_source=None) for joint in ir.joints]
    model = build_model(ir)
    q = torch.zeros(model.nq)
    assert full_configuration(model, q) is q
    assert full_tangent(model, q) is q


def test_public_manifold_permutation_and_plans_skip_mimic_targets() -> None:
    model = build_model(_mimic_ir(), dtype=torch.float64)
    q = torch.tensor([0.1], dtype=torch.float64)
    v = torch.tensor([-0.2], dtype=torch.float64)
    torch.testing.assert_close(difference(model, q, integrate(model, q, v)), v)
    assert model.random_configuration(generator=torch.Generator().manual_seed(0)).shape == (model.nq,)

    perm_q, perm_v = model.q_permutation(("target_joint", "source_joint"))
    torch.testing.assert_close(perm_q, torch.tensor([0]))
    torch.testing.assert_close(perm_v, torch.tensor([0]))

    moved = model.to(dtype=torch.float32)
    plan = mimic_plan(moved, dtype=torch.float32, device=_CPU)
    assert plan.q_expansion.dtype == plan.v_expansion.dtype == torch.float32
    assert mimic_plan(moved, dtype=torch.float32, device=_CPU) is plan
