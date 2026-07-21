"""Opt-in stable joint ordering without changing the historical DFS default."""

from __future__ import annotations

import torch

import better_robot as br
from better_robot.io import ModelBuilder, build_model, load
from better_robot.io.builders.kinematic_tree import build_kinematic_tree_model

from tests.support.branching_tree import (
    JOINT_NAMES,
    PARENTS,
    make_branching_tree_body,
    make_branching_tree_model,
)


def _remap_by_name(source, target, values: torch.Tensor, *, tangent: bool):
    """Synthetic name-keyed oracle; production callers use q_permutation."""
    source_indices = source.idx_vs if tangent else source.idx_qs
    source_dims = source.nvs if tangent else source.nqs
    target_dims = target.nvs if tangent else target.nqs
    parts: list[torch.Tensor] = []
    for target_id, name in enumerate(target.joint_names):
        width = target_dims[target_id]
        if width == 0:
            continue
        source_id = source.joint_id(name)
        assert source_dims[source_id] == width
        start = source_indices[source_id]
        parts.append(values[..., start : start + width])
    return torch.cat(parts, dim=-1)


def test_branching_preserve_order_is_exact_and_dfs_default_is_unchanged():
    ir = make_branching_tree_body()
    source_order = tuple(joint.name for joint in ir.joints)

    preserved = build_model(ir, preserve_joint_order=True)
    default = build_model(ir)

    assert preserved.joint_names[1:] == source_order
    assert preserved.joint_names[1:] == ("root", *JOINT_NAMES[1:])
    assert default.joint_names[:6] == (
        "universe",
        "root",
        "j1",
        "j4",
        "j7",
        "j10",
    )
    assert default.joint_names[1:] != source_order
    assert all(parent < child for child, parent in enumerate(preserved.parents) if parent >= 0)


def test_preserved_branching_fk_and_rnea_match_default_dfs_model():
    ir = make_branching_tree_body()
    preserved = build_model(ir, preserve_joint_order=True, dtype=torch.float64)
    default = build_model(ir, dtype=torch.float64)
    generator = torch.Generator().manual_seed(35)

    delta = torch.randn(preserved.nv, generator=generator, dtype=torch.float64) * 0.03
    q_preserved = preserved.integrate(preserved.q_neutral, delta)
    v_preserved = (
        torch.randn(
            preserved.nv,
            generator=generator,
            dtype=torch.float64,
        )
        * 0.05
    )
    a_preserved = (
        torch.randn(
            preserved.nv,
            generator=generator,
            dtype=torch.float64,
        )
        * 0.05
    )

    q_default = _remap_by_name(preserved, default, q_preserved, tangent=False)
    v_default = _remap_by_name(preserved, default, v_preserved, tangent=True)
    a_default = _remap_by_name(preserved, default, a_preserved, tangent=True)

    fk_preserved = br.forward_kinematics(preserved, q_preserved)
    fk_default = br.forward_kinematics(default, q_default)
    for body_name in preserved.body_names:
        preserved_id = preserved.body_id(body_name)
        default_id = default.body_id(body_name)
        torch.testing.assert_close(
            fk_preserved.joint_pose_world[preserved_id],
            fk_default.joint_pose_world[default_id],
            rtol=1e-12,
            atol=1e-12,
        )

    tau_preserved = br.rnea(
        preserved,
        q_preserved,
        v_preserved,
        a_preserved,
    )
    tau_default = br.rnea(
        default,
        q_default,
        v_default,
        a_default,
    )
    torch.testing.assert_close(
        tau_preserved,
        _remap_by_name(default, preserved, tau_default, tangent=True),
        rtol=1e-11,
        atol=1e-11,
    )


def test_stable_kahn_repairs_non_topological_input_deterministically():
    builder = ModelBuilder("stable_kahn")
    for body in ("base", "left", "left_tip", "right", "right_tip"):
        builder.add_body(body)

    # Deliberately place a child before its parent in the flat IR.
    builder.add_revolute_z("left_tip", parent="left", child="left_tip")
    builder.add_revolute_z("right", parent="base", child="right")
    builder.add_revolute_z("left", parent="base", child="left")
    builder.add_revolute_z("right_tip", parent="right", child="right_tip")

    model = build_model(builder.finalize(), preserve_joint_order=True)
    assert model.joint_names == (
        "universe",
        "root_joint",
        "right",
        "left",
        "left_tip",
        "right_tip",
    )


def test_preserve_order_threads_through_load_and_public_model_builders():
    loaded = load(make_branching_tree_body, preserve_joint_order=True)
    fixture_model = make_branching_tree_model(preserve_joint_order=True)
    tree_model = build_kinematic_tree_model(
        name="tree",
        joint_names=JOINT_NAMES,
        parents=PARENTS,
        translations=torch.zeros(len(JOINT_NAMES), 3),
        preserve_joint_order=True,
    )

    expected = ("root", *JOINT_NAMES[1:])
    assert loaded.joint_names[1:] == expected
    assert fixture_model.joint_names[1:] == expected
    assert tree_model.joint_names[1:] == expected
