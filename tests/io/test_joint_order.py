"""One joint order: parents first, and the description's own order wherever it is already topological."""

from __future__ import annotations

import dataclasses

import torch

import better_robot as br
from better_robot.io import ModelBuilder, build_model, load

from tests.support.branching_tree import JOINT_NAMES, make_branching_tree_body, make_branching_tree_model


def test_a_topological_description_keeps_its_order_exactly():
    ir = make_branching_tree_body()
    model = build_model(ir)
    assert model.joint_names[1:] == tuple(joint.name for joint in ir.joints) == ("root", *JOINT_NAMES[1:])
    assert all(parent < joint for joint, parent in enumerate(model.parent_joint_ids) if joint)


def test_a_child_declared_before_its_parent_is_repaired_deterministically():
    builder = ModelBuilder("kahn")
    for body in ("base", "left", "left_tip", "right", "right_tip"):
        builder.add_body(body)
    builder.add_revolute("left_tip", axis=(0.0, 0.0, 1.0), parent="left", child="left_tip")
    builder.add_revolute("right", axis=(0.0, 0.0, 1.0), parent="base", child="right")
    builder.add_revolute("left", axis=(0.0, 0.0, 1.0), parent="base", child="left")
    builder.add_revolute("right_tip", axis=(0.0, 0.0, 1.0), parent="right", child="right_tip")
    model = build_model(builder.finalize())
    assert model.joint_names == ("universe", "root_joint", "right", "left", "left_tip", "right_tip")


def test_the_order_changes_no_physics():
    """The same tree declared in reverse builds a model that agrees by name: poses and inverse dynamics."""
    ir = make_branching_tree_body()
    model = build_model(ir, dtype=torch.float64)
    other = build_model(dataclasses.replace(ir, joints=list(reversed(ir.joints))), dtype=torch.float64)
    assert other.joint_names != model.joint_names
    assert sorted(other.joint_names) == sorted(model.joint_names)

    generator = torch.Generator().manual_seed(35)
    delta, v, a = (torch.randn(other.nv, generator=generator, dtype=torch.float64) * 0.05 for _ in range(3))
    q = br.kinematics.integrate(other, other.q_neutral, delta)
    perm_q, perm_v = model.q_permutation(other.joint_names)

    state = br.forward_kinematics(model, q[perm_q])
    other_state = br.forward_kinematics(other, q)
    for name in model.body_names:
        torch.testing.assert_close(
            state.joint_poses[model.body_id(name)],
            other_state.joint_poses[other.body_id(name)],
            rtol=1e-12,
            atol=1e-12,
            msg=f"joint pose of body {name}",
        )
    tau = br.inverse_dynamics(model, state, v=v[perm_v], a=a[perm_v])
    other_tau = br.inverse_dynamics(other, other_state, v=v, a=a)
    torch.testing.assert_close(tau, other_tau[perm_v], rtol=1e-11, atol=1e-11, msg="inverse dynamics by name")


def test_load_and_the_fixture_builder_agree():
    expected = ("universe", "root", *JOINT_NAMES[1:])
    assert load(make_branching_tree_body).joint_names == expected
    assert make_branching_tree_model().joint_names == expected
