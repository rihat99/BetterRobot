"""Joint order is a naming question only. Oracle: Pinocchio (``buildModelFromUrdf`` with a free flyer).

G1's movable joints come in the same order from BetterRobot and from Pinocchio, so the ``g1_both`` cases gather by
an identity permutation. Here the G1 description's joints are reversed before ``build_model``: the Kahn sort then
puts the joints in an order that differs from Pinocchio's depth-first one, and inverse dynamics and every joint's
Jacobian still agree with Pinocchio once its vectors are gathered by joint name (``Model.q_permutation``).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pinocchio as pin
import pytest
import torch

import better_robot as br
from better_robot.dynamics import inverse_dynamics
from better_robot.io import build_model, parse
from better_robot.kinematics import forward_kinematics, joint_jacobian

from tests.support.robots import g1_urdf

TOLERANCE = 1e-10  # float64 end to end, as the g1_both cases


@pytest.fixture(scope="module")
def g1_reordered():
    """G1 built from its description with the joints reversed, Pinocchio's G1, and the gather indices."""
    ir = parse(g1_urdf())
    ir = dataclasses.replace(ir, joints=list(reversed(ir.joints)))
    br_model = build_model(ir, root_joint=br.model.JointFreeFlyer(), dtype=torch.float64)
    pin_model = pin.buildModelFromUrdf(str(g1_urdf()), pin.JointModelFreeFlyer())
    perm_q, perm_v = br_model.q_permutation([pin_model.names[i] for i in range(1, pin_model.njoints)])
    assert perm_q.tolist() != list(range(br_model.nq)), "the reversed description must reorder the coordinates"
    return br_model, pin_model, pin_model.createData(), perm_q, perm_v


def _sample(br_model: br.Model, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``q``, ``v``, ``a`` in Pinocchio's layout: base pose, unit quaternion, joints in [-0.3, 0.3]."""
    generator = torch.Generator().manual_seed(seed)
    q = torch.zeros(br_model.nq, dtype=torch.float64)
    q[:3] = torch.rand(3, generator=generator, dtype=torch.float64) - 0.5
    quaternion = torch.randn(4, generator=generator, dtype=torch.float64)
    q[3:7] = quaternion / quaternion.norm()
    q[7:] = torch.rand(br_model.nq - 7, generator=generator, dtype=torch.float64) * 0.6 - 0.3
    v = torch.randn(br_model.nv, generator=generator, dtype=torch.float64) * 0.2
    a = torch.randn(br_model.nv, generator=generator, dtype=torch.float64) * 0.2
    return q, v, a


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rnea_of_a_reordered_g1_matches_pinocchio(g1_reordered, seed):
    br_model, pin_model, pin_data, perm_q, perm_v = g1_reordered
    q, v, a = _sample(br_model, seed)
    state = forward_kinematics(br_model, q[perm_q])
    tau = inverse_dynamics(br_model, state, v=v[perm_v], a=a[perm_v]).numpy()
    tau_pin = np.asarray(pin.rnea(pin_model, pin_data, q.numpy(), v.numpy(), a.numpy()))
    np.testing.assert_allclose(
        tau, tau_pin[perm_v.numpy()], rtol=0.0, atol=TOLERANCE, err_msg=f"reordered G1 rnea by joint name, seed {seed}"
    )


@pytest.mark.parametrize("seed", [0])
def test_every_joint_jacobian_of_a_reordered_g1_matches_pinocchio(g1_reordered, seed):
    br_model, pin_model, pin_data, perm_q, perm_v = g1_reordered
    q, _, _ = _sample(br_model, seed)
    state = forward_kinematics(br_model, q[perm_q])
    pin.computeJointJacobians(pin_model, pin_data, q.numpy())
    for name in (name for name, width in zip(br_model.joint_names, br_model.nvs, strict=True) if width):
        jacobian = joint_jacobian(br_model, state, joint=br_model.joint_id(name), reference="world").numpy()
        jacobian_pin = pin.getJointJacobian(pin_model, pin_data, pin_model.getJointId(name), pin.ReferenceFrame.WORLD)
        np.testing.assert_allclose(
            jacobian,
            jacobian_pin[:, perm_v.numpy()],
            rtol=0.0,
            atol=TOLERANCE,
            err_msg=f"reordered G1 joint Jacobian of {name}, seed {seed}",
        )
