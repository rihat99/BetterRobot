"""Jacobian time-variation (J̇) tests — fp32, builder models.

Central finite differences of the same-frame Jacobian along ``v`` (via the
``model.integrate`` retraction) cross-check the analytic getters in all three
reference frames, for joints and frames. Non-degeneracy is mandatory: ``v`` is
nonzero in every coordinate, the tested joint/frame sits at a nonzero world
origin with angular DOF above it (so the LWA ``-v_pt × J_ang`` transport term is
O(1)), and one case asserts ``WORLD J̇ != LWA J̇`` to prove the fixture exercises
that term rather than greening a stripped implementation.

See ``docs/concepts/kinematics_and_jacobians.md``.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.exceptions import StaleCacheError
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics.forward import forward_kinematics, forward_kinematics_raw
from better_robot.kinematics.jacobian import (
    compute_joint_jacobians,
    compute_joint_jacobians_time_variation,
    frame_jacobian_time_variation_raw,
    get_frame_jacobian,
    get_frame_jacobian_time_variation,
    get_joint_jacobian,
    get_joint_jacobian_time_variation,
    joint_jacobians_raw,
    joint_jacobians_time_variation_raw,
)

_REFERENCES = ("world", "local_world_aligned", "local")
_FD_DT = 1e-3  # fp32 central-difference step
_FD_ATOL = 3e-4
_FD_RTOL = 1e-2


def _chain_with_frame():
    """3-revolute chain with nonzero joint origins and an offset tool frame.

    Mixed axes (Z, Y, X) give a rich Jacobian; the origins push every body away
    from the world origin so the LWA point-transport term does not vanish. The
    ``tool`` frame carries a nontrivial local placement (offset + rotation).
    """
    b = ModelBuilder("jdot_chain")
    for name in ("root", "l1", "l2", "l3"):
        b.add_body(name)
    b.add_revolute_z(
        "j1", parent="root", child="l1",
        origin=torch.tensor([0.1, 0.2, 0.5, 0.0, 0.0, 0.0, 1.0]), lower=-math.pi, upper=math.pi,
    )
    b.add_revolute_y(
        "j2", parent="l1", child="l2",
        origin=torch.tensor([0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]), lower=-math.pi, upper=math.pi,
    )
    b.add_revolute_x(
        "j3", parent="l2", child="l3",
        origin=torch.tensor([0.0, 0.4, 0.5, 0.0, 0.0, 0.0, 1.0]), lower=-math.pi, upper=math.pi,
    )
    b.add_frame(
        "tool",
        parent_body="l3",
        placement=torch.tensor([0.15, -0.1, 0.2, 0.0, 0.3826834, 0.0, 0.9238795]),
    )
    return build_model(b.finalize())


@pytest.fixture(scope="module")
def chain():
    return _chain_with_frame()


def _q_v(model):
    """Fixed, fully nonzero (q, v) for the chain (fp32)."""
    q = torch.tensor([0.3, -0.5, 0.7])
    v = torch.tensor([0.9, -1.1, 0.6])
    assert (v != 0).all()
    return q, v


def _static_jacobian(model, q, *, joint_id=None, frame_id=None, reference):
    data = forward_kinematics(model, q, compute_frames=True)
    compute_joint_jacobians(model, data)
    if joint_id is not None:
        return get_joint_jacobian(model, data, joint_id, reference=reference)
    return get_frame_jacobian(model, data, frame_id, reference=reference)


def _finite_diff_dot(model, q, v, *, joint_id=None, frame_id=None, reference):
    q_plus = model.integrate(q, v * _FD_DT)
    q_minus = model.integrate(q, -v * _FD_DT)
    j_plus = _static_jacobian(model, q_plus, joint_id=joint_id, frame_id=frame_id, reference=reference)
    j_minus = _static_jacobian(model, q_minus, joint_id=joint_id, frame_id=frame_id, reference=reference)
    return (j_plus - j_minus) / (2.0 * _FD_DT)


def _getter_dot(model, q, v, *, joint_id=None, frame_id=None, reference):
    data = forward_kinematics(model, q, compute_frames=True)
    data.v = v
    if joint_id is not None:
        return get_joint_jacobian_time_variation(model, data, joint_id, reference=reference)
    return get_frame_jacobian_time_variation(model, data, frame_id, reference=reference)


# ── finite-difference cross-checks ────────────────────────────────────────────


@pytest.mark.parametrize("reference", _REFERENCES)
def test_joint_time_variation_matches_finite_diff(chain, reference):
    q, v = _q_v(chain)
    joint_id = chain.joint_id("j3")
    analytic = _getter_dot(chain, q, v, joint_id=joint_id, reference=reference)
    finite = _finite_diff_dot(chain, q, v, joint_id=joint_id, reference=reference)
    torch.testing.assert_close(analytic, finite, atol=_FD_ATOL, rtol=_FD_RTOL)


@pytest.mark.parametrize("reference", _REFERENCES)
def test_frame_time_variation_matches_finite_diff(chain, reference):
    q, v = _q_v(chain)
    frame_id = chain.frame_id("tool")
    analytic = _getter_dot(chain, q, v, frame_id=frame_id, reference=reference)
    finite = _finite_diff_dot(chain, q, v, frame_id=frame_id, reference=reference)
    torch.testing.assert_close(analytic, finite, atol=_FD_ATOL, rtol=_FD_RTOL)


def test_world_and_lwa_are_non_degenerate(chain):
    """The fixture must actually exercise the LWA transport term (O(1) gap)."""
    q, v = _q_v(chain)
    joint_id = chain.joint_id("j3")
    world = _getter_dot(chain, q, v, joint_id=joint_id, reference="world")
    lwa = _getter_dot(chain, q, v, joint_id=joint_id, reference="local_world_aligned")
    assert (world - lwa).abs().max() > 0.1


# ── raw / workspace / reuse equivalence ───────────────────────────────────────


def test_raw_matches_workspace(chain):
    q, v = _q_v(chain)
    poses = forward_kinematics_raw(chain.structure, chain.values, q).joint_pose_world
    raw = joint_jacobians_time_variation_raw(chain.structure, q, v, poses)

    data = forward_kinematics(chain, q)
    data.v = v
    compute_joint_jacobians_time_variation(chain, data)
    torch.testing.assert_close(raw.joint_jacobians_dot, data.joint_jacobians_dot)
    torch.testing.assert_close(raw.joint_jacobians, data.joint_jacobians)


def test_reuse_path_matches_fresh(chain):
    q, v = _q_v(chain)
    poses = forward_kinematics_raw(chain.structure, chain.values, q).joint_pose_world
    frame_id = chain.frame_id("tool")

    fresh = joint_jacobians_time_variation_raw(chain.structure, q, v, poses)
    joint_jacobians = joint_jacobians_raw(chain.structure, q, poses).joint_jacobians
    reused = joint_jacobians_time_variation_raw(chain.structure, q, v, poses, joint_jacobians=joint_jacobians)
    torch.testing.assert_close(fresh.joint_jacobians_dot, reused.joint_jacobians_dot)

    frame_fresh = frame_jacobian_time_variation_raw(
        chain.structure, chain.values, q, v, poses, frame_id, reference="local"
    )
    frame_reused = frame_jacobian_time_variation_raw(
        chain.structure, chain.values, q, v, poses, frame_id,
        reference="local",
        joint_jacobians=joint_jacobians,
        joint_jacobians_dot=fresh.joint_jacobians_dot,
    )
    torch.testing.assert_close(frame_fresh, frame_reused)


def test_compute_fills_joint_jacobians_byproduct(chain):
    q, v = _q_v(chain)
    data = forward_kinematics(chain, q)
    data.v = v
    assert data.joint_jacobians is None
    compute_joint_jacobians_time_variation(chain, data)
    assert data.joint_jacobians is not None
    assert data.joint_jacobians_dot is not None


# ── shapes ────────────────────────────────────────────────────────────────────


def test_batched_shapes(chain):
    q, v = _q_v(chain)
    batch = 3
    q_batched = q.unsqueeze(0).expand(batch, -1).clone()
    v_batched = v.unsqueeze(0).expand(batch, -1).clone()
    data = forward_kinematics(chain, q_batched)
    data.v = v_batched

    joint_dot = get_joint_jacobian_time_variation(chain, data, chain.joint_id("j3"))
    frame_dot = get_frame_jacobian_time_variation(chain, data, chain.frame_id("tool"))
    assert joint_dot.shape == (batch, 6, chain.nv)
    assert frame_dot.shape == (batch, 6, chain.nv)


# ── autograd / compile safety ─────────────────────────────────────────────────


def test_jacrev_equals_jacfwd_through_raw(chain):
    q, v = _q_v(chain)

    def evaluate(q_value, v_value):
        poses = forward_kinematics_raw(chain.structure, chain.values, q_value).joint_pose_world
        return joint_jacobians_time_variation_raw(
            chain.structure, q_value, v_value, poses
        ).joint_jacobians_dot

    for argnums in (0, 1):
        reverse = torch.func.jacrev(evaluate, argnums=argnums)(q, v)
        forward = torch.func.jacfwd(evaluate, argnums=argnums)(q, v)
        torch.testing.assert_close(reverse, forward, atol=1e-4, rtol=1e-4)


def test_raw_compiles_fullgraph(chain):
    q, v = _q_v(chain)
    q = q.expand(2, -1).clone()
    v = v.expand(2, -1).clone()
    poses = forward_kinematics_raw(chain.structure, chain.values, q).joint_pose_world

    def evaluate(q_value, v_value, joint_poses):
        return joint_jacobians_time_variation_raw(
            chain.structure, q_value, v_value, joint_poses
        ).joint_jacobians_dot

    compiled = torch.compile(evaluate, fullgraph=True, backend="eager")
    torch.testing.assert_close(compiled(q, v, poses), evaluate(q, v, poses))


# ── cache semantics ───────────────────────────────────────────────────────────


def test_velocity_reassignment_invalidates_dot(chain):
    """``data.joint_jacobians_dot`` clears on ``v`` reassignment, survives ``a``."""
    q, v = _q_v(chain)
    data = forward_kinematics(chain, q)
    data.v = v
    compute_joint_jacobians_time_variation(chain, data)
    assert data.joint_jacobians_dot is not None

    data.a = torch.zeros_like(v)
    assert data.joint_jacobians_dot is not None  # a invalidates only accelerations

    data.v = v * 1.1
    assert data.joint_jacobians_dot is None  # v invalidates the velocity bucket


def test_missing_velocity_raises(chain):
    q, _ = _q_v(chain)
    data = forward_kinematics(chain, q)
    with pytest.raises(StaleCacheError):
        compute_joint_jacobians_time_variation(chain, data)
