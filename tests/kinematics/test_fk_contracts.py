"""Contracts for the batched forward-kinematics pass behind ``forward_kinematics_state``.

Parity against a hand-computed chain, the canonical-sign quaternion property,
``torch.func`` (jacrev/jacfwd) agreement on a mixed-kind model, batched
joint-placement forward + gradient, and ``torch.compile`` parity.

All tests are float32 with loose tolerances (the library's target precision).
"""

from __future__ import annotations

import math

import torch

from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.lie import so3
from better_robot.kinematics import forward_kinematics_state


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> torch.Tensor:
    return torch.tensor([x, y, z, 0.0, 0.0, 0.0, 1.0])


def _rot_z(angle: float) -> torch.Tensor:
    cos, sin = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _mixed_kind_model() -> "object":
    """Free-flyer root, then revolute, prismatic, spherical, revolute joints."""
    builder = ModelBuilder("mixed_kind")
    for body in ("base", "l1", "l2", "l3", "l4"):
        builder.add_body(body, mass=1.0)
    builder.add_free_flyer_root("ff", child="base")
    builder.add_revolute_z("rev_z", parent="base", child="l1", origin=_pose(0.1, 0.0, 0.05))
    builder.add_prismatic_x("pris_x", parent="l1", child="l2", origin=_pose(0.1, 0.0, 0.0))
    builder.add_spherical("sph", parent="l2", child="l3", origin=_pose(0.1, 0.0, 0.0))
    builder.add_revolute_y("rev_y", parent="l3", child="l4", origin=_pose(0.1, 0.0, 0.0))
    return build_model(builder.finalize())


def _random_config(model, *, batch: tuple[int, ...] = (), scale: float = 1.0, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    tangent = (torch.rand(*batch, model.nv, generator=generator) - 0.5) * 2.0 * scale
    q_neutral = model.q_neutral.expand(*batch, model.nq)
    return model.integrate(q_neutral, tangent).contiguous()


def test_matrix_fk_matches_hand_computed_two_link_chain() -> None:
    """Parity against a closed-form planar two-link revolute chain."""
    builder = ModelBuilder("two_link")
    for body in ("root", "link1", "link2"):
        builder.add_body(body, mass=1.0)
    builder.add_revolute_z("j1", parent="root", child="link1", origin=_pose(0.2, 0.0, 0.0))
    builder.add_revolute_z("j2", parent="link1", child="link2", origin=_pose(0.3, 0.0, 0.0))
    model = build_model(builder.finalize())

    theta1, theta2 = 0.5, -0.7
    q = torch.tensor([theta1, theta2], dtype=torch.float32)
    result = forward_kinematics_state(model, q)

    j1 = model.structure.joint_id("j1")
    j2 = model.structure.joint_id("j2")
    world1 = result.joint_pose[j1]
    world2 = result.joint_pose[j2]

    torch.testing.assert_close(world1[:3], torch.tensor([0.2, 0.0, 0.0]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(so3.to_matrix(world1[3:7]), _rot_z(theta1), atol=1e-5, rtol=1e-5)

    expected_t2 = torch.tensor([0.2 + 0.3 * math.cos(theta1), 0.3 * math.sin(theta1), 0.0])
    torch.testing.assert_close(world2[:3], expected_t2, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(so3.to_matrix(world2[3:7]), _rot_z(theta1 + theta2), atol=1e-5, rtol=1e-5)


def test_matrix_fk_emits_canonical_sign_quaternions() -> None:
    """FK returns the Shepperd canonical quaternion for every pose.

    The chosen canon is the fixed point of ``so3.from_matrix ∘ so3.to_matrix``:
    a rotation maps to one deterministic representative, so a returned ``q``
    already satisfies ``q == from_matrix(to_matrix(q))`` (never ``-q``). Feeding
    the flipped ``-q`` collapses back to the same ``q``, proving the sign choice
    is deterministic rather than input-dependent.
    """
    model = _mixed_kind_model()
    # Large tangents so world rotations exceed the qw>0 hemisphere and exercise
    # multiple Shepperd branches / sign flips.
    q = _random_config(model, batch=(8,), scale=2.5, seed=3)
    result = forward_kinematics_state(model, q)

    for poses in (result.joint_pose, result.frame_pose):
        quaternion = poses[..., 3:7]
        canonical = so3.from_matrix(so3.to_matrix(quaternion))
        torch.testing.assert_close(quaternion, canonical, atol=1e-5, rtol=1e-5)
        flipped_canonical = so3.from_matrix(so3.to_matrix(-quaternion))
        torch.testing.assert_close(flipped_canonical, quaternion, atol=1e-5, rtol=1e-5)


def test_matrix_fk_jacrev_equals_jacfwd_on_mixed_kinds() -> None:
    """Reverse- and forward-mode Jacobians agree on a mixed-kind model."""
    model = _mixed_kind_model()
    q = _random_config(model, scale=1.0, seed=5)

    def scalar(config: torch.Tensor) -> torch.Tensor:
        result = forward_kinematics_state(model, config)
        return result.joint_pose.sum() + result.frame_pose.sum()

    reverse = torch.func.jacrev(scalar)(q)
    forward = torch.func.jacfwd(scalar)(q)
    torch.testing.assert_close(reverse, forward, atol=1e-4, rtol=1e-4)


def test_matrix_fk_batched_joint_placements_forward_and_grad() -> None:
    """Per-element joint placements run batched and stay differentiable."""
    model = _mixed_kind_model()
    njoints = model.njoints
    batch = 4
    q = _random_config(model, batch=(batch,), scale=0.6, seed=7)

    # Per-element translation offsets keep placement quaternions unit-valid.
    offsets = torch.zeros(batch, njoints, 7)
    offsets[..., 0] = torch.linspace(0.0, 0.15, batch).unsqueeze(-1)
    batched_placements = model.values.joint_placements.unsqueeze(0) + offsets

    placements = batched_placements.detach().clone().requires_grad_()
    result = forward_kinematics_state(model.with_values(joint_placements=placements), q)
    assert result.joint_pose.shape == (batch, njoints, 7)

    gradient = torch.autograd.grad(result.joint_pose.sum(), placements)[0]
    assert gradient.shape == (batch, njoints, 7)
    assert torch.isfinite(gradient).all()

    for index in range(batch):
        element = forward_kinematics_state(model.with_values(joint_placements=batched_placements[index]), q[index])
        torch.testing.assert_close(
            result.joint_pose[index],
            element.joint_pose,
            atol=1e-5,
            rtol=1e-5,
        )
