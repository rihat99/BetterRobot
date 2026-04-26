"""Tests for ``dynamics.integrators`` (P11 D0–D4+).

Currently only ``integrate_q`` (D0) is implemented. The dynamic
integrators (semi-implicit Euler, symplectic Euler, RK4) still raise
``NotImplementedError``.

See ``docs/concepts/dynamics.md §8``.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.dynamics.integrators import (
    integrate_q,
    rk4,
    semi_implicit_euler,
    symplectic_euler,
)
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder


def _two_link_planar():
    b = ModelBuilder("two_link")
    b.add_body("base", mass=0.0)
    b.add_body("l1", mass=1.0)
    b.add_body("l2", mass=1.0)
    origin = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
    b.add_revolute_z("j1", parent="base", child="l1", origin=origin,
                     lower=-math.pi, upper=math.pi)
    b.add_revolute_z("j2", parent="l1", child="l2", origin=origin,
                     lower=-math.pi, upper=math.pi)
    return build_model(b.finalize())


@pytest.fixture(scope="module")
def model():
    return _two_link_planar()


def test_integrate_q_matches_model_integrate_for_revolute(model):
    """For a fixed-base revolute robot, ``integrate_q(q, v, dt)`` is just
    ``q + dt * v``. The retraction must agree with that closed form
    component-wise.
    """
    q = torch.tensor([0.4, -1.2])
    v = torch.tensor([1.5, -0.3])
    dt = 0.05

    q_new = integrate_q(model, q, v, dt)
    expected = q + dt * v
    assert q_new.shape == q.shape
    torch.testing.assert_close(q_new, expected, rtol=0.0, atol=1e-8)


def test_integrate_q_zero_velocity_is_identity(model):
    """``v = 0`` ⇒ ``q_new == q`` to bit-precision."""
    q = torch.tensor([0.4, -1.2])
    v = torch.zeros_like(q)
    out = integrate_q(model, q, v, dt=0.1)
    torch.testing.assert_close(out, q, rtol=0.0, atol=0.0)


def test_integrate_q_batched(model):
    """Leading batch dims pass through ``model.integrate``."""
    q = torch.zeros(4, model.nq)
    v = torch.ones(4, model.nv)
    out = integrate_q(model, q, v, dt=0.01)
    assert out.shape == q.shape
    torch.testing.assert_close(out, 0.01 * v, rtol=0.0, atol=1e-8)


def test_integrate_q_handles_free_flyer():
    """Free-flyer base: first 7 components of q are an SE(3) pose; the
    integration must round-trip through the SO(3) exp map for the
    rotation part. Sanity check: a small twist preserves the unit-quat
    constraint."""
    b = ModelBuilder("ff_arm")
    b.add_body("body", mass=1.0)
    b.add_body("link", mass=0.5)
    origin = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
    b.add_free_flyer_root("ff", child="body")
    b.add_revolute_z("j1", parent="body", child="link", origin=origin,
                     lower=-math.pi, upper=math.pi)
    model = build_model(b.finalize())

    q = model.q_neutral.clone()
    v = torch.zeros(model.nv)
    v[5] = 0.1   # small angular component about z
    v[6] = 0.2   # revolute joint velocity

    q_new = integrate_q(model, q, v, dt=0.01)
    # Quaternion stays unit-norm.
    quat = q_new[3:7]
    torch.testing.assert_close(quat.norm(), torch.tensor(1.0), rtol=0.0, atol=1e-6)


# ── stubs still expected to raise ────────────────────────────────────────────


@pytest.mark.parametrize("fn", [semi_implicit_euler, symplectic_euler, rk4])
def test_dynamic_integrators_still_stubs(model, fn):
    """The dynamic integrators (SE / Symplectic / RK4) wait on the rest of
    the dynamics milestone; calling them today must raise
    ``NotImplementedError`` so accidental use surfaces immediately.
    """
    q = torch.zeros(model.nq)
    v = torch.zeros(model.nv)
    tau = torch.zeros(model.nv)
    data = model.create_data()
    with pytest.raises(NotImplementedError):
        fn(model, data, q, v, tau, dt=0.01)
