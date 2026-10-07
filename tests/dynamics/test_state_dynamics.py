"""Dynamics on ``KinematicState``: consistency between the three functions and autograd."""

from __future__ import annotations

import pytest
import torch

import better_robot as br
from better_robot.dynamics.state import centroidal_map, forward_dynamics, inverse_dynamics, mass_matrix
from better_robot.kinematics.state import forward_kinematics_state

robot_descriptions = pytest.importorskip("robot_descriptions")


@pytest.fixture(scope="module")
def g1() -> br.Model:
    from robot_descriptions import g1_description  # noqa: PLC0415 - optional test dependency

    return br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)


def _inputs(model: br.Model, batch: tuple[int, ...], seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    q = torch.stack([model.random_configuration(generator) for _ in range(int(torch.tensor(batch).prod()))])
    q = q.view(*batch, model.nq)
    v = torch.randn(*batch, model.nv, dtype=torch.float64, generator=generator)
    fext = torch.randn(*batch, model.njoints, 6, dtype=torch.float64, generator=generator)
    return q, v, fext


def test_forward_dynamics_inverts_inverse_dynamics_with_leading_batch_axes(g1: br.Model) -> None:
    q, v, fext = _inputs(g1, (2, 3))
    a = torch.randn(2, 3, g1.nv, dtype=torch.float64)
    state = forward_kinematics_state(g1, q)
    tau = inverse_dynamics(g1, state, v=v, a=a, fext=fext)
    assert tau.shape == (2, 3, g1.nv)
    torch.testing.assert_close(forward_dynamics(g1, state, v=v, tau=tau, fext=fext), a, rtol=1e-9, atol=1e-9)


def test_mass_matrix_is_the_acceleration_jacobian_of_inverse_dynamics(g1: br.Model) -> None:
    # The mass matrix comes from composite inertias, the torque from body wrenches: two formulas.
    q, v, _ = _inputs(g1, (2,), seed=1)
    state = forward_kinematics_state(g1, q)
    mass = mass_matrix(g1, state)
    torch.testing.assert_close(mass, mass.mT, rtol=0, atol=1e-12)
    assert (torch.linalg.eigvalsh(mass) > 0).all()
    jacobian = torch.func.jacrev(lambda a: inverse_dynamics(g1, state, v=v, a=a))(torch.zeros(2, g1.nv, dtype=torch.float64))
    # (2, nv, 2, nv): batch elements do not interact, so the diagonal blocks are the mass matrices.
    torch.testing.assert_close(torch.stack([jacobian[i, :, i] for i in range(2)]), mass, rtol=1e-9, atol=1e-9)
    momentum = (centroidal_map(g1, state) @ v.unsqueeze(-1)).squeeze(-1)
    assert momentum.shape == (2, 6)


def test_inverse_dynamics_gradcheck_through_the_state(g1: br.Model) -> None:
    q, v, fext = _inputs(g1, (1,), seed=2)
    a = torch.randn(1, g1.nv, dtype=torch.float64)

    def function(q_: torch.Tensor, v_: torch.Tensor, a_: torch.Tensor) -> torch.Tensor:
        return inverse_dynamics(g1, forward_kinematics_state(g1, q_), v=v_, a=a_, fext=fext)

    # q is a point on the free-flyer manifold; gradcheck perturbs it in the ambient space, which
    # is fine for finite differences of a smooth map.
    assert torch.autograd.gradcheck(function, (q.requires_grad_(), v.requires_grad_(), a.requires_grad_()))


def test_float32_accuracy_does_not_depend_on_distance_from_the_world_origin(g1: br.Model) -> None:
    # World quantities are taken about the first joint, so a robot 100 m away keeps short lever arms.
    q, v, _ = _inputs(g1, (4,), seed=3)
    q = q.clone()
    q[:, :3] += torch.tensor([100.0, 50.0, 0.8], dtype=torch.float64)
    reference = mass_matrix(g1, forward_kinematics_state(g1, q))
    g1_32 = g1.to(dtype=torch.float32)
    mass = mass_matrix(g1_32, forward_kinematics_state(g1_32, q.float()))
    assert (torch.linalg.eigvalsh(mass) > 0).all()
    assert ((mass.double() - reference).abs().max() / reference.abs().max()) < 1e-5


def test_state_reuse_sees_edited_inertias_and_other_models(g1: br.Model) -> None:
    q, v, _ = _inputs(g1, (1,), seed=4)
    a = torch.zeros(1, g1.nv, dtype=torch.float64)
    state = forward_kinematics_state(g1, q)
    before = inverse_dynamics(g1, state, v=v, a=a)
    g1.values.body_inertias[..., 0].mul_(2.0)
    try:
        edited = inverse_dynamics(g1, state, v=v, a=a)
        assert not torch.allclose(edited, before)
        torch.testing.assert_close(edited, inverse_dynamics(g1, forward_kinematics_state(g1, q), v=v, a=a))
    finally:
        g1.values.body_inertias[..., 0].mul_(0.5)
    torch.testing.assert_close(inverse_dynamics(g1, state, v=v, a=a), before)
