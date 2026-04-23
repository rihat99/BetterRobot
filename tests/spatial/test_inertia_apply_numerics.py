"""Numerical correctness tests for ``Inertia.apply`` and ``Inertia.se3_action``.

Covers the linear-first 6×6 block layout of ``Inertia._to_6x6`` — regression
guard for the former angular-first mix-up that silently falsified every
``I @ v`` call whenever ``com`` was non-zero.
"""
from __future__ import annotations

import torch

from better_robot.lie import se3
from better_robot.spatial.inertia import Inertia
from better_robot.spatial.motion import Motion


def _inertia_apply_closed_form(
    mass: torch.Tensor,
    com: torch.Tensor,
    I_c: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Reference ``f = I · v`` expanded without the 6×6 matrix.

    Matches Pinocchio's ``Inertia::__mult__(v, h)``::

        f_lin = m · (v_lin − c × ω)
        f_ang = I_c · ω + c × f_lin
    """
    v_lin, omega = v[..., :3], v[..., 3:]
    c_cross_w = torch.linalg.cross(com, omega)
    f_lin = mass * (v_lin - c_cross_w)
    f_ang = (I_c @ omega.unsqueeze(-1)).squeeze(-1) + torch.linalg.cross(com, f_lin)
    return torch.cat([f_lin, f_ang], dim=-1)


def test_inertia_apply_matches_closed_form_nonzero_com():
    mass = torch.tensor(3.0, dtype=torch.float64)
    com = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    I_c = torch.diag(torch.tensor([0.5, 0.7, 0.9], dtype=torch.float64))

    sym3 = torch.tensor([0.5, 0.7, 0.9, 0.0, 0.0, 0.0], dtype=torch.float64)
    I = Inertia.from_mass_com_sym3(mass, com, sym3)  # noqa: E741

    torch.manual_seed(0)
    v = Motion(torch.randn(6, dtype=torch.float64))

    f_actual = I.apply(v).data
    f_expected = _inertia_apply_closed_form(mass, com, I_c, v.data)

    assert torch.allclose(f_actual, f_expected, atol=1e-12)


def test_inertia_apply_matches_closed_form_zero_com():
    """Sanity: with c=0 the formula collapses to diag(m·I3, I_c)."""
    mass = torch.tensor(2.0, dtype=torch.float64)
    com = torch.zeros(3, dtype=torch.float64)
    I_c = torch.diag(torch.tensor([0.4, 0.5, 0.6], dtype=torch.float64))

    sym3 = torch.tensor([0.4, 0.5, 0.6, 0.0, 0.0, 0.0], dtype=torch.float64)
    I = Inertia.from_mass_com_sym3(mass, com, sym3)  # noqa: E741

    v = Motion(torch.tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=torch.float64))
    f = I.apply(v).data

    expected = torch.tensor(
        [2.0 * 1.0, 2.0 * 2.0, 2.0 * 3.0,
         0.4 * 0.1, 0.5 * 0.2, 0.6 * 0.3],
        dtype=torch.float64,
    )
    assert torch.allclose(f, expected, atol=1e-12)


def test_inertia_apply_batched():
    """Batched call matches per-element closed form."""
    torch.manual_seed(1)
    B = 5
    mass = torch.tensor(3.0, dtype=torch.float64)
    com = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    I_c = torch.diag(torch.tensor([0.5, 0.7, 0.9], dtype=torch.float64))

    sym3 = torch.tensor([0.5, 0.7, 0.9, 0.0, 0.0, 0.0], dtype=torch.float64)
    I = Inertia.from_mass_com_sym3(mass, com, sym3)  # noqa: E741

    v_data = torch.randn(B, 6, dtype=torch.float64)
    f = I.apply(Motion(v_data)).data
    for b in range(B):
        expected = _inertia_apply_closed_form(mass, com, I_c, v_data[b])
        assert torch.allclose(f[b], expected, atol=1e-12), f"mismatch at batch {b}"


def test_inertia_se3_action_roundtrip():
    """I.se3_action(T).se3_action(T^-1) ≈ I — composition identity."""
    torch.manual_seed(2)
    mass = torch.tensor(2.5, dtype=torch.float64)
    com = torch.tensor([0.3, -0.1, 0.25], dtype=torch.float64)
    sym3 = torch.tensor([0.4, 0.6, 0.8, 0.05, -0.02, 0.01], dtype=torch.float64)
    I = Inertia.from_mass_com_sym3(mass, com, sym3)  # noqa: E741

    xi = torch.tensor([0.1, 0.2, -0.15, 0.3, -0.2, 0.1], dtype=torch.float64)
    T = se3.exp(xi)
    T_inv = se3.inverse(T)

    I_back = I.se3_action(T).se3_action(T_inv)
    assert torch.allclose(I_back.data, I.data, atol=1e-10)


def test_inertia_se3_action_translation_preserves_mass_and_inertia_c():
    """Pure translation: mass unchanged, inertia-about-COM unchanged, com shifts."""
    mass = torch.tensor(4.0, dtype=torch.float64)
    com = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    sym3 = torch.tensor([0.5, 0.6, 0.7, 0.0, 0.0, 0.0], dtype=torch.float64)
    I = Inertia.from_mass_com_sym3(mass, com, sym3)  # noqa: E741

    # Pure translation by t: T = [t, identity quat]
    t = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float64)
    T = torch.cat([t, torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)])

    I_new = I.se3_action(T)

    assert torch.allclose(I_new.mass, mass, atol=1e-12)
    # Under a pure translation, the inertia-about-COM is unchanged.
    assert torch.allclose(I_new.inertia_matrix, I.inertia_matrix, atol=1e-10)
