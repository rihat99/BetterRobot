"""Tests for ``src/better_robot/lie/tangents.py`` — hat/vee operators and Jacobians.

Key property checked: Jr(xi) satisfies the finite-difference definition
    log(exp(-xi) * exp(xi + eps*ej)) / eps ≈ Jr(xi)[:, j]
Random cases draw from a ``torch.Generator`` seeded by the ``seed`` in the test id.
"""

import math
import torch
import pytest
from better_robot.lie import so3 as _so3
from better_robot.lie import tangents


SEEDS = [0, 1, 2]


def _randn(seed: int, *shape: int, scale: float = 1.0) -> torch.Tensor:
    """Float32 ``N(0, scale^2)`` draws from a generator seeded with ``seed``."""
    return scale * torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


# ──────────────────────────── hat / vee ──────────────────────────────────


@pytest.mark.parametrize("seed", SEEDS)
def test_hat_so3_antisymmetric(seed):
    W = tangents.hat_so3(_randn(seed, 3))
    assert W.shape == (3, 3)
    assert torch.equal(W, -W.T)


@pytest.mark.parametrize("seed", SEEDS)
def test_hat_vee_so3_roundtrip(seed):
    w = _randn(seed, 3)
    assert torch.equal(tangents.vee_so3(tangents.hat_so3(w)), w)


@pytest.mark.parametrize("seed", SEEDS)
def test_hat_se3_structure(seed):
    """hat_se3 top-left 3x3 should be skew-sym, top-right column is v."""
    xi = _randn(seed, 6)
    X = tangents.hat_se3(xi)
    assert X.shape == (4, 4)
    W = X[:3, :3]
    assert torch.equal(W, -W.T)
    assert torch.equal(X[:3, 3], xi[:3])  # translation column
    assert torch.equal(X[3], torch.zeros(4))


@pytest.mark.parametrize("seed", SEEDS)
def test_hat_vee_se3_roundtrip(seed):
    xi = _randn(seed, 6)
    assert torch.equal(tangents.vee_se3(tangents.hat_se3(xi)), xi)


# ──────────────────────── SO3 Jacobians ──────────────────────────────────


def _fd_jr_so3(omega, eps=1e-5):
    """Finite-difference right Jacobian of SO3 exp."""
    n = 3
    # Promote to float64 to avoid float32 round-off error (≈1e-7/eps ≈ 0.01 at eps=1e-5)
    omega_d = omega.double()
    Jr_d = torch.zeros(n, n, dtype=torch.float64)
    q0 = _so3.exp(omega_d)
    q0_inv = _so3.inverse(q0)
    for j in range(n):
        delta = torch.zeros(n, dtype=torch.float64)
        delta[j] = eps
        q1 = _so3.exp(omega_d + delta)
        diff_q = _so3.compose(q0_inv, q1)
        Jr_d[:, j] = _so3.log(_so3.normalize(diff_q)) / eps
    return Jr_d.to(dtype=omega.dtype)


@pytest.mark.parametrize("scale", [0.3, 1.0, 2.5])
@pytest.mark.parametrize("seed", SEEDS)
def test_right_jacobian_so3_fd(seed, scale):
    omega = _randn(seed, 3, scale=scale)
    Jr_analytic = tangents.right_jacobian_so3(omega)
    Jr_fd = _fd_jr_so3(omega)
    assert torch.allclose(Jr_analytic, Jr_fd, atol=1e-4), (
        f"Jr_so3 mismatch at scale={scale}:\n{Jr_analytic}\nvs FD:\n{Jr_fd}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_right_jacobian_so3_small_angle(seed):
    """Near zero: Jr ≈ I - 1/2 * hat(omega)."""
    omega = _randn(seed, 3, scale=1e-3)
    Jr = tangents.right_jacobian_so3(omega)
    Jr_approx = torch.eye(3) - 0.5 * tangents.hat_so3(omega)
    assert torch.allclose(Jr, Jr_approx, atol=1e-5)


@pytest.mark.parametrize(
    "op",
    [tangents.right_jacobian_so3, tangents.right_jacobian_inverse_so3],
    ids=["right_jacobian", "right_jacobian_inverse"],
)
@pytest.mark.parametrize("theta", [0.0, 1e-9], ids=["identity", "theta_1e-9"])
def test_right_jacobian_singularities_gradcheck(op, theta):
    """SO3 Jacobians have finite fp64 derivatives through second order."""
    omega = torch.tensor([theta, 0.0, 0.0], dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(op, (omega,), atol=1e-6, rtol=1e-5)
    assert torch.autograd.gradgradcheck(op, (omega,), atol=1e-6, rtol=1e-5)

    (gradient,) = torch.autograd.grad(op(omega).sum(), omega)
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_jr_jrinv_so3_product(seed):
    """Jr(omega) @ Jr_inv(omega) == I."""
    omega = _randn(seed, 3, scale=0.8)
    Jr = tangents.right_jacobian_so3(omega)
    Jr_inv = tangents.right_jacobian_inverse_so3(omega)
    assert torch.allclose(Jr @ Jr_inv, torch.eye(3), atol=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_left_right_so3_relation(seed):
    """Jl(phi) == Jr(-phi) and Jl^-1(phi) == Jr^-1(-phi)."""
    omega = _randn(seed, 3, scale=0.5)
    assert torch.equal(tangents.left_jacobian_so3(omega), tangents.right_jacobian_so3(-omega))
    assert torch.equal(tangents.left_jacobian_inverse_so3(omega), tangents.right_jacobian_inverse_so3(-omega))


# ──────────────────────── SE3 Jacobians ──────────────────────────────────


@pytest.mark.parametrize("seed", SEEDS)
def test_jr_jrinv_se3_product(seed):
    """Jr(xi) @ Jr_inv(xi) == I."""
    xi = _randn(seed, 6, scale=0.4)
    Jr = tangents.right_jacobian_se3(xi)
    Jr_inv = tangents.right_jacobian_inverse_se3(xi)
    assert torch.allclose(Jr @ Jr_inv, torch.eye(6), atol=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_left_right_se3_relation(seed):
    """Jl(xi) == Jr(-xi) and Jl^-1(xi) == Jr^-1(-xi)."""
    xi = _randn(seed, 6, scale=0.5)
    assert torch.equal(tangents.left_jacobian_se3(xi), tangents.right_jacobian_se3(-xi))
    assert torch.equal(tangents.left_jacobian_inverse_se3(xi), tangents.right_jacobian_inverse_se3(-xi))


# ───────────── SE3 Jacobians at large rotation angles ────────────────────


def _ad(xi: torch.Tensor) -> torch.Tensor:
    """The se(3) adjoint ``ad([v, w]) = [[ŵ, v̂], [0, ŵ]]``, ``(..., 6) → (..., 6, 6)``."""
    linear, angular = tangents.hat_so3(xi[..., :3]), tangents.hat_so3(xi[..., 3:])
    return torch.cat((torch.cat((angular, linear), -1), torch.cat((torch.zeros_like(angular), angular), -1)), -2)


def _series_right_jacobian_se3(xi: torch.Tensor, terms: int = 60) -> torch.Tensor:
    """Reference ``Jr`` from the full power series in float64."""
    ad = _ad(xi)
    result = torch.eye(6, dtype=xi.dtype).expand_as(ad).clone()
    power = result.clone()
    for k in range(1, terms):
        power = power @ ad * (-1.0 / (k + 1))
        result = result + power
    return result


def _xi_with_angle(theta: float, batch: int = 16) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    axis = torch.randn(batch, 3, generator=generator, dtype=torch.float64)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    translation = torch.randn(batch, 3, generator=generator, dtype=torch.float64)
    return torch.cat([translation, theta * axis], dim=-1)


@pytest.mark.parametrize("theta", [1e-7, 1e-3, 0.19, 0.21, 1.0, 2.0, 3.0, math.pi, 4.5])
def test_right_jacobian_se3_exact_at_any_angle(theta):
    xi = _xi_with_angle(theta)
    reference = _series_right_jacobian_se3(xi)
    torch.testing.assert_close(tangents.right_jacobian_se3(xi), reference, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(
        tangents.right_jacobian_inverse_se3(xi), torch.linalg.inv(reference), atol=1e-9, rtol=0.0
    )


@pytest.mark.parametrize("theta", [1e-4, 0.05, 0.5, 3.0])
def test_right_jacobian_se3_float32(theta):
    xi = _xi_with_angle(theta)
    reference = _series_right_jacobian_se3(xi)
    result = tangents.right_jacobian_se3(xi.float())
    assert result.dtype == torch.float32
    torch.testing.assert_close(result.double(), reference, atol=2e-5, rtol=0.0)


@pytest.mark.parametrize("theta", [3.5, 5.0])
def test_right_jacobian_inverse_so3_beyond_pi(theta):
    omega = _xi_with_angle(theta)[:, 3:]
    product = tangents.right_jacobian_so3(omega) @ tangents.right_jacobian_inverse_so3(omega)
    torch.testing.assert_close(product, torch.eye(3, dtype=torch.float64).expand_as(product), atol=1e-10, rtol=0.0)
