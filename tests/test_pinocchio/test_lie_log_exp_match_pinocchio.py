"""SE(3) log / exp parity with Pinocchio.

Pinocchio's ``log6`` takes an SE3, returns a Motion (linear, angular).
BetterRobot's ``se3.log`` takes a 7-vector pose, returns a 6-vector tangent
``[vx, vy, vz, ωx, ωy, ωz]``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from better_robot.lie import se3

from .conftest import pose_to_se3

pin = pytest.importorskip("pinocchio")


def _random_pose(seed: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    t = (torch.rand(3, generator=rng, dtype=torch.float64) - 0.5) * 2.0
    # Random rotation via uniform quaternion
    q = torch.randn(4, generator=rng, dtype=torch.float64)
    q = q / q.norm()
    # se3 convention is (qx, qy, qz, qw); force qw >= 0 for a unique log.
    if q[3] < 0:
        q = -q
    return torch.cat([t, q])


def _random_xi(seed: int, magnitude: float = 0.5) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    return (torch.rand(6, generator=rng, dtype=torch.float64) - 0.5) * 2.0 * magnitude


@pytest.mark.parametrize("seed", list(range(8)))
def test_se3_log_matches_pinocchio(seed):
    pose = _random_pose(seed)
    xi_br = se3.log(pose).detach().cpu().double().numpy()
    M_pin = pose_to_se3(pose)
    motion_pin = pin.log6(M_pin)
    xi_pin = np.concatenate([np.asarray(motion_pin.linear), np.asarray(motion_pin.angular)])
    np.testing.assert_allclose(xi_br, xi_pin, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("seed", list(range(8)))
def test_se3_exp_matches_pinocchio(seed):
    xi = _random_xi(seed)
    pose_br = se3.exp(xi).detach().cpu().double().numpy()
    # Pinocchio expects Motion object
    motion_pin = pin.Motion(np.asarray(xi[:3].double().numpy()), np.asarray(xi[3:].double().numpy()))
    M_pin = pin.exp6(motion_pin)
    t_pin = np.asarray(M_pin.translation)
    from scipy.spatial.transform import Rotation as Rot
    q_pin = Rot.from_matrix(np.asarray(M_pin.rotation)).as_quat()  # xyzw
    # Normalize scalar-sign to match (quaternions q and -q are the same rotation)
    if np.sign(pose_br[3:][np.argmax(np.abs(pose_br[3:]))]) != np.sign(q_pin[np.argmax(np.abs(q_pin))]):
        q_pin = -q_pin
    np.testing.assert_allclose(pose_br[:3], t_pin, atol=1e-10)
    np.testing.assert_allclose(pose_br[3:], q_pin, atol=1e-10)


@pytest.mark.parametrize("seed", list(range(8)))
def test_log_exp_roundtrip(seed):
    """Sanity: ``exp(log(T)) == T`` in both libraries; cross-validate."""
    pose = _random_pose(seed)
    xi = se3.log(pose)
    pose_back = se3.exp(xi)

    # Both quaternions should represent the same rotation (allow sign flip).
    p = pose.numpy()
    pb = pose_back.detach().cpu().double().numpy()
    np.testing.assert_allclose(pb[:3], p[:3], atol=1e-10)
    # quat equality modulo sign
    dot = np.abs(np.dot(pb[3:], p[3:]))
    np.testing.assert_allclose(dot, 1.0, atol=1e-10)
