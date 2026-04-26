"""``Trajectory`` shape contract — accepts ``(T, nq)`` and ``(*B, T, nq)``.

Covers:
* Construction from unbatched ``(T, nq)`` and batched ``(*B, T, nq)``.
* ``batch_shape == ()`` for unbatched.
* ``with_batch_dims(1)`` normalises an unbatched trajectory.
* Mismatched shapes raise :class:`ShapeError`.
* SLERP-resampled quaternion is unit-norm.
* ``slice`` / ``downsample`` operate on the time axis.

See ``docs/concepts/tasks.md §2``.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.exceptions import ShapeError
from better_robot.tasks.trajectory import Trajectory


def test_unbatched_construction() -> None:
    t = torch.linspace(0.0, 1.0, 5)
    q = torch.zeros(5, 4)
    traj = Trajectory(t=t, q=q)
    assert traj.batch_shape == ()
    assert traj.num_knots == 5


def test_batched_construction() -> None:
    t = torch.linspace(0.0, 1.0, 5).unsqueeze(0)  # (1, 5)
    q = torch.zeros(1, 5, 4)
    traj = Trajectory(t=t, q=q)
    assert traj.batch_shape == (1,)
    assert traj.num_knots == 5


def test_multi_batch_construction() -> None:
    t = torch.linspace(0.0, 1.0, 5).expand(2, 3, 5)
    q = torch.zeros(2, 3, 5, 4)
    traj = Trajectory(t=t, q=q)
    assert traj.batch_shape == (2, 3)


def test_mismatched_T_raises() -> None:
    with pytest.raises(ShapeError, match=r"T="):
        Trajectory(t=torch.zeros(5), q=torch.zeros(6, 4))


def test_mismatched_batch_raises() -> None:
    with pytest.raises(ShapeError, match=r"batch shapes disagree"):
        Trajectory(t=torch.zeros(5), q=torch.zeros(2, 5, 4))


def test_v_shape_validated() -> None:
    t = torch.linspace(0.0, 1.0, 5)
    q = torch.zeros(5, 4)
    bad_v = torch.zeros(4, 4)  # wrong T
    with pytest.raises(ShapeError, match=r"v.shape"):
        Trajectory(t=t, q=q, v=bad_v)


def test_with_batch_dims_adds_singleton() -> None:
    t = torch.linspace(0.0, 1.0, 5)
    q = torch.zeros(5, 4)
    traj = Trajectory(t=t, q=q)
    batched = traj.with_batch_dims(1)
    assert batched.batch_shape == (1,)
    assert batched.q.shape == (1, 5, 4)
    assert batched.t.shape == (1, 5)


def test_with_batch_dims_idempotent_for_already_batched() -> None:
    traj = Trajectory(t=torch.zeros(1, 5), q=torch.zeros(1, 5, 4))
    out = traj.with_batch_dims(1)
    assert out.batch_shape == (1,)


def test_with_batch_dims_squeezes_singleton() -> None:
    traj = Trajectory(t=torch.zeros(1, 5), q=torch.zeros(1, 5, 4))
    out = traj.with_batch_dims(0)
    assert out.batch_shape == ()


def test_downsample_halves_T() -> None:
    traj = Trajectory(t=torch.arange(10.0), q=torch.zeros(10, 3))
    half = traj.downsample(2)
    assert half.num_knots == 5


def test_slice_by_time_window() -> None:
    t = torch.linspace(0.0, 1.0, 11)
    q = torch.arange(11).float().unsqueeze(-1).expand(-1, 3).contiguous()
    traj = Trajectory(t=t, q=q)
    sub = traj.slice(0.3, 0.7)
    assert sub.num_knots > 0
    assert float(sub.t[0]) >= 0.3 - 1e-6
    assert float(sub.t[-1]) <= 0.7 + 1e-6


def test_resample_linear() -> None:
    t = torch.linspace(0.0, 1.0, 4)
    q = t.unsqueeze(-1).expand(-1, 2).contiguous()  # q[k] = (t_k, t_k)
    traj = Trajectory(t=t, q=q)
    new_t = torch.tensor([0.1, 0.5, 0.9])
    out = traj.resample(new_t, kind="linear")
    expected = new_t.unsqueeze(-1).expand(-1, 2).contiguous()
    torch.testing.assert_close(out.q, expected, atol=1e-5, rtol=1e-5)


def test_resample_sclerp_keeps_unit_quat() -> None:
    """SLERP-resampled quaternions remain unit-norm."""
    T = 5
    t = torch.linspace(0.0, 1.0, T)
    q = torch.zeros(T, 7)
    q[:, 0] = t  # translation: linear in t
    # quaternion: rotate about z from 0 to π/2.
    angles = torch.linspace(0.0, math.pi / 2, T)
    q[:, 5] = torch.sin(angles / 2)   # qz
    q[:, 6] = torch.cos(angles / 2)   # qw
    traj = Trajectory(t=t, q=q)
    new_t = torch.tensor([0.25, 0.5, 0.75])
    out = traj.resample(new_t, kind="sclerp")
    quats = out.q[..., 3:7]
    norms = quats.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)
