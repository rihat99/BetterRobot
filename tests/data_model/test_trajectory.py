"""``Trajectory``: shape validation, ``len`` and per-step access."""

from __future__ import annotations

import dataclasses

import pytest
import torch

import better_robot as br
from better_robot.exceptions import ShapeError


def test_trajectory_shapes_len_and_at() -> None:
    q = torch.arange(12.0).reshape(4, 3)
    v = torch.ones(4, 2)
    times = torch.linspace(0.0, 0.3, 4)
    trajectory = br.Trajectory(q=q, times=times, v=v)

    assert len(trajectory) == 4
    q_step, v_step = trajectory.at(2)
    assert torch.equal(q_step, q[2]) and torch.equal(v_step, v[2])
    assert br.Trajectory(q=q, times=times).at(1)[1] is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        trajectory.q = q  # type: ignore[misc]

    with pytest.raises(ShapeError):
        br.Trajectory(q=q[0], times=times)
    with pytest.raises(ShapeError):
        br.Trajectory(q=q, times=times[:3])
    with pytest.raises(ShapeError):
        br.Trajectory(q=q, times=times, v=v[:3])


@pytest.mark.parametrize("times", [[0.0, 0.1, 0.1], [0.0, 0.2, 0.1]])
def test_trajectory_rejects_times_that_are_not_strictly_increasing(times: list[float]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        br.Trajectory(q=torch.zeros(3, 2), times=torch.tensor(times))
