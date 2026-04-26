"""V1 ``TrajectoryPlayer`` covers only straight-through, frame-indexed
playback.  Manifold interpolation and transport controls (seek, step,
speed, loop, ghost, trace) come back with ``docs/concepts/viewer.md §10.3``.

This file covers the V1 surface; the richer-interpolation tests return
in future work under the same filename.
"""

from __future__ import annotations

import math

import pytest
import torch

import better_robot as br
from better_robot.tasks.trajectory import Trajectory
from better_robot.viewer.scene import Scene
from better_robot.viewer.trajectory_player import TrajectoryPlayer
from better_robot.viewer.renderers.testing import MockBackend
from robot_descriptions import panda_description


@pytest.fixture(scope="module")
def panda():
    return br.load(panda_description.URDF_PATH)


def _make_trajectory(model, q_start, q_end, T=5):
    """Build a (1, T, nq) Trajectory between q_start and q_end."""
    t = torch.linspace(0.0, 1.0, T)
    qs = torch.stack(
        [q_start + (q_end - q_start) * (k / (T - 1)) for k in range(T)]
    ).unsqueeze(0)  # (1, T, nq)
    return Trajectory(t=t.unsqueeze(0), q=qs)


def _make_scene(model) -> tuple[Scene, MockBackend]:
    backend = MockBackend()
    scene = Scene.default(model, backend=backend)
    return scene, backend


def test_show_frame_zero_updates_scene(panda):
    scene, backend = _make_scene(panda)
    q0 = panda.q_neutral.clamp(panda.lower_pos_limit, panda.upper_pos_limit)
    q1 = q0.clone()
    q1[0] += 0.3
    q1 = q1.clamp(panda.lower_pos_limit, panda.upper_pos_limit)
    traj = _make_trajectory(panda, q0, q1)

    player = TrajectoryPlayer(scene, traj)
    backend.reset()
    player.show_frame(0)
    # Scene should have been pushed at least one transform update
    assert len(backend.calls_for("set_transform")) > 0


def test_show_frame_last(panda):
    scene, backend = _make_scene(panda)
    q0 = panda.q_neutral.clamp(panda.lower_pos_limit, panda.upper_pos_limit)
    q1 = q0.clone()
    q1[0] += 0.3
    q1 = q1.clamp(panda.lower_pos_limit, panda.upper_pos_limit)
    traj = _make_trajectory(panda, q0, q1)

    player = TrajectoryPlayer(scene, traj)
    backend.reset()
    player.show_frame(traj.horizon - 1)
    assert len(backend.calls_for("set_transform")) > 0


def test_show_frame_out_of_range_clamps(panda):
    scene, _ = _make_scene(panda)
    q0 = panda.q_neutral.clamp(panda.lower_pos_limit, panda.upper_pos_limit)
    traj = _make_trajectory(panda, q0, q0, T=3)
    player = TrajectoryPlayer(scene, traj)
    # Should not raise
    player.show_frame(-5)
    player.show_frame(999)


def test_horizon_matches_traj(panda):
    scene, _ = _make_scene(panda)
    q0 = panda.q_neutral.clamp(panda.lower_pos_limit, panda.upper_pos_limit)
    traj = _make_trajectory(panda, q0, q0, T=7)
    player = TrajectoryPlayer(scene, traj)
    assert player.horizon == 7


def test_play_completes_without_error(panda):
    scene, backend = _make_scene(panda)
    q0 = panda.q_neutral.clamp(panda.lower_pos_limit, panda.upper_pos_limit)
    traj = _make_trajectory(panda, q0, q0, T=3)
    player = TrajectoryPlayer(scene, traj)
    # fps=0 skips the sleep so the test is instant
    player.play(fps=0)
    # We expect at least horizon set_transform calls (init plus play)
    assert len(backend.calls_for("set_transform")) > 0


def test_seek_frame_alias(panda):
    """``seek_frame`` is kept as an alias for ``show_frame`` in V1."""
    scene, backend = _make_scene(panda)
    q0 = panda.q_neutral.clamp(panda.lower_pos_limit, panda.upper_pos_limit)
    traj = _make_trajectory(panda, q0, q0, T=4)
    player = TrajectoryPlayer(scene, traj)
    backend.reset()
    player.seek_frame(2)
    assert len(backend.calls_for("set_transform")) > 0


def test_seek_is_future_work(panda):
    """The normalised-cursor seek is §10.3 and raises NotImplementedError."""
    scene, _ = _make_scene(panda)
    q0 = panda.q_neutral.clamp(panda.lower_pos_limit, panda.upper_pos_limit)
    traj = _make_trajectory(panda, q0, q0, T=3)
    player = TrajectoryPlayer(scene, traj)
    with pytest.raises(NotImplementedError, match="§10.3"):
        player.seek(0.5)
