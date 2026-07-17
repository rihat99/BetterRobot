"""Headless tests for the public viewer playback and styling surface."""

from __future__ import annotations

import torch

import better_robot as br
from better_robot.tasks.trajectory import Trajectory
from better_robot.viewer import Scene, TrajectoryPlayer, Visualizer
from better_robot.viewer.overlays import ForceVectorsOverlay
from better_robot.viewer.renderers.testing import MockBackend
from robot_descriptions import panda_description


def _trajectory(model) -> Trajectory:
    q0 = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit)
    q1 = q0.clone()
    q1[0] += 0.25
    q1 = q1.clamp(model.lower_pos_limit, model.upper_pos_limit)
    q = torch.stack((q0, q1)).unsqueeze(0)
    t = torch.tensor([[0.0, 1.0]], dtype=q.dtype, device=q.device)
    return Trajectory(t=t, q=q)


def test_visualizer_and_force_vectors_remain_public() -> None:
    assert Visualizer.__name__ == "Visualizer"
    assert ForceVectorsOverlay.__name__ == "ForceVectorsOverlay"


def test_scene_handle_styles_primitive_while_player_updates_frame() -> None:
    model = br.load(panda_description.URDF_PATH)
    backend = MockBackend()
    scene = Scene.default(model, backend=backend)
    player = TrajectoryPlayer(scene, _trajectory(model))

    handle = scene.joint_primitive(2)
    rgba = (0.15, 0.35, 0.75, 0.9)
    handle.set_color(rgba)
    handle.set_scale(1.5)

    before = backend.last_transform(handle.name)
    assert before is not None
    before = before.clone()
    transform_count = len(backend.calls_for("set_transform"))

    player.show_frame(1)

    after = backend.last_transform(handle.name)
    assert after is not None
    assert not torch.allclose(after, before)
    assert len(backend.calls_for("set_transform")) > transform_count
    assert backend.colors[handle.name] == rgba
    assert backend.scales[handle.name] == 1.5
