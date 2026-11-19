"""Skinned mesh (client bones + numpy LBS) on a synthetic two-bone mesh."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from better_robot.lie import so3

from better_robot.viewer import SkinnedMeshLayer, Viewer
from better_robot.viewer.layers.skinned import top_k_weights

from better_robot.testing import FakeServer


def test_bounds_hold_every_skinned_vertex_without_skinning_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chain of six bones in a tube of 600 vertices, each weighted to its two nearest bones, bent at random: the
    bones' boxes hold every skinned vertex, and reading them skins nothing (bounds are cheap, viewer/CLAUDE.md)."""
    rng = np.random.default_rng(0)
    joints = np.stack([np.zeros(6), np.zeros(6), np.arange(6) * 0.3], axis=1)  # (6, 3) a chain along +z
    vertices = (rng.random((600, 3)) * [0.2, 0.2, 1.8] - [0.1, 0.1, 0.0]).astype(np.float32)
    nearest = np.argsort(np.abs(vertices[:, 2:] - joints[:, 2]), axis=1)[:, :2]  # (V, 2) the two nearest joints
    weights = np.zeros((600, 6), np.float32)
    np.put_along_axis(weights, nearest, np.array([0.7, 0.3], np.float32)[None], axis=1)
    bind = np.concatenate([joints, np.tile([0.0, 0.0, 0.0, 1.0], (6, 1))], axis=1).astype(np.float32)
    steps = 20
    turns = so3.exp(torch.from_numpy(rng.normal(scale=0.5, size=(steps, 6, 3))))  # each bone's turn at its joint
    poses = np.repeat(bind[None], steps, axis=0)
    for j in range(1, 6):  # forward kinematics of the chain: a child sits on its parent's turned bone
        poses[:, j, 3:] = so3.compose(torch.from_numpy(poses[:, j - 1, 3:]), turns[:, j].float()).numpy()
        offset = torch.from_numpy(joints[j] - joints[j - 1]).float().expand(steps, 3)
        poses[:, j, :3] = poses[:, j - 1, :3] + so3.act(torch.from_numpy(poses[:, j - 1, 3:]), offset).numpy()
    faces = np.array([[0, 1, 2]])
    layer = SkinnedMeshLayer("tube", vertices, faces=faces, weights=weights, bind_poses=bind, bone_poses=poses)
    skinned = [layer.skin(step) for step in range(steps)]
    monkeypatch.setattr(layer, "skin", lambda step: pytest.fail("bounds skinned the vertices"))
    for step, posed in enumerate(skinned):
        low, high = layer.bounds(step)
        assert (low.min(axis=0) <= posed.min(axis=0) + 1e-5).all() and (
            high.max(axis=0) >= posed.max(axis=0) - 1e-5
        ).all()


def test_two_bone_mesh_bends_at_the_second_bone() -> None:
    # Vertices along +z from 0 to 2; bone 0 at the origin, bone 1 at z = 1 (bind pose, no rotation).
    vertices = np.array([[0.1, 0.0, z] for z in np.linspace(0.0, 2.0, 9)], dtype=np.float32)
    faces = np.array([[i, i + 1, (i + 2) % 9] for i in range(7)])
    weights = np.zeros((9, 2), dtype=np.float32)
    weights[:, 0] = np.clip(1.0 - vertices[:, 2], 0.0, 1.0)
    weights[:, 1] = 1.0 - weights[:, 0]
    bind = np.array([[0, 0, 0, 0, 0, 0, 1.0], [0, 0, 1.0, 0, 0, 0, 1.0]], dtype=np.float32)
    # Frame 1: bone 1 rotated 90° about x at its own position → the upper half folds towards -y.
    half = math.sqrt(0.5)
    posed = np.stack([bind, np.array([[0, 0, 0, 0, 0, 0, 1.0], [0, 0, 1.0, half, 0, 0, half]], dtype=np.float32)])
    layer = SkinnedMeshLayer("body", vertices, faces=faces, weights=weights, bind_poses=bind, bone_poses=posed)
    assert layer.nsteps == 2
    np.testing.assert_allclose(layer.skin(0), vertices, atol=1e-6)
    bent = layer.skin(1)
    np.testing.assert_allclose(bent[[0, 4]], vertices[[0, 4]], atol=1e-6)  # on bone 0 only / at bone 1
    np.testing.assert_allclose(bent[8], [0.1, -1.0, 1.0], atol=1e-6)  # tip: (0.1, 0, 2) rotated about bone 1

    server = FakeServer()
    viewer = Viewer(server=server)
    viewer.add([layer], tags="g")
    node = server.scene.nodes["/g/body/mesh"]
    # viser needs four bones: two unweighted bones at the origin pad the mesh and never move.
    assert node.props["skin_weights"].shape == (9, 4) and not node.props["skin_weights"][:, 2:].any()
    np.testing.assert_allclose(node.props["bone_wxyzs"][1], [1.0, 0.0, 0.0, 0.0])
    viewer.show(1)
    assert len(node.bones) == 4 and not node.bones[2].writes and not node.bones[3].writes
    np.testing.assert_allclose(node.bones[1].wxyz, [half, half, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(node.bones[1].position, [0.0, 0.0, 1.0])

    sparse = top_k_weights(np.array([[0.4, 0.3, 0.2, 0.05, 0.05]], dtype=np.float32), 4)
    np.testing.assert_allclose(sparse.sum(), 1.0, atol=1e-6)
    assert (sparse > 0).sum() == 4
