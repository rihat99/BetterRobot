"""Tests for Scene mode add/remove/visible/unavailable logic.

Uses MockBackend — no viser, no pyrender.

See ``docs/12_VIEWER.md §13``.
"""

from __future__ import annotations

import pytest
import torch

import better_robot as br
from better_robot.viewer.scene import Scene
from better_robot.viewer.render_modes.skeleton import SkeletonMode
from better_robot.viewer.render_modes.urdf_mesh import URDFMeshMode
from better_robot.viewer.renderers.testing import MockBackend
from robot_descriptions import panda_description


@pytest.fixture(scope="module")
def panda():
    return br.load(panda_description.URDF_PATH)


@pytest.fixture
def backend():
    return MockBackend()


@pytest.fixture
def scene(panda, backend):
    return Scene.default(panda, backend=backend)


def test_skeleton_mode_is_available(panda, backend):
    assert SkeletonMode.is_available(panda, panda.create_data())


def test_urdf_mesh_mode_available_from_urdf(panda, backend):
    # Panda loaded from URDF via build_model — meta["ir"] is set → available
    q0 = panda.q_neutral
    data = br.forward_kinematics(panda, q0)
    assert URDFMeshMode.is_available(panda, data)


def test_urdf_mesh_mode_unavailable_programmatic(backend):
    # A programmatic model has no ir in meta → unavailable
    from better_robot.io.builders.smpl_like import make_smpl_like_body
    model = br.load(make_smpl_like_body)
    data = br.forward_kinematics(model, model.q_neutral)
    assert not URDFMeshMode.is_available(model, data)


def test_scene_default_has_urdf_mesh(panda, backend):
    # Panda has meta["ir"] → Scene.default picks URDFMeshMode over Skeleton
    scene = Scene.default(panda, backend=backend)
    avail = scene.available_modes()
    assert "URDF mesh" in avail


def test_scene_default_has_skeleton(backend):
    # Programmatic model has no ir → Scene.default falls back to SkeletonMode
    from better_robot.io.builders.smpl_like import make_smpl_like_body
    model = br.load(make_smpl_like_body)
    scene = Scene.default(model, backend=backend)
    avail = scene.available_modes()
    assert "Skeleton" in avail


def test_add_mode_available(panda, backend):
    scene = Scene(panda, backend=backend)
    mode = SkeletonMode()
    scene.add_mode(mode)
    assert "Skeleton" in scene.available_modes()


def test_set_mode_visible_false(scene, backend):
    # Panda default scene uses URDFMeshMode
    assert "URDF mesh" in scene.available_modes()
    scene.set_mode_visible("URDF mesh", False)
    # Verify backend received set_visible calls
    vis_calls = backend.calls_for("set_visible")
    assert any(not c.args[1] for c in vis_calls)


def test_set_mode_visible_true(scene, backend):
    scene.set_mode_visible("URDF mesh", True)
    vis_calls = backend.calls_for("set_visible")
    assert any(c.args[1] for c in vis_calls)


def test_remove_mode(panda, backend):
    b2 = MockBackend()
    scene = Scene.default(panda, backend=b2)
    scene.remove_mode("URDF mesh")
    assert "URDF mesh" not in scene.available_modes()
    # remove() should have been called for mesh nodes
    remove_calls = b2.calls_for("remove")
    assert len(remove_calls) > 0


def test_update_calls_set_transform(panda, backend):
    b2 = MockBackend()
    scene = Scene.default(panda, backend=b2)
    q = panda.q_neutral.clone().clamp(panda.lower_pos_limit, panda.upper_pos_limit)
    scene.update_from_q(q)
    assert len(b2.calls_for("set_transform")) > 0


def test_unavailable_mode_not_attached(backend):
    """Programmatic model: URDFMeshMode unavailable, Skeleton is the fallback."""
    from better_robot.io.builders.smpl_like import make_smpl_like_body
    model = br.load(make_smpl_like_body)
    b2 = MockBackend()
    scene = Scene.default(model, backend=b2)
    assert "URDF mesh" not in scene.available_modes()
    assert "Skeleton" in scene.available_modes()
