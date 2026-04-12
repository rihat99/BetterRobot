"""Tests that URDFMeshMode and CollisionMode is_available logic is correct.

See ``docs/12_VIEWER.md §13``.
"""

from __future__ import annotations

import pytest
import torch

import better_robot as br
from better_robot.viewer.render_modes.urdf_mesh import URDFMeshMode
from better_robot.viewer.render_modes.skeleton import SkeletonMode
from better_robot.viewer.render_modes.collision import CollisionMode
from robot_descriptions import panda_description


@pytest.fixture(scope="module")
def panda():
    return br.load(panda_description.URDF_PATH)


@pytest.fixture(scope="module")
def panda_data(panda):
    return br.forward_kinematics(panda, panda.q_neutral)


def test_skeleton_always_available(panda, panda_data):
    assert SkeletonMode.is_available(panda, panda_data)


def test_urdf_mesh_available_with_meta(panda, panda_data):
    # Panda loaded via br.load() → build_model sets meta["ir"] → available
    assert URDFMeshMode.is_available(panda, panda_data)


def test_urdf_mesh_unavailable_no_meta():
    # A model built programmatically has no ir in meta → unavailable
    from better_robot.io.builders.smpl_like import make_smpl_like_body
    model = br.load(make_smpl_like_body)
    data = br.forward_kinematics(model, model.q_neutral)
    assert not URDFMeshMode.is_available(model, data)


def test_collision_mode_unavailable_without_rc(panda, panda_data):
    assert not CollisionMode.is_available(panda, panda_data, robot_collision=None)


def test_collision_mode_available_with_rc(panda, panda_data):
    # Pass a dummy non-None robot_collision object
    class _DummyRC:
        pass
    assert CollisionMode.is_available(panda, panda_data, robot_collision=_DummyRC())


def test_programmatic_model_skeleton_available():
    from better_robot.io.builders.smpl_like import make_smpl_like_body
    model = br.load(make_smpl_like_body)
    data = br.forward_kinematics(model, model.q_neutral)
    assert SkeletonMode.is_available(model, data)
    assert not URDFMeshMode.is_available(model, data)
