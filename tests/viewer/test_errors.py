"""Bad input to the viewer raises the library's error tree, at construction when the layer knows the answer, naming
the argument and the fix."""

from __future__ import annotations

import socket
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import better_robot as br
from better_robot.exceptions import ArgumentError, OptionalDependencyError, ShapeError, UnknownNameError
from better_robot.testing import FakeServer
from better_robot.viewer import (
    ArrowsLayer,
    CameraLayer,
    GroundLayer,
    MeshLayer,
    MeshSpec,
    PointsLayer,
    RigidMeshLayer,
    SkeletonLayer,
    SkinnedMeshLayer,
    Theme,
    Viewer,
    robot_skeleton,
)
from better_robot.viewer.capture import write_frames
from better_robot.viewer.robot import robot_meshes
from better_robot.viewer.tools import PoseGizmo
from tests.support.robots import panda_urdf

_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
_TRIANGLE = np.array([[0, 1, 2]])


def test_add_and_show_refuse_what_is_not_a_layer_or_a_step() -> None:
    viewer = Viewer(server=FakeServer())
    for bad in ("abc", 5, None):
        with pytest.raises(ArgumentError, match=r"add takes an iterable of layers; pass \[layer, \.\.\.\]"):
            viewer.add(bad)  # type: ignore[arg-type]
    with pytest.raises(ArgumentError, match="add takes Layer objects, got int at index 1"):
        viewer.add([PointsLayer("p", np.zeros((2, 3))), 5])  # type: ignore[list-item]
    assert viewer.layers == []
    with pytest.raises(ArgumentError, match="step must be an int, got 'x'"):
        viewer.show("x")  # type: ignore[arg-type]


_BAD_LAYERS: list[tuple[Callable[[], object], type[Exception], str]] = [
    (lambda: PointsLayer("p", np.zeros((2, 4, 2))), ShapeError, "points must have 2 or 3 axes with a last axis of 3"),
    (
        lambda: PointsLayer("p", np.zeros((4, 3)), colors=np.zeros((5, 3))),
        ShapeError,
        r"colors must be one RGB or \(4, 3\)",
    ),
    (
        lambda: ArrowsLayer("a", np.zeros((2, 3)), np.ones((2, 3)), colors=np.zeros((3, 3))),
        ShapeError,
        "colors must be",
    ),
    (lambda: MeshLayer("m", np.zeros((3, 3)), np.array([[0, 1, 2, 0]])), ShapeError, r"faces must be \(F, 3\)"),
    (
        lambda: CameraLayer("c", np.zeros((2, 6)), fov=1.0, aspect=1.0),
        ShapeError,
        "poses must have 1 or 2 axes with a last axis of 7",
    ),
    (
        lambda: SkinnedMeshLayer(
            "k", np.zeros((3, 3)), _TRIANGLE, np.ones((3, 2)), np.zeros((2, 6)), np.zeros((1, 2, 7))
        ),
        ShapeError,
        r"bind_poses must be \(J, 7\)",
    ),
    (
        lambda: SkeletonLayer("s", np.zeros((3, 3)), [-1, 0, 9]),
        ArgumentError,
        r"parent_ids must be joint indices in \[0, 3\)",
    ),
    (
        lambda: SkeletonLayer("s", np.zeros((2, 3)), [-1, 0], axes=np.zeros((3, 3))),
        ShapeError,
        r"axes must be \(3,\) or \(J, 3\)",
    ),
    (
        lambda: SkeletonLayer("s", np.zeros((3, 3)), [-1, 0, 1], bone_radius=np.zeros(5)),
        ShapeError,
        "bone_radius must be one float",
    ),
    (
        lambda: SkeletonLayer("s", np.zeros((2, 3)), [-1, 0], kinds=["revolute", "bogus"]),
        UnknownNameError,
        "unknown joint kind 'bogus'",
    ),
    (
        lambda: RigidMeshLayer("r", [MeshSpec(np.zeros((3, 3)), _TRIANGLE, 5, _POSE)], np.zeros((10, 2, 7))),
        ArgumentError,
        r"MeshSpec.body \[5\] out of range for body_poses with 2 bodies",
    ),
    (lambda: GroundLayer(plane="ab"), ArgumentError, "plane must be 'xy', 'xz', 'yz' or None"),
    (lambda: Theme(up="+w"), ArgumentError, r"up must be one of \['\+x'"),
]


def test_layers_refuse_bad_input_at_construction() -> None:
    for build, error, match in _BAD_LAYERS:
        with pytest.raises(error, match=match):
            build()


def test_robot_skeleton_names_joint_poses_and_tau_when_their_shape_is_wrong() -> None:
    model = br.load(panda_urdf())
    poses = br.forward_kinematics(model, model.q_neutral).joint_poses.numpy()
    with pytest.raises(ShapeError, match=rf"joint_poses must be \(T, njoints, 7\) = \(T, {model.njoints}, 7\)"):
        robot_skeleton(model, poses[:-1])
    with pytest.raises(ShapeError, match=rf"tau must be \(T, nv\) or \(nv,\) with nv = {model.nv}"):
        robot_skeleton(model, poses, tau=np.zeros(model.nv + 1))


def test_pose_gizmo_takes_one_pose() -> None:
    with pytest.raises(ShapeError, match=r"pose must be \(7,\)"):
        PoseGizmo(Viewer(server=FakeServer()), "g", np.zeros(6))


def test_a_missing_optional_package_raises_its_install_line(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = [
        ("viser", "viewer", Viewer),
        ("PIL", "viewer", lambda: write_frames([np.zeros((2, 2, 3), np.uint8)], tmp_path / "a.gif", fps=1.0)),
        ("trimesh", "meshes", lambda: robot_meshes(br.io.load_geometry(panda_urdf(), br.load(panda_urdf())))),
    ]
    for module, extra, call in calls:
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, module, None)
            with pytest.raises(
                OptionalDependencyError, match=f"{module} is not installed; run: uv sync --extra {extra}"
            ):
                call()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.slow
def test_a_skinned_mesh_with_fewer_than_four_bones_builds_on_a_real_server() -> None:
    viewer = Viewer(port=_free_port())
    try:
        vertices = np.random.default_rng(0).normal(size=(6, 3))
        bind = np.tile(_POSE, (2, 1))
        layer = SkinnedMeshLayer(
            "skin", vertices, np.array([[0, 1, 2], [3, 4, 5]]), np.ones((6, 2)) / 2, bind, bind[None]
        )
        viewer.add([layer])
        viewer.show()
    finally:
        viewer.server.stop()
