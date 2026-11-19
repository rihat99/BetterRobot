"""Bad input to the viewer raises the library's error tree, at construction when the layer knows the answer, naming
the argument and the fix."""

from __future__ import annotations

import socket
import sys
from collections.abc import Callable

import numpy as np
import pytest

import better_robot as br
from better_robot.exceptions import ArgumentError, OptionalDependencyError, ShapeError
from better_robot.testing import FakeServer
from better_robot.viewer import (
    ArrowsLayer,
    CameraLayer,
    FramesLayer,
    GroundLayer,
    MeshLayer,
    MeshSpec,
    PointsLayer,
    RigidMeshLayer,
    RobotVisual,
    Skin,
    SkinnedMeshLayer,
    Theme,
    Viewer,
)
from better_robot.viewer.tools import PoseGizmo
from tests.support.robots import panda_urdf

_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
_TRIANGLE = np.array([[0, 1, 2]])


def test_add_and_show_refuse_what_is_not_a_layer_or_a_step() -> None:
    viewer = Viewer(server=FakeServer(), ground=False)
    for bad in ("abc", 5, None):
        with pytest.raises(ArgumentError, match=r"add takes a layer or an iterable of layers, got"):
            viewer.add(bad)  # type: ignore[arg-type]
    with pytest.raises(ArgumentError, match="add takes Layer objects, got int at index 1"):
        viewer.add([PointsLayer("p", np.zeros((2, 3))), 5])  # type: ignore[list-item]
    assert viewer.layers == []
    with pytest.raises(ArgumentError, match="step must be an int, got 'x'"):
        viewer.show("x")  # type: ignore[arg-type]
    with pytest.raises(ArgumentError, match="step must be an int, got True"):
        viewer.show(True)  # a bool is an int to Python, never a step
    for fps in (0.0, float("nan")):
        with pytest.raises(ArgumentError, match="fps > 0"):
            Viewer(server=FakeServer(), fps=fps)


_BAD_LAYERS: list[tuple[Callable[[], object], type[Exception], str]] = [
    (lambda: PointsLayer("p", np.zeros((2, 4, 2))), ShapeError, "points must have 2 or 3 axes with a last axis of 3"),
    (
        lambda: PointsLayer("p", np.zeros((4, 3)), colors=np.zeros((5, 3))),
        ShapeError,
        r"colors must be one RGB or \(4, 3\)",
    ),
    (
        lambda: ArrowsLayer("a", np.zeros((2, 3)), vectors=np.ones((2, 3)), colors=np.zeros((3, 3))),
        ShapeError,
        "colors must be",
    ),
    (lambda: MeshLayer("m", np.zeros((3, 3)), faces=np.array([[0, 1, 2, 0]])), ShapeError, r"faces must be \(F, 3\)"),
    (lambda: FramesLayer("f", np.zeros((2, 7)), scales=np.ones(3)), ShapeError, r"scales must be \(F,\) = \(2,\)"),
    (
        lambda: MeshLayer("m", np.zeros((3, 3)), faces=np.array([[0, 1, 3]])),
        ArgumentError,
        "faces must index the 3 vertices, from 0 to 2; got 0 to 3",
    ),
    (
        lambda: SkinnedMeshLayer(
            "k",
            np.zeros((3, 3)),
            faces=[[-1, 1, 2]],
            weights=np.ones((3, 1)),
            bind_poses=np.zeros((1, 7)),
            bone_poses=np.zeros((1, 1, 7)),
        ),
        ArgumentError,
        "faces must index the 3 vertices, from 0 to 2; got -1 to 2",
    ),
    (
        lambda: RigidMeshLayer(
            "r",
            [MeshSpec(np.zeros((3, 3)), np.array([[0, 1, 2], [2, 3, 0]]), body=0, origin=_POSE)],
            body_poses=np.zeros((1, 7)),
        ),
        ArgumentError,
        "faces must index the 3 vertices",
    ),
    (
        lambda: CameraLayer("c", np.zeros((2, 6)), intrinsics=np.eye(3), size=(4, 3)),
        ShapeError,
        "poses must have 1 or 2 axes with a last axis of 7",
    ),
    (
        lambda: CameraLayer("c", np.zeros((2, 7)), intrinsics=np.eye(4), size=(4, 3)),
        ShapeError,
        r"intrinsics must be the \(3, 3\) matrix K",
    ),
    (
        lambda: SkinnedMeshLayer(
            "k",
            np.zeros((3, 3)),
            faces=_TRIANGLE,
            weights=np.ones((3, 2)),
            bind_poses=np.zeros((2, 6)),
            bone_poses=np.zeros((1, 2, 7)),
        ),
        ShapeError,
        r"bind_poses must be \(J, 7\)",
    ),
    (
        lambda: RigidMeshLayer(
            "r", [MeshSpec(np.zeros((3, 3)), _TRIANGLE, body=5, origin=_POSE)], body_poses=np.zeros((10, 2, 7))
        ),
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


def _panda(**options: object) -> RobotVisual:
    return RobotVisual(br.load(panda_urdf()), **options)  # type: ignore[arg-type]


_NAN = float("nan")
_BAD_ROBOT_INPUT: list[tuple[Callable[[], object], type[Exception], str]] = [
    (lambda: _panda().set_frames("panda_hand"), ArgumentError, "frames must be True, False or a sequence of frame"),
    (lambda: _panda().set_collision("yes"), ArgumentError, "collision must be True or False, got 'yes'"),
    (lambda: _panda(collision="yes"), ArgumentError, "collision must be True or False, got 'yes'"),
    (lambda: _panda(hue=(2.0, 0.0, 0.0)), ArgumentError, r"hue must be three numbers in \[0, 1\]"),
    (lambda: _panda(hue=(_NAN, 0.0, 0.0)), ArgumentError, r"hue must be three numbers in \[0, 1\]"),
    (lambda: _panda(hue=("1", "0", "0")), ArgumentError, r"hue must be three numbers in \[0, 1\]"),
    (lambda: _panda(opacity="0.5"), ArgumentError, r"opacity must be in \(0, 1\], got '0.5'"),
    (lambda: _panda().set_opacity(_NAN), ArgumentError, r"opacity must be in \(0, 1\], got nan"),
    (lambda: _panda(joint_values=["0.5"] * 10), ArgumentError, "joint_values must be numbers"),
    (lambda: _panda().set_joint_values(None, value_range=("0", "1")), ArgumentError, "value_range must be two finite"),
    (lambda: Theme(ramp=((1.0, 0.0, 0.0),)), ShapeError, r"ramp must be two RGB colours \(low, high\)"),
    (lambda: Theme(ramp="ab"), ShapeError, r"ramp must be two RGB colours \(low, high\)"),
    (lambda: Theme(ramp=((2.0, 0.0, 0.0), (0.0, 0.0, 0.0))), ArgumentError, r"ramp must be two colours of numbers"),
    (lambda: FramesLayer("f", np.zeros((2, 7)), scales=[1.0, -1.0]), ArgumentError, "scales must be positive"),
    (lambda: FramesLayer("f", np.zeros((2, 7)), scales=[0.0, 1.0]), ArgumentError, "scales must be positive"),
    (lambda: FramesLayer("f", np.zeros((2, 7)), scales=[_NAN, 1.0]), ArgumentError, "scales must be positive"),
    (lambda: FramesLayer("f", np.zeros((2, 7)), scales=["1", "1"]), ArgumentError, "scales must be numbers"),
]


@pytest.mark.parametrize(("call", "error", "match"), _BAD_ROBOT_INPUT)
def test_the_robot_surface_refuses_bad_input_naming_the_argument(
    call: Callable[[], object], error: type[Exception], match: str
) -> None:
    """R2 m4: a string where a switch or numbers belong, a colour outside [0, 1], a ramp of one colour and frame
    scales that are not positive were accepted, read letter by letter or met a builtin error later."""
    with pytest.raises(error, match=match):
        call()


def test_robot_visual_names_q_when_its_shape_is_wrong() -> None:
    model = br.load(panda_urdf())
    with pytest.raises(ShapeError, match=rf"q must be \(nq,\) or \(T, nq\) with nq = {model.nq}"):
        RobotVisual(model, np.zeros(model.nq + 1))


def test_a_skin_with_a_column_per_body_joint_not_per_model_joint_names_the_weights() -> None:
    """A body's weights over its own joints, not scattered into the model's columns (the universe included): the
    error names the weights and the shape they need, not the bone poses the robot visual made."""
    model = br.load(panda_urdf())
    skin = Skin(np.zeros((3, 3)), _TRIANGLE, weights=np.ones((3, model.njoints - 1)))
    with pytest.raises(ShapeError, match=rf"weights must be \(V, J\) = \(3, {model.njoints}\), one column per bone"):
        Viewer(server=FakeServer(), ground=False).add(RobotVisual(model, skin=skin))
    with pytest.raises(ShapeError, match=r"bone_poses must be \(T, J, 7\) with the J = 2 bones of bind_poses"):
        SkinnedMeshLayer(
            "k", np.zeros((3, 3)), faces=_TRIANGLE, weights=np.ones((3, 2)), bind_poses=np.zeros((2, 7)),
            bone_poses=np.zeros((1, 3, 7)),
        )  # fmt: skip


def test_pose_gizmo_takes_one_pose() -> None:
    with pytest.raises(ShapeError, match=r"pose must be \(7,\)"):
        PoseGizmo(Viewer(server=FakeServer()), "g", pose=np.zeros(6))


def _panda_collision_overlay() -> None:
    model = br.load(panda_urdf())
    geometry = br.io.load_geometry(panda_urdf(), model)
    robot = RobotVisual(model, geometry=geometry, root=panda_urdf(), style="maximal", collision=True)
    Viewer(server=FakeServer(), ground=False).add(robot)  # the overlay's meshes are loaded when it is built


def test_a_missing_optional_package_raises_its_install_line(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = [
        ("viser", "viewer", Viewer),
        ("trimesh", "meshes", _panda_collision_overlay),
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
            "skin",
            vertices,
            faces=np.array([[0, 1, 2], [3, 4, 5]]),
            weights=np.ones((6, 2)) / 2,
            bind_poses=bind,
            bone_poses=bind[None],
        )
        viewer.add([layer])
        viewer.show()
    finally:
        viewer.server.stop()
