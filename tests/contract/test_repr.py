"""Every public type has a short repr (principle 4.9, decision record 0009): one line, under 200 characters, its name,
and its own (the default ``<module.Class object at 0x...>`` tells a reader nothing).

``BUILDERS`` builds each public class of the ``__all__`` tree (``better_robot.testing.public_objects``, each class
once, under the first path that lists it) from the test fixtures, with arrays large enough that a repr printing
them fails. ``NO_INSTANCE`` holds the permanent, reasoned exceptions. ``repr(model)`` also names its sizes, dtype
and device (record 0009).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np
import pytest
import torch

import better_robot as br
from better_robot import collision, io, optim, terms, viewer
from better_robot import model as robot_model
from better_robot.testing import FakeServer, public_objects
from tests.support.robots import panda_urdf

EXCEPTION = "an exception's repr is its message, which contract 06 shapes"
ABSTRACT = "an abstract base: every concrete subclass is built"
NO_INSTANCE: dict[str, str] = {
    **{
        f"better_robot.exceptions.{name}": EXCEPTION
        for name in (
            "BetterRobotError",
            "ShapeError",
            "DtypeMismatchError",
            "DeviceMismatchError",
            "ArgumentError",
            "ModelError",
            "QuaternionNormError",
            "UnknownNameError",
            "OptionalDependencyError",
        )
    },
    **{
        path: ABSTRACT
        for path in (
            "better_robot.model.JointModel",
            "better_robot.optim.Manifold",
            "better_robot.optim.RobustKernel",
            "better_robot.optim.Solver",
            "better_robot.viewer.Layer",
        )
    },
}
POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
AXIS = (0.0, 0.0, 1.0)
POINTS = np.zeros((300, 3))
POSES = np.tile(np.array(POSE), (2, 1))
FACES = np.array([[0, 1, 2]])


def _panda() -> robot_model.Model:
    return br.load(panda_urdf())


def _problems() -> tuple[optim.Problem, optim.Problem, optim.Problem]:
    """A dense, a banded and an arrowhead problem: one per normal-equation type."""
    generator = torch.Generator().manual_seed(0)
    path = torch.randn(2, 5, 3, generator=generator, dtype=torch.float64)
    scale = torch.randn(2, 4, generator=generator, dtype=torch.float64)
    dense = optim.Problem({"x": torch.zeros(2, 3)})
    dense.residual("offset", lambda x: x - 1.0)
    banded = optim.Problem({"path": path}, time=("path",))
    banded.residual("velocity", lambda path: path[:, 1:] - path[:, :-1], window=2)
    arrow = optim.Problem({"path": path, "scale": scale}, time=("path",))
    arrow.residual("scaled", lambda path, scale: path * scale[:, None, :3], window=1)
    return dense, banded, arrow


def _state() -> br.KinematicState:
    model = _panda()
    return br.forward_kinematics(model, model.q_neutral)


def _result() -> optim.Result:
    return optim.solve(_problems()[0], optim.LevenbergMarquardt(), iterations=2)


def _mesh_spec() -> viewer.MeshSpec:
    return viewer.MeshSpec(vertices=np.zeros((3, 3)), faces=FACES, body=0, origin=np.array(POSE))


BUILDERS: dict[str, Callable[[], object]] = {
    "better_robot.model.AttachedShape": lambda: robot_model.AttachedShape(
        body=1, shape=robot_model.Sphere(radius=0.1, placement=POSE), collision=True
    ),
    "better_robot.model.Box": lambda: robot_model.Box(size=(1.0, 2.0, 3.0), placement=POSE),
    "better_robot.model.Capsule": lambda: robot_model.Capsule(radius=0.1, length=0.2, placement=POSE),
    "better_robot.model.Cylinder": lambda: robot_model.Cylinder(radius=0.1, length=0.2, placement=POSE),
    "better_robot.model.Geometry": lambda: io.load_geometry(panda_urdf(), _panda()),
    "better_robot.model.JointFixed": robot_model.JointFixed,
    "better_robot.model.JointFreeFlyer": robot_model.JointFreeFlyer,
    "better_robot.model.JointHelical": lambda: robot_model.JointHelical(AXIS, 0.1),
    "better_robot.model.JointPlanar": robot_model.JointPlanar,
    "better_robot.model.JointPrismatic": lambda: robot_model.JointPrismatic(AXIS),
    "better_robot.model.JointRevolute": lambda: robot_model.JointRevolute(AXIS),
    "better_robot.model.JointRevoluteUnbounded": lambda: robot_model.JointRevoluteUnbounded(AXIS),
    "better_robot.model.JointSpherical": robot_model.JointSpherical,
    "better_robot.model.JointTranslation": robot_model.JointTranslation,
    "better_robot.model.JointManifold": lambda: robot_model.JointFreeFlyer().manifold,
    "better_robot.model.Mesh": lambda: robot_model.Mesh(uri="link0.stl", scale=(1.0, 1.0, 1.0), placement=POSE),
    "better_robot.model.Model": _panda,
    "better_robot.model.Sphere": lambda: robot_model.Sphere(radius=0.1, placement=POSE),
    "better_robot.kinematics.KinematicState": _state,
    "better_robot.collision.RobotCapsules": lambda: collision.RobotCapsules(
        frame_ids=torch.tensor([1, 2]), starts=torch.zeros(2, 3), ends=torch.ones(2, 3), radii=torch.full((2,), 0.1)
    ),
    "better_robot.collision.RobotSpheres": lambda: collision.RobotSpheres(
        frame_ids=torch.tensor([1, 2]), offsets=torch.zeros(2, 3), radii=torch.full((2,), 0.1)
    ),
    "better_robot.io.ModelBuilder": lambda: io.ModelBuilder("arm"),
    "better_robot.io.IRModel": lambda: io.IRModel(name="arm", bodies=[io.IRBody(name="link0")], joints=[]),
    "better_robot.io.IRBody": lambda: io.IRBody(name="link0"),
    "better_robot.io.IRJoint": lambda: io.IRJoint(
        name="joint1", parent_body="link0", child_body="link1", joint=robot_model.JointRevolute(AXIS)
    ),
    "better_robot.io.IRFrame": lambda: io.IRFrame(name="tool", parent_body="link0", placement=torch.tensor(POSE)),
    "better_robot.io.IRShape": lambda: io.IRShape(
        body="link0", shape=robot_model.Sphere(radius=0.1, placement=POSE), collision=True
    ),
    "better_robot.optim.ArrowNormal": lambda: _problems()[2].linearize()[0],
    "better_robot.optim.BandedNormal": lambda: _problems()[1].linearize()[0],
    "better_robot.optim.Cauchy": optim.Cauchy,
    "better_robot.optim.Euclidean": optim.Euclidean,
    "better_robot.optim.GaussNewton": optim.GaussNewton,
    "better_robot.optim.GemanMcClure": optim.GemanMcClure,
    "better_robot.optim.Huber": optim.Huber,
    "better_robot.optim.LBFGS": optim.LBFGS,
    "better_robot.optim.LevenbergMarquardt": optim.LevenbergMarquardt,
    "better_robot.optim.Normal": lambda: _problems()[0].linearize()[0],
    "better_robot.optim.Phase": lambda: optim.Phase(optim.LevenbergMarquardt(), 10),
    "better_robot.optim.Problem": lambda: _problems()[0],
    "better_robot.optim.Result": _result,
    "better_robot.optim.State": lambda: _result().state,
    "better_robot.optim.Term": lambda: _problems()[0].terms["offset"],
    "better_robot.optim.TermFn": lambda: optim.TermFn(fn=lambda x: x, reads=("x",)),
    "better_robot.optim.TorchOptim": lambda: optim.TorchOptim(torch.optim.Adam),
    "better_robot.optim.Tukey": optim.Tukey,
    "better_robot.terms.RobotManifold": lambda: terms.RobotManifold(_panda()),
    "better_robot.viewer.ArrowsLayer": lambda: viewer.ArrowsLayer(name="arrows", origins=POINTS, vectors=POINTS),
    "better_robot.viewer.CameraLayer": lambda: viewer.CameraLayer("camera", POSES, fov=1.0, aspect=1.5),
    "better_robot.viewer.FramesLayer": lambda: viewer.FramesLayer("frames", POSES),
    "better_robot.viewer.GroundLayer": viewer.GroundLayer,
    "better_robot.viewer.ImagePanel": lambda: viewer.ImagePanel("image", np.zeros((2, 4, 4, 3), dtype=np.uint8)),
    "better_robot.viewer.MeshLayer": lambda: viewer.MeshLayer(name="mesh", vertices=np.zeros((3, 3)), faces=FACES),
    "better_robot.viewer.MeshSpec": _mesh_spec,
    "better_robot.viewer.PlotPanel": lambda: viewer.PlotPanel("plot", np.zeros((10, 2))),
    "better_robot.viewer.PointsLayer": lambda: viewer.PointsLayer("points", POINTS),
    "better_robot.viewer.RigidMeshLayer": lambda: viewer.RigidMeshLayer(
        name="rigid", meshes=[_mesh_spec()], body_poses=POSES[None, :1]
    ),
    "better_robot.viewer.SkeletonLayer": lambda: viewer.SkeletonLayer(
        name="skeleton", joints=np.zeros((3, 3)), parent_ids=[-1, 0, 1]
    ),
    "better_robot.viewer.SkinnedMeshLayer": lambda: viewer.SkinnedMeshLayer(
        name="skin",
        vertices=np.zeros((3, 3)),
        faces=FACES,
        weights=np.ones((3, 1)),
        bind_poses=POSES[:1],
        bone_poses=POSES[None, :1],
    ),
    "better_robot.viewer.SpheresLayer": lambda: viewer.SpheresLayer(name="spheres", centers=POINTS, radii=np.ones(300)),
    "better_robot.viewer.Theme": viewer.Theme,
    "better_robot.viewer.Timeline": lambda: viewer.Timeline(10),
    "better_robot.viewer.Viewer": lambda: viewer.Viewer(server=FakeServer()),
    "better_robot.testing.FakeServer": FakeServer,
}


def _public_classes() -> dict[str, type]:
    """Each public class once, under the first path that lists it."""
    found: dict[int, tuple[str, type]] = {}
    for path, value in public_objects("better_robot").items():
        if inspect.isclass(value):
            found.setdefault(id(value), (path, value))
    return dict(found.values())


def test_every_public_class_is_built_or_has_a_reason() -> None:
    """A new public class needs a builder here (or a reason in ``NO_INSTANCE``)."""
    classes = _public_classes()
    assert not BUILDERS.keys() & NO_INSTANCE.keys()
    assert sorted(classes.keys() - BUILDERS.keys() - NO_INSTANCE.keys()) == []
    assert sorted((BUILDERS.keys() | NO_INSTANCE.keys()) - classes.keys()) == []


@pytest.mark.parametrize("path", sorted(BUILDERS))
def test_repr_is_one_short_line_naming_the_type(path: str) -> None:
    value = BUILDERS[path]()
    name = path.rsplit(".", 1)[-1]
    assert type(value).__name__ == name
    text = repr(value)
    assert "\n" not in text and len(text) < 200 and name in text, f"{path}: {len(text)} characters: {text[:120]!r}"
    assert "object at 0x" not in text, f"{path} prints the default object repr; give it a one-line __repr__: {text!r}"


def test_panda_repr_names_its_sizes_dtype_and_device() -> None:
    model = _panda()
    text = repr(model)
    assert text.startswith("Model(") and repr(model.name) in text
    assert f"nq={model.nq}" in text and f"nv={model.nv}" in text and f"njoints={model.njoints}" in text
    assert str(model.dtype).removeprefix("torch.") in text and str(model.device) in text
