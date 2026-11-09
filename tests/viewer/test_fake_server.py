"""The stand-in server against viser: every method of ``FakeServer`` binds like viser's, and every attribute and method
the viewer uses on a stand-in handle exists on the viser class it stands for."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import viser

from better_robot.testing import FakeServer
from better_robot.testing.fake_server import FakeGui, FakeHandle, FakeScene
from better_robot.viewer import (
    ArrowsLayer,
    CameraLayer,
    FramesLayer,
    GroundLayer,
    ImagePanel,
    MeshLayer,
    MeshSpec,
    PlotPanel,
    PointsLayer,
    RigidMeshLayer,
    SkeletonLayer,
    SkinnedMeshLayer,
    SpheresLayer,
    Viewer,
)
from better_robot.viewer.tools import PoseGizmo

#: Methods that play the test or the browser, not viser: ``control`` finds a control, ``fire`` sends an event.
STAND_IN_ONLY = {"control", "fire"}


def test_a_stand_in_handle_refuses_writes_after_remove() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    viewer.add([PointsLayer("cloud", np.zeros((2, 4, 3)))], group="a")
    cloud = server.scene.nodes["/a/cloud/points"]
    viewer.remove_group("a")
    with pytest.raises(AttributeError, match="'points': this PointCloudHandle was removed"):
        cloud.points = np.ones((4, 3))


def test_removing_a_pointer_callback_that_is_not_held_does_nothing_as_on_viser() -> None:
    scene = FakeServer().scene
    scene.on_click()(print)
    scene.on_rect_select()(repr)
    scene.remove_click_callback(repr)
    assert scene.pointers == [("click", print), ("rect-select", repr)]
    scene.remove_click_callback()
    assert scene.pointers == [("rect-select", repr)]


@pytest.mark.slow
@pytest.mark.parametrize(
    ("fake", "real"), [(FakeScene, viser.SceneApi), (FakeGui, viser.GuiApi), (FakeServer, viser.ViserServer)]
)
def test_every_stand_in_method_binds_like_viser(fake: type, real: type) -> None:
    methods = [name for name, member in vars(fake).items() if callable(member) and not name.startswith("_")]
    assert methods
    for name in methods:
        if name in STAND_IN_ONLY:
            continue
        assert hasattr(real, name), f"{fake.__name__}.{name} is not on viser's {real.__name__}"
        ours = list(inspect.signature(getattr(fake, name)).parameters.values())[1:]
        theirs = list(inspect.signature(getattr(real, name)).parameters.values())[1:]
        by_name = {parameter.name: parameter for parameter in theirs}
        for index, parameter in enumerate(ours):
            twin = by_name.get(parameter.name)
            assert twin is not None, f"{fake.__name__}.{name}({parameter.name}=) is not a parameter of viser's"
            assert parameter.default == twin.default, f"{fake.__name__}.{name}({parameter.name}=) default differs"
            if parameter.kind is parameter.POSITIONAL_OR_KEYWORD:  # a positional call must bind the same way
                assert theirs[index] == twin, f"{fake.__name__}.{name}: {parameter.name} sits at another position"


@pytest.mark.slow
def test_every_handle_attribute_and_method_the_viewer_uses_exists_on_viser() -> None:
    server = FakeServer()
    server.clients[0] = SimpleNamespace(camera=FakeHandle(None, "CameraHandle", "camera", {}))
    viewer = Viewer(server=server)
    rng = np.random.default_rng(0)
    points = rng.normal(size=(3, 8, 3)).astype(np.float32)
    faces = np.array([[0, 1, 2], [2, 3, 4]])
    poses = np.tile(np.array([0, 0, 0, 0, 0, 0, 1.0], np.float32), (3, 4, 1))
    spec = MeshSpec(vertices=points[0], faces=faces, body=1, origin=poses[0, 0])
    layers = [
        GroundLayer(),
        PointsLayer("points", points),
        MeshLayer("mesh", points, faces=faces, opacity=0.5),
        RigidMeshLayer("rigid", [spec], body_poses=poses),
        SkeletonLayer(
            "skeleton", points[:, :4], parent_ids=[-1, 0, 1, 2], kinds=["revolute"] * 4, axes=poses[0, :, :3] + 1.0
        ),
        FramesLayer("frames", poses),
        ArrowsLayer("arrows", points[:, :3], vectors=points[:, 3:6]),
        ArrowsLayer("solid", points[:, :3], vectors=points[:, 3:6], shaft_radius=0.01),
        SpheresLayer("spheres", points[:, :3], radii=np.full(3, 0.1)),
        SkinnedMeshLayer(
            "skinned", points[0], faces=faces, weights=rng.uniform(size=(8, 4)), bind_poses=poses[0], bone_poses=poses
        ),
        CameraLayer("camera", poses[:, 0], fov=1.0, aspect=1.5, image=np.zeros((3, 4, 4, 3), np.uint8)),
        ImagePanel("image", np.zeros((3, 4, 4, 3), np.uint8)),
        PlotPanel("plot", rng.normal(size=(3, 2))),
    ]
    viewer.add(layers)
    viewer.add([SpheresLayer("valid", points[:, :3], radii=np.full(3, 0.1))], group="v", valid=np.ones((3, 3), bool))
    for step in range(3):
        viewer.show(step)
    for layer in (layers[2], layers[3], layers[8], layers[9]):
        layer.set_opacity(0.3)
    for layer in layers:
        viewer.set_visible(layer, False)
        viewer.set_visible(layer, True)
    viewer.rebuild(layers[3])
    viewer.replace(layers[1], PointsLayer("points2", points))
    server.gui.control("Play").fire()
    server.gui.control("Step").fire()
    viewer.timeline.pause()
    gizmo = PoseGizmo(viewer, "target", pose=poses[0, 0])
    gizmo.pose = poses[1, 0]
    gizmo.remove()
    viewer.on_pointer("click", print)
    viewer.clear_pointer()
    viewer.aim()
    viewer.set_fps(10.0)
    viewer.remove_group("v")
    viewer.play(block=True)

    handles = [*server.scene.nodes.values(), *server.gui.controls, server.initial_camera, server.clients[0].camera]
    handles += [bone for handle in server.scene.nodes.values() for bone in handle.__dict__.get("bones", ())]
    used = set()
    for handle in handles:
        real = getattr(viser, handle.kind)
        for attribute in handle.writes:
            used.add(f"{handle.kind}.{attribute}")
            assert any(
                attribute in vars(cls) or attribute in getattr(cls, "__annotations__", {}) for cls in real.__mro__
            ), f"the viewer writes {attribute!r} on a {handle.kind}, which viser's class lacks"
        for method in handle.calls:
            used.add(f"{handle.kind}.{method}()")
            assert callable(getattr(real, method, None)), f"the viewer calls {method}() on a {handle.kind}"
    assert {"MeshSkinnedBoneHandle.wxyz", "GuiFolderHandle.__enter__()", "TransformControlsHandle.on_update()"} <= used
