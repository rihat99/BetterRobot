"""The stand-in server against viser: every method of ``FakeServer`` binds like viser's, every attribute and method
the viewer (or an app on it, BetterHuman's studio) uses on a stand-in handle exists on the viser class it stands for,
and a control reads back the value it was given."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import viser

import better_robot as br
from better_robot.testing import FakeServer
from better_robot.exceptions import ArgumentError
from better_robot.testing.fake_server import FakeClient, FakeGui, FakeScene
from better_robot.viewer import (
    DARK,
    ArrowsLayer,
    CameraLayer,
    FramesLayer,
    ImagePanel,
    LinesLayer,
    MeshLayer,
    MeshSpec,
    PlotPanel,
    PointsLayer,
    RigidMeshLayer,
    RobotVisual,
    SkeletonRecipe,
    SkinnedMeshLayer,
    SpheresLayer,
    Viewer,
)
from better_robot.viewer.tools import PoseGizmo
from tests.support.robots import panda_urdf

#: Methods that play the test or the browser, not viser: ``control`` finds a control, ``fire`` sends an event.
STAND_IN_ONLY = {"control", "fire"}


def test_a_stand_in_handle_refuses_writes_after_remove() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    viewer.add([PointsLayer("cloud", np.zeros((2, 4, 3)))], tags="a")
    cloud = server.scene.nodes["/a/cloud/points"]
    viewer.remove("a")
    with pytest.raises(AttributeError, match="'points': this PointCloudHandle was removed"):
        cloud.points = np.ones((4, 3))


def test_a_stand_in_keeps_a_copy_of_each_array_it_is_sent_as_viser_does() -> None:
    """The stand-in kept the arrays it was sent by reference, so the array-rule test of test_core passed its
    points half with the push removed. viser serialises an array when it is sent."""
    server = FakeServer()
    viewer = Viewer(server=server, ground=False)
    cloud = viewer.add(PointsLayer("cloud", np.zeros((1, 4, 3), np.float32)))
    node = server.scene.nodes["/cloud/cloud/points"]
    cloud.points[0, :, 2] = 0.5  # the layer's attribute: the array the add sent a slice of
    assert (node.points[:, 2] == 0.0).all() and (node.props["points"][:, 2] == 0.0).all()
    viewer.show()
    np.testing.assert_allclose(node.points[:, 2], 0.5)


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
    ("fake", "real"),
    [
        (FakeScene, viser.SceneApi),
        (FakeGui, viser.GuiApi),
        (FakeServer, viser.ViserServer),
        (FakeClient, viser.ClientHandle),
    ],
)
def test_every_stand_in_method_binds_like_viser(fake: type, real: type) -> None:
    methods = [name for name, member in vars(fake).items() if callable(member) and not name.startswith("_")]
    assert methods
    for name in methods:
        if name in STAND_IN_ONLY:
            continue
        assert hasattr(real, name), f"{fake.__name__}.{name} is not on viser's {real.__name__}"
        _assert_binds_like(getattr(fake, name), getattr(real, name), f"{fake.__name__}.{name}")


def test_the_stand_in_theme_takes_no_parameter_viser_deprecates() -> None:
    """``configure_theme(control_layout=)``, which viser 1.1 deprecates for the main panel's placement, stayed
    on the stand-in although the viewer never passes it."""
    assert "control_layout" not in inspect.signature(FakeGui.configure_theme).parameters


def _drive_the_sidebar(viewer: Viewer) -> None:
    """Each kind of sidebar control, and a refused change that turns into a notification."""

    def refuse(value: object) -> None:
        raise ArgumentError(f"cannot show {value}")

    sidebar, server = viewer.sidebar, viewer.server
    sidebar.dropdown("Robot", "Style", options=("meshes", "medium"), value="meshes", on_change=refuse)
    sidebar.slider("Robot", "Size", low=0.5, high=2.0, step=0.1, value=1.0, on_change=print)
    sidebar.checkbox("Robot", "Frames", value=False, on_change=print).value = True
    sidebar.button("Robot", "Reset", on_click=print).fire()
    style = server.gui.control("Style")
    style.value = "medium"
    style.fire(SimpleNamespace(client=server.clients[0]))  # refused: back to "meshes", a notification shown
    assert style.value == "meshes" and server.clients[0].notifications[0]["color"] == "red"
    robot = RobotVisual(br.load(panda_urdf()), medium=SkeletonRecipe(limits=True), frames=True)
    viewer.add(robot).set_style("minimal")  # a robot joins the Robot section made above; both its skeletons are built
    sidebar.remove("Robot")
    viewer.layers[1].redraw()
    viewer.set_theme(DARK)


def _assert_binds_like(fake: Callable[..., Any], real: Callable[..., Any], where: str) -> None:
    """Every parameter of ``fake`` is one of ``real``'s with its default, at its position when it can be positional."""
    ours = list(inspect.signature(fake).parameters.values())[1:]
    theirs = list(inspect.signature(real).parameters.values())[1:]
    by_name = {parameter.name: parameter for parameter in theirs}
    for index, parameter in enumerate(ours):
        twin = by_name.get(parameter.name)
        assert twin is not None, f"{where}({parameter.name}=) is not a parameter of viser's"
        assert parameter.default == twin.default, f"{where}({parameter.name}=) default differs"
        if parameter.kind is parameter.POSITIONAL_OR_KEYWORD:  # a positional call must bind the same way
            assert theirs[index] == twin, f"{where}: {parameter.name} sits at another position"


def test_the_controls_an_app_adds_read_back_their_initial_values() -> None:
    """BetterHuman's studio builds its panel on these calls; a dropdown starts at its first option unless told
    otherwise, as on viser."""
    gui = FakeServer().gui
    assert gui.add_dropdown("Family", ("smpl", "mhr")).value == "smpl"
    assert gui.add_dropdown("Gender", ("neutral", "male"), initial_value="male", disabled=True).value == "male"
    assert gui.add_slider("LOD", min=0, max=2, step=1, initial_value=1, disabled=True).value == 1
    assert gui.add_text("Pose name", initial_value="pose").value == "pose"
    assert gui.add_number("Density", initial_value=985.0).value == 985.0
    assert gui.add_markdown("**smpl**").content == "**smpl**"


@pytest.mark.slow
def test_every_handle_attribute_and_method_the_viewer_uses_exists_on_viser() -> None:
    server = FakeServer()
    server.clients[0] = FakeClient()
    viewer = Viewer(server=server)  # its ground and the theme's backdrop included
    rng = np.random.default_rng(0)
    points = rng.normal(size=(3, 8, 3)).astype(np.float32)
    faces = np.array([[0, 1, 2], [2, 3, 4]])
    poses = np.tile(np.array([0, 0, 0, 0, 0, 0, 1.0], np.float32), (3, 4, 1))
    spec = MeshSpec(vertices=points[0], faces=faces, body=1, origin=poses[0, 0])
    layers = [
        PointsLayer("points", points),
        MeshLayer("mesh", points, faces=faces, opacity=0.5),
        RigidMeshLayer("rigid", [spec], body_poses=poses),
        FramesLayer("frames", poses),
        ArrowsLayer("arrows", points[:, :3], vectors=points[:, 3:6]),
        ArrowsLayer("solid", points[:, :3], vectors=points[:, 3:6], shaft_radius=0.01),
        SpheresLayer("spheres", points[:, :3], radii=np.full(3, 0.1)),
        SkinnedMeshLayer(
            "skinned", points[0], faces=faces, weights=rng.uniform(size=(8, 4)), bind_poses=poses[0], bone_poses=poses
        ),
        CameraLayer("camera", poses[:, 0], intrinsics=np.eye(3), size=(4, 4), image=np.zeros((3, 4, 4, 3), np.uint8)),
        ImagePanel("image", np.zeros((3, 4, 4, 3), np.uint8)),
        PlotPanel("plot", rng.normal(size=(3, 2))),
        LinesLayer("lines", points.reshape(3, 4, 2, 3)),
        MeshLayer("painted", points[0], faces=faces, colors=rng.uniform(size=(8, 3))),
        MeshLayer("bands", points, faces=faces, colors=np.eye(3)[np.arange(8) % 2]),
        PointsLayer("ragged", [points[0], points[1, :2], points[2, :0]]),
    ]
    viewer.add(layers)
    viewer.add([SpheresLayer("valid", points[:, :3], radii=np.full(3, 0.1))], tags="v", valid=np.ones((3, 3), bool))
    for step in range(3):
        viewer.show(step)
    for layer in (layers[1], layers[2], layers[6], layers[7]):
        layer.set_opacity(0.3)
    for layer in layers:
        viewer.set_visible(layer, False)
        viewer.set_visible(layer, True)
    viewer.replace(layers[0], PointsLayer("points2", points))

    _drive_the_sidebar(viewer)
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
    viewer.remove("v")
    viewer.serve()
    # the GUI calls of an app built on the viewer (BetterHuman's studio)
    server.gui.add_dropdown("Family", options=("smpl", "mhr"), initial_value="smpl").value = "mhr"
    server.gui.add_dropdown("Gender", options=("neutral",), initial_value="neutral", disabled=True).disabled = False
    server.gui.add_slider("LOD", min=0, max=2, step=1, initial_value=0, disabled=False).disabled = True
    server.gui.add_text("Pose name", initial_value="pose").value = "pose 2"
    server.gui.add_number("Density", initial_value=985.0).value = 1000.0
    server.gui.add_markdown("").content = "**smpl**"

    handles = [*server.scene.nodes.values(), *server.gui.controls, server.initial_camera, server.clients[0].camera]
    handles += [bone for handle in server.scene.nodes.values() for bone in handle.__dict__.get("bones", ())]
    handles.append(server.gui.main_panel)
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
    assert "GuiMarkdownHandle.content" in used
    assert {"GuiDropdownHandle.disabled", "GuiSliderHandle.disabled"} <= used
