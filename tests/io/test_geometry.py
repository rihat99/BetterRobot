"""A URDF with every shape kind round-trips into ``Geometry`` (record 0016); the builder attaches the capsule."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from better_robot.exceptions import UnknownNameError
from better_robot.io import ModelBuilder, build_geometry, build_model, load, load_geometry
from better_robot.model import AttachedShape, Box, Capsule, Cylinder, Mesh, Sphere

IDENTITY = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

URDF = """<robot name="shapes">
  <material name="red"><color rgba="1 0 0 1"/></material>
  <link name="base"/>
  <link name="arm">
    <visual><origin xyz="0 0 0.1"/><geometry><sphere radius="0.05"/></geometry><material name="red"/></visual>
    <visual><geometry><box size="0.1 0.2 0.3"/></geometry>
      <material name="inline"><color rgba="0 1 0 0.5"/></material></visual>
    <visual><geometry><cylinder radius="0.02" length="0.4"/></geometry></visual>
    <visual><geometry><mesh filename="package://shapes/arm.stl" scale="1 2 3"/></geometry></visual>
    <collision><origin xyz="0 0 0.2" rpy="0 0 1.5707963"/><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
  </link>
  <joint name="j" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>"""


def test_urdf_shapes_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "shapes.urdf"
    path.write_text(URDF)
    model = load(path)
    geometry = load_geometry(path, model)
    assert [attached.body for attached in geometry.shapes] == [model.body_id("arm")] * 5
    assert [attached.collision for attached in geometry.shapes] == [False, False, False, False, True]
    sphere, box, cylinder, mesh, collision_box = (attached.shape for attached in geometry.shapes)
    assert sphere == Sphere(radius=0.05, placement=(0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0), color=(1.0, 0.0, 0.0, 1.0))
    assert box == Box(size=(0.1, 0.2, 0.3), placement=IDENTITY, color=(0.0, 1.0, 0.0, 0.5))
    assert cylinder == Cylinder(radius=0.02, length=0.4, placement=IDENTITY)
    assert mesh == Mesh(uri="package://shapes/arm.stl", scale=(1.0, 2.0, 3.0), placement=IDENTITY)
    assert isinstance(collision_box, Box) and collision_box.size == (0.1, 0.1, 0.1)
    half = math.sin(math.pi / 4)
    assert collision_box.placement == pytest.approx((0.0, 0.0, 0.2, 0.0, 0.0, half, half), abs=1e-6)


def _capsule_arm() -> ModelBuilder:
    builder = ModelBuilder("arm")
    builder.add_body("base")
    builder.add_body("link")
    builder.add_fixed("j", parent="base", child="link")
    builder.add_shape("link", Capsule(radius=0.1, length=0.5, placement=IDENTITY))
    return builder


def test_builder_capsule_reaches_geometry() -> None:
    ir = _capsule_arm().finalize()
    model = build_model(ir)
    capsule = AttachedShape(
        body=model.body_id("link"), shape=Capsule(radius=0.1, length=0.5, placement=IDENTITY), collision=True
    )
    assert build_geometry(ir, model).shapes == (capsule,)


def test_build_geometry_names_a_body_the_model_lacks() -> None:
    other = ModelBuilder("other")
    other.add_body("base")
    with pytest.raises(UnknownNameError, match="'link'"):
        build_geometry(_capsule_arm().finalize(), build_model(other.finalize()))
