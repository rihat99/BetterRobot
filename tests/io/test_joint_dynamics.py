"""URDF ``<dynamics>`` and MJCF ``damping``/``frictionloss``/``armature`` reach ``ModelValues``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from better_robot.io import build_model, load, parse

URDF = """<robot name="r"><link name="b0"/><link name="b1"><inertial><mass value="1"/>
<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
<joint name="j1" type="revolute"><parent link="b0"/><child link="b1"/><axis xyz="0 0 1"/>
<limit lower="-1" upper="1" velocity="2" effort="3"/><dynamics damping="0.5" friction="0.2"/></joint></robot>"""

MJCF = """<mujoco><worldbody><body name="b1"><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
<joint name="j1" type="hinge" axis="0 0 1" damping="0.5" frictionloss="0.2" armature="0.1"/>
</body></worldbody></mujoco>"""


def test_urdf_dynamics_reach_model_values(tmp_path: Path) -> None:
    path = tmp_path / "a.urdf"
    path.write_text(URDF)
    values = load(path, dtype=torch.float64).values
    torch.testing.assert_close(values.damping, torch.tensor([0.5], dtype=torch.float64))
    torch.testing.assert_close(values.friction, torch.tensor([0.2], dtype=torch.float64))
    torch.testing.assert_close(values.armature, torch.zeros(1, dtype=torch.float64))


def test_mjcf_dynamics_reach_model_values(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    path = tmp_path / "a.xml"
    path.write_text(MJCF)
    values = load(path, dtype=torch.float64).values
    torch.testing.assert_close(values.damping, torch.tensor([0.5], dtype=torch.float64))
    torch.testing.assert_close(values.friction, torch.tensor([0.2], dtype=torch.float64))
    torch.testing.assert_close(values.armature, torch.tensor([0.1], dtype=torch.float64))


def test_parse_returns_ir_that_build_model_consumes(tmp_path: Path) -> None:
    path = tmp_path / "a.urdf"
    path.write_text(URDF)
    ir = parse(path)
    assert [joint.damping for joint in ir.joints] == [0.5]
    assert build_model(ir).nv == load(path).nv


MJCF_DEFAULTS = """<mujoco><default><joint damping="0.3" armature="0.05"/></default><worldbody>
<body name="b1"><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/><joint name="j1" type="hinge" axis="0 0 1"/>
<body name="b2" pos="0 0 1"><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
<joint name="j2" type="ball" frictionloss="0.4"/></body></body></worldbody></mujoco>"""


def test_mjcf_default_class_inheritance_and_ball_joint_per_dof(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    path = tmp_path / "b.xml"
    path.write_text(MJCF_DEFAULTS)
    values = load(path, dtype=torch.float64).values
    torch.testing.assert_close(values.damping, torch.full((4,), 0.3, dtype=torch.float64))
    torch.testing.assert_close(values.friction, torch.tensor([0.0, 0.4, 0.4, 0.4], dtype=torch.float64))
    torch.testing.assert_close(values.armature, torch.full((4,), 0.05, dtype=torch.float64))
