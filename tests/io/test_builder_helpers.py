"""``ModelBuilder`` named-helper tests.

Per ``docs/UPDATE_PHASES.md §P5``: every named helper round-trips through
``build_model`` and produces the expected joint kind. The legacy
stringly-typed ``add_joint(kind="revolute_z")`` raises ``TypeError``
pointing at the named helper.
"""

from __future__ import annotations

import pytest
import torch

from better_robot.io import ModelBuilder, build_model


def _three_link_axis(helper_name: str) -> tuple[str, str]:
    """Build a tiny <world → base → tip> chain, return (kind, neutral q)."""
    b = ModelBuilder("rrr")
    base = b.add_body("base", mass=0.0)
    tip = b.add_body("tip", mass=1.0)
    helper = getattr(b, helper_name)
    helper("j", parent=base, child=tip, lower=-1.0, upper=1.0)
    ir = b.finalize()
    return ir, base


@pytest.mark.parametrize(
    "helper_name, expected_kind",
    [
        ("add_revolute_x", "revolute_rx"),
        ("add_revolute_y", "revolute_ry"),
        ("add_revolute_z", "revolute_rz"),
        ("add_prismatic_x", "prismatic_px"),
        ("add_prismatic_y", "prismatic_py"),
        ("add_prismatic_z", "prismatic_pz"),
    ],
)
def test_axis_named_helpers_select_correct_joint_kind(helper_name, expected_kind):
    ir, _ = _three_link_axis(helper_name)
    model = build_model(ir)
    # Joint 0 is universe, 1 is auto root, 2 is the test joint.
    assert model.joint_models[2].kind == expected_kind


def test_add_revolute_with_explicit_axis():
    b = ModelBuilder("r")
    base = b.add_body("base")
    tip = b.add_body("tip", mass=1.0)
    b.add_revolute(
        "j", parent=base, child=tip, axis=torch.tensor([0.0, 1.0, 0.0]),
    )
    model = build_model(b.finalize())
    # Y-axis revolute resolves to JointRY.
    assert model.joint_models[2].kind == "revolute_ry"


def test_add_spherical():
    b = ModelBuilder("s")
    base = b.add_body("base")
    tip = b.add_body("tip", mass=1.0)
    b.add_spherical("j", parent=base, child=tip)
    model = build_model(b.finalize())
    assert model.joint_models[2].kind == "spherical"


def test_add_free_flyer_root():
    b = ModelBuilder("ff")
    base = b.add_body("base", mass=1.0)
    tip = b.add_body("tip", mass=1.0)
    b.add_free_flyer_root(child=base)
    b.add_revolute_z("j2", parent=base, child=tip)
    model = build_model(b.finalize())
    # Joint 1 is the free-flyer root.
    assert model.joint_models[1].kind == "free_flyer"


def test_add_fixed():
    b = ModelBuilder("f")
    base = b.add_body("base", mass=0.0)
    a = b.add_body("a", mass=1.0)
    b.add_fixed("j", parent=base, child=a)
    model = build_model(b.finalize())
    assert model.joint_models[2].kind == "fixed"


def test_legacy_string_kind_raises_with_helper_pointer():
    b = ModelBuilder("legacy")
    base = b.add_body("base")
    tip = b.add_body("tip", mass=1.0)
    with pytest.raises(TypeError, match="add_revolute_z"):
        b.add_joint(
            "j", kind="revolute_z", parent=base, child=tip,
            origin=torch.zeros(7),
        )


def test_add_joint_without_kind_raises():
    b = ModelBuilder("no_kind")
    base = b.add_body("base")
    tip = b.add_body("tip", mass=1.0)
    with pytest.raises(TypeError):
        b.add_joint("j", parent=base, child=tip, origin=torch.zeros(7))
