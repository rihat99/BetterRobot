"""Skeleton signatures — every public symbol exists and is introspectable.

Doesn't run any numerical code; just verifies that importing the symbol
works and that it's a class / function / dataclass as expected.

See ``docs/concepts/architecture.md §Phase 1 pass criteria``.
"""

from __future__ import annotations

import inspect

import better_robot as br

# Classes (including dataclasses / protocols / enums)
EXPECTED_CLASSES = {
    "Model",
    "Data",
    "Frame",
    "Body",
    "Joint",
    "SE3",
    "ModelBuilder",
}

# Plain callables / functions
EXPECTED_CALLABLES = {
    "load",
    "forward_kinematics",
    "update_frame_placements",
    "compute_joint_jacobians",
    "get_joint_jacobian",
    "get_frame_jacobian",
    "rnea",
    "aba",
    "crba",
    "center_of_mass",
    "compute_centroidal_map",
}


def test_classes_are_classes() -> None:
    for name in EXPECTED_CLASSES:
        obj = getattr(br, name)
        assert inspect.isclass(obj), f"{name} should be a class, got {type(obj).__name__}"


def test_callables_are_callable() -> None:
    for name in EXPECTED_CALLABLES:
        obj = getattr(br, name)
        assert callable(obj), f"{name} should be callable"


def test_model_has_frozen_dataclass_shape() -> None:
    fields = tuple(br.Model.__dataclass_fields__)
    assert fields == ("structure", "values", "reference_configurations", "meta")


def test_data_has_core_fields() -> None:
    """Data exposes the readable field names defined in docs/conventions/naming.md."""
    data_cls = br.Data
    fields = {f.name for f in data_cls.__dataclass_fields__.values()}
    required = {
        "q",
        "v",
        "a",
        "tau",
        "joint_pose_local",
        "joint_pose_world",
        "frame_pose_world",
        "mass_matrix",
        "joint_jacobians",
    }
    missing = required - fields
    assert not missing, f"Data missing dataclass fields: {missing}"


def test_data_legacy_aliases_are_removed() -> None:
    old_names = {
        "oMi",
        "oMf",
        "liMi",
        "ov",
        "oa",
        "v_joint",
        "a_joint",
        "M",
        "C",
        "g",
        "nle",
        "J",
        "dJ",
        "Ag",
        "hg",
        "com",
        "vcom",
        "acom",
    }
    assert all(not hasattr(br.Data, name) for name in old_names)


def test_solve_ik_signature_shape() -> None:
    from better_robot.terms import solve_ik  # noqa: PLC0415

    params = inspect.signature(solve_ik).parameters
    assert list(params)[:3] == ["model", "targets", "q0"]
    assert params["solver"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["iters"].kind is inspect.Parameter.KEYWORD_ONLY
