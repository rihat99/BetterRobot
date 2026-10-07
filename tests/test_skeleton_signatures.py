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
    "KinematicState",
    "Frame",
    "Body",
    "Joint",
    "SE3",
    "ModelBuilder",
}

# Plain callables / functions
EXPECTED_CALLABLES = {
    "load",
    "forward_kinematics_state",
    "frame_jacobian",
    "joint_jacobian",
    "frame_jacobian_time_variation",
    "joint_jacobian_time_variation",
    "inverse_dynamics",
    "mass_matrix",
    "forward_dynamics",
    "center_of_mass",
    "centroidal_map",
    "centroidal_momentum",
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


def test_solve_ik_signature_shape() -> None:
    from better_robot.terms import solve_ik  # noqa: PLC0415

    params = inspect.signature(solve_ik).parameters
    assert list(params)[:3] == ["model", "targets", "q0"]
    assert params["solver"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["iters"].kind is inspect.Parameter.KEYWORD_ONLY
