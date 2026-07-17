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
    "CostStack",
    "LeastSquaresProblem",
    "Trajectory",
    "JacobianStrategy",
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
    "solve_ik",
    "solve_trajopt",
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
    # Model is a frozen @dataclass — must expose the canonical field names.
    model_cls = br.Model
    fields = {f.name for f in model_cls.__dataclass_fields__.values()}
    required = {
        "njoints",
        "nbodies",
        "nframes",
        "nq",
        "nv",
        "parents",
        "topo_order",
        "joint_models",
        "idx_qs",
        "idx_vs",
        "joint_placements",
        "lower_pos_limit",
        "upper_pos_limit",
    }
    missing = required - fields
    assert not missing, f"Model missing dataclass fields: {missing}"


def test_data_has_core_fields() -> None:
    """Data exposes the readable field names defined in docs/conventions/naming.md."""
    data_cls = br.Data
    fields = {f.name for f in data_cls.__dataclass_fields__.values()}
    required = {
        "q", "v", "a", "tau",
        "joint_pose_local", "joint_pose_world", "frame_pose_world",
        "mass_matrix", "joint_jacobians",
    }
    missing = required - fields
    assert not missing, f"Data missing dataclass fields: {missing}"


def test_data_legacy_aliases_are_removed() -> None:
    old_names = {
        "oMi", "oMf", "liMi", "ov", "oa", "v_joint", "a_joint",
        "M", "C", "g", "nle", "J", "dJ", "Ag", "hg", "com",
        "vcom", "acom",
    }
    assert all(not hasattr(br.Data, name) for name in old_names)


def test_jacobian_strategy_enum_values() -> None:
    values = {e.value for e in br.JacobianStrategy}
    assert values == {"analytic", "finite_diff", "auto"}
    assert not hasattr(br.JacobianStrategy, "AUTODIFF")
    assert not hasattr(br.JacobianStrategy, "FUNCTIONAL")


def test_solve_ik_signature_shape() -> None:
    sig = inspect.signature(br.solve_ik)
    params = sig.parameters
    assert "model" in params
    assert "targets" in params
    assert "initial_q" in params
    assert "cost_cfg" in params
    assert "optimizer_cfg" in params
    assert "robot_collision" not in params


def test_cost_stack_basic_api() -> None:
    stack = br.CostStack()
    assert hasattr(stack, "add")
    assert hasattr(stack, "items")
    assert isinstance(stack.items, dict)
    assert len(stack.items) == 0
