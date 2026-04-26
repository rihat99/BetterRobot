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
    "register_residual",
    "solve_ik",
    "solve_trajopt",
    "retarget",
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


def test_data_exposes_deprecated_aliases() -> None:
    """Old cryptic names (oMi / oMf / liMi / nle / Ag / M / J …) still resolve
    via the one-release deprecation shim.

    See docs/concepts/model_and_data.md §11 and docs/conventions/naming.md §6.
    """
    import warnings
    d = br.Data(_model_id=0, q=__import__("torch").zeros(3))
    for old in ("oMi", "oMf", "liMi", "nle", "Ag", "hg", "M", "J", "com"):
        descriptor = getattr(br.Data, old, None)
        assert isinstance(descriptor, property), f"{old} should be a @property shim"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = getattr(d, old)
        assert any(issubclass(w.category, DeprecationWarning) for w in caught), old


def test_jacobian_strategy_enum_values() -> None:
    values = {e.value for e in br.JacobianStrategy}
    assert values == {"analytic", "autodiff", "functional", "finite_diff", "auto"}


def test_solve_ik_signature_shape() -> None:
    sig = inspect.signature(br.solve_ik)
    params = sig.parameters
    assert "model" in params
    assert "targets" in params
    assert "initial_q" in params
    assert "cost_cfg" in params
    assert "optimizer_cfg" in params
    assert "robot_collision" in params


def test_register_residual_is_decorator_factory() -> None:
    # register_residual("name") must return a decorator
    deco = br.register_residual("test_dummy_residual")
    assert callable(deco)

    # Decorating a minimal class should succeed and tag it
    class _Dummy:  # noqa: D401 — test dummy
        """Dummy residual for signature check."""

        dim = 1

        def __call__(self, state):
            return None

        def jacobian(self, state):
            return None

    out = deco(_Dummy)
    assert out is _Dummy
    assert _Dummy.name == "test_dummy_residual"


def test_cost_stack_basic_api() -> None:
    stack = br.CostStack()
    assert hasattr(stack, "add")
    assert hasattr(stack, "items")
    assert isinstance(stack.items, dict)
    assert len(stack.items) == 0
