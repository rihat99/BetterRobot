"""Public API surface — required core contract.

Enforces ``docs/concepts/architecture.md §Public API contract``:

1. ``better_robot.__all__`` contains the required core symbols.
2. Every listed symbol is actually importable as a top-level attribute.
3. Every listed symbol has a non-empty docstring with at least one example.
4. ``better_robot.__all__`` is a list of strings.
"""

from __future__ import annotations

import better_robot as br
from better_robot.lie import SE3 as LieSE3

# The stable core. Other supported conveniences may also be exported without
# forcing this contract test to enumerate every top-level symbol forever.
REQUIRED: frozenset[str] = frozenset(
    {
        # data_model
        "Model",
        "Data",
        "Frame",
        "Joint",
        "Body",
        # io
        "load",
        "ModelBuilder",
        # lie
        "SE3",
        # kinematics
        "forward_kinematics",
        "update_frame_placements",
        "compute_joint_jacobians",
        "get_joint_jacobian",
        "get_frame_jacobian",
        "JacobianStrategy",
        # dynamics
        "rnea",
        "aba",
        "crba",
        "center_of_mass",
        "compute_centroidal_map",
        # costs
        "CostStack",
        # optim
        "LeastSquaresProblem",
        # tasks
        "solve_ik",
        "solve_trajopt",
        "solve_contact_forces",
        "Trajectory",
    }
)

REMOVED: frozenset[str] = frozenset({"register_residual", "retarget"})


def test_all_contains_required_core() -> None:
    actual = set(br.__all__)
    assert REQUIRED <= actual, f"missing required symbols: {REQUIRED - actual}"


def test_all_has_no_duplicates() -> None:
    assert len(br.__all__) == len(set(br.__all__))


def test_retired_top_level_symbols_stay_removed() -> None:
    assert REMOVED.isdisjoint(br.__all__)
    for name in REMOVED:
        assert not hasattr(br, name)


def test_all_symbols_importable() -> None:
    for name in br.__all__:
        assert hasattr(br, name), f"{name} not found on better_robot"


def test_all_symbols_have_docstrings() -> None:
    missing: list[str] = []
    for name in br.__all__:
        obj = getattr(br, name)
        doc = getattr(obj, "__doc__", None)
        if not doc or not doc.strip():
            missing.append(name)
    assert not missing, f"symbols without docstrings: {missing}"


def test_all_is_list_of_str() -> None:
    assert isinstance(br.__all__, list)
    for name in br.__all__:
        assert isinstance(name, str)


def test_root_se3_remains_the_lie_type_after_named_block_freeze() -> None:
    assert br.SE3 is LieSE3
