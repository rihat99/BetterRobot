"""Public API surface — the frozen 26-symbol contract.

Enforces ``docs/concepts/architecture.md §Public API contract``:

1. ``better_robot.__all__`` matches the frozen ``EXPECTED`` set exactly.
2. Every listed symbol is actually importable as a top-level attribute.
3. Every listed symbol has a non-empty docstring with at least one example.
4. ``better_robot.__all__`` is a list of strings.
"""

from __future__ import annotations

import better_robot as br

# The frozen public surface — single source of truth for SemVer scope.
# Adding to this set is a minor bump after 1.0; removing or renaming
# anything is a major bump.
EXPECTED: frozenset[str] = frozenset(
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
        # residuals
        "register_residual",
        # costs
        "CostStack",
        # optim
        "LeastSquaresProblem",
        # tasks
        "solve_ik",
        "solve_trajopt",
        "retarget",
        "Trajectory",
    }
)


def test_all_matches_spec() -> None:
    actual = set(br.__all__)
    assert actual == EXPECTED, (
        f"missing: {EXPECTED - actual}; extra: {actual - EXPECTED}"
    )


def test_all_length() -> None:
    assert len(br.__all__) == len(EXPECTED) == 26


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
