"""Public API surface — the 25-symbol contract.

Enforces ``docs/01_ARCHITECTURE.md §Public API contract``:

1. ``better_robot.__all__`` has exactly 25 entries.
2. Every listed symbol is actually importable as a top-level attribute.
3. Every listed symbol has a non-empty docstring.
4. The entries match the canonical list from the architecture doc.
"""

from __future__ import annotations

import better_robot as br

EXPECTED = {
    # data_model
    "Model",
    "Data",
    "Frame",
    "Joint",
    "Body",
    # io
    "load",
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
    "solve",
    # tasks
    "solve_ik",
    "solve_trajopt",
    "retarget",
    "Trajectory",
}


def test_all_length() -> None:
    assert len(br.__all__) == 25, f"expected 25 symbols, got {len(br.__all__)}"


def test_all_matches_spec() -> None:
    assert set(br.__all__) == EXPECTED, (
        f"missing: {EXPECTED - set(br.__all__)}; extra: {set(br.__all__) - EXPECTED}"
    )


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
