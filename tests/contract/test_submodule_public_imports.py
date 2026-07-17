"""Submodule reachability — symbols documented as living under
``better_robot.<submodule>`` must actually resolve there.

Per ``docs/concepts/architecture.md §Public API contract``: anything
*not* in ``better_robot.__all__`` is reachable only via its submodule
path (``Symmetric3``, ``Inertia``, etc.). This test pins those paths.
"""

from __future__ import annotations

import importlib

import better_robot
import pytest
from better_robot import optim


SUBMODULE_PATHS: list[tuple[str, str]] = [
    # lie
    ("better_robot.lie", "SE3"),
    ("better_robot.lie", "SO3"),
    ("better_robot.lie", "Pose"),
    ("better_robot.lie", "se3"),
    ("better_robot.lie", "so3"),
    ("better_robot.lie", "tangents"),
    # spatial
    ("better_robot.spatial", "Motion"),
    ("better_robot.spatial", "Force"),
    ("better_robot.spatial", "Inertia"),
    ("better_robot.spatial", "Symmetric3"),
    ("better_robot.spatial", "SE3"),
    ("better_robot.spatial", "SO3"),
    ("better_robot.spatial", "Pose"),
    # kinematics
    ("better_robot.kinematics", "ReferenceFrame"),
    ("better_robot.kinematics", "JacobianStrategy"),
    # data_model
    ("better_robot.data_model", "KinematicsLevel"),
    # io
    ("better_robot.io", "ModelBuilder"),
    ("better_robot.io", "build_model"),
    ("better_robot.io", "IRModel"),
    # tasks
    ("better_robot.tasks.ik", "IKResult"),
    ("better_robot.tasks.ik", "IKCostConfig"),
    ("better_robot.tasks.ik", "OptimizerConfig"),
    # optim named-block evaluation (M2a)
    ("better_robot.optim", "Bounds"),
    ("better_robot.optim", "Euclidean"),
    ("better_robot.optim", "SO3Manifold"),
    ("better_robot.optim", "SE3Manifold"),
    ("better_robot.optim", "RobotConfig"),
    ("better_robot.optim", "Values"),
    ("better_robot.optim", "VarSpec"),
    ("better_robot.optim", "Problem"),
    ("better_robot.optim", "ResidualItem"),
    ("better_robot.optim", "ObjectiveItem"),
    ("better_robot.optim", "RobotStateProvider"),
    ("better_robot.optim", "detach_values"),
    ("better_robot.optim", "Adam"),
    ("better_robot.optim", "AdamState"),
    ("better_robot.optim", "AdamStatus"),
    ("better_robot.optim", "Phase"),
    ("better_robot.optim", "PhaseResult"),
    ("better_robot.optim", "run_phases"),
    # optim-owned legacy flat-cost composition
    ("better_robot.optim", "CostStack"),
    ("better_robot.optim", "CostItem"),
    ("better_robot.optim", "CostKind"),
    # exceptions
    ("better_robot.exceptions", "StaleCacheError"),
]


@pytest.mark.parametrize("module_path, attr", SUBMODULE_PATHS)
def test_submodule_attribute_resolves(module_path: str, attr: str) -> None:
    mod = importlib.import_module(module_path)
    assert hasattr(mod, attr), f"{module_path} is missing {attr}"


def test_symmetric3_is_submodule_only() -> None:
    """``Symmetric3`` is reachable via ``better_robot.spatial`` but **not**
    as a top-level attribute of ``better_robot``.
    """
    with pytest.raises(ImportError):
        from better_robot import Symmetric3  # noqa: F401, PLC0415


def test_named_block_api_is_qualified_and_has_no_lie_name_collision() -> None:
    block_names = {
        "Bounds",
        "Euclidean",
        "SO3Manifold",
        "SE3Manifold",
        "RobotConfig",
        "Values",
        "VarSpec",
        "Problem",
        "ResidualItem",
        "ObjectiveItem",
        "RobotStateProvider",
        "detach_values",
        "Adam",
        "AdamState",
        "AdamStatus",
        "Phase",
        "PhaseResult",
        "run_phases",
    }

    assert block_names <= set(optim.__all__)
    assert not hasattr(optim, "SO3")
    assert not hasattr(optim, "SE3")
    assert block_names.isdisjoint(better_robot.__all__)
    for name in block_names:
        assert not hasattr(better_robot, name)


def test_optimizer_owned_cost_stack_exports_are_public() -> None:
    cost_names = {"CostStack", "CostItem", "CostKind"}

    assert cost_names <= set(optim.__all__)
    assert "CostStack" in better_robot.__all__
    assert better_robot.CostStack is optim.CostStack
