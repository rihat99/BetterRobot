"""Submodule reachability — symbols documented as living under
``better_robot.<submodule>`` must actually resolve there.

Per ``docs/design/01_ARCHITECTURE.md §Public API contract``: anything
*not* in ``better_robot.__all__`` is reachable only via its submodule
path (``Symmetric3``, ``Inertia``, etc.). This test pins those paths.
"""

from __future__ import annotations

import importlib

import pytest


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
    # backends
    ("better_robot.backends", "default_backend"),
    ("better_robot.backends", "set_backend"),
    ("better_robot.backends", "get_backend"),
    ("better_robot.backends", "Backend"),
    # exceptions
    ("better_robot.exceptions", "IRSchemaVersionError"),
    ("better_robot.exceptions", "StaleCacheError"),
    ("better_robot.exceptions", "BackendNotAvailableError"),
]


@pytest.mark.parametrize("module_path, attr", SUBMODULE_PATHS)
def test_submodule_attribute_resolves(module_path: str, attr: str) -> None:
    mod = importlib.import_module(module_path)
    assert hasattr(mod, attr), f"{module_path} is missing {attr}"


def test_symmetric3_is_submodule_only() -> None:
    """``Symmetric3`` is reachable via ``better_robot.spatial`` but **not**
    as a top-level attribute of ``better_robot``.
    """
    import better_robot

    with pytest.raises(ImportError):
        from better_robot import Symmetric3  # noqa: F401
