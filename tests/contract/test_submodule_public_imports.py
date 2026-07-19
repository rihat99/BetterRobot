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


OPTIM_V2_PUBLIC: frozenset[str] = frozenset(
    {
        "Bounds",
        "Residual",
        "Weight",
        "ScaleWeight",
        "DiagonalWeight",
        "Difference",
        "residual",
        "RobustKernel",
        "L2",
        "Huber",
        "Cauchy",
        "Tukey",
        "GemanMcClure",
        "LinearSystem",
        "LinearSolver",
        "InformativeLinearSolver",
        "LinearSolveResult",
        "LinearSolveStatus",
        "Cholesky",
        "LU",
        "BandedCholesky",
        "BlockBandedMatrix",
        "Variable",
        "SO3Variable",
        "SE3Variable",
        "RobotVariable",
        "TemporalPattern",
        "TemporalAnalysis",
        "StructuredNormal",
        "LinearizationMode",
        "LinearizationReason",
        "LinearizationDecision",
        "Problem",
        "JacobianStrategy",
        "Optimizer",
        "OptimizerInfo",
        "OptimizerStatus",
        "TorchOptimizer",
        "LevenbergMarquardt",
        "GaussNewton",
        "ImplicitDiffConfig",
        "ImplicitDifferentiationError",
    }
)

RETIRED_OPTIM_PUBLIC: frozenset[str] = frozenset(
    {
        "Manifold",
        "Euclidean",
        "SO3Manifold",
        "SE3Manifold",
        "RobotConfig",
        "Values",
        "VarSpec",
        "ResidualItem",
        "EvaluationContext",
        "RobotStateProvider",
        "detach_values",
        "LMState",
        "FirstOrderResult",
        "OptimizerFactory",
        "run_first_order",
    }
)

SUBMODULE_PATHS: list[tuple[str, str]] = [
    # lie
    ("better_robot.lie", "SE3"),
    ("better_robot.lie", "SO3"),
    ("better_robot.lie", "Pose"),
    ("better_robot.lie", "se3"),
    ("better_robot.lie", "so3"),
    ("better_robot.lie", "tangents"),
    ("better_robot.lie", "umeyama"),
    # spatial
    ("better_robot.spatial", "Motion"),
    ("better_robot.spatial", "Force"),
    ("better_robot.spatial", "Inertia"),
    ("better_robot.spatial", "Symmetric3"),
    ("better_robot.spatial", "SE3"),
    ("better_robot.spatial", "SO3"),
    ("better_robot.spatial", "Pose"),
    # kinematics
    ("better_robot.kinematics", "forward_kinematics_raw"),
    ("better_robot.kinematics", "frame_placements_raw"),
    ("better_robot.kinematics", "joint_jacobians_raw"),
    ("better_robot.kinematics", "frame_jacobian_raw"),
    # dynamics
    ("better_robot.dynamics", "rnea_raw"),
    ("better_robot.dynamics", "aba_raw"),
    ("better_robot.dynamics", "crba_raw"),
    ("better_robot.dynamics", "ccrba_raw"),
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
    ("better_robot.tasks", "smooth_trajectory"),
    ("better_robot.tasks", "solve_contact_forces"),
    ("better_robot.tasks", "ContactForceResult"),
    ("better_robot.tasks", "ContactForceWeights"),
    # M4 vision residuals
    ("better_robot.residuals", "Node"),
    ("better_robot.residuals", "RobotState"),
    ("better_robot.residuals", "ProjectionResidual"),
    ("better_robot.residuals", "MaskedChamferResidual"),
    ("better_robot.residuals", "SceneSDFState"),
    ("better_robot.residuals", "SceneSDFResult"),
    ("better_robot.residuals", "ScenePenetrationResidual"),
    ("better_robot.residuals", "SceneAttractionResidual"),
    ("better_robot.residuals", "SceneClearanceResidual"),
    ("better_robot.optim.kernels", "GemanMcClure"),
    # Optimization API v2
    *(("better_robot.optim", name) for name in sorted(OPTIM_V2_PUBLIC)),
    # exceptions
    ("better_robot.exceptions", "StaleCacheError"),
]


@pytest.mark.parametrize("module_path, attr", SUBMODULE_PATHS)
def test_submodule_attribute_resolves(module_path: str, attr: str) -> None:
    mod = importlib.import_module(module_path)
    assert hasattr(mod, attr), f"{module_path} is missing {attr}"


@pytest.mark.parametrize(
    "module_path, raw_names",
    [
        (
            "better_robot.kinematics",
            {
                "forward_kinematics_raw",
                "frame_placements_raw",
                "joint_jacobians_raw",
                "frame_jacobian_raw",
            },
        ),
        (
            "better_robot.dynamics",
            {"rnea_raw", "aba_raw", "crba_raw", "ccrba_raw"},
        ),
    ],
)
def test_raw_passes_are_in_package_all(module_path: str, raw_names: set[str]) -> None:
    mod = importlib.import_module(module_path)
    assert raw_names <= set(mod.__all__)


def test_reference_frame_enum_stays_removed() -> None:
    kinematics = importlib.import_module("better_robot.kinematics")
    assert "ReferenceFrame" not in kinematics.__all__
    assert not hasattr(kinematics, "ReferenceFrame")


def test_symmetric3_is_submodule_only() -> None:
    """``Symmetric3`` is reachable via ``better_robot.spatial`` but **not**
    as a top-level attribute of ``better_robot``.
    """
    with pytest.raises(ImportError):
        from better_robot import Symmetric3  # noqa: F401, PLC0415


def test_optim_v2_api_is_qualified_and_has_no_lie_name_collision() -> None:
    assert OPTIM_V2_PUBLIC <= set(optim.__all__)
    assert not hasattr(optim, "SO3")
    assert not hasattr(optim, "SE3")
    assert OPTIM_V2_PUBLIC.isdisjoint(better_robot.__all__)
    for name in OPTIM_V2_PUBLIC:
        assert not hasattr(better_robot, name)


def test_retired_optim_api_stays_removed() -> None:
    assert RETIRED_OPTIM_PUBLIC.isdisjoint(optim.__all__)
    for name in RETIRED_OPTIM_PUBLIC:
        assert not hasattr(optim, name)
