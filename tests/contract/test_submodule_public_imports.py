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


OPTIM_PUBLIC: frozenset[str] = frozenset(
    {
        "ArrowNormal",
        "BandedNormal",
        "Cauchy",
        "Euclidean",
        "GaussNewton",
        "GemanMcClure",
        "Huber",
        "LBFGS",
        "LevenbergMarquardt",
        "Manifold",
        "Normal",
        "Phase",
        "Problem",
        "Result",
        "Solver",
        "State",
        "TermFn",
        "TorchOptim",
        "Tukey",
        "check_jacobian",
        "run",
        "solve",
        "solve_constrained",
        "solve_layer",
    }
)

TERMS_PUBLIC: frozenset[str] = frozenset(
    {
        "RobotManifold",
        "finite_difference",
        "frame_pose",
        "frame_position",
        "ik_problem",
        "joint_limits",
        "kinematic_state",
        "random_start",
        "rest",
        "self_clearance",
        "solve_ik",
        "sphere_clearance",
    }
)

RETIRED_OPTIM_PUBLIC: frozenset[str] = frozenset(
    {
        "AutodiffFallbackWarning",
        "BandedCholesky",
        "BlockBandedMatrix",
        "Bounds",
        "Cholesky",
        "DiagonalWeight",
        "Difference",
        "EvaluationContext",
        "FirstOrderResult",
        "ImplicitDiffConfig",
        "ImplicitDifferentiationError",
        "InformativeLinearSolver",
        "JacobianStrategy",
        "L2",
        "LMState",
        "LinearSolveResult",
        "LinearSolveStatus",
        "LinearSolver",
        "LinearSystem",
        "LinearizationDecision",
        "LinearizationMode",
        "LinearizationReason",
        "Optimizer",
        "OptimizerFactory",
        "OptimizerInfo",
        "OptimizerStatus",
        "Residual",
        "ResidualItem",
        "RobotConfig",
        "RobotStateProvider",
        "RobotVariable",
        "RobustKernel",
        "SE3Manifold",
        "SE3Variable",
        "SO3Manifold",
        "SO3Variable",
        "ScalarCost",
        "ScaleWeight",
        "StructuredNormal",
        "TemporalAnalysis",
        "TemporalPattern",
        "TorchOptimizer",
        "Values",
        "VarSpec",
        "Variable",
        "Weight",
        "detach_values",
        "residual",
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
    ("better_robot.kinematics", "KinematicState"),
    ("better_robot.kinematics", "forward_kinematics_state"),
    ("better_robot.kinematics", "frame_jacobian"),
    ("better_robot.kinematics", "joint_jacobian"),
    ("better_robot.kinematics", "frame_jacobian_time_variation"),
    ("better_robot.kinematics", "joint_jacobian_time_variation"),
    # dynamics
    ("better_robot.dynamics", "inverse_dynamics"),
    ("better_robot.dynamics", "mass_matrix"),
    ("better_robot.dynamics", "forward_dynamics"),
    ("better_robot.dynamics", "center_of_mass"),
    ("better_robot.dynamics", "centroidal_map"),
    ("better_robot.dynamics", "centroidal_momentum"),
    # data_model
    # io
    ("better_robot.io", "ModelBuilder"),
    ("better_robot.io", "build_model"),
    ("better_robot.io", "IRModel"),
    ("better_robot.optim.kernels", "GemanMcClure"),
    # Optimization API
    *(("better_robot.terms", name) for name in sorted(TERMS_PUBLIC)),
    *(("better_robot.optim", name) for name in sorted(OPTIM_PUBLIC)),
    # exceptions
]


@pytest.mark.parametrize("module_path, attr", SUBMODULE_PATHS)
def test_submodule_attribute_resolves(module_path: str, attr: str) -> None:
    mod = importlib.import_module(module_path)
    assert hasattr(mod, attr), f"{module_path} is missing {attr}"


def test_raw_kernels_are_not_public() -> None:
    """Raw tensor passes stay below the state API: no ``*_raw`` name in a package ``__all__``."""
    for module_path in ("better_robot.kinematics", "better_robot.dynamics"):
        mod = importlib.import_module(module_path)
        assert not [name for name in mod.__all__ if name.endswith("_raw")], module_path


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


def test_optim_api_is_qualified_and_has_no_lie_name_collision() -> None:
    assert OPTIM_PUBLIC <= set(optim.__all__)
    assert not hasattr(optim, "SO3")
    assert not hasattr(optim, "SE3")
    assert OPTIM_PUBLIC.isdisjoint(better_robot.__all__)
    for name in OPTIM_PUBLIC:
        assert not hasattr(better_robot, name)


def test_retired_optim_api_stays_removed() -> None:
    assert RETIRED_OPTIM_PUBLIC.isdisjoint(optim.__all__)
    for name in RETIRED_OPTIM_PUBLIC:
        assert not hasattr(optim, name)
