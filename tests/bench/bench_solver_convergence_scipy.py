"""Tracked CPU convergence reference for M2b LM against SciPy TRF.

This benchmark is advisory: it records iteration counts and final costs for
the real Panda P1/P3 problems and warns on a greater-than-2x iteration
regression.  The timed unit is one BetterRobot solve plus one SciPy solve;
the per-backend convergence results live in pytest-benchmark ``extra_info``.

SciPy promotes its parameter vector to float64 internally.  Each residual and
Jacobian callback is explicitly evaluated by BetterRobot in float32, matching
the milestone's objective arithmetic while retaining SciPy's required driver
dtype.  This limitation is recorded rather than disguised as pure fp32 TRF.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import warnings

import numpy as np
import pytest
import torch

from better_robot.optim import LevenbergMarquardt, LMStatus, Problem

from ..optim.solver_quality_support import (
    bounded_start,
    make_panda_problem,
    sample_nearby_configurations,
    target_poses,
)


pytest.importorskip("pytest_benchmark")
scipy_optimize = pytest.importorskip("scipy.optimize")
pytestmark = [pytest.mark.bench, pytest.mark.slow]

WARMUP_ROUNDS = 1
MEASURED_ROUNDS = 3
PANDA_SEED = 20_250_717


@dataclass(frozen=True)
class _Measurement:
    iterations: int
    final_cost: float
    position_error_m: float
    status: str


def _position_error(model, q: torch.Tensor, target: torch.Tensor) -> float:
    solved = target_poses(model, q)
    return float((solved[..., :3] - target[..., :3]).norm(dim=-1).amax())


def _run_better_robot(model, problem: Problem, start: torch.Tensor, target: torch.Tensor) -> _Measurement:
    solver = LevenbergMarquardt(
        max_iter=100,
        gtol=1e-5,
        xtol=1e-8,
        ftol=1e-8,
    )
    values, state = solver.run({"q": start.clone()}, problem)
    return _Measurement(
        iterations=int(state.iterations),
        final_cost=float(state.cost),
        position_error_m=_position_error(model, values["q"], target),
        status=LMStatus(int(state.status)).name.lower(),
    )


def _run_scipy_trf(model, problem: Problem, start: torch.Tensor, target: torch.Tensor) -> _Measurement:
    def residual(q_numpy: np.ndarray) -> np.ndarray:
        # scipy.optimize.least_squares promotes x to float64.  The objective
        # callback is deliberately rounded back to the benchmark's fp32 lane.
        q = torch.from_numpy(q_numpy).to(dtype=torch.float32)
        return problem.residual({"q": q}).detach().numpy()

    def jacobian(q_numpy: np.ndarray) -> np.ndarray:
        q = torch.from_numpy(q_numpy).to(dtype=torch.float32)
        return problem.dense_jacobian({"q": q}, strategy="analytic").detach().numpy()

    result = scipy_optimize.least_squares(
        residual,
        start.detach().numpy(),
        jac=jacobian,
        bounds=(
            model.lower_pos_limit.detach().numpy(),
            model.upper_pos_limit.detach().numpy(),
        ),
        method="trf",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-5,
        max_nfev=200,
    )
    if result.njev is None:  # pragma: no cover - an explicit Jacobian always sets it
        raise RuntimeError("SciPy TRF did not report a Jacobian evaluation count")
    q = torch.from_numpy(result.x).to(dtype=torch.float32)
    return _Measurement(
        iterations=int(result.njev),
        final_cost=float(result.cost),
        position_error_m=_position_error(model, q, target),
        status="converged" if result.success else "failed",
    )


def _case(model, name: str) -> tuple[Problem, torch.Tensor, torch.Tensor]:
    start = bounded_start(model)
    if name == "p1_bounded_interior":
        target_q = 0.7 * model.lower_pos_limit + 0.3 * model.upper_pos_limit
        regularized = False
    elif name == "p3_feasible_regularized":
        target_q = sample_nearby_configurations(
            model,
            1,
            seed=PANDA_SEED,
            fraction=0.1,
        )[0]
        regularized = True
    else:  # pragma: no cover - parametrization is static
        raise ValueError(name)
    target = target_poses(model, target_q)
    return make_panda_problem(model, target, regularized=regularized), start, target


@pytest.mark.parametrize("case_name", ("p1_bounded_interior", "p3_feasible_regularized"))
def test_lm_convergence_vs_scipy_trf(benchmark, panda, case_name: str) -> None:
    """Record fp32-objective convergence under a fixed warmup/statistics protocol."""
    problem, start, target = _case(panda, case_name)

    def solve_pair() -> tuple[_Measurement, _Measurement]:
        better_robot = _run_better_robot(panda, problem, start, target)
        scipy = _run_scipy_trf(panda, problem, start, target)
        return better_robot, scipy

    better_robot, scipy = benchmark.pedantic(
        solve_pair,
        iterations=1,
        rounds=MEASURED_ROUNDS,
        warmup_rounds=WARMUP_ROUNDS,
    )
    benchmark.extra_info.update(
        {
            "case": case_name,
            "device": "cpu",
            "objective_dtype": "torch.float32",
            "scipy_driver_dtype": "numpy.float64",
            "q_shape": [panda.nq],
            "batch_shape": [],
            "warmup_rounds": WARMUP_ROUNDS,
            "measured_rounds": MEASURED_ROUNDS,
            "statistic": "pytest-benchmark distribution of paired solves",
            "better_robot": asdict(better_robot),
            "scipy_trf": asdict(scipy),
        }
    )

    assert math.isfinite(better_robot.final_cost)
    assert math.isfinite(scipy.final_cost)
    if better_robot.iterations > 2 * scipy.iterations:
        warnings.warn(
            f"{case_name}: BetterRobot used {better_robot.iterations} iterations "
            f"vs SciPy TRF's {scipy.iterations} (>2x advisory threshold)",
            RuntimeWarning,
            stacklevel=2,
        )
