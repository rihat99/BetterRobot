"""M2b solver-quality probes P1 and P3--P6.

P2 is intentionally absent: pushing one configuration outside a redundant
arm's joint box does not prove that its end-effector pose is unreachable from
all other configurations.  A future P2 must supply a geometrically certified
unreachable target before asserting ``stalled_at_bounds``.
"""

from __future__ import annotations

import pytest
import torch

from better_robot.io import load
from better_robot.kinematics import forward_kinematics
from better_robot.optim.blocks import (
    GaussNewton,
    LevenbergMarquardt,
    LMStatus,
    Problem,
    ResidualItem,
    VarSpec,
)
from better_robot.optim.kernels import Huber, L2
from better_robot.tasks.ik import IKCostConfig, OptimizerConfig, solve_ik

from .solver_quality_support import (
    bounded_start,
    make_panda_problem,
    panda_frame_id,
    sample_nearby_configurations,
    target_poses,
)


PANDA_SEED = 20_250_717


@pytest.fixture(scope="module")
def panda_model():
    panda_description = pytest.importorskip("robot_descriptions.panda_description")
    return load(panda_description.URDF_PATH, dtype=torch.float32)


def _solver(*, max_iter: int = 60) -> LevenbergMarquardt:
    return LevenbergMarquardt(
        max_iter=max_iter,
        gtol=1e-5,
        xtol=1e-8,
        ftol=1e-8,
    )


def _position_error(model, q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    solved = forward_kinematics(model, q, compute_frames=True)
    pose = solved.frame_pose_world[..., panda_frame_id(model), :]
    return (pose[..., :3] - target[..., :3]).norm(dim=-1)


def test_p1_bounded_interior_panda_regression(panda_model) -> None:
    """The M0 projection-only stall now reaches the known interior target."""
    lower = panda_model.lower_pos_limit
    upper = panda_model.upper_pos_limit
    q_target = 0.7 * lower + 0.3 * upper
    target = target_poses(panda_model, q_target)
    problem = make_panda_problem(panda_model, target, regularized=False)

    values, state = _solver(max_iter=60).run(
        {"q": bounded_start(panda_model)},
        problem,
    )

    assert int(state.status) == int(LMStatus.CONVERGED)
    assert state.cost < 1e-8
    assert _position_error(panda_model, values["q"], target) < 1e-3
    assert state.iterations <= 60


def test_p3_block_level_feasible_panda_target(panda_model) -> None:
    """Pose + default-ish limit/rest terms reach a seeded feasible target."""
    q_target = sample_nearby_configurations(
        panda_model,
        1,
        seed=PANDA_SEED,
        fraction=0.1,
    )[0]
    target = target_poses(panda_model, q_target)
    problem = make_panda_problem(panda_model, target, regularized=True)

    values, state = _solver().run({"q": bounded_start(panda_model)}, problem)

    assert int(state.status) == int(LMStatus.CONVERGED)
    assert _position_error(panda_model, values["q"], target) < 1e-3


def test_p3_feasible_panda_target_through_solve_ik_facade(panda_model) -> None:
    """The M2c facade reaches the same default-regularized feasible target."""
    q_target = sample_nearby_configurations(
        panda_model,
        1,
        seed=PANDA_SEED,
        fraction=0.1,
    )[0]
    target = target_poses(panda_model, q_target)
    frame_name = panda_model.frame_names[panda_frame_id(panda_model)]

    result = solve_ik(
        panda_model,
        {frame_name: target},
        initial_q=bounded_start(panda_model),
        optimizer_cfg=OptimizerConfig(max_iter=60, tol=1e-5),
    )

    assert result.converged is True
    assert _position_error(panda_model, result.q, target) < 1e-3


def test_p4_gauss_newton_is_monotone_and_reports_an_honest_status(panda_model) -> None:
    """The guarded GN preset never recreates unconditional-accept divergence."""
    q_target = sample_nearby_configurations(
        panda_model,
        1,
        seed=PANDA_SEED,
        fraction=0.1,
    )[0]
    target = target_poses(panda_model, q_target)
    problem = make_panda_problem(panda_model, target, regularized=True)
    initial = {"q": bounded_start(panda_model)}
    solver = GaussNewton(
        max_iter=60,
        gtol=1e-5,
        xtol=1e-8,
        ftol=1e-8,
    )
    initial_cost = solver.init_state(initial, problem).cost

    _values, state = solver.run(initial, problem)

    assert state.cost <= initial_cost
    assert int(state.status) in {int(LMStatus.CONVERGED), int(LMStatus.MAXITER)}


class _PointFitResidual:
    name = "points"
    reads = ("x", "points")

    def __init__(self, count: int) -> None:
        self.dim = count

    def __call__(self, ctx) -> torch.Tensor:
        return ctx["x"][..., :1] - ctx["points"]

    def jacobian_blocks(self, ctx) -> dict[str, torch.Tensor]:
        x = ctx["x"]
        return {
            "x": torch.ones(
                (*x.shape[:-1], self.dim, 1),
                dtype=x.dtype,
                device=x.device,
            )
        }


def _point_problem(points: torch.Tensor, kernel) -> Problem:
    residual = _PointFitResidual(points.numel())
    return Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("points", residual, kernel=kernel),),
        parameters={"points": points},
    )


def test_p5_huber_point_fit_rejects_gross_outlier_bias() -> None:
    generator = torch.Generator().manual_seed(1_847)
    clean = 2.0 + 0.08 * torch.randn(16, generator=generator)
    points = torch.cat((clean, torch.tensor([25.0, 30.0, 35.0, 40.0])))
    initial = {"x": torch.zeros(1)}
    solver = LevenbergMarquardt(max_iter=80, gtol=2e-5)

    l2_values, _l2_state = solver.run(initial, _point_problem(points, L2()))
    huber_values, huber_state = solver.run(
        initial,
        _point_problem(points, Huber(delta=0.2)),
    )
    clean_optimum = clean.mean()
    l2_error = (l2_values["x"][0] - clean_optimum).abs()
    huber_error = (huber_values["x"][0] - clean_optimum).abs()

    assert int(huber_state.status) == int(LMStatus.CONVERGED)
    assert huber_error <= 0.1 * clean_optimum.abs()
    assert l2_error > 0.1 * clean_optimum.abs()
    assert l2_error > 10.0 * huber_error


@pytest.mark.slow
def test_p6_solve_ik_facade_one_call_matches_128_sequential_solves(panda_model) -> None:
    """The public facade preserves the M2b one-call B=128 parity contract."""
    batch = 128
    target_q = sample_nearby_configurations(
        panda_model,
        batch,
        seed=PANDA_SEED,
        fraction=0.04,
    )
    targets = target_poses(panda_model, target_q)
    start = bounded_start(panda_model)
    frame_name = panda_model.frame_names[panda_frame_id(panda_model)]
    costs = IKCostConfig(limit_weight=0.0, rest_weight=0.0)
    optimizer = OptimizerConfig(max_iter=40, tol=1e-5)

    batched = solve_ik(
        panda_model,
        {frame_name: targets},
        initial_q=start,
        cost_cfg=costs,
        optimizer_cfg=optimizer,
    )
    sequential = [
        solve_ik(
            panda_model,
            {frame_name: targets[index]},
            initial_q=start,
            cost_cfg=costs,
            optimizer_cfg=optimizer,
        )
        for index in range(batch)
    ]
    sequential_q = torch.stack([result.q for result in sequential])
    sequential_cost = torch.stack([0.5 * result.residual.square().sum() for result in sequential])
    sequential_converged = torch.tensor(
        [result.converged for result in sequential],
        dtype=torch.bool,
        device=batched.q.device,
    )
    sequential_iterations = torch.tensor(
        [result.iters for result in sequential],
        dtype=torch.int64,
        device=batched.q.device,
    )
    batched_cost = 0.5 * batched.residual.square().sum(dim=-1)

    assert batched.q.shape == (batch, panda_model.nq)
    assert isinstance(batched.converged, torch.Tensor)
    assert isinstance(batched.iters, torch.Tensor)
    assert batched.converged.shape == batched.iters.shape == (batch,)
    exact_convergence = batched.converged == sequential_converged
    iteration_carveout = batched.iters != sequential_iterations
    assert exact_convergence.float().mean() >= 0.90
    assert bool((exact_convergence | iteration_carveout).all())

    both_converged = batched.converged & sequential_converged
    assert bool(both_converged.any())
    torch.testing.assert_close(
        batched_cost[both_converged],
        sequential_cost[both_converged],
        atol=1e-6,
        rtol=1e-2,
    )
    batched_error = _position_error(panda_model, batched.q, targets)
    sequential_error = _position_error(panda_model, sequential_q, targets)
    assert bool((batched_error[both_converged] < 1e-3).all())
    assert bool((sequential_error[both_converged] < 1e-3).all())
