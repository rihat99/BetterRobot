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
from better_robot.optim.kernels import Huber, L2, Tukey

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


def test_p5_huber_accepts_a_robust_decrease_that_increases_raw_l2() -> None:
    points = torch.cat((torch.zeros(16), torch.full((4,), 100.0)))
    problem = _point_problem(points, Huber(delta=1.0))
    solver = LevenbergMarquardt(max_iter=1, gtol=1e-8)
    values = {"x": torch.tensor([20.0])}  # the raw-L2 optimum
    state = solver.init_state(values, problem)

    next_values, next_state = solver.update(values, state, problem)
    raw_before = 0.5 * problem.residual(values).square().sum()
    raw_after = 0.5 * problem.residual(next_values).square().sum()

    assert next_state.cost < state.cost
    assert raw_after > raw_before
    assert next_state.gain_ratio > 0.0
    assert not torch.equal(next_values["x"], values["x"])


class _QuadraticTukeyResidual:
    name = "quadratic"
    reads = ("x",)
    dim = 2

    def __call__(self, ctx) -> torch.Tensor:
        x = ctx["x"][..., :1]
        return torch.cat((-3.0 * x.square() - x - 6.0, x.square() + 3.0 * x + 5.0), dim=-1)

    def jacobian_blocks(self, ctx) -> dict[str, torch.Tensor]:
        x = ctx["x"][..., 0]
        rows = torch.stack((-6.0 * x - 1.0, 2.0 * x + 3.0), dim=-1)
        return {"x": rows.unsqueeze(-1)}


def test_p5_tukey_rejects_a_robust_increase_that_decreases_raw_l2() -> None:
    """At x=-2, IRLS proposes x=1: raw L2 falls but Tukey rho rises."""
    kernel = Tukey(c=10.0)
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("quadratic", _QuadraticTukeyResidual(), kernel=kernel),),
    )
    solver = LevenbergMarquardt(max_iter=1, gtol=1e-8)
    values = {"x": torch.tensor([-2.0])}
    candidate = {"x": torch.tensor([1.0])}
    current_residual = problem.residual(values)
    candidate_residual = problem.residual(candidate)
    raw_before = 0.5 * current_residual.square().sum()
    raw_candidate = 0.5 * candidate_residual.square().sum()
    robust_before = kernel.rho(current_residual.square()).sum()
    robust_candidate = kernel.rho(candidate_residual.square()).sum()
    state = solver.init_state(values, problem)

    next_values, next_state = solver.update(values, state, problem)

    assert raw_candidate < raw_before
    assert robust_candidate > robust_before
    torch.testing.assert_close(next_values["x"], values["x"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(next_state.cost, state.cost, rtol=0.0, atol=0.0)
    assert next_state.gain_ratio < 0.0
    assert next_state.mu > state.mu


@pytest.mark.slow
def test_p6_one_call_batched_panda_matches_128_sequential_solves(panda_model) -> None:
    """One B=128 call matches 128 B=1 calls of the identical solver."""
    batch = 128
    target_q = sample_nearby_configurations(
        panda_model,
        batch,
        seed=PANDA_SEED,
        fraction=0.04,
    )
    targets = target_poses(panda_model, target_q)
    start = bounded_start(panda_model)
    solver = _solver(max_iter=40)

    batched_problem = make_panda_problem(panda_model, targets, regularized=False)
    batched_values, batched_state = solver.run(
        {"q": start.expand(batch, -1).clone()},
        batched_problem,
    )

    sequential_q: list[torch.Tensor] = []
    sequential_status: list[torch.Tensor] = []
    sequential_cost: list[torch.Tensor] = []
    sequential_converged: list[torch.Tensor] = []
    sequential_kkt: list[torch.Tensor] = []
    sequential_iterations: list[torch.Tensor] = []
    for index in range(batch):
        problem = make_panda_problem(
            panda_model,
            targets[index : index + 1],
            regularized=False,
        )
        values, state = solver.run({"q": start.unsqueeze(0).clone()}, problem)
        sequential_q.append(values["q"])
        sequential_status.append(state.status)
        sequential_cost.append(state.cost)
        sequential_converged.append(state.converged)
        sequential_kkt.append(state.projected_grad_norm)
        sequential_iterations.append(state.iterations)

    seq_q = torch.cat(sequential_q)
    seq_status = torch.cat(sequential_status)
    seq_cost = torch.cat(sequential_cost)
    seq_converged = torch.cat(sequential_converged)
    seq_kkt = torch.cat(sequential_kkt)
    seq_iterations = torch.cat(sequential_iterations)

    assert batched_state.status.shape == (batch,)
    assert batched_state.cost.shape == (batch,)
    assert batched_state.converged.shape == (batch,)

    exact_status = batched_state.status == seq_status
    threshold_carveout = (
        (batched_state.projected_grad_norm <= 10.0 * solver.gtol)
        | (seq_kkt <= 10.0 * solver.gtol)
        | (batched_state.iterations != seq_iterations)
    )
    assert exact_status.float().mean() >= 0.90
    assert bool((exact_status | threshold_carveout).all())
    assert torch.equal(
        batched_state.status == LMStatus.FAILED,
        seq_status == LMStatus.FAILED,
    )

    both_converged = batched_state.converged & seq_converged
    assert bool(both_converged.any())
    scale = torch.maximum(batched_state.cost, seq_cost)
    assert bool(((batched_state.cost - seq_cost).abs() <= 1e-6 + 1e-2 * scale)[both_converged].all())
    batched_error = _position_error(panda_model, batched_values["q"], targets)
    sequential_error = _position_error(panda_model, seq_q, targets)
    assert bool((batched_error[both_converged] < 1e-3).all())
    assert bool((sequential_error[both_converged] < 1e-3).all())
