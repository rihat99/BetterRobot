"""BVR-shaped staged-fit vertical slice across orders 01--05."""

from __future__ import annotations

import torch

from better_robot.optim import OptimizerStatus, TorchOptimizer

from .slice_support import (
    ADAM_PHASE_STEPS,
    COARSE_WEIGHTS,
    FINAL_WEIGHTS,
    SEED,
    TIME,
    WARMUP_STEPS,
    SliceCounters,
    configure_phase,
    make_problem,
    make_slice_data,
)


def test_staged_articulated_body_fit_converges_with_exact_evaluation_count() -> None:
    """Fit a composed body through a frozen warm-up and two resumed phases."""
    torch.manual_seed(SEED)
    data = make_slice_data()
    counters = SliceCounters()
    warm_problem, warm_q, warm_scale, _warm_terms = make_problem(
        data,
        counters,
        frozen_root=True,
    )
    full_problem, full_q, full_scale, full_terms = make_problem(
        data,
        counters,
        frozen_root=False,
    )
    counters.reset()

    warm_info = TorchOptimizer(
        warm_problem,
        torch.optim.Adam,
        lr=0.035,
        max_iterations=WARMUP_STEPS,
        tolerance=0.0,
    ).optimize()
    assert int(warm_info.status) == int(OptimizerStatus.MAXITER)
    assert int(warm_info.iterations) == WARMUP_STEPS
    assert torch.equal(warm_q.tensor[..., :7], data.initial_q[..., :7])
    assert not torch.equal(warm_scale.tensor, data.initial_log_scale)

    full_problem.update(
        {
            "q": warm_q.tensor.detach().clone(),
            "log_scale": warm_scale.tensor.detach().clone(),
        }
    )
    configure_phase(full_terms, COARSE_WEIGHTS, projection_enabled=False)
    initial_terms = full_problem.term_costs()
    assert tuple(initial_terms) == ("scene", "chamfer", "projection", "scale_prior")
    torch.testing.assert_close(initial_terms["projection"], torch.tensor(0.0))

    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []

    def scheduler_factory(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        schedulers.append(scheduler)
        return scheduler

    optimizer = TorchOptimizer(
        full_problem,
        torch.optim.Adam,
        lr=0.035,
        max_iterations=ADAM_PHASE_STEPS,
        tolerance=0.0,
        scheduler=scheduler_factory,
    )
    first_info = optimizer.optimize()
    assert int(first_info.status) == int(OptimizerStatus.MAXITER)
    assert int(first_info.iterations) == ADAM_PHASE_STEPS
    assert len(schedulers) == 1
    assert schedulers[0].last_epoch == ADAM_PHASE_STEPS
    first_phase_lr = schedulers[0].get_last_lr()[0]

    configure_phase(full_terms, FINAL_WEIGHTS, projection_enabled=True)
    full_problem.update({"scene_points": data.target_scene_points.clone()})
    optimizer.resume()
    second_info = optimizer.optimize()
    final_terms = full_problem.term_costs()

    assert int(second_info.status) == int(OptimizerStatus.MAXITER)
    assert int(second_info.iterations) == 2 * ADAM_PHASE_STEPS
    assert len(schedulers) == 1
    assert schedulers[0].last_epoch == 2 * ADAM_PHASE_STEPS
    assert schedulers[0].get_last_lr()[0] < first_phase_lr
    assert all(torch.isfinite(cost) for cost in final_terms.values())

    initial_cost = torch.stack(tuple(initial_terms.values())).sum()
    final_cost = torch.stack(tuple(final_terms.values())).sum()
    torch.testing.assert_close(final_cost, second_info.cost)
    assert final_cost.item() < 0.02 * initial_cost.item()

    neutral = data.model.q_neutral.expand(TIME, -1)
    final_tangent = data.model.difference(neutral, full_q.tensor)
    torch.testing.assert_close(final_tangent, data.target_tangent, atol=1.5e-2, rtol=0.0)
    torch.testing.assert_close(full_scale.tensor, data.target_log_scale, atol=6.0e-3, rtol=0.0)

    expected_evaluations = WARMUP_STEPS + 2 * ADAM_PHASE_STEPS + 7
    assert counters.articulation == expected_evaluations
    assert counters.posed_body == expected_evaluations
