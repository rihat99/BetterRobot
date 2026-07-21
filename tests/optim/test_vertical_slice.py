"""M2a T2a.8 vertical slice — all seven gate ingredients in one consumer.

1. **Two blocks:** ``q`` has event shape ``(time, 2)`` and ``log_s`` is a
   scalar log-scale block in :func:`slice_support.make_problem`.
2. **Shared node:** counted synthetic kinematics feeds one detached-index
   nearest-neighbor node shared by penetration, attraction, and clearance.
3. **Custom residual:** ``PenetrationResidual`` is executed from the exact
   marked testcode fence in ``docs/guides/custom_residual.md``; it is not copied.
4. **Scale prior:** ``ScalePriorResidual`` is an ordinary one-row residual
   and optimized by the first-order loop.
5. **Batching:** B=3 residuals and gradients are compared with three sequential
   evaluations using fixed fp32 tolerances.
6. **Phase transition:** two manual Adam segments use different weight columns,
   with no library solver or Phase abstraction.

Friction log (resolved in test-local integration): the manual transition rebuilds
the Problem while copying owned tensors; TorchOptimizer drives persistent Adam
state, and shared Node objects own evaluation-scoped memos.
"""

from __future__ import annotations

import torch

from better_robot.optim import Problem, TorchOptimizer

from .slice_support import (
    COORDS,
    FRICTION_LOG,
    FULL_WEIGHTS,
    GUIDE_CUSTOM_RESIDUAL_SOURCE,
    POINTS,
    PROVIDER_INACTIVE_WEIGHTS,
    ROOT_WEIGHTS,
    TIME,
    PenetrationResidual,
    SliceCounters,
    make_batched_values,
    make_problem,
    make_slice_data,
)


def _run_adam_segment(
    problem: Problem,
    values: dict[str, torch.Tensor],
    *,
    iterations: int,
    learning_rate: float,
) -> dict[str, torch.Tensor]:
    """Run the public torch.optim adapter for one staged segment."""
    problem.update(values)
    TorchOptimizer(
        problem,
        torch.optim.Adam,
        lr=learning_rate,
        max_iterations=iterations,
        tolerance=0.0,
    ).optimize()
    return {variable.name: variable.tensor.clone() for variable in problem.vars}


def test_marked_guide_residual_and_provider_evaluation_counts() -> None:
    assert GUIDE_CUSTOM_RESIDUAL_SOURCE.startswith("import torch")
    assert "class PenetrationResidual(Residual):" in GUIDE_CUSTOM_RESIDUAL_SOURCE
    assert PenetrationResidual.__module__.endswith("slice_support")
    assert len(FRICTION_LOG) == 3

    data = make_slice_data()
    problem, counters = make_problem(data, weights=FULL_WEIGHTS)

    residual = problem.error()
    assert residual.shape == (TIME * POINTS * (COORDS + 2) + 1,)
    assert (counters.kinematics, counters.nearest_neighbor) == (1, 1)

    gradient = problem.gradient()
    assert set(gradient) == {"q", "log_s"}
    assert (counters.kinematics, counters.nearest_neighbor) == (2, 2)

    objective = problem.objective()
    assert torch.isfinite(objective)
    assert (counters.kinematics, counters.nearest_neighbor) == (3, 3)

    inactive_problem, inactive = make_problem(
        data,
        weights=PROVIDER_INACTIVE_WEIGHTS,
    )
    inactive_problem.error()
    inactive_problem.gradient()
    inactive_problem.objective()
    assert (inactive.kinematics, inactive.nearest_neighbor) == (0, 0)


def test_batched_residual_and_gradient_match_three_sequential_evaluations() -> None:
    data = make_slice_data()
    batched_values = make_batched_values(data)
    problem, counters = make_problem(
        data,
        values=batched_values,
        weights=FULL_WEIGHTS,
    )

    batched_residual = problem.error()
    assert counters.kinematics == counters.nearest_neighbor == 1
    batched_gradient = problem.gradient()
    assert counters.kinematics == counters.nearest_neighbor == 2

    assert batched_residual.shape == (3, TIME * POINTS * (COORDS + 2) + 1)
    assert batched_gradient["q"].shape == (3, TIME * COORDS)
    assert batched_gradient["log_s"].shape == (3, 1)
    for index in range(3):
        sequential = {name: value[index] for name, value in batched_values.items()}
        sequential_problem, _ = make_problem(
            data,
            values=sequential,
            weights=FULL_WEIGHTS,
        )
        expected_residual = sequential_problem.error()
        expected_gradient = sequential_problem.gradient()
        torch.testing.assert_close(
            batched_residual[index],
            expected_residual,
            rtol=2e-5,
            atol=2e-6,
        )
        for name in batched_gradient:
            torch.testing.assert_close(
                batched_gradient[name][index],
                expected_gradient[name],
                rtol=3e-5,
                atol=3e-6,
            )


def test_batched_jacobians_keep_shared_slice_parameters_intact() -> None:
    data = make_slice_data()
    batched_values = make_batched_values(data)
    problem, _ = make_problem(
        data,
        values=batched_values,
        weights=FULL_WEIGHTS,
    )

    for strategy in ("jacrev", "jacfwd"):
        batched = problem.jacobian_blocks(strategy=strategy)
        for index in range(3):
            sequential_values = {name: value[index] for name, value in batched_values.items()}
            sequential_problem, _ = make_problem(
                data,
                values=sequential_values,
                weights=FULL_WEIGHTS,
            )
            sequential = sequential_problem.jacobian_blocks(strategy=strategy)
            assert tuple(batched) == tuple(sequential)
            for key in batched:
                torch.testing.assert_close(
                    batched[key][index],
                    sequential[key],
                    rtol=4e-5,
                    atol=4e-6,
                )


def test_manual_weight_phase_transition_converges() -> None:
    torch.manual_seed(20260717)
    data = make_slice_data()
    counters = SliceCounters()
    first_problem, _ = make_problem(
        data,
        counters=counters,
        weights=ROOT_WEIGHTS,
    )
    full_problem, _ = make_problem(
        data,
        counters=counters,
        weights=FULL_WEIGHTS,
    )
    values = data.initial_values()
    first_problem._freeze()
    full_problem._freeze()

    assert first_problem.vars[0].free_dim == TIME * COORDS
    assert full_problem.vars[0].free_dim == TIME * COORDS
    assert ROOT_WEIGHTS != FULL_WEIGHTS

    initial_loss = full_problem.objective()
    root_iterations = 45
    full_iterations = 90
    values = _run_adam_segment(
        first_problem,
        values,
        iterations=root_iterations,
        learning_rate=0.04,
    )
    values = _run_adam_segment(
        full_problem,
        values,
        iterations=full_iterations,
        learning_rate=0.035,
    )
    final_loss = full_problem.objective()

    assert final_loss.item() < 2e-4
    assert final_loss.item() < 0.01 * initial_loss.item()
    torch.testing.assert_close(values["q"], data.target_q, rtol=0.0, atol=1.5e-2)
    torch.testing.assert_close(values["log_s"], data.target_log_s, rtol=0.0, atol=1.5e-2)
    # Per segment: one reset forward, one gradient forward per step, one final
    # cost refresh in optimize(); plus this test's two explicit objective calls.
    expected_active_evaluations = 2 + (root_iterations + 2) + (full_iterations + 2)
    assert counters.kinematics == expected_active_evaluations
    assert counters.nearest_neighbor == expected_active_evaluations
