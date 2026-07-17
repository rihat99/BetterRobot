"""M2a T2a.8 vertical slice — all seven gate ingredients in one consumer.

1. **Two blocks:** ``q`` has event shape ``(time, 2)`` and ``log_s`` is a
   scalar log-scale block in :func:`slice_support.make_problem`.
2. **Shared provider:** counted synthetic kinematics feeds one detached-index
   nearest-neighbor provider shared by penetration, attraction, and clearance.
3. **Custom residual:** ``PenetrationResidual`` is executed from the exact
   marked Python fence in ``docs/guides/custom_residuals.md``; it is not copied.
4. **Scalar term:** ``ScalePriorTerm`` is registered through ``ObjectiveItem``
   and optimized by the first-order loop.
5. **Masks:** the root phase retains one q coordinate per frame; the full phase
   rebuild retains every q tangent coordinate, proving elimination semantics.
6. **Batching:** B=3 residuals and gradients are compared with three sequential
   evaluations using fixed fp32 tolerances.
7. **Phase transition:** two manual Adam segments use different weight columns
   and q masks, with no library solver or Phase abstraction.

Friction log (resolved in test-local integration): VarSpec masks are static, so
the manual transition cheaply rebuilds Problem while preserving Values;
torch.optim.Adam needs leaf parameters, so persistent Adam state drives zeroed
reduced-tangent buffers whose updates are applied by Problem.retract; providers
remain structural objects because the guide deliberately requires no base class.
"""

from __future__ import annotations

import torch

from better_robot.optim import Values, detach_values

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
    problem,
    values: Values,
    *,
    weights,
    iterations: int,
    learning_rate: float,
) -> Values:
    """Use Adam only as a tangent-step generator; BetterRobot owns retraction."""
    tangent_buffers = {
        spec.name: torch.nn.Parameter(values[spec.name].new_zeros(spec.free_dim)) for spec in problem.vars
    }
    optimizer = torch.optim.Adam(tuple(tangent_buffers.values()), lr=learning_rate)
    current = detach_values(values)
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        gradient = problem.gradient(current, weights=weights)
        for name, buffer in tangent_buffers.items():
            buffer.grad = gradient[name].detach().clone()
        optimizer.step()
        steps = {name: buffer.detach().clone() for name, buffer in tangent_buffers.items()}
        current = detach_values(problem.retract(current, steps))
        with torch.no_grad():
            for buffer in tangent_buffers.values():
                buffer.zero_()
    return current


def test_marked_guide_residual_and_provider_evaluation_counts() -> None:
    assert GUIDE_CUSTOM_RESIDUAL_SOURCE.startswith("class PenetrationResidual:")
    assert PenetrationResidual.__module__.endswith("slice_support")
    assert len(FRICTION_LOG) == 3

    data = make_slice_data()
    problem, counters = make_problem(data, root_only=False)
    values = data.initial_values()

    residual = problem.residual(values, weights=FULL_WEIGHTS)
    assert residual.shape == (TIME * POINTS * (COORDS + 2),)
    assert (counters.kinematics, counters.nearest_neighbor) == (1, 1)

    gradient = problem.gradient(values, weights=FULL_WEIGHTS)
    assert set(gradient) == {"q", "log_s"}
    assert (counters.kinematics, counters.nearest_neighbor) == (2, 2)

    objective = problem.objective(values, weights=FULL_WEIGHTS)
    assert torch.isfinite(objective)
    assert (counters.kinematics, counters.nearest_neighbor) == (3, 3)

    inactive_problem, inactive = make_problem(data, root_only=False)
    inactive_problem.residual(values, weights=PROVIDER_INACTIVE_WEIGHTS)
    inactive_problem.gradient(values, weights=PROVIDER_INACTIVE_WEIGHTS)
    inactive_problem.objective(values, weights=PROVIDER_INACTIVE_WEIGHTS)
    assert (inactive.kinematics, inactive.nearest_neighbor) == (0, 0)


def test_batched_residual_and_gradient_match_three_sequential_evaluations() -> None:
    data = make_slice_data()
    problem, counters = make_problem(data, root_only=False)
    batched_values = make_batched_values(data)

    batched_residual = problem.residual(batched_values, weights=FULL_WEIGHTS)
    assert counters.kinematics == counters.nearest_neighbor == 1
    batched_gradient = problem.gradient(batched_values, weights=FULL_WEIGHTS)
    assert counters.kinematics == counters.nearest_neighbor == 2

    assert batched_residual.shape == (3, TIME * POINTS * (COORDS + 2))
    assert batched_gradient["q"].shape == (3, TIME * COORDS)
    assert batched_gradient["log_s"].shape == (3, 1)
    for index in range(3):
        sequential = {name: value[index] for name, value in batched_values.items()}
        expected_residual = problem.residual(sequential, weights=FULL_WEIGHTS)
        expected_gradient = problem.gradient(sequential, weights=FULL_WEIGHTS)
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
    problem, _ = make_problem(data, root_only=False)
    batched_values = make_batched_values(data)

    for strategy in ("jacrev", "jacfwd"):
        batched = problem.jacobian_blocks(
            batched_values,
            weights=FULL_WEIGHTS,
            strategy=strategy,
        )
        for index in range(3):
            sequential_values = {name: value[index] for name, value in batched_values.items()}
            sequential = problem.jacobian_blocks(
                sequential_values,
                weights=FULL_WEIGHTS,
                strategy=strategy,
            )
            assert tuple(batched) == tuple(sequential)
            for key in batched:
                torch.testing.assert_close(
                    batched[key][index],
                    sequential[key],
                    rtol=4e-5,
                    atol=4e-6,
                )


def test_manual_root_to_full_phase_transition_converges() -> None:
    torch.manual_seed(20260717)
    data = make_slice_data()
    counters = SliceCounters()
    root_problem, _ = make_problem(data, root_only=True, counters=counters)
    full_problem, _ = make_problem(data, root_only=False, counters=counters)
    values = data.initial_values()

    assert root_problem.vars[0].free_dim == TIME
    assert full_problem.vars[0].free_dim == TIME * COORDS
    assert root_problem.vars[0].free_indices.tolist() != full_problem.vars[0].free_indices.tolist()
    assert ROOT_WEIGHTS != FULL_WEIGHTS

    initial_loss = full_problem.objective(values, weights=FULL_WEIGHTS)
    root_iterations = 45
    full_iterations = 90
    values = _run_adam_segment(
        root_problem,
        values,
        weights=ROOT_WEIGHTS,
        iterations=root_iterations,
        learning_rate=0.04,
    )
    values = _run_adam_segment(
        full_problem,
        values,
        weights=FULL_WEIGHTS,
        iterations=full_iterations,
        learning_rate=0.035,
    )
    final_loss = full_problem.objective(values, weights=FULL_WEIGHTS)

    assert final_loss.item() < 2e-4
    assert final_loss.item() < 0.01 * initial_loss.item()
    torch.testing.assert_close(values["q"], data.target_q, rtol=0.0, atol=1.5e-2)
    torch.testing.assert_close(values["log_s"], data.target_log_s, rtol=0.0, atol=1.5e-2)
    expected_active_evaluations = 2 + root_iterations + full_iterations
    assert counters.kinematics == expected_active_evaluations
    assert counters.nearest_neighbor == expected_active_evaluations
