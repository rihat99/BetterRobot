"""Matrix-free, batched, manifold-aware named-block Adam tests."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import math
from typing import Any

import pytest
import torch
from torch.utils import _pytree

from better_robot.lie import so3
from better_robot.optim import (
    Adam,
    AdamState,
    AdamStatus,
    ObjectiveItem,
    Problem,
    ResidualItem,
    SO3Manifold,
    VarSpec,
)


class _TargetResidual:
    name = "target"
    reads = ("x", "target")

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"] - ctx["target"]


class _RotationResidual:
    name = "rotation"
    reads = ("rotation", "target")
    dim = 3

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return so3.log(ctx["rotation"]) - ctx["target"]


class _QuadraticObjective:
    name = "quadratic"
    reads = ("x",)

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return 0.5 * ctx["x"].square().sum(dim=-1)


class _PerElementInvalidResidual:
    name = "invalid"
    reads = ("x",)
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        x = ctx["x"]
        return torch.where(x > 0.5, torch.full_like(x, torch.nan), x + 1.0)


def _target_problem(target: torch.Tensor, *, mask: torch.Tensor | None = None) -> Problem:
    return Problem(
        vars=(VarSpec("x", (target.shape[-1],), mask=mask),),
        residuals=(ResidualItem("target", _TargetResidual(target.shape[-1])),),
        parameters={"target": target},
    )


def _assert_state_close(actual: AdamState, expected: AdamState) -> None:
    actual_leaves = _pytree.tree_leaves(actual)
    expected_leaves = _pytree.tree_leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        torch.testing.assert_close(actual_leaf, expected_leaf, equal_nan=True)


def test_solver_and_state_are_frozen_tensor_pytrees_with_reduced_moments() -> None:
    target = torch.tensor([[0.4, -0.2, 0.8], [-0.3, 0.6, 0.1]])
    mask = torch.tensor([True, False, True])
    problem = _target_problem(target, mask=mask)
    values = {"x": torch.zeros_like(target)}
    solver = Adam(lr=0.05, max_iter=4)

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(solver, "lr", 0.2)

    state = solver.init_state(values, problem)
    leaves = _pytree.tree_leaves(state)
    assert leaves and all(isinstance(leaf, torch.Tensor) for leaf in leaves)
    assert state.m["x"].shape == (2, 2)
    assert state.v["x"].shape == (2, 2)
    assert state.step.shape == (2,)
    assert state.step.dtype == torch.int64
    assert state.status.dtype == torch.int8

    values_before = {name: value.clone() for name, value in values.items()}
    state_before = _pytree.tree_map(torch.clone, state)
    values_next, state_next = solver.update(values, state, problem)
    values_repeat, state_repeat = solver.update(values, state, problem)

    torch.testing.assert_close(values, values_before)
    _assert_state_close(state, state_before)
    torch.testing.assert_close(values_next, values_repeat)
    _assert_state_close(state_next, state_repeat)
    torch.testing.assert_close(values_next["x"][..., 1], values["x"][..., 1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(state_next.cost, problem.objective(values_next))
    next_gradient = problem.gradient(values_next)["x"]
    torch.testing.assert_close(state_next.grad_norm, next_gradient.abs().amax(dim=-1))


def test_update_uses_only_prevalidated_matrix_free_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = torch.tensor([[0.7, -0.4], [-0.2, 0.5]])
    problem = _target_problem(target)
    values = {"x": torch.zeros_like(target)}
    solver = Adam(lr=0.1, max_iter=10)
    state = solver.init_state(values, problem)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Adam update reached validation or Jacobian assembly")

    for name in (
        "gradient",
        "jacobian_blocks",
        "dense_jacobian",
        "_dense_jacobian_prevalidated",
        "_validate_values",
        "_validate_weights",
        "_validate_runtime_weights",
    ):
        monkeypatch.setattr(Problem, name, forbidden)
    monkeypatch.setattr(VarSpec, "validate_value", forbidden)
    monkeypatch.setattr(VarSpec, "batch_shape", forbidden)
    monkeypatch.setattr(VarSpec, "retract", forbidden)

    values_next, state_next = solver.update(values, state, problem)

    assert state_next.cost.shape == (2,)
    assert torch.all(state_next.cost < state.cost)
    assert torch.all(values_next["x"] != values["x"])


def test_run_never_materializes_a_jacobian(monkeypatch: pytest.MonkeyPatch) -> None:
    target = torch.tensor([0.5, -0.25])
    problem = _target_problem(target)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("matrix-free Adam materialized a Jacobian")

    for name in (
        "jacobian_blocks",
        "dense_jacobian",
        "_dense_jacobian_prevalidated",
    ):
        monkeypatch.setattr(Problem, name, forbidden)

    values, state = Adam(lr=0.08, max_iter=250, tol=1e-5).run(
        {"x": torch.zeros_like(target)},
        problem,
    )

    assert bool(state.converged)
    assert AdamStatus(int(state.status)) is AdamStatus.CONVERGED
    torch.testing.assert_close(values["x"], target, atol=2e-5, rtol=2e-5)


def test_so3_identity_step_is_finite_and_stays_on_manifold() -> None:
    target = torch.tensor([0.2, -0.1, 0.15])
    problem = Problem(
        vars=(VarSpec("rotation", (4,), manifold=SO3Manifold()),),
        residuals=(ResidualItem("rotation", _RotationResidual()),),
        parameters={"target": target},
    )
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    solver = Adam(lr=0.05, max_iter=1)
    state = solver.init_state({"rotation": identity}, problem)

    values_next, state_next = solver.update({"rotation": identity}, state, problem)

    assert torch.isfinite(values_next["rotation"]).all()
    assert torch.isfinite(state_next.m["rotation"]).all()
    torch.testing.assert_close(values_next["rotation"].norm(), torch.tensor(1.0), atol=2e-6, rtol=2e-6)
    assert state_next.cost < state.cost


@pytest.mark.parametrize("batch_shape", [(1,), (8,), (2, 3)])
def test_arbitrary_batch_axes_match_sequential_runs(batch_shape: tuple[int, ...]) -> None:
    dim = 2
    count = math.prod(batch_shape)
    target = torch.linspace(-0.7, 0.8, count * dim).reshape(*batch_shape, dim)
    problem = _target_problem(target)
    solver = Adam(lr=0.06, max_iter=45, tol=0.0)

    batched_values, batched_state = solver.run({"x": torch.zeros_like(target)}, problem)
    flat_target = target.reshape(count, dim)
    sequential = []
    sequential_steps = []
    for index in range(count):
        single_problem = _target_problem(flat_target[index])
        single_values, single_state = solver.run(
            {"x": torch.zeros_like(flat_target[index])},
            single_problem,
        )
        sequential.append(single_values["x"])
        sequential_steps.append(single_state.step)

    expected = torch.stack(sequential).reshape(*batch_shape, dim)
    expected_steps = torch.stack(sequential_steps).reshape(*batch_shape)
    torch.testing.assert_close(batched_values["x"], expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(batched_state.step, expected_steps, rtol=0.0, atol=0.0)


def test_external_update_loop_matches_run_values_and_moments() -> None:
    target = torch.tensor([[0.4, -0.7], [-0.3, 0.2]])
    problem = _target_problem(target)
    solver = Adam(lr=0.04, max_iter=7, tol=0.0)
    initial = {"x": torch.zeros_like(target)}
    manual_values = {"x": initial["x"].clone()}
    manual_state = solver.init_state(manual_values, problem)
    for _ in range(solver.max_iter):
        manual_values, manual_state = solver.update(manual_values, manual_state, problem)

    run_values, run_state = solver.run(initial, problem)

    torch.testing.assert_close(run_values, manual_values)
    torch.testing.assert_close(run_state.m, manual_state.m)
    torch.testing.assert_close(run_state.v, manual_state.v)
    torch.testing.assert_close(run_state.step, manual_state.step, rtol=0.0, atol=0.0)
    assert torch.all(run_state.status == AdamStatus.MAXITER.value)


def test_warm_start_retains_moments_but_refreshes_target_artifacts() -> None:
    old_target = torch.tensor([0.8, -0.5])
    old_problem = _target_problem(old_target)
    solver = Adam(lr=0.05, max_iter=5, tol=0.0)
    old_values, old_state = solver.run({"x": torch.zeros_like(old_target)}, old_problem)
    new_target = torch.tensor([-0.2, 0.7])
    new_problem = _target_problem(new_target)
    new_values = {"x": old_values["x"] + torch.tensor([0.1, -0.1])}

    _unused_values, warm_state = dataclasses.replace(solver, max_iter=0).run(
        new_values,
        new_problem,
        state=old_state,
    )

    torch.testing.assert_close(warm_state.m, old_state.m)
    torch.testing.assert_close(warm_state.v, old_state.v)
    torch.testing.assert_close(warm_state.step, old_state.step, rtol=0.0, atol=0.0)
    torch.testing.assert_close(warm_state.cost, new_problem.objective(new_values))
    assert not torch.equal(warm_state.cost, old_state.cost)
    assert AdamStatus(int(warm_state.status)) is AdamStatus.MAXITER

    bad_state = old_state._replace(
        m={"x": old_state.m["x"].unsqueeze(0)},
        v={"x": old_state.v["x"].unsqueeze(0)},
    )
    with pytest.raises(ValueError, match="reduced shape"):
        solver.run(new_values, new_problem, state=bad_state)


def test_compatible_warm_start_needs_fewer_new_steps_than_cold() -> None:
    old_target = torch.tensor([0.8, -0.5])
    seed_solver = Adam(lr=0.02, max_iter=10, tol=0.0)
    seed_values, seed_state = seed_solver.run(
        {"x": torch.zeros_like(old_target)},
        _target_problem(old_target),
    )
    new_target = old_target + torch.tensor([0.03, -0.03])
    new_problem = _target_problem(new_target)
    solver = Adam(lr=0.02, max_iter=300, tol=1e-5)

    _warm_values, warm_state = solver.run(seed_values, new_problem, state=seed_state)
    _cold_values, cold_state = solver.run(seed_values, new_problem)

    assert bool(warm_state.converged) and bool(cold_state.converged)
    warm_new_steps = warm_state.step - seed_state.step
    assert warm_new_steps < cold_state.step


def test_scalar_objectives_are_supported_and_default_run_is_detached() -> None:
    problem = Problem(
        vars=(VarSpec("x", (2,)),),
        objectives=(ObjectiveItem("quadratic", _QuadraticObjective()),),
    )
    initial = torch.tensor([1.2, -0.8], requires_grad=True)

    values, state = Adam(lr=0.05, max_iter=80, tol=1e-4).run({"x": initial}, problem)

    assert values["x"].grad_fn is None
    assert all(leaf.grad_fn is None for leaf in _pytree.tree_leaves(state))
    assert state.cost < 0.01


def test_invalid_batch_element_fails_without_moving_valid_neighbor() -> None:
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("invalid", _PerElementInvalidResidual()),),
    )
    initial = {"x": torch.tensor([[-0.5], [0.75]])}

    values, state = Adam(lr=0.05, max_iter=2).run(initial, problem)

    assert state.status.tolist()[1] == AdamStatus.FAILED.value
    torch.testing.assert_close(values["x"][1], initial["x"][1], rtol=0.0, atol=0.0)
    assert values["x"][0] < initial["x"][0]
