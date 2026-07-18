"""Step-API and batched-semantics contracts for the named-block LM solver."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import pytest
import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.optim import (
    GaussNewton,
    LevenbergMarquardt,
    LMState,
    LMStatus,
    Problem,
    ResidualItem,
    RobotConfig,
    RobotStateProvider,
    VarSpec,
)


class _LinearResidual:
    name = "linear"
    reads = ("x", "target")

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"] - ctx["target"]

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        x = ctx["x"]
        identity = torch.eye(self.dim, dtype=x.dtype, device=x.device)
        return {"x": identity.expand(*x.shape[:-1], self.dim, self.dim)}


class _ExponentialResidual:
    name = "exponential"
    reads = ("x", "target")
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"].exp() - ctx["target"]

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        return {"x": ctx["x"].exp().unsqueeze(-1)}


class _MatrixResidual:
    name = "matrix"
    reads = ("x", "matrix", "target")
    dim = 2

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return torch.einsum("...ij,...j->...i", ctx["matrix"], ctx["x"]) - ctx["target"]

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        return {"x": ctx["matrix"]}


class _FixedBaseConfigResidual:
    name = "fixed_base_config"
    reads = ("data", "target")
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["data"].q - ctx["target"]

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        q = ctx["data"].q
        return {
            "q": torch.ones(
                (*q.shape[:-1], 1, 1),
                dtype=q.dtype,
                device=q.device,
            )
        }


def _linear_problem(target: torch.Tensor) -> Problem:
    dim = target.shape[-1]
    return Problem(
        vars=(VarSpec("x", (dim,)),),
        residuals=(ResidualItem("linear", _LinearResidual(dim)),),
        parameters={"target": target},
    )


def _state_fields(state: LMState) -> dict[str, torch.Tensor]:
    if dataclasses.is_dataclass(state):
        fields = {field.name: getattr(state, field.name) for field in dataclasses.fields(state)}
    elif hasattr(state, "_fields"):
        fields = {name: getattr(state, name) for name in state._fields}
    else:  # pragma: no cover - assertion has a more useful message than an AttributeError
        pytest.fail("LMState must be a frozen dataclass or NamedTuple")
    assert fields
    assert all(isinstance(value, torch.Tensor) for value in fields.values())
    return fields


def _replace_state(state: LMState, **changes: torch.Tensor) -> LMState:
    if dataclasses.is_dataclass(state):
        return dataclasses.replace(state, **changes)
    return state._replace(**changes)


def _structure(state: LMState) -> tuple[tuple[str, torch.Size, torch.dtype, torch.device], ...]:
    return tuple((name, value.shape, value.dtype, value.device) for name, value in _state_fields(state).items())


def test_solver_hyperparameters_and_state_are_frozen_tensor_values() -> None:
    target = torch.tensor([0.5, -0.25])
    problem = _linear_problem(target)
    values = {"x": torch.zeros_like(target)}
    solver = LevenbergMarquardt(max_iter=4)

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(solver, "max_iter", 10)

    state = solver.init_state(values, problem)

    assert isinstance(state, LMState)
    assert all(isinstance(value, torch.Tensor) for value in _state_fields(state).values())
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        setattr(state, "cost", torch.zeros_like(state.cost))


def test_update_is_pure_and_preserves_state_structure() -> None:
    target = torch.tensor([[0.5, -0.25], [-0.2, 0.7]])
    problem = _linear_problem(target)
    values = {"x": torch.zeros_like(target)}
    solver = LevenbergMarquardt(max_iter=4)
    state = solver.init_state(values, problem)
    values_before = {name: value.clone() for name, value in values.items()}
    state_before = {name: value.clone() for name, value in _state_fields(state).items()}

    values_next, state_next = solver.update(values, state, problem)
    values_repeat, state_repeat = solver.update(values, state, problem)

    assert _structure(state) == _structure(state_next) == _structure(state_repeat)
    for name, before in values_before.items():
        torch.testing.assert_close(values[name], before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(values_next[name], values_repeat[name], rtol=0.0, atol=0.0)
    for name, before in state_before.items():
        torch.testing.assert_close(
            _state_fields(state)[name],
            before,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
        torch.testing.assert_close(
            _state_fields(state_next)[name],
            _state_fields(state_repeat)[name],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )


@pytest.mark.slow
def test_update_fullgraph_compile_smoke_on_fixed_base_model() -> None:
    """CPU fullgraph is a graph-break proxy, not CUDA capture certification."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable in this Torch build")
    builder = ModelBuilder("lm_compile_fixed_base")
    base = builder.add_body("base")
    tip = builder.add_body("tip")
    builder.add_revolute_z(
        "joint",
        parent=base,
        child=tip,
        lower=-math.pi,
        upper=math.pi,
    )
    model = build_model(builder.finalize())
    target = torch.tensor([[0.4], [-0.2]])
    problem = Problem(
        vars=(VarSpec("q", (model.nq,), manifold=RobotConfig(model)),),
        residuals=(
            ResidualItem(
                "fixed_base_config",
                _FixedBaseConfigResidual(),
            ),
        ),
        providers=(RobotStateProvider(model),),
        parameters={"target": target},
    )
    values = {"q": torch.zeros_like(target)}
    solver = LevenbergMarquardt(max_iter=2)
    state = solver.init_state(values, problem)

    eager_values, eager_state = solver.update(values, state, problem)
    compiled_update = torch.compile(solver.update, fullgraph=True, backend="eager")
    compiled_values, compiled_state = compiled_update(values, state, problem)

    torch.testing.assert_close(compiled_values["q"], eager_values["q"])
    for name, eager_leaf in _state_fields(eager_state).items():
        torch.testing.assert_close(
            _state_fields(compiled_state)[name],
            eager_leaf,
            equal_nan=True,
        )


def test_external_update_loop_is_identical_to_run() -> None:
    target = torch.tensor([[0.3, -0.8], [-0.4, 0.2], [0.7, 0.1]])
    problem = _linear_problem(target)
    initial = {"x": torch.zeros_like(target)}
    solver = LevenbergMarquardt(
        max_iter=8,
        gtol=1e-6,
        xtol=1e-8,
        ftol=1e-8,
    )

    manual_values = {name: value.clone() for name, value in initial.items()}
    manual_state = solver.init_state(manual_values, problem)
    for _ in range(solver.max_iter):
        manual_values, manual_state = solver.update(manual_values, manual_state, problem)
        if bool(manual_state.converged.all()):
            break
    manual_values, manual_state = solver.finalize(manual_values, manual_state, problem)

    run_values, run_state = solver.run(
        {name: value.clone() for name, value in initial.items()},
        problem,
    )

    for name in manual_values:
        torch.testing.assert_close(manual_values[name], run_values[name], rtol=0.0, atol=0.0)
    torch.testing.assert_close(manual_state.cost, run_state.cost, rtol=0.0, atol=0.0)
    torch.testing.assert_close(manual_state.status, run_state.status, rtol=0.0, atol=0.0)
    torch.testing.assert_close(manual_state.iterations, run_state.iterations, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("batch_shape", [(1,), (128,), (2, 3)])
def test_solver_supports_arbitrary_leading_batch_axes(batch_shape: tuple[int, ...]) -> None:
    dim = 2
    target = torch.linspace(
        -0.75,
        0.75,
        steps=math.prod(batch_shape) * dim,
        dtype=torch.float32,
    ).reshape(*batch_shape, dim)
    problem = _linear_problem(target)
    solver = LevenbergMarquardt(
        max_iter=8,
        gtol=1e-6,
        xtol=1e-8,
        ftol=1e-8,
    )

    values, state = solver.run({"x": torch.zeros_like(target)}, problem)

    assert values["x"].shape == (*batch_shape, dim)
    assert state.cost.shape == batch_shape
    assert state.mu.shape == batch_shape
    assert state.status.shape == batch_shape
    assert state.converged.shape == batch_shape
    assert state.factorization_ok.shape == batch_shape
    assert state.residual.shape == (*batch_shape, dim)
    assert state.active_mask.shape == (*batch_shape, dim)
    assert state.status.dtype == torch.int8
    assert state.converged.dtype == torch.bool
    assert bool(state.converged.all())
    torch.testing.assert_close(values["x"], target, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize(
    "solver",
    [
        LevenbergMarquardt(max_iter=8, gtol=1e-6),
        GaussNewton(max_iter=8, gtol=1e-6),
    ],
)
def test_lm_and_gn_converge_on_an_unbatched_linear_problem(solver: object) -> None:
    target = torch.tensor([0.6, -0.3])
    problem = _linear_problem(target)

    values, state = solver.run({"x": torch.zeros_like(target)}, problem)

    torch.testing.assert_close(values["x"], target, rtol=1e-5, atol=2e-6)
    assert state.status.shape == ()
    assert state.converged.shape == ()
    assert bool(state.converged)
    assert LMStatus(int(state.status)) is LMStatus.CONVERGED


def test_accept_reject_and_damping_are_independent_per_element() -> None:
    target = torch.tensor([[1.1], [10.0]])
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("exponential", _ExponentialResidual()),),
        parameters={"target": target},
    )
    values = {"x": torch.zeros_like(target)}
    solver = LevenbergMarquardt(max_iter=4)
    state = solver.init_state(values, problem)

    values_next, state_next = solver.update(values, state, problem)

    assert not torch.equal(values_next["x"][0], values["x"][0])
    assert torch.equal(values_next["x"][1], values["x"][1])
    assert state_next.cost[0] < state.cost[0]
    torch.testing.assert_close(state_next.cost[1], state.cost[1], rtol=0.0, atol=0.0)
    assert state_next.mu[0] < state.mu[0]
    assert state_next.mu[1] > state.mu[1]
    assert state_next.gain_ratio[0] > 0
    assert state_next.gain_ratio[1] <= 0


def test_converged_element_rides_along_without_moving() -> None:
    target = torch.tensor([[0.0], [2.0]])
    problem = _linear_problem(target)
    solver = LevenbergMarquardt(max_iter=4, gtol=1e-6)
    values = {"x": torch.zeros_like(target)}
    state = solver.init_state(values, problem)

    values, state = solver.update(values, state, problem)
    first_terminal_value = values["x"][0].clone()
    values, state = solver.update(values, state, problem)

    assert bool(state.converged[0])
    assert LMStatus(int(state.status[0])) is LMStatus.CONVERGED
    torch.testing.assert_close(values["x"][0], first_terminal_value, rtol=0.0, atol=0.0)


def test_cholesky_failure_is_isolated_within_a_batch() -> None:
    matrix = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ]
    )
    target = torch.tensor([[1.0, -1.0], [1.0, 1.0]])
    problem = Problem(
        vars=(VarSpec("x", (2,)),),
        residuals=(ResidualItem("matrix", _MatrixResidual()),),
        parameters={"matrix": matrix, "target": target},
    )
    values = {"x": torch.zeros_like(target)}
    solver = LevenbergMarquardt(max_iter=2)
    state = solver.init_state(values, problem)
    # Zero damping exposes the second element's rank deficiency while the
    # first element remains positive definite. This directly exercises the
    # fixed-work cholesky_ex info-mask path.
    state = _replace_state(state, mu=torch.zeros_like(state.mu))

    values_next, state_next = solver.update(values, state, problem)

    torch.testing.assert_close(
        state_next.factorization_ok,
        torch.tensor([True, False]),
    )
    assert not torch.equal(values_next["x"][0], values["x"][0])
    assert torch.equal(values_next["x"][1], values["x"][1])
    assert torch.isfinite(values_next["x"]).all()
    assert torch.isfinite(state_next.residual).all()
    assert torch.isfinite(state_next.cost).all()
