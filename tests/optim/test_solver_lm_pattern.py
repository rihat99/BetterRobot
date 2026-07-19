"""Private iteration-program and batched-semantics contracts for LM v2."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from better_robot.io import ModelBuilder, build_model
from better_robot.optim import (
    GaussNewton,
    LevenbergMarquardt,
    OptimizerStatus,
    Problem,
    Residual,
    RobotVariable,
    Variable,
)


class _LinearResidual(Residual):
    def __init__(self, x: Variable, target: Variable) -> None:
        self.x = x
        self.target = target
        self.width = target.shape[-1]
        super().__init__(x, target, dim=self.width, name="linear")

    def error(self) -> torch.Tensor:
        return self.x.tensor - self.target.tensor

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        x = self.x.tensor
        identity = torch.eye(self.width, dtype=x.dtype, device=x.device)
        identity = identity.index_select(-1, self.x.free_indices.to(x.device))
        return (identity.expand(*x.shape[:-1], self.width, self.x.free_dim),)


class _ExponentialResidual(Residual):
    def __init__(self, x: Variable, target: Variable) -> None:
        self.x = x
        self.target = target
        super().__init__(x, target, dim=1, name="exponential")

    def error(self) -> torch.Tensor:
        return self.x.tensor.exp() - self.target.tensor

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        return (self.x.tensor.exp().unsqueeze(-1),)


class _MatrixResidual(Residual):
    def __init__(self, x: Variable, matrix: Variable, target: Variable) -> None:
        self.x = x
        self.matrix = matrix
        self.target = target
        super().__init__(x, matrix, target, dim=2, name="matrix")

    def error(self) -> torch.Tensor:
        return torch.einsum("...ij,...j->...i", self.matrix.tensor, self.x.tensor) - self.target.tensor

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        return (self.matrix.tensor,)


class _FixedBaseConfigResidual(Residual):
    def __init__(self, q: RobotVariable, target: Variable) -> None:
        self.q = q
        self.target = target
        super().__init__(q, target, dim=1, name="fixed_base_config")

    def error(self) -> torch.Tensor:
        return self.q.tensor - self.target.tensor

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        q = self.q.tensor
        return (torch.ones((*q.shape[:-1], 1, 1), dtype=q.dtype, device=q.device),)


def _linear_problem(target: torch.Tensor) -> tuple[Variable, Problem]:
    batch_ndim = max(target.ndim - 1, 0)
    x = Variable(torch.zeros_like(target), name="x", batch_ndim=batch_ndim)
    target_variable = Variable(target, name="target", trainable=False, batch_ndim=batch_ndim)
    return x, Problem([_LinearResidual(x, target_variable)])


def _state_fields(state: Any) -> dict[str, torch.Tensor]:
    fields = {name: getattr(state, name) for name in state._fields}
    assert fields
    assert all(isinstance(value, torch.Tensor) for value in fields.values())
    return fields


def _structure(state: Any) -> tuple[tuple[str, torch.Size, torch.dtype, torch.device], ...]:
    return tuple((name, value.shape, value.dtype, value.device) for name, value in _state_fields(state).items())


def test_private_update_is_pure_and_preserves_state_structure() -> None:
    target = torch.tensor([[0.5, -0.25], [-0.2, 0.7]])
    x, problem = _linear_problem(target)
    values = {"x": x.tensor}
    optimizer = LevenbergMarquardt(problem, max_iterations=4)
    state = optimizer._init_state(values, problem)
    values_before = {name: value.clone() for name, value in values.items()}
    state_before = {name: value.clone() for name, value in _state_fields(state).items()}

    values_next, state_next = optimizer._update(values, state, problem)
    values_repeat, state_repeat = optimizer._update(values, state, problem)

    assert _structure(state) == _structure(state_next) == _structure(state_repeat)
    for name, before in values_before.items():
        torch.testing.assert_close(values[name], before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(values_next[name], values_repeat[name], rtol=0.0, atol=0.0)
    for name, before in state_before.items():
        torch.testing.assert_close(_state_fields(state)[name], before, rtol=0.0, atol=0.0, equal_nan=True)
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
    builder.add_revolute_z("joint", parent=base, child=tip, lower=-math.pi, upper=math.pi)
    model = build_model(builder.finalize())
    target_tensor = torch.tensor([[0.4], [-0.2]])
    q = RobotVariable(model, torch.zeros_like(target_tensor), name="q", batch_ndim=1)
    target = Variable(target_tensor, name="target", trainable=False, batch_ndim=1)
    problem = Problem([_FixedBaseConfigResidual(q, target)])
    values = {"q": q.tensor}
    optimizer = LevenbergMarquardt(problem, max_iterations=2)
    state = optimizer._init_state(values, problem)

    eager_values, eager_state = optimizer._update(values, state, problem)
    compiled_update = torch.compile(optimizer._update, fullgraph=True, backend="eager")
    compiled_values, compiled_state = compiled_update(values, state, problem)

    torch.testing.assert_close(compiled_values["q"], eager_values["q"])
    for name, eager_leaf in _state_fields(eager_state).items():
        torch.testing.assert_close(_state_fields(compiled_state)[name], eager_leaf, equal_nan=True)


@pytest.mark.parametrize("batch_shape", [(1,), (128,), (2, 3)])
def test_solver_supports_arbitrary_leading_batch_axes(batch_shape: tuple[int, ...]) -> None:
    dim = 2
    target = torch.linspace(
        -0.75,
        0.75,
        steps=math.prod(batch_shape) * dim,
        dtype=torch.float32,
    ).reshape(*batch_shape, dim)
    x, problem = _linear_problem(target)
    optimizer = LevenbergMarquardt(
        problem,
        max_iterations=8,
        tolerance=1e-6,
        step_tolerance=1e-8,
        relative_tolerance=1e-8,
    )

    info = optimizer.optimize()

    assert x.tensor.shape == (*batch_shape, dim)
    assert info.cost.shape == batch_shape
    assert info.status.shape == batch_shape
    assert info.converged.shape == batch_shape
    assert info.status.dtype == torch.int8
    assert info.converged.dtype == torch.bool
    assert bool(info.converged.all())
    torch.testing.assert_close(x.tensor, target, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("optimizer_type", [LevenbergMarquardt, GaussNewton])
def test_lm_and_gn_converge_on_an_unbatched_linear_problem(optimizer_type: type) -> None:
    target = torch.tensor([0.6, -0.3])
    x, problem = _linear_problem(target)
    optimizer = optimizer_type(problem, max_iterations=8, tolerance=1e-6)

    info = optimizer.optimize()

    torch.testing.assert_close(x.tensor, target, rtol=1e-5, atol=2e-6)
    assert info.status.shape == ()
    assert info.converged.shape == ()
    assert bool(info.converged)
    assert OptimizerStatus(int(info.status)) is OptimizerStatus.CONVERGED


def test_accept_reject_and_damping_are_independent_per_element() -> None:
    target_tensor = torch.tensor([[1.1], [10.0]])
    x = Variable(torch.zeros_like(target_tensor), name="x", batch_ndim=1)
    target = Variable(target_tensor, name="target", trainable=False, batch_ndim=1)
    problem = Problem([_ExponentialResidual(x, target)])
    values = {"x": x.tensor}
    optimizer = LevenbergMarquardt(problem, max_iterations=4)
    state = optimizer._init_state(values, problem)

    values_next, state_next = optimizer._update(values, state, problem)

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
    x, problem = _linear_problem(target)
    optimizer = LevenbergMarquardt(problem, max_iterations=4, tolerance=1e-6)
    values = {"x": x.tensor}
    state = optimizer._init_state(values, problem)

    values, state = optimizer._update(values, state, problem)
    first_terminal_value = values["x"][0].clone()
    values, state = optimizer._update(values, state, problem)

    assert bool(state.converged[0])
    assert OptimizerStatus(int(state.status[0])) is OptimizerStatus.CONVERGED
    torch.testing.assert_close(values["x"][0], first_terminal_value, rtol=0.0, atol=0.0)


def test_cholesky_failure_is_isolated_within_a_batch() -> None:
    matrix_tensor = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ]
    )
    target_tensor = torch.tensor([[1.0, -1.0], [1.0, 1.0]])
    x = Variable(torch.zeros_like(target_tensor), name="x", batch_ndim=1)
    matrix = Variable(matrix_tensor, name="matrix", trainable=False, batch_ndim=1)
    target = Variable(target_tensor, name="target", trainable=False, batch_ndim=1)
    problem = Problem([_MatrixResidual(x, matrix, target)])
    values = {"x": x.tensor}
    optimizer = LevenbergMarquardt(problem, max_iterations=2)
    state = optimizer._init_state(values, problem)._replace(mu=torch.zeros(2))

    values_next, state_next = optimizer._update(values, state, problem)

    torch.testing.assert_close(state_next.factorization_ok, torch.tensor([True, False]))
    assert not torch.equal(values_next["x"][0], values["x"][0])
    assert torch.equal(values_next["x"][1], values["x"][1])
    assert torch.isfinite(values_next["x"]).all()
    assert torch.isfinite(state_next.residual).all()
    assert torch.isfinite(state_next.cost).all()
