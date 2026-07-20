"""Bound-aware object-owned LM tests with analytic constrained optima.

The coupled linear system is intentionally not diagonal. At the upper bound
on ``x[0]``, restricting the normal system to the inactive coordinate gives
the true constrained minimizer ``x = [1, 1.6]``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from better_robot.data_model.joint_models import (
    JointComposite,
    JointHelical,
    JointPlanar,
    JointPX,
    JointTranslation,
)
from better_robot.io import ModelBuilder, build_model
from better_robot.optim import (
    Bounds,
    LevenbergMarquardt,
    OptimizerStatus,
    Problem,
    Residual,
    RobotVariable,
    Variable,
)


_A = torch.tensor([[1.0, 1.0], [1.0, 2.0]])
_LOWER = torch.tensor([-3.0, -3.0])
_UPPER = torch.tensor([1.0, 3.0])


class _CoupledLinearResidual(Residual):
    def __init__(self, x: Variable, target: Variable) -> None:
        self.x = x
        self.target = target
        super().__init__(x, target, dim=2, name="linear")

    def error(self) -> torch.Tensor:
        x = self.x.tensor
        matrix = _A.to(dtype=x.dtype, device=x.device)
        return torch.matmul(x, matrix.mT) - self.target.tensor

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        x = self.x.tensor
        matrix = _A.to(dtype=x.dtype, device=x.device)
        return (matrix.expand(*x.shape[:-1], 2, self.x.free_dim),)


class _RobotTrajectoryResidual(Residual):
    def __init__(self, q: RobotVariable, target: Variable) -> None:
        self.q = q
        self.target = target
        super().__init__(q, target, dim=2, name="joint_trajectory")

    def error(self) -> torch.Tensor:
        return self.q.tensor[..., :, 0] - self.target.tensor

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        q = self.q.tensor
        identity = torch.eye(2, dtype=q.dtype, device=q.device)
        return (identity.expand(*q.shape[:-2], 2, 2),)


class _RobotTangentTargetResidual(Residual):
    def __init__(self, q: RobotVariable, target: Variable) -> None:
        self.q = q
        self.target = target
        self.model = q.model
        super().__init__(q, target, dim=q.model.nv, name="robot_tangent_target")

    def error(self) -> torch.Tensor:
        q = self.q.tensor
        neutral = self.model.q_neutral.to(dtype=q.dtype, device=q.device)
        return self.model.difference(neutral, q) - self.target.tensor

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        q = self.q.tensor
        identity = torch.eye(self.model.nv, dtype=q.dtype, device=q.device)
        return (identity.expand(*q.shape[:-1], self.model.nv, self.q.free_dim),)


def _target_for(solution: torch.Tensor) -> torch.Tensor:
    return torch.matmul(solution, _A.mT)


def _linear_problem(target: torch.Tensor, *, initial: torch.Tensor | None = None) -> tuple[Variable, Problem]:
    value = torch.zeros_like(target) if initial is None else initial.clone()
    x = Variable(
        value,
        name="x",
        bounds=Bounds(lower=_LOWER.to(value), upper=_UPPER.to(value)),
        batch_ndim=max(value.ndim - 1, 0),
    )
    target_variable = Variable(
        target,
        name="target",
        trainable=False,
        batch_ndim=max(target.ndim - 1, 0),
    )
    return x, Problem([_CoupledLinearResidual(x, target_variable)])


def _optimizer(problem: Problem, *, max_iterations: int = 40) -> LevenbergMarquardt:
    return LevenbergMarquardt(
        problem,
        max_iterations=max_iterations,
        tolerance=1e-6,
        step_tolerance=1e-8,
        relative_tolerance=1e-8,
    )


def _build_qv_mapping_model(case: str) -> Any:
    builder = ModelBuilder(f"bounded_{case}")
    base = builder.add_body("base")
    tip = builder.add_body("tip")
    if case == "prismatic":
        builder.add_prismatic_x("joint", parent=base, child=tip)
    elif case == "helical":
        builder.add_helical(
            "joint",
            parent=base,
            child=tip,
            axis=torch.tensor([0.0, 0.0, 1.0]),
            pitch=0.25,
        )
    elif case == "translation":
        builder.add_joint("joint", kind=JointTranslation(), parent=base, child=tip)
    elif case == "planar":
        builder.add_planar("joint", parent=base, child=tip)
    elif case == "nested_composite":
        nested = JointComposite(
            (
                JointTranslation(),
                JointHelical(torch.tensor([0.0, 0.0, 1.0]), pitch=0.25),
            )
        )
        joint = JointComposite((JointPlanar(), nested, JointPX()))
        builder.add_joint("joint", kind=joint, parent=base, child=tip)
    else:  # pragma: no cover
        raise AssertionError(f"unknown q-v mapping case {case!r}")
    return build_model(builder.finalize())


def test_active_set_restricts_the_coupled_normal_system() -> None:
    """The first boundary step follows the reduced system, not clamp-only LM."""
    target = _target_for(torch.tensor([2.0, 1.0]))
    x, problem = _linear_problem(target, initial=torch.tensor([1.0, 0.0]))
    values = {"x": x.tensor}
    optimizer = _optimizer(problem, max_iterations=1)

    state = optimizer._init_state(values, problem)
    next_values, next_state = optimizer._update(values, state, problem)

    assert torch.equal(next_values["x"][0], values["x"][0])
    assert next_values["x"][1] > 1.5
    assert next_values["x"][1] < 1.7
    torch.testing.assert_close(next_state.active_mask, torch.tensor([True, False]))


def test_known_constrained_optimum_has_active_mask_and_kkt_status() -> None:
    target = _target_for(torch.tensor([2.0, 1.0]))
    x, problem = _linear_problem(target)
    optimizer = _optimizer(problem)

    info = optimizer.optimize()
    assert optimizer._state is not None
    state = optimizer._state

    torch.testing.assert_close(x.tensor, torch.tensor([1.0, 1.6]), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(state.active_mask, torch.tensor([True, False]))
    assert int(info.status) == int(OptimizerStatus.STALLED_AT_BOUNDS)
    assert state.projected_grad_norm <= 1e-6
    assert state.grad_norm > 0.1
    torch.testing.assert_close(state.gradient, torch.tensor([-0.2, 0.0]), atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(info.cost, torch.tensor(0.1), atol=3e-5, rtol=3e-5)


def test_kkt_terminal_point_is_not_reported_as_unconstrained_convergence() -> None:
    target = _target_for(torch.tensor([2.0, 1.0]))
    initial = torch.tensor([1.0, 1.6])
    x, problem = _linear_problem(target, initial=initial)
    optimizer = _optimizer(problem, max_iterations=5)

    info = optimizer.optimize()
    assert optimizer._state is not None
    state = optimizer._state

    torch.testing.assert_close(x.tensor, initial, atol=2e-6, rtol=0.0)
    assert int(info.status) == int(OptimizerStatus.STALLED_AT_BOUNDS)
    assert state.projected_grad_norm <= 1e-6
    assert state.grad_norm > 0.1
    assert info.iterations <= 1


def test_mixed_batch_keeps_interior_and_bound_statuses_independent() -> None:
    interior = torch.tensor([0.5, 0.25])
    outside = torch.tensor([2.0, 1.0])
    target = _target_for(torch.stack((interior, outside)))
    x, problem = _linear_problem(target)
    optimizer = _optimizer(problem)

    info = optimizer.optimize()
    assert optimizer._state is not None
    state = optimizer._state

    expected = torch.tensor([[0.5, 0.25], [1.0, 1.6]])
    torch.testing.assert_close(x.tensor, expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
        info.status,
        torch.tensor(
            [OptimizerStatus.CONVERGED, OptimizerStatus.STALLED_AT_BOUNDS],
            dtype=info.status.dtype,
        ),
    )
    torch.testing.assert_close(state.active_mask, torch.tensor([[False, False], [True, False]]))
    assert torch.all(state.projected_grad_norm <= 1e-6)
    assert state.grad_norm[0] <= 1e-6
    assert state.grad_norm[1] > 0.1


def test_robot_config_bounds_repeat_in_knot_major_tangent_layout() -> None:
    builder = ModelBuilder("bounded_revolute")
    base = builder.add_body("base")
    tip = builder.add_body("tip")
    builder.add_revolute_z("joint", parent=base, child=tip, lower=-1.0, upper=1.0)
    model = build_model(builder.finalize())
    q = RobotVariable(
        model,
        model.q_neutral.expand(2, -1).clone(),
        name="q",
        bounds=True,
        time_axis=0,
    )
    target = Variable(torch.tensor([0.25, 2.0]), name="target", trainable=False)
    problem = Problem([_RobotTrajectoryResidual(q, target)])
    optimizer = _optimizer(problem)

    info = optimizer.optimize()
    assert optimizer._state is not None
    state = optimizer._state

    torch.testing.assert_close(q.tensor, torch.tensor([[0.25], [1.0]]), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(state.active_mask, torch.tensor([False, True]))
    assert int(info.status) == int(OptimizerStatus.STALLED_AT_BOUNDS)
    assert state.projected_grad_norm <= 1e-6


@pytest.mark.parametrize(
    ("case", "q_for_v"),
    (
        pytest.param("prismatic", (0,), id="prismatic"),
        pytest.param("helical", (0,), id="helical"),
        pytest.param("translation", (0, 1, 2), id="translation"),
        pytest.param("planar", (0, 1, -1), id="planar_translation"),
        pytest.param("nested_composite", (0, 1, -1, 4, 5, 6, 7, 8), id="nested_composite"),
    ),
)
def test_supported_robot_config_bounds_map_q_to_v_and_project_steps(
    case: str,
    q_for_v: tuple[int, ...],
) -> None:
    """Every supported state-box axis maps to its physical local tangent."""
    model = _build_qv_mapping_model(case)
    assert model.nv == len(q_for_v)

    lower = model.q_neutral.new_full((model.nq,), -torch.inf)
    upper = model.q_neutral.new_full((model.nq,), torch.inf)
    for order, q_index in enumerate(index for index in q_for_v if index >= 0):
        lower[q_index] = -1.0 - 0.1 * order
        upper[q_index] = 1.0 + 0.1 * order

    target_tensor = model.q_neutral.new_full((model.nv,), 0.2)
    expected_taken = target_tensor.clone()
    expected_lower = target_tensor.new_full((model.nv,), -torch.inf)
    expected_upper = target_tensor.new_full((model.nv,), torch.inf)
    expected_bounded = torch.zeros(model.nv, dtype=torch.bool)
    for v_index, q_index in enumerate(q_for_v):
        if q_index < 0:
            continue
        target_tensor[v_index] = upper[q_index] + 0.5
        expected_taken[v_index] = upper[q_index]
        expected_lower[v_index] = lower[q_index]
        expected_upper[v_index] = upper[q_index]
        expected_bounded[v_index] = True

    q = RobotVariable(
        model,
        model.q_neutral.clone(),
        name="q",
        bounds=Bounds(lower=lower, upper=upper),
    )
    target = Variable(target_tensor, name="target", trainable=False)
    problem = Problem([_RobotTangentTargetResidual(q, target)])
    optimizer = _optimizer(problem, max_iterations=10)
    initial = {"q": model.q_neutral.clone()}

    state = optimizer._init_state(initial, problem)
    torch.testing.assert_close(state.bound_state_index, torch.tensor(q_for_v, dtype=torch.long))
    torch.testing.assert_close(state.bound_lower, expected_lower)
    torch.testing.assert_close(state.bound_upper, expected_upper)
    torch.testing.assert_close(state.bounded_mask, expected_bounded)

    next_values, next_state = optimizer._update(initial, state, problem)
    actual_taken = model.difference(model.q_neutral, next_values["q"])
    torch.testing.assert_close(actual_taken, expected_taken, atol=2e-5, rtol=2e-5)
    assert OptimizerStatus(int(next_state.status)) is OptimizerStatus.RUNNING

    info = optimizer.optimize()
    final_taken = model.difference(model.q_neutral, q.tensor)
    torch.testing.assert_close(final_taken, expected_taken, atol=2e-5, rtol=2e-5)
    assert optimizer._state is not None
    torch.testing.assert_close(optimizer._state.active_mask, expected_bounded)
    assert OptimizerStatus(int(info.status)) is OptimizerStatus.STALLED_AT_BOUNDS
    assert optimizer._state.projected_grad_norm <= optimizer.gtol


def test_finite_free_flyer_translation_bounds_fail_fast() -> None:
    """World-axis boxes cannot mask a right-local SE(3) translation tangent."""
    builder = ModelBuilder("bounded_free_flyer")
    base = builder.add_body("base")
    builder.add_free_flyer_root(child=base)
    model = build_model(builder.finalize())

    lower = torch.tensor([-1.0, -1.0, -1.0, -torch.inf, -torch.inf, -torch.inf, -torch.inf])
    upper = torch.tensor([1.0, 1.0, 1.0, torch.inf, torch.inf, torch.inf, torch.inf])
    q = RobotVariable(
        model,
        model.q_neutral,
        name="q",
        bounds=Bounds(lower=lower, upper=upper),
    )
    target = Variable(torch.zeros(model.nv), name="target", trainable=False)
    problem = Problem([_RobotTangentTargetResidual(q, target)])
    optimizer = _optimizer(problem)

    with pytest.raises(ValueError, match=r"(?i)free.flyer.*translation.*bound"):
        optimizer._init_state({"q": q.tensor}, problem)
