"""Outer-weight, reduction, activity, and IRLS algebra contracts."""

from __future__ import annotations

import pytest
import torch

from better_robot.optim import GemanMcClure, L2, LevenbergMarquardt, Problem, Residual, Variable


class _IdentityResidual(Residual):
    def __init__(self, x: Variable, *, name: str = "identity", **kwargs) -> None:
        self.x = x
        super().__init__(x, dim=x.shape[-1], name=name, **kwargs)

    def error(self) -> torch.Tensor:
        return self.x.tensor

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        value = self.x.tensor
        width = value.shape[-1]
        identity = torch.eye(width, dtype=value.dtype, device=value.device)
        return (identity.expand(*value.shape[:-1], width, width),)


class _ThresholdResidual(_IdentityResidual):
    def __init__(self, x: Variable, *, threshold: float, **kwargs) -> None:
        self.threshold = threshold
        super().__init__(x, **kwargs)

    def active_groups(self) -> torch.Tensor:
        groups = self.error().reshape(*self.x.batch_shape, -1, self.group_size)
        return groups.square().sum(dim=-1) > self.threshold * self.threshold


class _DeclaredMaskResidual(_IdentityResidual):
    def __init__(self, x: Variable, mask: torch.Tensor, **kwargs) -> None:
        self.mask = mask
        super().__init__(x, **kwargs)

    def active_groups(self) -> torch.Tensor:
        return self.mask


class _NonzeroAtOriginKernel:
    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        return squared_norm + 7.0

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(squared_norm)


def test_outer_weight_and_geman_mcclure_scale_are_independent() -> None:
    rows = torch.tensor([0.0, 0.5, 2.0, -3.0])
    x = Variable(rows, name="x")
    weight, scale = 2.4, 1.3
    problem = Problem(
        [
            _IdentityResidual(
                x,
                weight=weight,
                kernel=GemanMcClure(c=scale),
                reduce="mean",
            )
        ]
    )

    squared = rows.square()
    expected = weight * (0.5 * scale**2 * squared / (scale**2 + squared)).mean()
    torch.testing.assert_close(problem.objective(), expected)
    torch.testing.assert_close(problem.error(), rows)


def test_mean_active_uses_candidate_activity_from_the_same_evaluation_scope() -> None:
    current = torch.tensor([0.0, 2.0, 4.0, 6.0])
    candidate = torch.tensor([5.0, 1.0, 0.0, 7.0])
    x = Variable(current, name="x")
    problem = Problem([_ThresholdResidual(x, threshold=3.0, reduce="mean_active")])

    expected_current = 0.5 * current[2:].square().mean()
    expected_candidate = 0.5 * candidate[[0, 3]].square().mean()
    torch.testing.assert_close(problem.objective(), expected_current)
    torch.testing.assert_close(problem.objective({"x": candidate}), expected_candidate)
    torch.testing.assert_close(x.tensor, current)


def test_activity_mask_is_authoritative_when_kernel_rho_at_zero_is_nonzero() -> None:
    rows = torch.zeros(4)
    mask = torch.tensor([False, True, False, True])
    x = Variable(rows, name="x")
    problem = Problem(
        [
            _DeclaredMaskResidual(
                x,
                mask,
                kernel=_NonzeroAtOriginKernel(),
            )
        ]
    )

    torch.testing.assert_close(problem.objective(), torch.tensor(14.0))


def test_term_costs_sum_exactly_to_the_objective() -> None:
    x = Variable(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), name="x", batch_ndim=1)
    first = _IdentityResidual(x, weight=torch.tensor([0.5, 2.0]), name="first")
    second = _IdentityResidual(x, weight=3.0, reduce="mean", name="second")
    problem = Problem([first, second])

    costs = problem.term_costs()
    assert tuple(costs) == ("first", "second")
    torch.testing.assert_close(
        torch.stack(tuple(costs.values()), dim=-1).sum(dim=-1),
        problem.objective(),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("weight_form", ["python", "scalar", "batch", "group"])
@pytest.mark.parametrize("reduce", ["sum", "mean", "mean_active"])
@pytest.mark.parametrize("kernel", [L2(), GemanMcClure(c=1.3)], ids=["l2", "gm"])
@pytest.mark.parametrize("masked", [False, True], ids=["all_active", "masked"])
def test_lm_gradient_matches_exact_objective_on_fixed_active_sets(
    weight_form: str,
    reduce: str,
    kernel: object,
    masked: bool,
) -> None:
    value = torch.tensor([[0.4, -0.7, 1.1, -1.3], [0.8, 0.2, -0.5, 1.4]])
    x = Variable(value, name="x", batch_ndim=1)
    weights = {
        "python": 1.7,
        "scalar": torch.tensor(1.7),
        "batch": torch.tensor([0.8, 1.7]),
        "group": torch.tensor([[0.8, 1.2], [1.7, 0.6]]),
    }
    mask = (
        torch.tensor([[True, False], [False, True]])
        if masked
        else torch.ones((2, 2), dtype=torch.bool)
    )
    item = _DeclaredMaskResidual(
        x,
        mask,
        weight=weights[weight_form],
        row_weight=1.25,
        reduce=reduce,
        kernel=kernel,
        group_size=2,
    )
    problem = Problem([item])
    optimizer = LevenbergMarquardt(problem, jacobian_strategy="analytic")

    exact = problem.gradient()["x"]
    irls = optimizer._init_state({"x": x.tensor}, problem).gradient
    torch.testing.assert_close(irls, exact, rtol=2e-5, atol=2e-6)


def test_activity_threshold_crossing_still_descends() -> None:
    x = Variable(torch.tensor([-2.0]), name="x")

    class _ShiftedThreshold(_ThresholdResidual):
        def error(self) -> torch.Tensor:
            return self.x.tensor - 2.0

    item = _ShiftedThreshold(x, threshold=0.1, reduce="mean_active")
    problem = Problem([item])
    optimizer = LevenbergMarquardt(problem, max_iterations=2, jacobian_strategy="analytic")
    before = problem.objective().clone()

    optimizer.step()

    assert problem.objective() < before
    assert x.tensor > -2.0


def test_enabled_round_trip_preserves_lm_state_and_problem_serial() -> None:
    x = Variable(torch.tensor([1.0]), name="x")
    item = _IdentityResidual(x)
    problem = Problem([item])
    optimizer = LevenbergMarquardt(problem)
    state = optimizer._ensure_state()
    serial = problem._update_serial

    item.enabled = False
    assert optimizer._ensure_state() is state
    torch.testing.assert_close(problem.error(), torch.zeros(1))
    torch.testing.assert_close(problem.objective(), torch.tensor(0.0))
    torch.testing.assert_close(problem.gradient()["x"], torch.zeros(1))
    assert problem.jacobian_blocks(strategy="analytic") == {}
    item.enabled = True
    assert optimizer._ensure_state() is state
    assert problem._update_serial == serial


@pytest.mark.parametrize("shape", [(1,), (2, 1), (2, 2, 1)])
def test_outer_tensor_weight_rejects_inexact_shapes(shape: tuple[int, ...]) -> None:
    x = Variable(torch.ones(2, 4), name="x", batch_ndim=1)
    problem = Problem([_IdentityResidual(x, weight=torch.ones(shape), group_size=2)])

    with pytest.raises(ValueError, match="weight shape must be one of"):
        problem.objective()


@pytest.mark.parametrize(
    "mask",
    [torch.ones(2, 1, dtype=torch.bool), torch.ones(2, 2)],
    ids=["wrong_shape", "wrong_dtype"],
)
def test_activity_mask_rejects_wrong_shape_or_dtype(mask: torch.Tensor) -> None:
    x = Variable(torch.ones(2, 4), name="x", batch_ndim=1)
    problem = Problem([_DeclaredMaskResidual(x, mask, group_size=2)])

    with pytest.raises(ValueError, match="active_groups must"):
        problem.objective()


@pytest.mark.slow
def test_tensor_activity_mean_active_update_fullgraph_compile_smoke() -> None:
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable in this Torch build")
    x = Variable(torch.tensor([[-2.0], [-3.0]]), name="x", batch_ndim=1)

    class _ShiftedThreshold(_ThresholdResidual):
        def error(self) -> torch.Tensor:
            return self.x.tensor - 2.0

    problem = Problem([_ShiftedThreshold(x, threshold=0.1, reduce="mean_active")])
    optimizer = LevenbergMarquardt(problem, max_iterations=2, jacobian_strategy="analytic")
    values = {"x": x.tensor}
    state = optimizer._init_state(values, problem)

    eager_values, eager_state = optimizer._update(values, state, problem)
    compiled = torch.compile(optimizer._update, fullgraph=True, backend="eager")
    compiled_values, compiled_state = compiled(values, state, problem)

    torch.testing.assert_close(compiled_values["x"], eager_values["x"])
    for name in eager_state._fields:
        torch.testing.assert_close(getattr(compiled_state, name), getattr(eager_state, name), equal_nan=True)
