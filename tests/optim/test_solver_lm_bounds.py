"""Bound-aware LM tests with analytic constrained optima.

The coupled linear system is intentionally not diagonal.  At the upper bound
on ``x[0]``, solving the unrestricted problem and clamping afterward produces
``x[1] = 1``.  Restricting the normal system to the inactive coordinate gives
the true constrained minimizer ``x = [1, 1.6]`` instead.
"""

from __future__ import annotations

from collections.abc import Mapping
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
from better_robot.optim.blocks import (
    Bounds,
    LevenbergMarquardt,
    LMStatus,
    Problem,
    ResidualItem,
    RobotConfig,
    VarSpec,
)


_A = torch.tensor([[1.0, 1.0], [1.0, 2.0]])
_LOWER = torch.tensor([-3.0, -3.0])
_UPPER = torch.tensor([1.0, 3.0])


class _CoupledLinearResidual:
    name = "linear"
    reads = ("x", "target")
    dim = 2

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        x = ctx["x"]
        matrix = _A.to(dtype=x.dtype, device=x.device)
        return torch.matmul(x, matrix.mT) - ctx["target"]

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        x = ctx["x"]
        matrix = _A.to(dtype=x.dtype, device=x.device)
        return {"x": matrix.expand(*x.shape[:-1], 2, 2)}


class _FreeFlyerTranslationResidual:
    name = "translation"
    reads = ("q",)
    dim = 3

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["q"][..., :3]


class _RobotTrajectoryResidual:
    name = "joint_trajectory"
    reads = ("q", "target")
    dim = 2

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["q"][..., :, 0] - ctx["target"]

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        q = ctx["q"]
        identity = torch.eye(2, dtype=q.dtype, device=q.device)
        return {"q": identity.expand(*q.shape[:-2], 2, 2)}


class _RobotTangentTargetResidual:
    name = "robot_tangent_target"
    reads = ("q", "target")

    def __init__(self, model: Any) -> None:
        self.model = model
        self.dim = model.nv

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        q = ctx["q"]
        neutral = self.model.q_neutral.to(dtype=q.dtype, device=q.device)
        return self.model.difference(neutral, q) - ctx["target"]

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        q = ctx["q"]
        identity = torch.eye(self.dim, dtype=q.dtype, device=q.device)
        return {"q": identity.expand(*q.shape[:-1], self.dim, self.dim)}


def _target_for(solution: torch.Tensor) -> torch.Tensor:
    return torch.matmul(solution, _A.mT)


def _linear_problem(target: torch.Tensor) -> Problem:
    return Problem(
        vars=(
            VarSpec(
                "x",
                (2,),
                bounds=Bounds(lower=_LOWER, upper=_UPPER),
            ),
        ),
        residuals=(ResidualItem("linear", _CoupledLinearResidual()),),
        parameters={"target": target},
    )


def _solver(*, max_iter: int = 40) -> LevenbergMarquardt:
    return LevenbergMarquardt(
        max_iter=max_iter,
        gtol=1e-6,
        xtol=1e-8,
        ftol=1e-8,
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
        builder.add_joint(
            "joint",
            kind=JointTranslation(),
            parent=base,
            child=tip,
        )
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
    else:  # pragma: no cover - the parametrization below is exhaustive
        raise AssertionError(f"unknown q-v mapping case {case!r}")
    return build_model(builder.finalize())


def test_active_set_restricts_the_coupled_normal_system() -> None:
    """The first boundary step follows the reduced system, not clamp-only LM."""
    target = _target_for(torch.tensor([2.0, 1.0]))
    problem = _linear_problem(target)
    values = {"x": torch.tensor([1.0, 0.0])}
    solver = _solver(max_iter=1)

    state = solver.init_state(values, problem)
    next_values, next_state = solver.update(values, state, problem)

    # At x=[1, 0], x[0] is outward-active.  Restricting A.T@A to x[1]
    # gives delta[1] ~= 8/5.  An unrestricted step followed by projection
    # would instead land at x[1] ~= 1 and fail this regression.
    assert torch.equal(next_values["x"][0], values["x"][0])
    assert next_values["x"][1] > 1.5
    assert next_values["x"][1] < 1.7
    torch.testing.assert_close(next_state.active_mask, torch.tensor([True, False]))


def test_known_constrained_optimum_has_active_mask_and_kkt_status() -> None:
    target = _target_for(torch.tensor([2.0, 1.0]))
    problem = _linear_problem(target)

    values, state = _solver().run({"x": torch.zeros(2)}, problem)

    torch.testing.assert_close(values["x"], torch.tensor([1.0, 1.6]), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(state.active_mask, torch.tensor([True, False]))
    assert int(state.status) == int(LMStatus.STALLED_AT_BOUNDS)
    assert state.projected_grad_norm <= 1e-6
    assert state.kkt_norm is state.projected_grad_norm
    assert state.grad_norm > 0.1
    torch.testing.assert_close(state.gradient, torch.tensor([-0.2, 0.0]), atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(state.cost, torch.tensor(0.1), atol=3e-5, rtol=3e-5)


def test_kkt_terminal_point_is_not_reported_as_unconstrained_convergence() -> None:
    target = _target_for(torch.tensor([2.0, 1.0]))
    problem = _linear_problem(target)
    initial = {"x": torch.tensor([1.0, 1.6])}

    values, state = _solver(max_iter=5).run(initial, problem)

    torch.testing.assert_close(values["x"], initial["x"], atol=2e-6, rtol=0.0)
    assert int(state.status) == int(LMStatus.STALLED_AT_BOUNDS)
    assert state.projected_grad_norm <= 1e-6
    assert state.grad_norm > 0.1
    assert state.iterations <= 1


def test_mixed_batch_keeps_interior_and_bound_statuses_independent() -> None:
    interior = torch.tensor([0.5, 0.25])
    outside = torch.tensor([2.0, 1.0])
    target = _target_for(torch.stack((interior, outside)))
    problem = _linear_problem(target)

    values, state = _solver().run({"x": torch.zeros(2, 2)}, problem)

    expected = torch.tensor([[0.5, 0.25], [1.0, 1.6]])
    torch.testing.assert_close(values["x"], expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
        state.status,
        torch.tensor(
            [LMStatus.CONVERGED, LMStatus.STALLED_AT_BOUNDS],
            dtype=state.status.dtype,
        ),
    )
    torch.testing.assert_close(
        state.active_mask,
        torch.tensor([[False, False], [True, False]]),
    )
    assert torch.all(state.projected_grad_norm <= 1e-6)
    assert state.grad_norm[0] <= 1e-6
    assert state.grad_norm[1] > 0.1


def test_robot_config_bounds_repeat_in_knot_major_tangent_layout() -> None:
    builder = ModelBuilder("bounded_revolute")
    base = builder.add_body("base")
    tip = builder.add_body("tip")
    builder.add_revolute_z(
        "joint",
        parent=base,
        child=tip,
        lower=-1.0,
        upper=1.0,
    )
    model = build_model(builder.finalize())
    problem = Problem(
        vars=(
            VarSpec(
                "q",
                (2, model.nq),
                manifold=RobotConfig(model),
                bounds=Bounds(
                    lower=model.lower_pos_limit,
                    upper=model.upper_pos_limit,
                ),
            ),
        ),
        residuals=(ResidualItem("joint_trajectory", _RobotTrajectoryResidual()),),
        parameters={"target": torch.tensor([0.25, 2.0])},
    )

    values, state = _solver().run({"q": model.q_neutral.expand(2, -1).clone()}, problem)

    torch.testing.assert_close(
        values["q"],
        torch.tensor([[0.25], [1.0]]),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(state.active_mask, torch.tensor([False, True]))
    assert int(state.status) == int(LMStatus.STALLED_AT_BOUNDS)
    assert state.projected_grad_norm <= 1e-6


@pytest.mark.parametrize(
    ("case", "q_for_v"),
    (
        pytest.param("prismatic", (0,), id="prismatic"),
        pytest.param("helical", (0,), id="helical"),
        pytest.param("translation", (0, 1, 2), id="translation"),
        pytest.param("planar", (0, 1, -1), id="planar_translation"),
        pytest.param(
            "nested_composite",
            (0, 1, -1, 4, 5, 6, 7, 8),
            id="nested_composite",
        ),
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

    residual = _RobotTangentTargetResidual(model)
    target = model.q_neutral.new_full((model.nv,), 0.2)
    expected_taken = target.clone()
    expected_lower = target.new_full((model.nv,), -torch.inf)
    expected_upper = target.new_full((model.nv,), torch.inf)
    expected_bounded = torch.zeros(model.nv, dtype=torch.bool)
    for v_index, q_index in enumerate(q_for_v):
        if q_index < 0:
            continue
        target[v_index] = upper[q_index] + 0.5
        expected_taken[v_index] = upper[q_index]
        expected_lower[v_index] = lower[q_index]
        expected_upper[v_index] = upper[q_index]
        expected_bounded[v_index] = True

    problem = Problem(
        vars=(
            VarSpec(
                "q",
                (model.nq,),
                manifold=RobotConfig(model),
                bounds=Bounds(lower=lower, upper=upper),
            ),
        ),
        residuals=(ResidualItem(residual.name, residual),),
        parameters={"target": target},
    )
    solver = _solver(max_iter=10)
    initial = {"q": model.q_neutral.clone()}

    state = solver.init_state(initial, problem)

    torch.testing.assert_close(
        state.bound_state_index,
        torch.tensor(q_for_v, dtype=torch.long),
    )
    torch.testing.assert_close(state.bound_lower, expected_lower)
    torch.testing.assert_close(state.bound_upper, expected_upper)
    torch.testing.assert_close(state.bounded_mask, expected_bounded)

    next_values, next_state = solver.update(initial, state, problem)
    actual_taken = model.difference(model.q_neutral, next_values["q"])
    torch.testing.assert_close(actual_taken, expected_taken, atol=2e-5, rtol=2e-5)
    assert LMStatus(int(next_state.status)) is LMStatus.RUNNING

    final_values, final_state = solver.run(initial, problem)
    final_taken = model.difference(model.q_neutral, final_values["q"])

    torch.testing.assert_close(final_taken, expected_taken, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(final_state.active_mask, expected_bounded)
    assert LMStatus(int(final_state.status)) is LMStatus.STALLED_AT_BOUNDS
    assert final_state.projected_grad_norm <= solver.gtol


def test_finite_free_flyer_translation_bounds_fail_fast() -> None:
    """World-axis boxes cannot mask a right-local SE(3) translation tangent."""
    builder = ModelBuilder("bounded_free_flyer")
    base = builder.add_body("base")
    builder.add_free_flyer_root(child=base)
    model = build_model(builder.finalize())

    lower = torch.tensor([-1.0, -1.0, -1.0, -torch.inf, -torch.inf, -torch.inf, -torch.inf])
    upper = torch.tensor([1.0, 1.0, 1.0, torch.inf, torch.inf, torch.inf, torch.inf])
    problem = Problem(
        vars=(
            VarSpec(
                "q",
                (model.nq,),
                manifold=RobotConfig(model),
                bounds=Bounds(lower=lower, upper=upper),
            ),
        ),
        residuals=(ResidualItem("translation", _FreeFlyerTranslationResidual()),),
    )

    with pytest.raises(ValueError, match=r"(?i)free.flyer.*translation.*bound"):
        _solver().init_state({"q": model.q_neutral}, problem)
