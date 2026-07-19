"""Named-block trajectory facade routing, batching, and safety contracts."""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.optim import (
    LevenbergMarquardt,
    LinearizationMode,
    LinearizationReason,
    Problem,
    Residual,
)
from better_robot.residuals.regularization import ReferenceTrajectoryResidual
from better_robot.tasks.parameterization import BSplineTrajectory
from better_robot.tasks.trajopt import solve_trajopt


class _UndeclaredTrajectoryResidual(Residual):
    """Valid named residual that deliberately makes auto route dense."""

    def __init__(self, q, target: torch.Tensor) -> None:
        self.q = q
        self.target = target
        super().__init__(q, dim=target.shape[-2] * q.model.nv, name="internal_undeclared")

    def error(self) -> torch.Tensor:
        q = self.q.tensor
        return (q - self.target).reshape(*q.shape[:-2], self.dim)


def _fixed_arm():
    builder = ModelBuilder("trajopt_named_arm")
    builder.add_body("base", mass=0.5)
    builder.add_body("link", mass=1.0)
    builder.add_revolute_z(
        "joint",
        parent="base",
        child="link",
        origin=torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]),
        lower=-math.pi,
        upper=math.pi,
    )
    return build_model(builder.finalize())


def _floating_body():
    builder = ModelBuilder("trajopt_named_floating")
    builder.add_body("base", mass=1.0)
    builder.add_free_flyer_root("floating_root", child="base")
    return build_model(builder.finalize())


def _reference_residuals(reference: torch.Tensor):
    return (lambda q: ReferenceTrajectoryResidual(q, reference, name="reference"),)


def _solve_reference(model, seed, reference, *, linearization: LinearizationMode, **kwargs):
    return solve_trajopt(
        model,
        dt=0.05,
        initial_q_traj=seed,
        residuals=_reference_residuals(reference),
        optimizer=lambda problem: LevenbergMarquardt(
            problem,
            max_iterations=12,
            damping=1e-5,
            linearization=linearization,
        ),
        **kwargs,
    )


def test_optimizer_factory_must_own_the_facade_problem() -> None:
    model = _fixed_arm()
    seed = model.q_neutral.expand(3, -1).clone()

    def wrong_factory(problem: Problem) -> LevenbergMarquardt:
        return LevenbergMarquardt(Problem(problem.residuals), max_iterations=0)

    with pytest.raises(ValueError, match="owning the supplied Problem"):
        solve_trajopt(
            model,
            dt=0.05,
            initial_q_traj=seed,
            residuals=_reference_residuals(seed),
            optimizer=wrong_factory,
        )


def test_dense_and_structured_task_routes_match() -> None:
    model = _fixed_arm()
    horizon = 5
    seed = model.q_neutral.expand(horizon, -1).clone()
    reference = seed.clone()
    reference[:, 0] = torch.linspace(-0.15, 0.2, horizon)

    dense = _solve_reference(model, seed, reference, linearization="dense")
    structured = _solve_reference(model, seed, reference, linearization="structured")

    assert dense.trajectory.q.shape == (1, horizon, model.nq)
    assert dense.linearization_requested == "dense"
    assert dense.linearization_used == "dense"
    assert dense.linearization_reason is LinearizationReason.FORCED_DENSE
    assert structured.linearization_requested == "structured"
    assert structured.linearization_used == "banded"
    assert structured.linearization_reason is LinearizationReason.ELIGIBLE_BANDED
    torch.testing.assert_close(structured.trajectory.q, dense.trajectory.q, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(structured.residual, dense.residual, rtol=1e-4, atol=1e-5)
    assert structured.converged == dense.converged
    assert structured.status == dense.status


def test_auto_route_honors_optional_bounds_and_projects_seed() -> None:
    model = _fixed_arm()
    horizon = 6
    seed = model.q_neutral.expand(horizon, -1).clone()
    seed[:, 0] = 0.8
    reference = model.q_neutral.expand(horizon, -1).clone()
    reference[:, 0] = 0.15
    lower = torch.tensor([-0.2], dtype=seed.dtype)
    upper = torch.tensor([0.2], dtype=seed.dtype)

    result = solve_trajopt(
        model,
        dt=0.05,
        initial_q_traj=seed,
        residuals=_reference_residuals(reference),
        optimizer=lambda problem: LevenbergMarquardt(problem, max_iterations=12, linearization="auto"),
        lower=lower,
        upper=upper,
    )

    assert result.linearization_used == "banded"
    assert result.linearization_reason is LinearizationReason.ELIGIBLE_BANDED
    assert torch.all(result.trajectory.q >= lower)
    assert torch.all(result.trajectory.q <= upper)
    torch.testing.assert_close(
        result.trajectory.q[0, :, 0],
        reference[:, 0],
        rtol=1e-3,
        atol=1e-4,
    )


def test_arbitrary_leading_batches_keep_per_element_diagnostics() -> None:
    model = _fixed_arm()
    horizon = 5
    reference = model.q_neutral.expand(horizon, -1).clone()
    reference[:, 0] = torch.linspace(-0.1, 0.1, horizon)
    seed = model.q_neutral.expand(2, 3, horizon, -1).clone()
    offsets = torch.linspace(-0.04, 0.05, 6).reshape(2, 3, 1)
    seed[..., 0] += offsets

    result = _solve_reference(
        model,
        seed,
        reference,
        linearization="structured",
    )

    assert result.trajectory.q.shape == (2, 3, horizon, model.nq)
    assert result.trajectory.t.shape == (2, 3, horizon)
    assert result.residual.shape == (2, 3, horizon * model.nv)
    assert isinstance(result.iters, torch.Tensor) and result.iters.shape == (2, 3)
    assert isinstance(result.converged, torch.Tensor) and result.converged.shape == (2, 3)
    assert isinstance(result.status, torch.Tensor) and result.status.shape == (2, 3)
    torch.testing.assert_close(
        result.trajectory.q,
        reference.expand(2, 3, horizon, model.nq),
        rtol=1e-3,
        atol=1e-4,
    )


def test_quaternion_ingress_and_egress_are_hemisphere_aligned() -> None:
    model = _floating_body()
    horizon = 5
    reference = model.q_neutral.expand(horizon, -1).clone()
    reference[:, 0] = torch.linspace(0.0, 0.08, horizon)
    reference[1::2, 3:7].neg_()
    seed = model.q_neutral.expand(horizon, -1).clone()
    seed[::2, 3:7].neg_()
    seed_before = seed.clone()

    dense = _solve_reference(model, seed, reference, linearization="dense")
    structured = _solve_reference(model, seed, reference, linearization="structured")

    torch.testing.assert_close(seed, seed_before)
    for result in (dense, structured):
        quaternion = result.trajectory.q[..., 3:7]
        adjacent_dot = (quaternion[..., 1:, :] * quaternion[..., :-1, :]).sum(dim=-1)
        assert torch.all(adjacent_dot >= 0.0)
    torch.testing.assert_close(structured.trajectory.q, dense.trajectory.q, rtol=1e-3, atol=1e-4)


def test_auto_falls_back_to_dense_for_undeclared_temporal_item() -> None:
    model = _fixed_arm()
    horizon = 5
    seed = model.q_neutral.expand(horizon, -1).clone()
    target = seed.clone()
    target[:, 0] = 0.1
    result = solve_trajopt(
        model,
        dt=0.05,
        initial_q_traj=seed,
        residuals=(lambda q: _UndeclaredTrajectoryResidual(q, target),),
        optimizer=lambda problem: LevenbergMarquardt(problem, max_iterations=8, linearization="auto"),
    )

    assert result.linearization_requested == "auto"
    assert result.linearization_used == "dense"
    assert result.linearization_reason is LinearizationReason.UNDECLARED_TEMPORAL_RESIDUAL
    assert "undeclared" in result.linearization_detail


def test_bspline_fails_actionably() -> None:
    model = _fixed_arm()
    horizon = 5
    seed = model.q_neutral.expand(horizon, -1).clone()
    reference = seed.clone()

    with pytest.raises(NotImplementedError, match="component-space.*not manifold-safe"):
        solve_trajopt(
            model,
            dt=0.05,
            initial_q_traj=seed,
            residuals=_reference_residuals(reference),
            parameterization=BSplineTrajectory(num_control_points=4),
        )
