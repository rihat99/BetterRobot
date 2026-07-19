"""Camera-thin projection residual tests over frame and marker rows."""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.exceptions import DtypeMismatchError
from better_robot.io import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.optim.kernels import GemanMcClure
from better_robot.optim.problem import Problem
from better_robot.optim.variables import RobotVariable, Variable
from better_robot.residuals.projection import ProjectionResidual, _project_points, _projection_jacobian


def _camera_arm(*, dtype: torch.dtype = torch.float32):
    identity = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=dtype)
    builder = ModelBuilder("camera_arm")
    builder.add_body("base")
    builder.add_body("link")
    builder.add_revolute_z(
        "joint",
        parent="base",
        child="link",
        origin=torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0], dtype=dtype),
        lower=-math.pi,
        upper=math.pi,
    )
    builder.add_frame(
        "marker",
        parent_body="link",
        placement=torch.cat((torch.tensor([1.0], dtype=dtype), identity)),
    )
    return build_model(builder.finalize(), dtype=dtype)


def _camera_tensors(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = torch.tensor(
        [[100.0, 0.0, 320.0], [0.0, 120.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )
    return intrinsics, torch.eye(4, dtype=dtype)


def _point_ids(model) -> tuple[int, int]:
    return model.frame_id("body_link"), model.frame_id("marker")


def _problem(
    model,
    q_value: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    target: torch.Tensor | Variable,
    **kwargs,
) -> tuple[RobotVariable, ProjectionResidual, Problem]:
    q = RobotVariable(model, q_value, name="q")
    item = ProjectionResidual(q, _point_ids(model), intrinsics, extrinsics, target, **kwargs)
    return q, item, Problem([item])


def test_projection_value_confidence_and_valid_mask() -> None:
    model = _camera_arm()
    intrinsics, extrinsics = _camera_tensors(torch.float32)
    _q, _item, problem = _problem(
        model,
        torch.tensor([0.0]),
        intrinsics,
        extrinsics,
        torch.tensor([[300.0, 200.0], [300.0, 200.0]]),
        weights=torch.tensor([0.25, 0.75]),
        valid_mask=torch.tensor([True, False]),
    )

    # body_link is (0, 0, 3): (320, 240). marker is masked out.
    torch.testing.assert_close(problem.error(), torch.tensor([5.0, 10.0, 0.0, 0.0]))


@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
def test_projection_preserves_arbitrary_execution_batch(batch_shape: tuple[int, ...]) -> None:
    model = _camera_arm()
    intrinsics, extrinsics = _camera_tensors(torch.float32)
    q_value = torch.tensor([0.2]).expand(*batch_shape, model.nq).clone()
    _q, _item, problem = _problem(
        model,
        q_value,
        intrinsics,
        extrinsics,
        torch.zeros(2, 2),
    )

    assert problem.error().shape == (*batch_shape, 4)


def test_projection_analytic_jacobian_matches_both_ad_directions() -> None:
    model = _camera_arm()
    intrinsics, extrinsics = _camera_tensors(torch.float32)
    extrinsics = extrinsics.clone()
    angle = torch.tensor(0.3)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    extrinsics[:3, :3] = torch.tensor([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    extrinsics[:3, 3] = torch.tensor([0.2, -0.1, 0.5])
    _q, _item, problem = _problem(
        model,
        torch.tensor([[0.2], [-0.35]]),
        intrinsics,
        extrinsics,
        torch.tensor([[300.0, 200.0], [315.0, 215.0]]),
        weights=torch.tensor([0.4, 0.9]),
        valid_mask=torch.tensor([True, True]),
    )

    analytic = problem.dense_jacobian(strategy="analytic")
    jacrev = problem.dense_jacobian(strategy="jacrev")
    jacfwd = problem.dense_jacobian(strategy="jacfwd")
    torch.testing.assert_close(analytic, jacrev, atol=1.0e-3, rtol=1.0e-4)
    torch.testing.assert_close(analytic, jacfwd, atol=1.0e-3, rtol=1.0e-4)


def test_projection_gradcheck_through_configuration() -> None:
    model = _camera_arm(dtype=torch.float64)
    intrinsics, extrinsics = _camera_tensors(torch.float64)
    q, _item, problem = _problem(
        model,
        torch.tensor([0.31], dtype=torch.float64),
        intrinsics,
        extrinsics,
        torch.tensor([[300.0, 200.0], [315.0, 215.0]], dtype=torch.float64),
    )

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        problem.update({q.name: value})
        return problem.error()

    assert torch.autograd.gradcheck(
        evaluate,
        (q.tensor.requires_grad_(),),
        eps=1.0e-6,
        atol=1.0e-4,
        rtol=1.0e-3,
    )


def test_projection_clamps_points_behind_camera_without_nan() -> None:
    model = _camera_arm()
    intrinsics, extrinsics = _camera_tensors(torch.float32)
    extrinsics = extrinsics.clone()
    extrinsics[2, 3] = -4.0
    _q, _item, problem = _problem(
        model,
        torch.tensor([0.2]),
        intrinsics,
        extrinsics,
        torch.zeros(2, 2),
        min_depth=0.1,
    )

    assert bool(torch.isfinite(problem.error()).all())
    assert bool(torch.isfinite(problem.dense_jacobian(strategy="analytic")).all())


def test_projection_jacobian_matches_clamp_derivative_at_depth_floor() -> None:
    intrinsics, _ = _camera_tensors(torch.float64)
    point = torch.tensor([0.2, -0.1, 0.1], dtype=torch.float64)
    analytic = _projection_jacobian(point.unsqueeze(0), intrinsics, min_depth=0.1).squeeze(0)
    autodiff = torch.func.jacrev(
        lambda value: _project_points(value.unsqueeze(0), intrinsics, min_depth=0.1).squeeze(0)
    )(point)

    torch.testing.assert_close(analytic, autodiff)


def test_projection_invalid_nan_target_produces_finite_zero_rows() -> None:
    model = _camera_arm()
    intrinsics, extrinsics = _camera_tensors(torch.float32)
    _q, item, _problem_value = _problem(
        model,
        torch.tensor([0.2]),
        intrinsics,
        extrinsics,
        torch.full((2, 2), torch.nan),
        valid_mask=torch.zeros(2, dtype=torch.bool),
    )

    torch.testing.assert_close(item.error(), torch.zeros(4))


def test_projection_static_target_is_declared_and_graph_visible() -> None:
    model = _camera_arm(dtype=torch.float64)
    intrinsics, extrinsics = _camera_tensors(torch.float64)
    target_tensor = torch.tensor(
        [[300.0, 200.0], [315.0, 215.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = Variable(target_tensor, name="observed_px", trainable=False)
    _q, _item, problem = _problem(
        model,
        torch.tensor([0.2], dtype=torch.float64),
        intrinsics,
        extrinsics,
        target,
    )

    problem.error()
    assert problem.variables["observed_px"] is target
    gradient = problem.gradient(create_graph=True)["q"]
    target_vjp = torch.autograd.grad(gradient.sum(), target_tensor)[0]
    assert bool(torch.isfinite(gradient).all())
    assert bool(torch.isfinite(target_vjp).all())
    assert bool(torch.any(target_vjp != 0.0))


def test_projection_observations_validate_working_dtype() -> None:
    model = _camera_arm()
    intrinsics, extrinsics = _camera_tensors(torch.float32)
    target = Variable(torch.zeros(2, 2, dtype=torch.float64), name="observed_px", trainable=False)
    q = RobotVariable(model, torch.tensor([0.2]), name="q")

    with pytest.raises(DtypeMismatchError, match=r"target_px\.dtype"):
        ProjectionResidual(q, _point_ids(model), intrinsics, extrinsics, target)


def test_projection_groups_feed_geman_mcclure_per_point() -> None:
    model = _camera_arm()
    intrinsics, extrinsics = _camera_tensors(torch.float32)
    kernel = GemanMcClure(c=20.0)
    _q, item, problem = _problem(
        model,
        torch.tensor([0.1]),
        intrinsics,
        extrinsics,
        torch.tensor([[300.0, 200.0], [315.0, 215.0]]),
        kernel=kernel,
    )
    rows = problem.error().reshape(2, 2)
    expected = kernel.rho(rows.square().sum(dim=-1)).sum()

    assert item.group_size == 2
    torch.testing.assert_close(problem.objective(), expected)
