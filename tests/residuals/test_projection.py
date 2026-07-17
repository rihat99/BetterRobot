"""Camera-thin projection residual tests over frame and marker rows."""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.io import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.optim import Problem, ResidualItem, RobotConfig, RobotStateProvider, VarSpec
from better_robot.optim.kernels import GemanMcClure
from better_robot.residuals.projection import ProjectionResidual


def _camera_arm(*, dtype: torch.dtype = torch.float32):
    identity = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        dtype=dtype,
    )
    builder = ModelBuilder("camera_arm")
    builder.add_body("base")
    builder.add_body("link")
    builder.add_revolute_z(
        "joint",
        parent="base",
        child="link",
        origin=torch.tensor(
            [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            dtype=dtype,
        ),
        lower=-math.pi,
        upper=math.pi,
    )
    builder.add_frame(
        "marker",
        parent_body="link",
        placement=torch.cat((torch.tensor([1.0], dtype=dtype), identity)),
    )
    return build_model(builder.finalize(), dtype=dtype)


def _camera_tensors(dtype: torch.dtype):
    intrinsics = torch.tensor(
        [[100.0, 0.0, 320.0], [0.0, 120.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )
    extrinsics = torch.eye(4, dtype=dtype)
    return intrinsics, extrinsics


def _point_ids(model) -> tuple[int, int]:
    return model.frame_id("body_link"), model.frame_id("marker")


def _problem(model, residual: ProjectionResidual, *, kernel=None) -> Problem:
    return Problem(
        vars=(
            VarSpec(
                "q",
                (model.nq,),
                manifold=RobotConfig(model),
            ),
        ),
        residuals=(
            ResidualItem(
                residual.name,
                residual,
                kernel=kernel,
                group_size=2,
            ),
        ),
        providers=(RobotStateProvider(model),),
    )


def test_projection_value_confidence_and_valid_mask() -> None:
    model = _camera_arm()
    intrinsics, extrinsics = _camera_tensors(torch.float32)
    target = torch.tensor([[300.0, 200.0], [300.0, 200.0]])
    residual = ProjectionResidual(
        model,
        _point_ids(model),
        intrinsics,
        extrinsics,
        target,
        weights=torch.tensor([0.25, 0.75]),
        valid_mask=torch.tensor([True, False]),
    )
    q = torch.tensor([0.0])

    actual = _problem(model, residual).residual({"q": q})

    # body_link is (0, 0, 3): (320, 240). marker is masked out.
    expected = torch.tensor([5.0, 10.0, 0.0, 0.0])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 3)])
def test_projection_preserves_arbitrary_execution_batch(batch_shape: tuple[int, ...]) -> None:
    model = _camera_arm()
    intrinsics, extrinsics = _camera_tensors(torch.float32)
    residual = ProjectionResidual(
        model,
        _point_ids(model),
        intrinsics,
        extrinsics,
        torch.zeros(2, 2),
    )
    q = torch.tensor([0.2]).expand(*batch_shape, model.nq).clone()

    rows = _problem(model, residual).residual({"q": q})

    assert rows.shape == (*batch_shape, 4)


def test_projection_analytic_jacobian_matches_both_ad_directions() -> None:
    model = _camera_arm()
    intrinsics, extrinsics = _camera_tensors(torch.float32)
    extrinsics = extrinsics.clone()
    angle = torch.tensor(0.3)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    extrinsics[:3, :3] = torch.tensor(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    extrinsics[:3, 3] = torch.tensor([0.2, -0.1, 0.5])
    residual = ProjectionResidual(
        model,
        _point_ids(model),
        intrinsics,
        extrinsics,
        torch.tensor([[300.0, 200.0], [315.0, 215.0]]),
        weights=torch.tensor([0.4, 0.9]),
        valid_mask=torch.tensor([True, True]),
    )
    problem = _problem(model, residual)
    values = {"q": torch.tensor([[0.2], [-0.35]])}

    analytic = problem.jacobian_blocks(values, strategy="analytic")[("projection", "q")]
    jacrev = problem.jacobian_blocks(values, strategy="jacrev")[("projection", "q")]
    jacfwd = problem.jacobian_blocks(values, strategy="jacfwd")[("projection", "q")]

    torch.testing.assert_close(analytic, jacrev, atol=1.0e-3, rtol=1.0e-4)
    torch.testing.assert_close(analytic, jacfwd, atol=1.0e-3, rtol=1.0e-4)


def test_projection_gradcheck_through_configuration() -> None:
    model = _camera_arm(dtype=torch.float64)
    intrinsics, extrinsics = _camera_tensors(torch.float64)
    residual = ProjectionResidual(
        model,
        _point_ids(model),
        intrinsics,
        extrinsics,
        torch.tensor([[300.0, 200.0], [315.0, 215.0]], dtype=torch.float64),
    )
    problem = _problem(model, residual)
    q = torch.tensor([0.31], dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda value: problem.residual({"q": value}),
        (q,),
        eps=1.0e-6,
        atol=1.0e-4,
        rtol=1.0e-3,
    )


def test_projection_clamps_points_behind_camera_without_nan() -> None:
    model = _camera_arm()
    intrinsics, extrinsics = _camera_tensors(torch.float32)
    extrinsics = extrinsics.clone()
    extrinsics[2, 3] = -4.0
    residual = ProjectionResidual(
        model,
        _point_ids(model),
        intrinsics,
        extrinsics,
        torch.zeros(2, 2),
        min_depth=0.1,
    )
    problem = _problem(model, residual)
    values = {"q": torch.tensor([0.2])}

    rows = problem.residual(values)
    jacobian = problem.jacobian_blocks(values, strategy="analytic")[("projection", "q")]

    assert bool(torch.isfinite(rows).all())
    assert bool(torch.isfinite(jacobian).all())


def test_projection_groups_feed_geman_mcclure_per_point() -> None:
    model = _camera_arm()
    intrinsics, extrinsics = _camera_tensors(torch.float32)
    kernel = GemanMcClure(c=20.0)
    residual = ProjectionResidual(
        model,
        _point_ids(model),
        intrinsics,
        extrinsics,
        torch.tensor([[300.0, 200.0], [315.0, 215.0]]),
    )
    problem = _problem(model, residual, kernel=kernel)
    values = {"q": torch.tensor([0.1])}

    rows = problem.residual(values).reshape(2, 2)
    expected = kernel.rho(rows.square().sum(dim=-1)).sum()

    torch.testing.assert_close(problem.objective(values), expected)
