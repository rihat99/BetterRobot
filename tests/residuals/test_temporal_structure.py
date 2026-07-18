"""Named-block and temporal-structure contracts for trajectory residuals."""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics.forward import forward_kinematics
from better_robot.optim import Problem, ResidualItem, RobotConfig, VarSpec
from better_robot.residuals import (
    AccelerationResidual,
    ContactConsistencyResidual,
    PositionResidual,
    ReferenceTrajectoryResidual,
    TimeIndexedResidual,
    VelocityResidual,
)
from better_robot.residuals._temporal_jacobian import dense_temporal_jacobian
from better_robot.residuals.structure import TemporalPattern


class _Context(dict):
    def __init__(self, *args, temporal_indices: torch.Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._temporal_indices = temporal_indices

    def temporal_free_indices(self, variable_name: str) -> torch.Tensor:
        assert variable_name == "q"
        return self._temporal_indices


@pytest.fixture(scope="module")
def two_joint_model():
    pose = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    builder = ModelBuilder("temporal_structure")
    builder.add_body("base", mass=1.0)
    builder.add_body("first", mass=1.0)
    builder.add_body("second", mass=1.0)
    builder.add_revolute_z(
        "joint_1",
        parent="base",
        child="first",
        origin=pose,
        lower=-math.pi,
        upper=math.pi,
    )
    builder.add_revolute_y(
        "joint_2",
        parent="first",
        child="second",
        origin=pose,
        lower=-math.pi,
        upper=math.pi,
    )
    builder.add_frame("first_tip", parent_body="first", placement=pose)
    builder.add_frame("second_tip", parent_body="second", placement=pose)
    return build_model(builder.finalize(), dtype=torch.float64)


def _trajectory(model, batch_shape=(2, 3), horizon: int = 5) -> torch.Tensor:
    torch.manual_seed(8)
    neutral = model.q_neutral.expand(*batch_shape, horizon, model.nq).clone()
    tangent = torch.randn(*batch_shape, horizon, model.nv, dtype=neutral.dtype) * 0.03
    return model.integrate(neutral, tangent)


def _context(model, q: torch.Tensor, indices: torch.Tensor | None = None) -> _Context:
    data = forward_kinematics(model, q, compute_frames=True)
    if indices is None:
        indices = torch.arange(model.nv)
    return _Context({"q": q, "data": data}, temporal_indices=indices)


@pytest.mark.parametrize(
    ("kind", "expected"),
    (
        ("velocity", TemporalPattern(3, 2, 1, (-1, 1))),
        ("acceleration", TemporalPattern(3, 2, 1, (-1, 0, 1))),
        ("reference", TemporalPattern(5, 2, 0, (0,))),
    ),
)
def test_smoothness_and_reference_named_blocks_match_dense_oracle(
    two_joint_model,
    kind: str,
    expected: TemporalPattern,
) -> None:
    model = two_joint_model
    q = _trajectory(model)
    ctx = _context(model, q)
    if kind == "velocity":
        residual = VelocityResidual(model, dt=0.1, weight=0.7, horizon=5)
    elif kind == "acceleration":
        residual = AccelerationResidual(model, dt=0.1, weight=0.4, horizon=5)
    else:
        residual = ReferenceTrajectoryResidual(
            model,
            model.q_neutral.expand(5, -1).clone(),
            weight=0.3,
            weight_per_frame=torch.linspace(0.5, 1.0, 5, dtype=q.dtype),
        )

    assert residual.reads == ("q",)
    assert residual.temporal_structure("q") == expected
    assert residual.temporal_structure("other") is None
    result = residual(ctx)
    assert result.shape == (*q.shape[:-2], residual.dim)

    temporal = residual.temporal_jacobian_blocks(ctx, "q")
    assert tuple(temporal) == expected.offsets
    assert all(
        block.shape == (*q.shape[:-2], expected.rows, expected.row_width, model.nv) for block in temporal.values()
    )
    densified = dense_temporal_jacobian(expected, temporal, horizon=5)
    torch.testing.assert_close(residual.jacobian_blocks(ctx)["q"], densified)


@pytest.mark.parametrize("kind", ("velocity", "acceleration", "reference"))
def test_temporal_blocks_apply_one_knot_mask_reduction(two_joint_model, kind: str) -> None:
    model = two_joint_model
    q = _trajectory(model, batch_shape=())
    full_ctx = _context(model, q)
    reduced_ctx = _context(model, q, torch.tensor([1]))
    if kind == "velocity":
        residual = VelocityResidual(model, dt=0.2, horizon=5)
    elif kind == "acceleration":
        residual = AccelerationResidual(model, dt=0.2, horizon=5)
    else:
        residual = ReferenceTrajectoryResidual(model, model.q_neutral.expand(5, -1).clone())

    full = residual.jacobian_blocks(full_ctx)["q"]
    reduced = residual.jacobian_blocks(reduced_ctx)["q"]
    selected = torch.tensor([knot * model.nv + 1 for knot in range(5)])
    torch.testing.assert_close(reduced, full.index_select(-1, selected))


def test_constant_temporal_blocks_honor_create_graph(two_joint_model) -> None:
    model = two_joint_model
    q = _trajectory(model, batch_shape=()).requires_grad_()
    ctx = _context(model, q)
    residual = VelocityResidual(model, dt=0.1, horizon=5)
    block = residual.temporal_jacobian_blocks(ctx, "q")[-1]
    assert block.requires_grad
    gradient = torch.autograd.grad(block.sum(), q)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(q))


def test_time_indexed_named_mapping_slices_arbitrary_batches(two_joint_model) -> None:
    model = two_joint_model
    q = _trajectory(model)
    ctx = _context(model, q)
    frame_id = model.frame_id("second_tip")
    target = forward_kinematics(
        model,
        model.q_neutral,
        compute_frames=True,
    ).frame_pose_world[frame_id]
    inner = PositionResidual(frame_id=frame_id, target=target, model=model)
    residual = TimeIndexedResidual(inner, 2, horizon=5)

    assert residual.reads == ("q", "data")
    pattern = TemporalPattern(1, 3, 2, (0,))
    assert residual.temporal_structure("q") == pattern
    result = residual(ctx)
    assert result.shape == (*q.shape[:-2], 3)
    temporal = residual.temporal_jacobian_blocks(ctx, "q")
    assert temporal[0].shape == (*q.shape[:-2], 1, 3, model.nv)
    dense = dense_temporal_jacobian(pattern, temporal, horizon=5)
    torch.testing.assert_close(residual.jacobian_blocks(ctx)["q"], dense)


def test_contact_named_blocks_match_dense(two_joint_model) -> None:
    model = two_joint_model
    q = _trajectory(model)
    ctx = _context(model, q)
    frames = (model.frame_id("first_tip"), model.frame_id("second_tip"))
    weights = torch.tensor(
        [[1.0, 0.0], [1.0, 0.5], [0.7, 1.0], [0.2, 1.0], [0.0, 0.8]],
        dtype=q.dtype,
    )
    residual = ContactConsistencyResidual(model, frames, weights, dt=0.05, weight=0.3)
    pattern = TemporalPattern(4, 6, 0, (0, 1))

    assert residual.reads == ("q", "data")
    assert residual.temporal_structure("q") == pattern
    result = residual(ctx)
    assert result.shape == (*q.shape[:-2], residual.dim)
    temporal = residual.temporal_jacobian_blocks(ctx, "q")
    assert all(block.shape == (*q.shape[:-2], 4, 6, model.nv) for block in temporal.values())
    dense = dense_temporal_jacobian(pattern, temporal, horizon=5)
    torch.testing.assert_close(residual.jacobian_blocks(ctx)["q"], dense)


def test_contact_temporal_blocks_match_tangent_finite_difference(two_joint_model) -> None:
    model = two_joint_model
    q = _trajectory(model, batch_shape=())
    frames = (model.frame_id("first_tip"), model.frame_id("second_tip"))
    weights = torch.linspace(0.2, 1.0, 10, dtype=q.dtype).reshape(5, 2)
    residual = ContactConsistencyResidual(model, frames, weights, dt=0.1, weight=0.6)
    analytic = residual.jacobian_blocks(_context(model, q))["q"]
    finite_difference = torch.zeros_like(analytic)
    eps = 1e-6
    for knot in range(5):
        for coordinate in range(model.nv):
            delta = torch.zeros(5, model.nv, dtype=q.dtype)
            delta[knot, coordinate] = eps
            q_plus = model.integrate(q, delta)
            q_minus = model.integrate(q, -delta)
            plus = residual(_context(model, q_plus))
            minus = residual(_context(model, q_minus))
            finite_difference[:, knot * model.nv + coordinate] = (plus - minus) / (2.0 * eps)
    torch.testing.assert_close(analytic, finite_difference, rtol=2e-5, atol=2e-7)


def test_static_horizon_validation_is_eager_and_required(two_joint_model) -> None:
    model = two_joint_model
    with pytest.raises(ValueError, match="at least 3"):
        VelocityResidual(model, dt=0.1, horizon=2)
    with pytest.raises(ValueError, match="at least 3"):
        AccelerationResidual(model, dt=0.1, horizon=2)
    with pytest.raises(ValueError, match="positive"):
        TimeIndexedResidual(PositionResidual(frame_id=0, target=torch.zeros(7)), 0, horizon=0)
    with pytest.raises(ValueError, match="out of range"):
        TimeIndexedResidual(PositionResidual(frame_id=0, target=torch.zeros(7)), 3, horizon=3)
    with pytest.raises(ValueError, match="at least one timestep"):
        ReferenceTrajectoryResidual(model, torch.empty(0, model.nq))
    with pytest.raises(ValueError, match="at least two timesteps"):
        ContactConsistencyResidual(
            model,
            (model.frame_id("first_tip"),),
            torch.ones(1, 1),
            dt=0.1,
        )

    q = _trajectory(model, batch_shape=())
    with pytest.raises(ValueError, match="horizon"):
        VelocityResidual(model, dt=0.1)(_context(model, q))


def test_time_indexed_without_analytic_inner_falls_back_to_ad(two_joint_model) -> None:
    model = two_joint_model
    q = _trajectory(model, batch_shape=())

    class QuadraticResidual:
        name = "quadratic"
        dim = 1
        reads = ("q",)

        def __call__(self, ctx):
            return ctx["q"][..., :1].square()

    residual = TimeIndexedResidual(QuadraticResidual(), 2, horizon=5)
    problem = Problem(
        vars=(VarSpec("q", (5, model.nq), RobotConfig(model), time_axis=0),),
        residuals=(ResidualItem(residual.name, residual),),
    )

    auto = problem.jacobian_blocks({"q": q}, strategy="auto")[(residual.name, "q")]
    finite_difference = problem.jacobian_blocks(
        {"q": q},
        strategy="finite_difference",
        fd_eps=1e-6,
    )[(residual.name, "q")]
    torch.testing.assert_close(auto, finite_difference, rtol=2e-5, atol=2e-7)
