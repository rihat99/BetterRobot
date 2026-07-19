"""Focused numerical contracts for the second v2 residual migration group."""

from __future__ import annotations

import inspect
import math

import torch

from better_robot.io import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.lie import so3
from better_robot.optim.problem import Problem
from better_robot.optim.variables import RobotVariable, Variable
from better_robot.residuals.chamfer import MaskedChamferResidual
from better_robot.residuals.contact import ContactConsistencyResidual
from better_robot.residuals.human import SwingTwistLimitResidual
from better_robot.residuals.projection import ProjectionResidual
from better_robot.residuals import scene_sdf as scene_sdf_module
from better_robot.residuals.scene_sdf import (
    SceneAttractionResidual,
    SceneClearanceResidual,
    ScenePenetrationResidual,
    SceneSDFState,
)


def _camera_arm(*, dtype: torch.dtype = torch.float32):
    identity = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=dtype)
    builder = ModelBuilder("camera_arm_v2")
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


def _two_spherical_model():
    builder = ModelBuilder("two_spherical_v2")
    base = builder.add_body("base")
    first = builder.add_body("first", mass=1.0)
    second = builder.add_body("second", mass=1.0)
    builder.add_spherical("first_ball", parent=base, child=first)
    builder.add_spherical("second_ball", parent=first, child=second)
    return build_model(builder.finalize()).to(dtype=torch.float64)


def _temporal_model():
    pose = torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    builder = ModelBuilder("contact_v2")
    builder.add_body("base", mass=1.0)
    builder.add_body("first", mass=1.0)
    builder.add_body("second", mass=1.0)
    builder.add_revolute_z("joint_1", parent="base", child="first", origin=pose)
    builder.add_revolute_y("joint_2", parent="first", child="second", origin=pose)
    builder.add_frame("first_tip", parent_body="first", placement=pose)
    builder.add_frame("second_tip", parent_body="second", placement=pose)
    return build_model(builder.finalize(), dtype=torch.float64)


def test_chamfer_preserves_domain_weights_correspondence_gradient_and_outer_weight() -> None:
    source_tensor = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    target = torch.tensor([[[0.0, 1.0, 0.0], [10.0, 0.0, 0.0], [2.0, 0.1, 0.0]]])
    source = Variable(source_tensor, name="source")
    item = MaskedChamferResidual(
        source,
        target,
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[True, True, False]]),
        vertex_weights=torch.tensor([[2.0, 0.5]]),
        chunk_size=1,
        weight=0.25,
    )

    raw = item.error()
    torch.testing.assert_close(raw, torch.tensor([2.0, 0.5 * 5.0**0.5, 1.0, 8.0, 0.0]))
    torch.testing.assert_close(item.weighted_error(), raw * 0.25)
    gradient = torch.autograd.grad(raw.sum(), source_tensor)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_swing_twist_is_a_grouped_robot_variable_residual_and_matches_tangent_fd() -> None:
    model = _two_spherical_model()
    q = RobotVariable(model, model.q_neutral.clone(), name="q")
    item = SwingTwistLimitResidual(
        q,
        (3,),
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        0.4,
        (-0.3, 0.5),
        weight=0.5,
    )
    start = model.idx_qs[3]
    value = q.tensor.clone()
    value[start : start + 4] = so3.exp(torch.tensor([0.7, -0.2, 0.8], dtype=torch.float64))
    q.tensor = value
    problem = Problem([item])

    assert item.variables == (q,)
    assert item.group_size == 3
    torch.testing.assert_close(item.weighted_error(), item.error() * 0.5)
    reverse = problem.dense_jacobian(strategy="jacrev")
    finite_difference = problem.dense_jacobian(strategy="finite_difference")
    torch.testing.assert_close(reverse, finite_difference, atol=3e-5, rtol=3e-4)


def test_projection_preserves_confidence_grouping_and_analytic_q_block() -> None:
    model = _camera_arm()
    intrinsics, extrinsics = _camera_tensors(torch.float32)
    q = RobotVariable(model, torch.tensor([0.0]), name="q")
    target = Variable(
        torch.tensor([[300.0, 200.0], [300.0, 200.0]]),
        name="target_px",
        trainable=False,
    )
    item = ProjectionResidual(
        q,
        (model.frame_id("body_link"), model.frame_id("marker")),
        intrinsics,
        extrinsics,
        target,
        weights=torch.tensor([0.25, 0.75]),
        valid_mask=torch.tensor([True, False]),
        weight=2.0,
    )
    problem = Problem([item])

    torch.testing.assert_close(item.error(), torch.tensor([5.0, 10.0, 0.0, 0.0]))
    torch.testing.assert_close(problem.error(), torch.tensor([10.0, 20.0, 0.0, 0.0]))
    assert item.group_size == 2
    analytic = problem.dense_jacobian(strategy="analytic")
    reverse = problem.dense_jacobian(strategy="jacrev")
    torch.testing.assert_close(analytic, reverse, atol=1e-3, rtol=1e-4)


def test_projection_static_target_is_graph_visible_and_invalid_nan_rows_are_zero() -> None:
    model = _camera_arm(dtype=torch.float64)
    intrinsics, extrinsics = _camera_tensors(torch.float64)
    q = RobotVariable(model, torch.tensor([0.2], dtype=torch.float64), name="q")
    target_tensor = torch.tensor(
        [[300.0, 200.0], [315.0, 215.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = Variable(target_tensor, name="target", trainable=False)
    item = ProjectionResidual(
        q,
        (model.frame_id("body_link"), model.frame_id("marker")),
        intrinsics,
        extrinsics,
        target,
    )
    problem = Problem([item])

    gradient = problem.gradient(create_graph=True)["q"]
    target_vjp = torch.autograd.grad(gradient.sum(), target_tensor)[0]
    assert torch.isfinite(target_vjp).all()
    assert torch.count_nonzero(target_vjp) > 0

    invalid = ProjectionResidual(
        q,
        (model.frame_id("body_link"), model.frame_id("marker")),
        intrinsics,
        extrinsics,
        torch.full((2, 2), torch.nan, dtype=torch.float64),
        valid_mask=torch.zeros(2, dtype=torch.bool),
    )
    torch.testing.assert_close(invalid.error(), torch.zeros(4, dtype=torch.float64))


def _scene_inputs(*, requires_grad: bool = False) -> tuple[Variable, dict[str, torch.Tensor]]:
    query = torch.tensor(
        [
            [[0.0, -0.2, 0.0], [1.0, 0.3, 0.0], [2.0, 0.05, 0.0]],
            [[0.0, -0.4, 0.0], [1.0, 0.2, 0.0], [2.0, 0.1, 0.0]],
        ],
        requires_grad=requires_grad,
    )
    scene = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ]
    )
    normals = torch.zeros_like(scene)
    normals[..., 1] = 1.0
    return Variable(query, name="query"), {
        "query_validity": torch.ones(2, 3, dtype=torch.bool),
        "scene": scene,
        "normals": normals,
        "scene_validity": torch.tensor([[True, True, True], [False, False, False]]),
        "confidence": torch.tensor([[1.0, 0.5, 0.25], [1.0, 1.0, 1.0]]),
    }


def test_scene_state_matches_values_and_is_shared_once_by_explicit_identity(monkeypatch) -> None:
    query, values = _scene_inputs()
    state = SceneSDFState(
        query,
        values["query_validity"],
        values["scene"],
        values["normals"],
        values["scene_validity"],
        scene_confidence=values["confidence"],
        chunk_size=1,
    )
    calls = 0
    nearest = scene_sdf_module._detached_nearest

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return nearest(*args, **kwargs)

    monkeypatch.setattr(scene_sdf_module, "_detached_nearest", counted)
    heads = (
        ScenePenetrationResidual(state),
        SceneAttractionResidual(state),
        SceneClearanceResidual(state, clearance=0.1),
    )
    problem = Problem(heads)
    rows = problem.error().reshape(3, 2, 3)

    assert calls == 1
    torch.testing.assert_close(rows[0], torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    torch.testing.assert_close(rows[1], torch.tensor([[0.2, 0.15, 0.0125], [0.0, 0.0, 0.0]]))
    torch.testing.assert_close(rows[2], torch.tensor([[0.0, 0.1, 0.0], [0.0, 0.0, 0.0]]))
    gradient = problem.gradient()["query"]
    assert torch.isfinite(gradient).all()
    assert calls == 2


def test_scene_equivalent_distinct_nodes_do_not_merge(monkeypatch) -> None:
    query, values = _scene_inputs()

    def make_state() -> SceneSDFState:
        return SceneSDFState(
            query,
            values["query_validity"],
            values["scene"],
            values["normals"],
            values["scene_validity"],
        )

    calls = 0
    nearest = scene_sdf_module._detached_nearest

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return nearest(*args, **kwargs)

    monkeypatch.setattr(scene_sdf_module, "_detached_nearest", counted)
    problem = Problem(
        [
            ScenePenetrationResidual(make_state(), name="first"),
            ScenePenetrationResidual(make_state(), name="second"),
        ]
    )
    problem.error()
    assert calls == 2


def test_scene_forward_and_reverse_tangent_jacobians_match() -> None:
    query, values = _scene_inputs()
    state = SceneSDFState(
        query,
        values["query_validity"],
        values["scene"],
        values["normals"],
        values["scene_validity"],
        scene_confidence=values["confidence"],
        chunk_size=1,
    )
    problem = Problem([ScenePenetrationResidual(state)])

    reverse = problem.dense_jacobian(strategy="jacrev")
    forward = problem.dense_jacobian(strategy="jacfwd")
    torch.testing.assert_close(forward, reverse, rtol=2e-5, atol=2e-6)


def test_contact_value_and_analytic_rows_match_tangent_finite_difference() -> None:
    model = _temporal_model()
    torch.manual_seed(8)
    neutral = model.q_neutral.expand(5, model.nq).clone()
    trajectory = model.integrate(neutral, torch.randn(5, model.nv, dtype=torch.float64) * 0.03)
    q = RobotVariable(model, trajectory, name="q", time_axis=0)
    frames = (model.frame_id("first_tip"), model.frame_id("second_tip"))
    contacts = torch.linspace(0.2, 1.0, 10, dtype=torch.float64).reshape(5, 2)
    item = ContactConsistencyResidual(q, frames, contacts, dt=0.1, weight=0.6)
    problem = Problem([item])

    assert item.group_size == 3
    assert item.error().shape == (item.dim,)
    assert item.temporal_structure("q") is not None
    analytic = problem.dense_jacobian(strategy="analytic")
    finite_difference = problem.dense_jacobian(strategy="finite_difference")
    torch.testing.assert_close(analytic, finite_difference, rtol=2e-5, atol=2e-7)
    assert "angular" not in inspect.signature(ContactConsistencyResidual).parameters
