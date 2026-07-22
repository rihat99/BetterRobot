"""Shared scene-SDF state value, caching, and detached-gradient tests."""

from __future__ import annotations

import torch

from better_robot.optim.problem import Problem
from better_robot.optim.variables import Variable
from better_robot.residuals import scene_sdf as scene_sdf_module
from better_robot.residuals.scene_sdf import (
    SceneAttractionResidual,
    SceneClearanceResidual,
    ScenePenetrationResidual,
    SceneSDFState,
)
from better_robot.residuals.nodes import Node


FRAMES = 2
QUERIES = 3


def _scene_data() -> dict[str, torch.Tensor]:
    query = torch.tensor(
        [
            [[0.0, -0.2, 0.0], [1.0, 0.3, 0.0], [2.0, 0.05, 0.0]],
            [[0.0, -0.4, 0.0], [1.0, 0.2, 0.0], [2.0, 0.1, 0.0]],
        ]
    )
    scene = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ]
    )
    normals = torch.zeros_like(scene)
    normals[..., 1] = 1.0
    return {
        "query": query,
        "query_validity": torch.ones(FRAMES, QUERIES, dtype=torch.bool),
        "scene": scene,
        "normals": normals,
        "scene_validity": torch.tensor([[True, True, True], [False, False, False]]),
        "confidence": torch.tensor([[1.0, 0.5, 0.25], [1.0, 1.0, 1.0]]),
    }


def _state(data: dict[str, torch.Tensor], query: torch.Tensor | Variable | None = None) -> SceneSDFState:
    return SceneSDFState(
        data["query"] if query is None else query,
        data["query_validity"],
        data["scene"],
        data["normals"],
        data["scene_validity"],
        scene_confidence=data["confidence"],
        chunk_size=1,
    )


class _PosedPoints(Node):
    def __init__(self, q: Variable) -> None:
        self.q = q
        super().__init__(q)

    def compute(self) -> torch.Tensor:
        return self.q.tensor


def test_scene_sdf_values_and_three_penalty_heads_match_hand_computation() -> None:
    state = _state(_scene_data())
    result = state.value()
    expected_signed = torch.tensor([[-0.2, 0.3, 0.05], [0.0, 0.0, 0.0]])
    expected_confidence = torch.tensor([[1.0, 0.5, 0.25], [0.0, 0.0, 0.0]])

    torch.testing.assert_close(result.signed_distance, expected_signed)
    torch.testing.assert_close(result.dmin, expected_signed.abs())
    torch.testing.assert_close(result.confidence, expected_confidence)
    assert result.has_point.tolist() == [[True, True, True], [False, False, False]]
    torch.testing.assert_close(
        ScenePenetrationResidual(state).error(),
        torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        SceneAttractionResidual(state).error(),
        torch.tensor([0.2, 0.3 * 0.5**0.5, 0.025, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        SceneClearanceResidual(state, clearance=0.1).error(),
        torch.tensor([0.0, 0.2 * 0.5**0.5, 0.0, 0.0, 0.0, 0.0]),
    )


def test_point_and_plane_distance_agree_along_normal_and_differ_tangentially() -> None:
    query = torch.tensor([[[0.0, 0.2, 0.0], [0.3, 0.2, 0.0]]])
    validity = torch.ones(1, 2, dtype=torch.bool)
    scene = torch.zeros(1, 1, 3)
    scene_validity = torch.ones(1, 1, dtype=torch.bool)
    normals = torch.tensor([[[0.0, 1.0, 0.0]]])

    point = SceneSDFState(query, validity, scene, normals, scene_validity).value()
    plane = SceneSDFState(query, validity, scene, normals, scene_validity, distance="plane").value()

    torch.testing.assert_close(point.signed_distance[..., 0], plane.signed_distance[..., 0])
    assert plane.signed_distance[..., 1].abs().item() < point.signed_distance[..., 1].abs().item()
    torch.testing.assert_close(plane.signed_distance, torch.tensor([[0.2, 0.2]]))


def test_scene_penalty_gates_are_detached_and_rows_match_the_active_set() -> None:
    state = _state(_scene_data())
    attraction = SceneAttractionResidual(
        state,
        min_confidence=0.3,
        max_distance=0.25,
        band=0.25,
        reduce="mean_active",
    )
    penetration = ScenePenetrationResidual(state, max_penetration=0.25, margin=0.1)
    deep_rejected = ScenePenetrationResidual(state, max_penetration=0.15)
    clearance = SceneClearanceResidual(state, clearance=0.1, margin=0.05)

    active = attraction.active_groups()
    assert not active.requires_grad
    assert active.tolist() == [True, False, False, False, False, False]
    assert attraction.reduce == "mean_active"
    torch.testing.assert_close(attraction.error(), torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(penetration.error(), torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert deep_rejected.active_groups().tolist() == [False, True, True, False, False, False]
    torch.testing.assert_close(
        clearance.error(),
        torch.tensor([0.0, 0.15 * 0.5**0.5, 0.0, 0.0, 0.0, 0.0]),
    )


def test_scene_confidence_is_linear_in_cost_and_safe_at_zero() -> None:
    query = Variable(
        torch.tensor([[[0.0, -0.2, 0.0], [1.0, -0.2, 0.0], [2.0, -0.2, 0.0]]], requires_grad=True),
        name="query",
    )
    scene = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    normals = torch.zeros_like(scene)
    normals[..., 1] = 1.0
    state = SceneSDFState(
        query,
        torch.ones(1, 3, dtype=torch.bool),
        scene,
        normals,
        torch.ones(1, 3, dtype=torch.bool),
        scene_confidence=torch.tensor([[0.0, 0.25, 1.0]]),
    )
    item = SceneAttractionResidual(state)
    problem = Problem([item])

    torch.testing.assert_close(item.error(), torch.tensor([0.0, 0.1, 0.2]))
    cost = problem.objective()
    gradient = torch.autograd.grad(cost, query.tensor)[0]
    torch.testing.assert_close(cost, torch.tensor(0.025))
    torch.testing.assert_close(gradient[..., 0, :], torch.zeros(1, 3))
    assert torch.isfinite(gradient).all()


def test_update_reaches_node_query_candidate_scene_and_external_bool_mask() -> None:
    query = Variable(torch.tensor([[[0.0, -0.2, 0.0], [1.0, 0.3, 0.0]]]), name="q")
    scene = Variable(torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]), name="scene", trainable=False)
    validity = Variable(torch.ones(1, 2, dtype=torch.bool), name="validity", trainable=False)
    mask = Variable(torch.ones(1, 2, dtype=torch.bool), name="contact_mask", trainable=False)
    normals = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    state = SceneSDFState(_PosedPoints(query), validity, scene, normals, validity)
    item = ScenePenetrationResidual(state, mask=mask, reduce="mean_active")
    problem = Problem([item])

    torch.testing.assert_close(problem.error(), torch.tensor([0.2, 0.0]))
    torch.testing.assert_close(problem.objective(), torch.tensor(0.01))
    problem.update(
        {
            "scene": torch.tensor([[[0.0, -0.1, 0.0], [1.0, 0.2, 0.0]]]),
            "contact_mask": torch.tensor([[True, False]]),
        }
    )
    torch.testing.assert_close(problem.error(), torch.tensor([0.1, 0.0]))
    torch.testing.assert_close(problem.objective(), torch.tensor(0.005))


def test_one_scene_state_pass_feeds_all_three_heads_per_problem_evaluation(monkeypatch) -> None:
    data = _scene_data()
    query = Variable(data["query"], name="scene_query_points")
    state = _state(data, query)
    calls = 0
    real_nearest = scene_sdf_module._detached_nearest

    def counted_nearest(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_nearest(*args, **kwargs)

    monkeypatch.setattr(scene_sdf_module, "_detached_nearest", counted_nearest)
    problem = Problem(
        [
            ScenePenetrationResidual(state),
            SceneAttractionResidual(state),
            SceneClearanceResidual(state, clearance=0.1),
        ]
    )

    rows = problem.error()
    assert rows.shape == (3 * FRAMES * QUERIES,)
    assert calls == 1
    gradient = problem.gradient()[query.name]
    assert gradient.shape == (FRAMES * QUERIES * 3,)
    assert torch.isfinite(gradient).all()
    assert calls == 2


def test_scene_sdf_gradient_reaches_only_the_detached_nearest_match() -> None:
    query = torch.tensor([[[0.0, -0.2, 0.0]]], requires_grad=True)
    scene = torch.tensor([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]], requires_grad=True)
    state = SceneSDFState(
        query,
        torch.ones(1, 1, dtype=torch.bool),
        scene,
        torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]),
        torch.ones(1, 2, dtype=torch.bool),
        chunk_size=1,
    )
    value = ScenePenetrationResidual(state).error()
    query_gradient, scene_gradient = torch.autograd.grad(value.sum(), (query, scene))

    torch.testing.assert_close(value, torch.tensor([0.2]))
    torch.testing.assert_close(query_gradient, torch.tensor([[[0.0, -1.0, 0.0]]]))
    torch.testing.assert_close(scene_gradient[0, 0], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(scene_gradient[0, 1], torch.zeros(3))


def test_scene_sdf_masked_nan_padding_has_zero_value_and_gradients() -> None:
    query = torch.zeros((1, 1, 3), requires_grad=True)
    scene = torch.full((1, 1, 3), torch.nan, requires_grad=True)
    normals = torch.full((1, 1, 3), torch.nan, requires_grad=True)
    confidence = torch.full((1, 1), torch.nan, requires_grad=True)
    state = SceneSDFState(
        query,
        torch.ones(1, 1, dtype=torch.bool),
        scene,
        normals,
        torch.zeros(1, 1, dtype=torch.bool),
        scene_confidence=confidence,
        chunk_size=1,
    )
    result = state.value()
    value = SceneAttractionResidual(state).error()
    query_gradient, scene_gradient, normal_gradient, confidence_gradient = torch.autograd.grad(
        value.sum(), (query, scene, normals, confidence), allow_unused=True
    )

    torch.testing.assert_close(result.signed_distance, torch.zeros(1, 1))
    torch.testing.assert_close(result.dmin, torch.zeros(1, 1))
    torch.testing.assert_close(result.confidence, torch.zeros(1, 1))
    assert not result.has_point.any()
    torch.testing.assert_close(value, torch.zeros(1))
    for gradient in (query_gradient, scene_gradient):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))
    assert normal_gradient is None
    assert confidence_gradient is None


def test_scene_sdf_forward_and_reverse_tangent_jacobians_match() -> None:
    data = _scene_data()
    query = Variable(data["query"], name="scene_query_points")
    problem = Problem([ScenePenetrationResidual(_state(data, query))])

    reverse = problem.dense_jacobian(strategy="jacrev")
    forward = problem.dense_jacobian(strategy="jacfwd")
    torch.testing.assert_close(forward, reverse, rtol=2e-5, atol=2e-6)
