"""Shared scene-SDF provider value, caching, and detached-gradient tests."""

from __future__ import annotations

import torch

from better_robot.optim import Problem, ResidualItem, VarSpec
from better_robot.residuals import scene_sdf as scene_sdf_module
from better_robot.residuals.scene_sdf import (
    SceneAttractionResidual,
    SceneClearanceResidual,
    ScenePenetrationResidual,
    SceneSDFProvider,
)


FRAMES = 2
QUERIES = 3
SCENE_POINTS = 3


def _scene_data() -> dict[str, torch.Tensor]:
    query = torch.tensor(
        [
            [[0.0, -0.2, 0.0], [1.0, 0.3, 0.0], [2.0, 0.05, 0.0]],
            [[0.0, -0.4, 0.0], [1.0, 0.2, 0.0], [2.0, 0.1, 0.0]],
        ],
        dtype=torch.float32,
    )
    scene = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    normals = torch.zeros_like(scene)
    normals[..., 1] = 1.0
    return {
        "scene_query_points": query,
        "scene_query_validity": torch.ones(FRAMES, QUERIES, dtype=torch.bool),
        "scene_points": scene,
        "scene_normals": normals,
        "scene_validity": torch.tensor(
            [[True, True, True], [False, False, False]],
        ),
        "scene_confidence": torch.tensor(
            [[1.0, 0.5, 0.25], [1.0, 1.0, 1.0]],
        ),
    }


def _provider() -> SceneSDFProvider:
    return SceneSDFProvider(scene_confidence="scene_confidence", chunk_size=1)


def test_scene_sdf_values_and_three_penalty_heads_match_hand_computation() -> None:
    data = _scene_data()
    result = _provider()(data)["scene_sdf"]

    expected_signed = torch.tensor(
        [[-0.2, 0.3, 0.05], [0.0, 0.0, 0.0]],
    )
    expected_confidence = torch.tensor(
        [[1.0, 0.5, 0.25], [0.0, 0.0, 0.0]],
    )
    torch.testing.assert_close(result.signed_distance, expected_signed)
    torch.testing.assert_close(result.dmin, expected_signed.abs())
    torch.testing.assert_close(result.confidence, expected_confidence)
    assert result.has_point.tolist() == [[True, True, True], [False, False, False]]

    ctx = {"scene_sdf": result}
    penetration = ScenePenetrationResidual(FRAMES, QUERIES)(ctx)
    attraction = SceneAttractionResidual(FRAMES, QUERIES)(ctx)
    clearance = SceneClearanceResidual(FRAMES, QUERIES, clearance=0.1)(ctx)
    torch.testing.assert_close(
        penetration,
        torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        attraction,
        torch.tensor([0.2, 0.15, 0.0125, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        clearance,
        torch.tensor([0.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
    )


def test_one_scene_provider_pass_feeds_all_three_heads_per_problem_evaluation(monkeypatch) -> None:
    data = _scene_data()
    calls = 0
    real_nearest = scene_sdf_module._detached_nearest

    def counted_nearest(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_nearest(*args, **kwargs)

    monkeypatch.setattr(scene_sdf_module, "_detached_nearest", counted_nearest)
    residuals = (
        ScenePenetrationResidual(FRAMES, QUERIES),
        SceneAttractionResidual(FRAMES, QUERIES),
        SceneClearanceResidual(FRAMES, QUERIES, clearance=0.1),
    )
    problem = Problem(
        vars=(VarSpec("scene_query_points", (FRAMES, QUERIES, 3)),),
        residuals=tuple(ResidualItem(residual.name, residual) for residual in residuals),
        providers=(_provider(),),
        parameters={name: value for name, value in data.items() if name != "scene_query_points"},
    )
    values = {"scene_query_points": data["scene_query_points"]}

    rows = problem.residual(values)
    assert rows.shape == (3 * FRAMES * QUERIES,)
    assert calls == 1
    gradient = problem.gradient(values)["scene_query_points"]
    assert gradient.shape == (FRAMES * QUERIES * 3,)
    assert torch.isfinite(gradient).all()
    assert calls == 2


def test_scene_sdf_gradient_reaches_only_the_detached_nearest_match() -> None:
    query = torch.tensor([[[0.0, -0.2, 0.0]]], requires_grad=True)
    scene = torch.tensor(
        [[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    normals = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    provider = SceneSDFProvider(chunk_size=1)
    result = provider(
        {
            "scene_query_points": query,
            "scene_query_validity": torch.ones(1, 1, dtype=torch.bool),
            "scene_points": scene,
            "scene_normals": normals,
            "scene_validity": torch.ones(1, 2, dtype=torch.bool),
        }
    )["scene_sdf"]
    value = ScenePenetrationResidual(1, 1)({"scene_sdf": result})
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
    provider = SceneSDFProvider(scene_confidence="scene_confidence", chunk_size=1)

    result = provider(
        {
            "scene_query_points": query,
            "scene_query_validity": torch.ones(1, 1, dtype=torch.bool),
            "scene_points": scene,
            "scene_normals": normals,
            "scene_validity": torch.zeros(1, 1, dtype=torch.bool),
            "scene_confidence": confidence,
        }
    )["scene_sdf"]
    value = SceneAttractionResidual(1, 1)({"scene_sdf": result})
    gradients = torch.autograd.grad(
        value.sum(),
        (query, scene, normals, confidence),
    )

    torch.testing.assert_close(result.signed_distance, torch.zeros(1, 1))
    torch.testing.assert_close(result.dmin, torch.zeros(1, 1))
    torch.testing.assert_close(result.confidence, torch.zeros(1, 1))
    assert not result.has_point.any()
    torch.testing.assert_close(value, torch.zeros(1))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))


def test_scene_sdf_forward_and_reverse_tangent_jacobians_match() -> None:
    data = _scene_data()
    residual = ScenePenetrationResidual(FRAMES, QUERIES)
    problem = Problem(
        vars=(VarSpec("scene_query_points", (FRAMES, QUERIES, 3)),),
        residuals=(ResidualItem(residual.name, residual),),
        providers=(_provider(),),
        parameters={name: value for name, value in data.items() if name != "scene_query_points"},
    )
    values = {"scene_query_points": data["scene_query_points"]}

    reverse = problem.jacobian_blocks(values, strategy="jacrev")[(residual.name, "scene_query_points")]
    forward = problem.jacobian_blocks(values, strategy="jacfwd")[(residual.name, "scene_query_points")]
    torch.testing.assert_close(forward, reverse, rtol=2e-5, atol=2e-6)
