"""Shared point-cloud signed-distance state and reusable penalty residuals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch

from .._validation import check_tensor
from ._point_cloud import _detached_nearest, _gather_rows, _validate_clouds
from .utils import VariableLike as _VariableLike, value, variables
from .base import Residual, Weight
from .nodes import Node


@dataclass(frozen=True)
class SceneSDFResult:
    """Fixed-shape outputs shared by all scene signed-distance heads."""

    signed_distance: torch.Tensor
    dmin: torch.Tensor
    confidence: torch.Tensor
    has_point: torch.Tensor


class SceneSDFState(Node):
    """Approximate point-cloud SDF cached for one evaluation scope.

    This multi-input node is never identity-merged. Reuse is explicit: pass
    the same :class:`SceneSDFState` object to every penalty residual that
    should share one detached nearest-neighbour computation.
    """

    def __init__(
        self,
        query_points: _VariableLike | torch.Tensor,
        query_validity: _VariableLike | torch.Tensor,
        scene_points: _VariableLike | torch.Tensor,
        scene_normals: _VariableLike | torch.Tensor,
        scene_validity: _VariableLike | torch.Tensor,
        *,
        scene_confidence: _VariableLike | torch.Tensor | None = None,
        chunk_size: int = 4096,
        eps: float = 1e-8,
    ) -> None:
        self.query_points, self.query_validity = query_points, query_validity
        self.scene_points, self.scene_normals = scene_points, scene_normals
        self.scene_validity, self.scene_confidence = scene_validity, scene_confidence
        query, _scene, _query_validity, _scene_validity, _normals, _confidence = self._inputs()
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
        if not isinstance(eps, float) or not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be a positive finite float, got {eps!r}")

        self.frames = int(query.shape[-3])
        self.points = int(query.shape[-2])
        self.chunk_size = chunk_size
        self.eps = eps
        super().__init__(
            *variables(
                query_points,
                query_validity,
                scene_points,
                scene_normals,
                scene_validity,
                scene_confidence,
            )
        )

    def _inputs(self):
        query, scene, query_validity, scene_validity = _validate_clouds(
            value(self.query_points, "query_points"),
            value(self.scene_points, "scene_points"),
            value(self.query_validity, "query_validity"),
            value(self.scene_validity, "scene_validity"),
        )
        if query.ndim < 3:
            raise ValueError(f"query_points must include a frames axis, got {tuple(query.shape)}")
        normals = check_tensor(
            "scene_normals",
            value(self.scene_normals, "scene_normals"),
            shape=tuple(scene.shape[-2:]),
            floating=True,
            dtype=query.dtype,
            device=query.device,
        )
        confidence = None
        if self.scene_confidence is not None:
            confidence = check_tensor(
                "scene_confidence",
                value(self.scene_confidence, "scene_confidence"),
                shape=(scene.shape[-2],),
                floating=True,
                dtype=query.dtype,
                device=query.device,
            )
        return query, scene, query_validity, scene_validity, normals, confidence

    def compute(self) -> SceneSDFResult:
        query, scene, query_validity, scene_validity, normals, scene_confidence = self._inputs()
        correspondence = _detached_nearest(
            query,
            scene,
            query_validity=query_validity,
            reference_validity=scene_validity,
            chunk_size=self.chunk_size,
        )
        nearest_normal = _gather_rows(normals, correspondence.index)
        nearest_normal = torch.where(
            correspondence.valid.unsqueeze(-1),
            nearest_normal,
            torch.zeros_like(nearest_normal),
        )
        epsilon = max(self.eps, torch.finfo(query.dtype).eps)
        normal_norm = torch.linalg.vector_norm(nearest_normal, dim=-1, keepdim=True)
        unit_normal = nearest_normal / normal_norm.clamp_min(epsilon)
        normal_projection = (correspondence.delta * unit_normal).sum(dim=-1)
        sign = torch.where(
            normal_projection.detach() < 0.0,
            -torch.ones_like(normal_projection),
            torch.ones_like(normal_projection),
        )
        signed_distance = correspondence.distance * sign
        confidence = normal_projection.abs() / correspondence.distance.clamp_min(epsilon)
        confidence = confidence.clamp(min=0.0, max=1.0)

        if scene_confidence is not None:
            gathered = _gather_rows(scene_confidence.unsqueeze(-1), correspondence.index).squeeze(-1)
            gathered = torch.where(correspondence.valid, gathered, torch.zeros_like(gathered))
            confidence = confidence * gathered.clamp(min=0.0, max=1.0)

        valid = correspondence.valid & (normal_norm.squeeze(-1) > epsilon)
        zeros = torch.zeros_like(correspondence.distance)
        return SceneSDFResult(
            signed_distance=torch.where(valid, signed_distance, zeros),
            dmin=torch.where(valid, correspondence.distance, zeros),
            confidence=torch.where(valid, confidence, zeros),
            has_point=valid,
        )


class _ScenePenaltyResidual(Residual):
    def __init__(
        self,
        state: SceneSDFState,
        *,
        weight: Real | torch.Tensor,
        row_weight: Weight | Real | torch.Tensor,
        kernel: object | None,
        name: str,
    ) -> None:
        if not isinstance(state, SceneSDFState):
            raise TypeError(f"state must be a SceneSDFState, got {type(state).__name__}")
        self.state = state
        self.nodes = (state,)
        self.frames = state.frames
        self.points = state.points
        super().__init__(
            dim=self.frames * self.points,
            weight=weight,
            row_weight=row_weight,
            kernel=kernel,
            name=name,
        )

    def _result(self) -> SceneSDFResult:
        result = self.state._checked_value(SceneSDFResult)
        suffix = (self.frames, self.points)
        for label, tensor in (
            ("signed_distance", result.signed_distance),
            ("dmin", result.dmin),
            ("confidence", result.confidence),
            ("has_point", result.has_point),
        ):
            check_tensor(f"SceneSDFResult.{label}", tensor, shape=suffix)
        check_tensor("SceneSDFResult.has_point", result.has_point, dtype=torch.bool)
        return result

    def _finish(self, result: SceneSDFResult, penalty: torch.Tensor) -> torch.Tensor:
        weighted = penalty * result.confidence
        weighted = torch.where(result.has_point, weighted, torch.zeros_like(weighted))
        return weighted.reshape(*weighted.shape[:-2], self.dim)


class ScenePenetrationResidual(_ScenePenaltyResidual):
    """Positive signed depth for query points behind the scene surface."""

    def __init__(
        self,
        state: SceneSDFState,
        *,
        weight: Real | torch.Tensor = 1.0,
        row_weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str = "scene_penetration",
    ) -> None:
        super().__init__(state, weight=weight, row_weight=row_weight, kernel=kernel, name=name)

    def error(self) -> torch.Tensor:
        result = self._result()
        return self._finish(result, torch.relu(-result.signed_distance))


class SceneAttractionResidual(_ScenePenaltyResidual):
    """Absolute signed-distance error toward a requested surface offset."""

    def __init__(
        self,
        state: SceneSDFState,
        *,
        target_distance: float = 0.0,
        weight: Real | torch.Tensor = 1.0,
        row_weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str = "scene_attraction",
    ) -> None:
        if not isinstance(target_distance, float):
            raise TypeError("target_distance must be float")
        self.target_distance = target_distance
        super().__init__(state, weight=weight, row_weight=row_weight, kernel=kernel, name=name)

    def error(self) -> torch.Tensor:
        result = self._result()
        penalty = (result.signed_distance - self.target_distance).abs()
        return self._finish(result, penalty)


class SceneClearanceResidual(_ScenePenaltyResidual):
    """One-sided excess above an allowed positive signed clearance."""

    def __init__(
        self,
        state: SceneSDFState,
        *,
        clearance: float,
        weight: Real | torch.Tensor = 1.0,
        row_weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str = "scene_clearance",
    ) -> None:
        if not isinstance(clearance, float) or clearance < 0.0:
            raise ValueError("clearance must be a non-negative float")
        self.clearance = clearance
        super().__init__(state, weight=weight, row_weight=row_weight, kernel=kernel, name=name)

    def error(self) -> torch.Tensor:
        result = self._result()
        return self._finish(result, torch.relu(result.signed_distance - self.clearance))


__all__ = [
    "SceneAttractionResidual",
    "SceneClearanceResidual",
    "ScenePenetrationResidual",
    "SceneSDFResult",
    "SceneSDFState",
]
