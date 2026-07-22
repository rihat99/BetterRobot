"""Shared point-cloud signed-distance state and reusable penalty residuals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal

import torch

from .._validation import check_tensor
from ._point_cloud import _detached_nearest, _gather_rows, _validate_clouds
from .utils import VariableLike as _VariableLike, value as _leaf_value
from .base import Residual, Weight
from .nodes import Node


@dataclass(frozen=True)
class SceneSDFResult:
    """Fixed-shape outputs shared by all scene signed-distance heads."""

    signed_distance: torch.Tensor
    dmin: torch.Tensor
    confidence: torch.Tensor
    has_point: torch.Tensor


_Input = _VariableLike | Node | torch.Tensor


def _value(source: _Input, name: str) -> torch.Tensor:
    if isinstance(source, Node):
        return check_tensor(name, source.value())
    return _leaf_value(source, name)


def _optional_nonnegative(value: float | None, name: str) -> float | None:
    if value is not None and (not isinstance(value, float) or not math.isfinite(value) or value < 0.0):
        raise ValueError(f"{name} must be a non-negative finite float or None")
    return value


class SceneSDFState(Node):
    """Approximate point-cloud SDF cached for one evaluation scope.

    Child Nodes are read once per evaluation epoch, static Variables are
    updatable through ``Problem.update()``, and bare tensors are captured at
    construction. This node is never identity-merged: pass the same instance
    to every residual that should share its detached nearest-neighbour pass.
    """

    def __init__(
        self,
        query_points: _VariableLike | Node | torch.Tensor,
        query_validity: _VariableLike | Node | torch.Tensor,
        scene_points: _VariableLike | Node | torch.Tensor,
        scene_normals: _VariableLike | Node | torch.Tensor,
        scene_validity: _VariableLike | Node | torch.Tensor,
        *,
        scene_confidence: _VariableLike | Node | torch.Tensor | None = None,
        distance: Literal["point", "plane"] = "point",
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
        if not isinstance(distance, str) or distance not in {"point", "plane"}:
            raise ValueError(f"distance must be 'point' or 'plane', got {distance!r}")

        self.frames = int(query.shape[-3])
        self.points = int(query.shape[-2])
        self.distance = distance
        self.chunk_size = chunk_size
        self.eps = eps
        sources = (
            query_points,
            query_validity,
            scene_points,
            scene_normals,
            scene_validity,
            scene_confidence,
        )
        super().__init__(*(source for source in sources if isinstance(source, (_VariableLike, Node))))

    def _inputs(self):
        query, scene, query_validity, scene_validity = _validate_clouds(
            _value(self.query_points, "query_points"),
            _value(self.scene_points, "scene_points"),
            _value(self.query_validity, "query_validity"),
            _value(self.scene_validity, "scene_validity"),
        )
        if query.ndim < 3:
            raise ValueError(f"query_points must include a frames axis, got {tuple(query.shape)}")
        normals = check_tensor(
            "scene_normals",
            _value(self.scene_normals, "scene_normals"),
            shape=tuple(scene.shape[-2:]),
            floating=True,
            dtype=query.dtype,
            device=query.device,
        )
        confidence = None
        if self.scene_confidence is not None:
            confidence = check_tensor(
                "scene_confidence",
                _value(self.scene_confidence, "scene_confidence"),
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
        signed_distance = correspondence.distance * sign if self.distance == "point" else normal_projection
        confidence = normal_projection.abs() / correspondence.distance.clamp_min(epsilon)
        confidence = confidence.clamp(min=0.0, max=1.0)

        if scene_confidence is not None:
            gathered = _gather_rows(scene_confidence.unsqueeze(-1), correspondence.index).squeeze(-1)
            gathered = torch.where(correspondence.valid, gathered, torch.zeros_like(gathered))
            confidence = confidence * gathered.clamp(min=0.0, max=1.0)
        confidence = confidence.detach()

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
        mask: _Input | None,
        min_confidence: float | None,
        max_distance: float | None,
        weight: Real | torch.Tensor,
        row_weight: Weight | Real | torch.Tensor,
        reduce: Literal["sum", "mean", "mean_active"],
        kernel: object | None,
        name: str,
        enabled: bool,
    ) -> None:
        if not isinstance(state, SceneSDFState):
            raise TypeError(f"state must be a SceneSDFState, got {type(state).__name__}")
        self.state = state
        self.mask = mask
        self.min_confidence = _optional_nonnegative(min_confidence, "min_confidence")
        self.max_distance = _optional_nonnegative(max_distance, "max_distance")
        self.nodes = (state, *((mask,) if isinstance(mask, Node) else ()))
        self.frames = state.frames
        self.points = state.points
        mask_variables = (mask,) if isinstance(mask, _VariableLike) else ()
        super().__init__(
            *mask_variables,
            dim=self.frames * self.points,
            weight=weight,
            row_weight=row_weight,
            reduce=reduce,
            kernel=kernel,
            name=name,
            enabled=enabled,
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

    def _active(self, result: SceneSDFResult) -> torch.Tensor:
        active = result.has_point.detach()
        if self.min_confidence is not None:
            active = active & (result.confidence.detach() >= self.min_confidence)
        if self.max_distance is not None:
            active = active & (result.dmin.detach() <= self.max_distance)
        if self.mask is not None:
            mask = check_tensor(
                "mask",
                _value(self.mask, "mask"),
                shape=(self.frames, self.points),
                dtype=torch.bool,
                device=result.signed_distance.device,
            )
            active = active & mask.detach()
        return active

    def active_groups(self) -> torch.Tensor:
        """Return the detached gate shared with exact-zero output rows."""
        active = self._active(self._result())
        return active.reshape(*active.shape[:-2], self.dim)

    def _finish(self, result: SceneSDFResult, penalty: torch.Tensor) -> torch.Tensor:
        confidence = result.confidence.detach()
        positive = confidence > 0.0
        safe = torch.where(positive, confidence, torch.ones_like(confidence))
        scale = torch.where(positive, torch.sqrt(safe), torch.zeros_like(confidence))
        weighted = torch.where(self._active(result), penalty * scale, torch.zeros_like(penalty))
        return weighted.reshape(*weighted.shape[:-2], self.dim)


class ScenePenetrationResidual(_ScenePenaltyResidual):
    """Positive signed depth for query points behind the scene surface."""

    def __init__(
        self,
        state: SceneSDFState,
        *,
        min_confidence: float | None = None,
        max_distance: float | None = None,
        mask: _VariableLike | Node | torch.Tensor | None = None,
        max_penetration: float | None = None,
        margin: float | None = None,
        weight: Real | torch.Tensor = 1.0,
        row_weight: Weight | Real | torch.Tensor = 1.0,
        reduce: Literal["sum", "mean", "mean_active"] = "sum",
        kernel: object | None = None,
        name: str = "scene_penetration",
        enabled: bool = True,
    ) -> None:
        self.max_penetration = _optional_nonnegative(max_penetration, "max_penetration")
        self.margin = _optional_nonnegative(margin, "margin") or 0.0
        super().__init__(state, mask, min_confidence, max_distance, weight, row_weight, reduce, kernel, name, enabled)

    def _active(self, result: SceneSDFResult) -> torch.Tensor:
        active = super()._active(result)
        if self.max_penetration is not None:
            active = active & (result.signed_distance.detach() >= -self.max_penetration)
        return active

    def error(self) -> torch.Tensor:
        result = self._result()
        return self._finish(result, torch.relu(-result.signed_distance - self.margin))


class SceneAttractionResidual(_ScenePenaltyResidual):
    """Absolute signed-distance error toward a requested surface offset."""

    def __init__(
        self,
        state: SceneSDFState,
        *,
        target_distance: float = 0.0,
        min_confidence: float | None = None,
        max_distance: float | None = None,
        mask: _VariableLike | Node | torch.Tensor | None = None,
        band: float | None = None,
        weight: Real | torch.Tensor = 1.0,
        row_weight: Weight | Real | torch.Tensor = 1.0,
        reduce: Literal["sum", "mean", "mean_active"] = "sum",
        kernel: object | None = None,
        name: str = "scene_attraction",
        enabled: bool = True,
    ) -> None:
        if not isinstance(target_distance, float):
            raise TypeError("target_distance must be float")
        self.target_distance = target_distance
        self.band = _optional_nonnegative(band, "band")
        super().__init__(state, mask, min_confidence, max_distance, weight, row_weight, reduce, kernel, name, enabled)

    def _active(self, result: SceneSDFResult) -> torch.Tensor:
        active = super()._active(result)
        if self.band is not None:
            active = active & ((result.signed_distance.detach() - self.target_distance).abs() <= self.band)
        return active

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
        min_confidence: float | None = None,
        max_distance: float | None = None,
        mask: _VariableLike | Node | torch.Tensor | None = None,
        margin: float | None = None,
        weight: Real | torch.Tensor = 1.0,
        row_weight: Weight | Real | torch.Tensor = 1.0,
        reduce: Literal["sum", "mean", "mean_active"] = "sum",
        kernel: object | None = None,
        name: str = "scene_clearance",
        enabled: bool = True,
    ) -> None:
        if not isinstance(clearance, float) or clearance < 0.0:
            raise ValueError("clearance must be a non-negative float")
        self.clearance = clearance
        self.margin = _optional_nonnegative(margin, "margin") or 0.0
        super().__init__(state, mask, min_confidence, max_distance, weight, row_weight, reduce, kernel, name, enabled)

    def error(self) -> torch.Tensor:
        result = self._result()
        return self._finish(result, torch.relu(result.signed_distance - self.clearance - self.margin))


__all__ = [
    "SceneAttractionResidual",
    "SceneClearanceResidual",
    "ScenePenetrationResidual",
    "SceneSDFResult",
    "SceneSDFState",
]
