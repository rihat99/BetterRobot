"""Point-cloud signed-distance provider and reusable penalty residuals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from ._point_cloud import _detached_nearest, _gather_rows


@dataclass(frozen=True)
class SceneSDFResult:
    """Fixed-shape outputs shared by all scene signed-distance heads.

    All fields have shape ``(B..., frames, query_points)``. ``has_point``
    is boolean; the other fields preserve the query-point floating dtype.
    """

    signed_distance: torch.Tensor
    dmin: torch.Tensor
    confidence: torch.Tensor
    has_point: torch.Tensor


@dataclass(frozen=True)
class SceneSDFProvider:
    """Approximate point-cloud SDF with one detached nearest-neighbour pass.

    Queries and scenes use the padded ``(tensor, validity_mask)`` convention.
    Scene normals orient the unsigned nearest distance.  Both nearest indices
    and the side-of-surface sign are detached discrete choices; distance is
    reconstructed from graph-carrying query/scene points.  ``confidence`` is
    the absolute normal alignment, optionally multiplied by a gathered scene
    confidence table.
    """

    query_points: str = "scene_query_points"
    query_validity: str = "scene_query_validity"
    scene_points: str = "scene_points"
    scene_normals: str = "scene_normals"
    scene_validity: str = "scene_validity"
    scene_confidence: str | None = None
    output: str = "scene_sdf"
    name: str = "scene_sdf_provider"
    chunk_size: int = 4096
    eps: float = 1e-8

    def __post_init__(self) -> None:
        required = (
            self.query_points,
            self.query_validity,
            self.scene_points,
            self.scene_normals,
            self.scene_validity,
        )
        if any(not isinstance(value, str) or not value for value in required):
            raise ValueError("scene SDF context names must be non-empty strings")
        if len(set(required)) != len(required):
            raise ValueError("scene SDF input context names must be unique")
        if self.scene_confidence is not None and (
            not isinstance(self.scene_confidence, str) or not self.scene_confidence
        ):
            raise ValueError("scene_confidence must be a non-empty context name or None")
        if self.scene_confidence is not None and self.scene_confidence in required:
            raise ValueError("scene_confidence must name a distinct context entry")
        if not isinstance(self.output, str) or not self.output:
            raise ValueError("output must be a non-empty string")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {self.chunk_size!r}")
        if not isinstance(self.eps, float) or self.eps <= 0.0:
            raise ValueError(f"eps must be a positive float, got {self.eps!r}")

    @property
    def inputs(self) -> tuple[str, ...]:
        base = (
            self.query_points,
            self.query_validity,
            self.scene_points,
            self.scene_normals,
            self.scene_validity,
        )
        return (*base, *((self.scene_confidence,) if self.scene_confidence is not None else ()))

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output,)

    def __call__(self, ctx: Mapping[str, Any]) -> dict[str, SceneSDFResult]:
        query = ctx[self.query_points]
        scene = ctx[self.scene_points]
        query_validity = ctx[self.query_validity]
        scene_validity = ctx[self.scene_validity]
        correspondence = _detached_nearest(
            query,
            scene,
            query_validity=query_validity,
            reference_validity=scene_validity,
            chunk_size=self.chunk_size,
        )

        normals = ctx[self.scene_normals]
        if not isinstance(normals, torch.Tensor) or not normals.is_floating_point():
            raise TypeError("scene_normals must be a floating tensor")
        if tuple(normals.shape[-2:]) != tuple(scene.shape[-2:]):
            raise ValueError(
                "scene_normals must match the scene point/count suffix, "
                f"got {tuple(normals.shape)} for scene {tuple(scene.shape)}"
            )
        if normals.dtype != query.dtype or normals.device != query.device:
            raise ValueError("scene_normals must share query point dtype/device")
        nearest_normal = _gather_rows(normals, correspondence.index)
        epsilon = max(self.eps, torch.finfo(query.dtype).eps)
        normal_norm = torch.linalg.vector_norm(nearest_normal, dim=-1, keepdim=True)
        unit_normal = nearest_normal / normal_norm.clamp_min(epsilon)
        normal_projection = (correspondence.delta * unit_normal).sum(dim=-1)

        # Side selection is as discrete as the NN identity.  Keeping its sign
        # detached avoids gradients through a branch while dmin remains live.
        sign = torch.where(
            normal_projection.detach() < 0.0,
            -torch.ones_like(normal_projection),
            torch.ones_like(normal_projection),
        )
        signed_distance = correspondence.distance * sign
        confidence = normal_projection.abs() / correspondence.distance.clamp_min(epsilon)
        confidence = confidence.clamp(min=0.0, max=1.0)

        if self.scene_confidence is not None:
            scene_confidence = ctx[self.scene_confidence]
            if not isinstance(scene_confidence, torch.Tensor) or not scene_confidence.is_floating_point():
                raise TypeError("scene_confidence must be a floating tensor")
            if scene_confidence.shape[-1] != scene.shape[-2]:
                raise ValueError(
                    f"scene_confidence must end in ({scene.shape[-2]},), got {tuple(scene_confidence.shape)}"
                )
            if scene_confidence.dtype != query.dtype or scene_confidence.device != query.device:
                raise ValueError("scene_confidence must share query point dtype/device")
            gathered_confidence = _gather_rows(
                scene_confidence.unsqueeze(-1),
                correspondence.index,
            ).squeeze(-1)
            confidence = confidence * gathered_confidence.clamp(min=0.0, max=1.0)

        valid = correspondence.valid & (normal_norm.squeeze(-1) > epsilon)
        zeros = torch.zeros_like(correspondence.distance)
        result = SceneSDFResult(
            signed_distance=torch.where(valid, signed_distance, zeros),
            dmin=torch.where(valid, correspondence.distance, zeros),
            confidence=torch.where(valid, confidence, zeros),
            has_point=valid,
        )
        return {self.output: result}


class _ScenePenaltyResidual:
    """Common fixed-shape validation and confidence masking for three heads."""

    def __init__(
        self,
        frames: int,
        points: int,
        *,
        scene_sdf: str,
        name: str,
    ) -> None:
        for label, value in (("frames", frames), ("points", points)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer, got {value!r}")
        if not isinstance(scene_sdf, str) or not scene_sdf:
            raise ValueError("scene_sdf must be a non-empty context name")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        self.frames = frames
        self.points = points
        self.name = name
        self.reads = (scene_sdf,)
        self.dim = frames * points

    def _result(self, ctx: Mapping[str, Any]) -> SceneSDFResult:
        result = ctx[self.reads[0]]
        if not isinstance(result, SceneSDFResult):
            raise TypeError(f"{self.reads[0]} must be a SceneSDFResult")
        suffix = (self.frames, self.points)
        for label, value in (
            ("signed_distance", result.signed_distance),
            ("dmin", result.dmin),
            ("confidence", result.confidence),
            ("has_point", result.has_point),
        ):
            if value.ndim < 2 or tuple(value.shape[-2:]) != suffix:
                raise ValueError(f"SceneSDFResult.{label} must end in {suffix}, got {tuple(value.shape)}")
        if result.has_point.dtype != torch.bool:
            raise TypeError("SceneSDFResult.has_point must use bool dtype")
        return result

    def _finish(self, result: SceneSDFResult, penalty: torch.Tensor) -> torch.Tensor:
        weighted = penalty * result.confidence
        weighted = torch.where(result.has_point, weighted, torch.zeros_like(weighted))
        return weighted.reshape(*weighted.shape[:-2], self.dim)


class ScenePenetrationResidual(_ScenePenaltyResidual):
    """Positive signed depth for query points behind the scene surface."""

    def __init__(
        self,
        frames: int,
        points: int,
        *,
        scene_sdf: str = "scene_sdf",
        name: str = "scene_penetration",
    ) -> None:
        super().__init__(frames, points, scene_sdf=scene_sdf, name=name)

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        result = self._result(ctx)
        return self._finish(result, torch.relu(-result.signed_distance))


class SceneAttractionResidual(_ScenePenaltyResidual):
    """Absolute signed-distance error toward a requested surface offset."""

    def __init__(
        self,
        frames: int,
        points: int,
        *,
        target_distance: float = 0.0,
        scene_sdf: str = "scene_sdf",
        name: str = "scene_attraction",
    ) -> None:
        if not isinstance(target_distance, float):
            raise TypeError("target_distance must be float")
        super().__init__(frames, points, scene_sdf=scene_sdf, name=name)
        self.target_distance = target_distance

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        result = self._result(ctx)
        penalty = (result.signed_distance - self.target_distance).abs()
        return self._finish(result, penalty)


class SceneClearanceResidual(_ScenePenaltyResidual):
    """One-sided excess above an allowed positive signed clearance."""

    def __init__(
        self,
        frames: int,
        points: int,
        *,
        clearance: float,
        scene_sdf: str = "scene_sdf",
        name: str = "scene_clearance",
    ) -> None:
        if not isinstance(clearance, float) or clearance < 0.0:
            raise ValueError("clearance must be a non-negative float")
        super().__init__(frames, points, scene_sdf=scene_sdf, name=name)
        self.clearance = clearance

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        result = self._result(ctx)
        return self._finish(result, torch.relu(result.signed_distance - self.clearance))


__all__ = [
    "SceneAttractionResidual",
    "SceneClearanceResidual",
    "ScenePenetrationResidual",
    "SceneSDFProvider",
    "SceneSDFResult",
]
