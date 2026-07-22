"""Synthetic articulated-body components for the optimizer vertical slice."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch

from better_robot.data_model import Model
from better_robot.io import ModelBuilder, build_model
from better_robot.optim import Problem, Residual, RobotVariable, ScalarCost, Variable
from better_robot.residuals import (
    MaskedChamferResidual,
    Node,
    PointProjectionResidual,
    SceneAttractionResidual,
    SceneSDFState,
)


SEED = 20260722
TIME = 3
POINTS = 4
WARMUP_STEPS = 24
ADAM_PHASE_STEPS = 90

COARSE_WEIGHTS = {
    "scene": 0.35,
    "chamfer": 1.0,
    "projection": 0.3,
    "scale_prior": 0.2,
}
FINAL_WEIGHTS = {
    "scene": 0.2,
    "chamfer": 1.25,
    "projection": 0.45,
    "scale_prior": 0.15,
}


@dataclass
class SliceCounters:
    """Observable node invocation counts across evaluation scopes."""

    #: Calls to the configuration-to-tangent node.
    articulation: int = 0
    #: Calls to the tangent-to-vertex node.
    posed_body: int = 0

    def reset(self) -> None:
        """Clear constructor-time shape probes before the fit starts."""
        self.articulation = 0
        self.posed_body = 0


def _skin_points(
    tangent: torch.Tensor,
    log_scale: torch.Tensor,
    template_points: torch.Tensor,
) -> torch.Tensor:
    """Apply a small differentiable linear-blend pose to ``(T, nv)`` values."""
    scaled = (log_scale.exp() * template_points).unsqueeze(0).expand(tangent.shape[0], -1, -1)
    translation = tangent[:, :3].unsqueeze(1)
    rotation = tangent[:, 3:6].unsqueeze(1).expand_as(scaled)
    rotation_offset = torch.cross(rotation, scaled, dim=-1)
    hinge_axis = torch.stack(
        (0.7 * template_points[:, 1], -0.5 * template_points[:, 0], 0.2 * template_points[:, 0]),
        dim=-1,
    )
    influence = torch.linspace(0.0, 1.0, POINTS, dtype=tangent.dtype, device=tangent.device)
    hinge_offset = tangent[:, 6, None, None] * influence[None, :, None] * hinge_axis[None]
    return translation + scaled + rotation_offset + hinge_offset


class TangentPoseNode(Node):
    """Convert a robot trajectory to neutral-relative tangent coordinates."""

    def __init__(self, q: RobotVariable, counters: SliceCounters) -> None:
        self.q = q
        self.counters = counters
        super().__init__(q)

    def compute(self) -> torch.Tensor:
        """Return ``(T, nv)`` neutral-relative tangent coordinates."""
        self.counters.articulation += 1
        neutral = self.q.model.q_neutral.to(self.q.tensor).expand_as(self.q.tensor)
        return self.q.model.difference(neutral, self.q.tensor)


class PosedBodyNode(Node):
    """Pose template vertices from a child articulation node and body scale."""

    def __init__(
        self,
        articulation: TangentPoseNode,
        log_scale: Variable,
        template_points: torch.Tensor,
        counters: SliceCounters,
    ) -> None:
        self.articulation = articulation
        self.log_scale = log_scale
        self.template_points = template_points
        self.counters = counters
        super().__init__(articulation, log_scale)

    def compute(self) -> torch.Tensor:
        """Return posed body points with shape ``(T, P, 3)``."""
        self.counters.posed_body += 1
        return _skin_points(self.articulation.value(), self.log_scale.tensor, self.template_points)


@dataclass(frozen=True)
class SliceData:
    """Deterministic fp32 observations and optimizer states."""

    #: Synthetic free-flyer-plus-hinge model.
    model: Model
    #: Neutral body points with shape ``(P, 3)``.
    template_points: torch.Tensor
    #: Coarse scene used by the first Adam phase.
    coarse_scene_points: torch.Tensor
    #: Final scene observations.
    target_scene_points: torch.Tensor
    #: Final image observations with shape ``(T, P, 2)``.
    target_pixels: torch.Tensor
    #: Shared pinhole intrinsics.
    intrinsics: torch.Tensor
    #: Desired trajectory tangent with shape ``(T, nv)``.
    target_tangent: torch.Tensor
    #: Desired one-element log-scale tensor.
    target_log_scale: torch.Tensor
    #: Initial robot trajectory with shape ``(T, nq)``.
    initial_q: torch.Tensor
    #: Initial one-element log-scale tensor.
    initial_log_scale: torch.Tensor


def _make_model() -> Model:
    """Build the smallest articulated floating-base model used by the slice."""
    builder = ModelBuilder("vertical_slice_body")
    base = builder.add_body("base")
    tip = builder.add_body("tip")
    builder.add_free_flyer_root("root", child=base)
    builder.add_revolute_z("hinge", parent=base, child=tip)
    return build_model(builder.finalize(), dtype=torch.float32)


def _project(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Project ``(T, P, 3)`` world points through an identity camera pose."""
    homogeneous = torch.matmul(intrinsics, points.unsqueeze(-1)).squeeze(-1)
    return homogeneous[..., :2] / points[..., 2:3]


def make_slice_data() -> SliceData:
    """Return well-separated observations with stable nearest neighbours."""
    model = _make_model()
    template_points = torch.tensor(
        [
            [-0.75, -0.25, 2.20],
            [-0.15, 0.45, 2.35],
            [0.55, -0.35, 2.50],
            [0.95, 0.35, 2.65],
        ],
        dtype=torch.float32,
    )
    target_tangent = torch.tensor(
        [
            [-0.08, 0.03, 0.02, 0.025, -0.015, 0.020, -0.12],
            [-0.02, 0.05, 0.00, -0.015, 0.020, -0.010, 0.05],
            [0.06, 0.07, -0.015, 0.010, 0.015, -0.020, 0.14],
        ],
        dtype=torch.float32,
    )
    initial_offset = torch.tensor(
        [0.045, -0.035, 0.018, 0.018, -0.015, 0.012, -0.075],
        dtype=torch.float32,
    )
    initial_tangent = target_tangent + initial_offset
    target_log_scale = torch.tensor([math.log(1.04)], dtype=torch.float32)
    initial_log_scale = torch.tensor([math.log(0.94)], dtype=torch.float32)
    neutral = model.q_neutral.expand(TIME, -1)
    target_points = _skin_points(target_tangent, target_log_scale, template_points)
    intrinsics = torch.tensor(
        [[80.0, 0.0, 4.0], [0.0, 75.0, -3.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return SliceData(
        model=model,
        template_points=template_points,
        coarse_scene_points=target_points + torch.tensor([0.0, 0.025, 0.0]),
        target_scene_points=target_points,
        target_pixels=_project(target_points, intrinsics),
        intrinsics=intrinsics,
        target_tangent=target_tangent,
        target_log_scale=target_log_scale,
        initial_q=model.integrate(neutral, initial_tangent),
        initial_log_scale=initial_log_scale,
    )


def make_problem(
    data: SliceData,
    counters: SliceCounters,
    *,
    frozen_root: bool,
) -> tuple[Problem, RobotVariable, Variable, dict[str, Residual]]:
    """Build one frozen- or free-root view of the same staged fit."""
    q = RobotVariable(
        data.model,
        data.initial_q.clone(),
        name="q",
        time_axis=0,
        frozen_groups=("root",) if frozen_root else (),
    )
    log_scale = Variable(data.initial_log_scale.clone(), name="log_scale")
    articulation = TangentPoseNode(q, counters)
    posed_body = PosedBodyNode(articulation, log_scale, data.template_points, counters)

    point_validity = Variable(
        torch.tensor(
            [[True, True, True, True], [True, True, True, False], [True, True, True, True]],
        ),
        name="point_validity",
        trainable=False,
    )
    contact_mask = Variable(
        torch.tensor(
            [[True, True, False, True], [True, False, True, False], [False, True, True, True]],
        ),
        name="contact_mask",
        trainable=False,
    )
    scene_points = Variable(data.coarse_scene_points.clone(), name="scene_points", trainable=False)
    scene_normals = torch.zeros_like(data.target_scene_points)
    scene_normals[..., 1] = 1.0
    confidence = torch.tensor(
        [[1.0, 0.8, 0.6, 1.0], [0.9, 0.7, 1.0, 0.0], [0.6, 1.0, 0.8, 0.9]],
    )
    state = SceneSDFState(
        posed_body,
        point_validity,
        scene_points,
        scene_normals,
        point_validity,
        distance="plane",
    )
    scene = SceneAttractionResidual(
        state,
        mask=contact_mask,
        max_distance=0.6,
        band=0.6,
        weight=COARSE_WEIGHTS["scene"],
        reduce="mean_active",
        name="scene",
    )
    chamfer = MaskedChamferResidual(
        posed_body,
        scene_points,
        point_validity,
        point_validity,
        vertex_weights=confidence,
        bidirectional=False,
        weight=COARSE_WEIGHTS["chamfer"],
        name="chamfer",
    )
    projection = PointProjectionResidual(
        posed_body,
        data.intrinsics,
        torch.eye(4),
        data.target_pixels,
        confidence=confidence,
        visibility=point_validity,
        time_axis=0,
        weight=COARSE_WEIGHTS["projection"],
        reduce="mean_active",
        name="projection",
    )
    projection.enabled = False
    scale_prior = ScalarCost(
        lambda value: (value - data.target_log_scale).square().sum(dim=-1),
        log_scale,
        weight=COARSE_WEIGHTS["scale_prior"],
        name="scale_prior",
    )
    terms: dict[str, Residual] = {item.name: item for item in (scene, chamfer, projection, scale_prior)}
    return Problem(list(terms.values())), q, log_scale, terms


def configure_phase(
    terms: Mapping[str, Residual],
    weights: Mapping[str, float],
    *,
    projection_enabled: bool,
) -> None:
    """Apply plain-data phase settings without introducing a Phase object."""
    for name, weight in weights.items():
        terms[name].weight = weight
    terms["projection"].enabled = projection_enabled


__all__ = [
    "ADAM_PHASE_STEPS",
    "COARSE_WEIGHTS",
    "FINAL_WEIGHTS",
    "SEED",
    "SliceCounters",
    "SliceData",
    "TIME",
    "WARMUP_STEPS",
    "configure_phase",
    "make_problem",
    "make_slice_data",
]
