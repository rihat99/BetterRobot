"""Synthetic BVR-shaped consumer components for the M2a vertical slice."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from better_robot.optim import Problem, Residual, Variable
from better_robot.residuals import Node


SEED = 20260717
TIME = 4
POINTS = 3
COORDS = 2

ROOT_WEIGHTS = {
    "penetration": 0.0625,
    "attraction": 1.0,
    "clearance": 0.0,
    "scale_prior": 0.6,
}
FULL_WEIGHTS = {
    "penetration": 0.16,
    "attraction": 1.0,
    "clearance": 0.0225,
    "scale_prior": 0.25,
}
PROVIDER_INACTIVE_WEIGHTS = {
    "penetration": 0.0,
    "attraction": 0.0,
    "clearance": 0.0,
    "scale_prior": 1.0,
}

FRICTION_LOG = (
    "Masks are static on Variable, so the root-to-full transition cheaply rebuilds "
    "Problem while copying owned tensors; phase orchestration remains outside M2a.",
    "TorchOptimizer owns persistent Adam state over rebased reduced-tangent buffers.",
    "Shared Node objects memoize synthetic kinematics and nearest-neighbor work for "
    "one evaluation epoch without retaining an old autograd graph.",
)


def _extract_guide_custom_residual() -> tuple[str, type]:
    guide = Path(__file__).resolve().parents[2] / "docs" / "guides" / "custom_residual.md"
    text = guide.read_text()
    start_marker = "<!-- custom-residual-example:start -->"
    end_marker = "<!-- custom-residual-example:end -->"
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise RuntimeError("custom residual guide must contain exactly one marked example")
    marked = text.split(start_marker, 1)[1].split(end_marker, 1)[0]
    fence = "```{testcode}"
    if marked.count(fence) != 1 or marked.count("```") != 2:
        raise RuntimeError("marked custom residual example must contain one testcode fence")
    source = marked.split(fence, 1)[1].split("```", 1)[0].strip()
    namespace: dict[str, Any] = {"torch": torch, "__name__": __name__}
    exec(compile(source, str(guide), "exec"), namespace)  # noqa: S102 - executable guide contract
    return source, namespace["PenetrationResidual"]


GUIDE_CUSTOM_RESIDUAL_SOURCE, PenetrationResidual = _extract_guide_custom_residual()


@dataclass
class SliceCounters:
    """Observable provider invocation counts across evaluation contexts."""

    kinematics: int = 0
    nearest_neighbor: int = 0


class SyntheticKinematicsNode(Node):
    """One cheap FK-shaped pass producing each frame's translated origin."""

    def __init__(self, q: Variable, counters: SliceCounters) -> None:
        self.q = q
        self.counters = counters
        super().__init__(q)

    def compute(self) -> torch.Tensor:
        self.counters.kinematics += 1
        return self.q.tensor.unsqueeze(-2)


class DetachedNearestNeighborNode(Node):
    """Detach only discrete NN indices; preserve gradients through selected deltas."""

    def __init__(
        self,
        kinematics: SyntheticKinematicsNode,
        log_s: Variable,
        template_points: torch.Tensor,
        scene_points: torch.Tensor,
        counters: SliceCounters,
    ) -> None:
        self.kinematics = kinematics
        self.log_s = log_s
        self.template_points = template_points
        self.scene_points = scene_points
        self.counters = counters
        super().__init__(*kinematics.variables, log_s)

    def compute(self) -> dict[str, torch.Tensor]:
        self.counters.nearest_neighbor += 1
        origins = self.kinematics.value()
        scale = self.log_s.tensor.exp()
        while scale.ndim < origins.ndim:
            scale = scale.unsqueeze(-1)
        points = origins + scale * self.template_points
        candidates = points.unsqueeze(-2) - self.scene_points.unsqueeze(-3)
        squared = candidates.square().sum(dim=-1)
        nearest_index = squared.detach().argmin(dim=-1)
        gather_index = nearest_index[..., None, None].expand(
            *nearest_index.shape,
            1,
            candidates.shape[-1],
        )
        nearest_delta = candidates.gather(-2, gather_index).squeeze(-2)
        signed_distance = nearest_delta[..., 1]
        return {
            "signed_distance": signed_distance,
            "nearest_delta": nearest_delta,
            "nearest_squared_distance": nearest_delta.square().sum(dim=-1),
        }


class AttractionResidual(Residual):
    """Vector displacement from every synthetic body point to its detached NN."""

    def __init__(
        self,
        kinematics: SyntheticKinematicsNode,
        nearest: DetachedNearestNeighborNode,
        *,
        weight: float,
    ) -> None:
        self.nearest = nearest
        self.nodes = (kinematics, nearest)
        super().__init__(
            *nearest.variables,
            dim=TIME * POINTS * COORDS,
            group_size=COORDS,
            name="attraction",
            weight=weight,
        )

    def error(self) -> torch.Tensor:
        delta = self.nearest.value()["nearest_delta"]
        return delta.reshape(*delta.shape[:-3], self.dim)


class ClearanceResidual(Residual):
    """One-sided positive-side clearance excess around the target surface."""

    def __init__(
        self,
        kinematics: SyntheticKinematicsNode,
        nearest: DetachedNearestNeighborNode,
        *,
        weight: float,
    ) -> None:
        self.nearest = nearest
        self.nodes = (kinematics, nearest)
        super().__init__(
            *nearest.variables,
            dim=TIME * POINTS,
            name="clearance",
            weight=weight,
        )

    def error(self) -> torch.Tensor:
        signed = self.nearest.value()["signed_distance"]
        return torch.relu(signed - 0.02).reshape(*signed.shape[:-2], self.dim)


class ScalePriorResidual(Residual):
    """One-row least-squares prior keeping log-scale near the target."""

    def __init__(self, log_s: Variable, target: torch.Tensor, *, weight: float) -> None:
        self.log_s = log_s
        self.target = target
        super().__init__(log_s, dim=1, name="scale_prior", weight=weight)

    def error(self) -> torch.Tensor:
        return math.sqrt(2.0) * (self.log_s.tensor - self.target)


@dataclass(frozen=True)
class SliceData:
    """Fixed synthetic observations and initial/target states."""

    template_points: torch.Tensor
    scene_points: torch.Tensor
    target_q: torch.Tensor
    target_log_s: torch.Tensor
    initial_q: torch.Tensor
    initial_log_s: torch.Tensor

    def initial_values(self) -> dict[str, torch.Tensor]:
        return {
            "q": self.initial_q.clone(),
            "log_s": self.initial_log_s.clone(),
        }


def make_slice_data() -> SliceData:
    """Return deterministic CPU fp32 data with stable nearest-neighbor identities."""
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    template = torch.tensor(
        [[-0.40, -0.20], [0.00, 0.35], [0.45, -0.10]],
        dtype=torch.float32,
    )
    target_q = torch.tensor(
        [[-0.12, 0.02], [-0.04, 0.07], [0.05, 0.12], [0.14, 0.17]],
        dtype=torch.float32,
    )
    target_log_s = torch.tensor([math.log(1.15)], dtype=torch.float32)
    scene_points = target_q.unsqueeze(-2) + target_log_s.exp() * template
    perturbation = 0.01 * torch.randn(TIME, COORDS, generator=generator)
    initial_q = target_q + torch.tensor([0.13, -0.09]) + perturbation
    initial_log_s = torch.tensor([math.log(0.88)], dtype=torch.float32)
    return SliceData(
        template_points=template,
        scene_points=scene_points,
        target_q=target_q,
        target_log_s=target_log_s,
        initial_q=initial_q,
        initial_log_s=initial_log_s,
    )


def make_problem(
    data: SliceData,
    *,
    counters: SliceCounters | None = None,
    values: Mapping[str, torch.Tensor] | None = None,
    weights: Mapping[str, float] = FULL_WEIGHTS,
) -> tuple[Problem, SliceCounters]:
    """Build one phase's problem with caller-selected residual weights."""
    counters = counters or SliceCounters()
    initial = data.initial_values() if values is None else dict(values)
    q_value, log_s_value = initial["q"], initial["log_s"]
    q = Variable(q_value, name="q", batch_ndim=q_value.ndim - 2)
    log_s = Variable(log_s_value, name="log_s", batch_ndim=log_s_value.ndim - 1)
    kinematics = SyntheticKinematicsNode(q, counters)
    nearest = DetachedNearestNeighborNode(
        kinematics,
        log_s,
        data.template_points,
        data.scene_points,
        counters,
    )
    penetration = PenetrationResidual(
        nearest,
        time=TIME,
        points=POINTS,
        weight=weights["penetration"],
    )
    # The custom guide class knows its immediate node. Register the upstream
    # node too so both epoch memos invalidate together.
    penetration.nodes = (kinematics, nearest)
    problem = Problem(
        [
            penetration,
            AttractionResidual(kinematics, nearest, weight=weights["attraction"]),
            ClearanceResidual(kinematics, nearest, weight=weights["clearance"]),
            ScalePriorResidual(log_s, data.target_log_s, weight=weights["scale_prior"]),
        ]
    )
    return problem, counters


def make_batched_values(data: SliceData) -> dict[str, torch.Tensor]:
    """Three distinct starting points for batched/sequential parity."""
    q_offsets = torch.tensor(
        [[[0.00, 0.00]], [[0.02, -0.01]], [[-0.015, 0.025]]],
        dtype=torch.float32,
    )
    scale_offsets = torch.tensor([[0.0], [0.03], [-0.02]], dtype=torch.float32)
    return {
        "q": data.initial_q.unsqueeze(0) + q_offsets,
        "log_s": data.initial_log_s.unsqueeze(0) + scale_offsets,
    }


__all__ = [
    "COORDS",
    "FRICTION_LOG",
    "FULL_WEIGHTS",
    "GUIDE_CUSTOM_RESIDUAL_SOURCE",
    "POINTS",
    "PROVIDER_INACTIVE_WEIGHTS",
    "PenetrationResidual",
    "ROOT_WEIGHTS",
    "SEED",
    "SliceCounters",
    "SliceData",
    "TIME",
    "make_batched_values",
    "make_problem",
    "make_slice_data",
]
