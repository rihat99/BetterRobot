"""Synthetic BVR-shaped consumer components for the M2a vertical slice."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from better_robot.optim import Problem, ResidualItem, VarSpec


SEED = 20260717
TIME = 4
POINTS = 3
COORDS = 2

ROOT_WEIGHTS = {
    "penetration": 0.25,
    "attraction": 1.0,
    "clearance": 0.0,
    "scale_prior": math.sqrt(0.6),
}
FULL_WEIGHTS = {
    "penetration": 0.4,
    "attraction": 1.0,
    "clearance": 0.15,
    "scale_prior": 0.5,
}
PROVIDER_INACTIVE_WEIGHTS = {
    "penetration": 0.0,
    "attraction": 0.0,
    "clearance": 0.0,
    "scale_prior": 1.0,
}

FRICTION_LOG = (
    "Masks are static on VarSpec, so the root-to-full transition cheaply rebuilds "
    "Problem while preserving Values; phase orchestration remains outside M2a.",
    "torch.optim.Adam consumes leaf gradients, while Problem.gradient returns named "
    "reduced tangents; zeroed tangent buffers adapt Adam updates into Problem.retract.",
    "The guide intentionally keeps providers structural rather than prescribing a "
    "base class; the slice supplies name/reads/outputs/__call__ without inheritance.",
)


def _extract_guide_custom_residual() -> tuple[str, type]:
    guide = Path(__file__).resolve().parents[2] / "docs" / "guides" / "custom_residuals.md"
    text = guide.read_text()
    start_marker = "<!-- custom-residual-example:start -->"
    end_marker = "<!-- custom-residual-example:end -->"
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise RuntimeError("custom residual guide must contain exactly one marked example")
    marked = text.split(start_marker, 1)[1].split(end_marker, 1)[0]
    if marked.count("```python") != 1 or marked.count("```") != 2:
        raise RuntimeError("marked custom residual example must contain one Python fence")
    source = marked.split("```python", 1)[1].split("```", 1)[0].strip()
    namespace: dict[str, Any] = {"torch": torch, "__name__": __name__}
    exec(compile(source, str(guide), "exec"), namespace)  # noqa: S102 - executable guide contract
    return source, namespace["PenetrationResidual"]


GUIDE_CUSTOM_RESIDUAL_SOURCE, PenetrationResidual = _extract_guide_custom_residual()


@dataclass
class SliceCounters:
    """Observable provider invocation counts across evaluation contexts."""

    kinematics: int = 0
    nearest_neighbor: int = 0


@dataclass(frozen=True)
class SyntheticKinematicsProvider:
    """One cheap FK-shaped pass producing each frame's translated origin."""

    counters: SliceCounters
    name: str = "synthetic_kinematics"
    reads: tuple[str, ...] = ("q",)
    outputs: tuple[str, ...] = ("kinematic_origins",)

    def __call__(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        self.counters.kinematics += 1
        return {"kinematic_origins": ctx["q"].unsqueeze(-2)}


@dataclass(frozen=True)
class DetachedNearestNeighborProvider:
    """Detach only discrete NN indices; preserve gradients through selected deltas."""

    counters: SliceCounters
    name: str = "scene_nearest_neighbor"
    reads: tuple[str, ...] = (
        "kinematic_origins",
        "log_s",
        "template_points",
        "scene_points",
    )
    outputs: tuple[str, ...] = (
        "signed_distance",
        "nearest_delta",
        "nearest_squared_distance",
    )

    def __call__(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        self.counters.nearest_neighbor += 1
        origins = ctx["kinematic_origins"]
        scale = ctx["log_s"].exp()
        while scale.ndim < origins.ndim:
            scale = scale.unsqueeze(-1)
        points = origins + scale * ctx["template_points"]
        candidates = points.unsqueeze(-2) - ctx["scene_points"].unsqueeze(-3)
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


class AttractionResidual:
    """Vector displacement from every synthetic body point to its detached NN."""

    name = "attraction"
    reads = ("nearest_delta",)
    dim = TIME * POINTS * COORDS

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        delta = ctx["nearest_delta"]
        return delta.reshape(*delta.shape[:-3], self.dim)


class ClearanceResidual:
    """One-sided positive-side clearance excess around the target surface."""

    name = "clearance"
    reads = ("signed_distance",)
    dim = TIME * POINTS

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        signed = ctx["signed_distance"]
        return torch.relu(signed - 0.02).reshape(*signed.shape[:-2], self.dim)


class ScalePriorResidual:
    """One-row least-squares prior keeping log-scale near the target."""

    name = "scale_prior"
    reads = ("log_s", "target_log_s")
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return math.sqrt(2.0) * (ctx["log_s"] - ctx["target_log_s"])


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


def q_mask(*, root_only: bool) -> torch.Tensor:
    """Return BVR-like root-only or full tangent activity for the q block."""
    mask = torch.ones(TIME, COORDS, dtype=torch.bool)
    if root_only:
        mask[:, 1] = False
    return mask.flatten()


def make_problem(
    data: SliceData,
    *,
    root_only: bool,
    counters: SliceCounters | None = None,
) -> tuple[Problem, SliceCounters]:
    """Build one phase's problem; rebuilding is the M2a mask transition."""
    counters = counters or SliceCounters()
    penetration = PenetrationResidual(TIME, POINTS)
    problem = Problem(
        vars=(
            VarSpec("q", (TIME, COORDS), mask=q_mask(root_only=root_only)),
            VarSpec("log_s", (1,)),
        ),
        residuals=(
            ResidualItem("penetration", penetration),
            ResidualItem("attraction", AttractionResidual(), group_size=COORDS),
            ResidualItem("clearance", ClearanceResidual()),
            ResidualItem("scale_prior", ScalePriorResidual()),
        ),
        providers=(
            SyntheticKinematicsProvider(counters),
            DetachedNearestNeighborProvider(counters),
        ),
        parameters={
            "template_points": data.template_points,
            "scene_points": data.scene_points,
            "target_log_s": data.target_log_s,
        },
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
    "q_mask",
]
