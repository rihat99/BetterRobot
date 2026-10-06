"""Kinematic terms and the configuration manifold of a robot model.

Each factory returns a :class:`~better_robot.optim2.TermFn`. Kinematic terms read a cached
:class:`~better_robot.kinematics.KinematicState` (register it with :func:`kinematic_state`)
so one forward-kinematics pass serves every term.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..data_model.model import Model
from ..kinematics.state import KinematicState, forward_kinematics_state, frame_jacobian
from ..lie import se3
from ..lie.tangents import right_jacobian_inv_se3
from ..optim2 import TermFn


class RobotManifold:
    """Configuration manifold of ``model``: ``q: (B, nq)`` or ``(B, T, nq)``, tangent ``nv`` per configuration."""

    def __init__(self, model: Model) -> None:
        self.model = model

    def tangent_dim(self, x: torch.Tensor) -> int:
        return x[0].numel() // self.model.nq * self.model.nv

    def retract(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return self.model.integrate(x, delta.reshape(*x.shape[:-1], self.model.nv))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.integrate(x, x.new_zeros(*x.shape[:-1], self.model.nv))

    def tangent_bounds(self, x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Step range of the vector-space joints; quaternion coordinates are unbounded."""
        structure = self.model.structure
        q_index, v_index = structure.manifold_euclidean_q_indices, structure.manifold_euclidean_v_indices
        low = x.new_full((*x.shape[:-1], self.model.nv), -float("inf"))
        high = x.new_full((*x.shape[:-1], self.model.nv), float("inf"))
        low[..., v_index] = (lower - x)[..., q_index]
        high[..., v_index] = (upper - x)[..., q_index]
        return low.flatten(1), high.flatten(1)


def kinematic_state(model: Model, *, q: str = "q") -> TermFn:
    """Cached value: forward kinematics of variable ``q``. Register with ``problem.cached(..., name="state")``."""
    return TermFn(fn=lambda value: forward_kinematics_state(model, value), reads=(q,))


def _frame_id(model: Model, frame: int | str) -> int:
    return model.frame_id(frame) if isinstance(frame, str) else frame


def frame_pose(model: Model, frame: int | str, *, target: str, state: str = "state", q: str = "q") -> TermFn:
    """Pose error of ``frame`` against data ``target: (B, 7)``. Residual ``(B, 6)``, ``[linear, angular]``."""
    frame_id = _frame_id(model, frame)

    def error(kinematics: KinematicState, goal: torch.Tensor) -> torch.Tensor:
        return se3.log(se3.compose(se3.inverse(goal), kinematics.frame_pose[..., frame_id, :]))

    def jac(kinematics: KinematicState, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        local = frame_jacobian(model, kinematics, frame_id, reference="local")
        return {q: right_jacobian_inv_se3(error(kinematics, goal)) @ local}

    return TermFn(fn=error, reads=(state, target), jac=jac)


def frame_position(model: Model, frame: int | str, *, target: str, state: str = "state", q: str = "q") -> TermFn:
    """Position error of ``frame`` against data ``target: (B, 3)``. Residual ``(B, 3)``."""
    frame_id = _frame_id(model, frame)

    def error(kinematics: KinematicState, goal: torch.Tensor) -> torch.Tensor:
        return kinematics.frame_pose[..., frame_id, :3] - goal

    def jac(kinematics: KinematicState, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        return {q: frame_jacobian(model, kinematics, frame_id)[..., :3, :]}

    return TermFn(fn=error, reads=(state, target), jac=jac)


def _identity_jac(model: Model, q: str, sign_fn: Callable[[torch.Tensor], torch.Tensor]):
    """Diagonal Jacobian for models whose configuration is its own tangent (``nq == nv``); else autodiff."""
    if model.nq != model.nv:
        return None
    return lambda value: {q: torch.diag_embed(sign_fn(value))}


def joint_limits(model: Model, *, q: str = "q") -> TermFn:
    """Violation of the position limits, zero inside them. Residual ``(B, nq)``."""
    lower, upper = model.lower_pos_limit, model.upper_pos_limit

    def violation(value: torch.Tensor) -> torch.Tensor:
        return (value - upper).clamp_min(0.0) - (lower - value).clamp_min(0.0)

    def slope(value: torch.Tensor) -> torch.Tensor:
        return ((value > upper) | (value < lower)).to(value.dtype)

    return TermFn(fn=violation, reads=(q,), jac=_identity_jac(model, q, slope))


def rest(model: Model, q_rest: torch.Tensor, *, q: str = "q") -> TermFn:
    """Tangent difference to a rest configuration ``q_rest: (nq,)`` or ``(B, nq)``. Residual ``(B, nv)``."""

    def offset(value: torch.Tensor) -> torch.Tensor:
        return model.difference(q_rest.expand_as(value), value)

    return TermFn(fn=offset, reads=(q,), jac=_identity_jac(model, q, torch.ones_like))


__all__ = ["RobotManifold", "frame_pose", "frame_position", "joint_limits", "kinematic_state", "rest"]
