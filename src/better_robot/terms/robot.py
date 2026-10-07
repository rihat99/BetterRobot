"""Kinematic terms and the configuration manifold of a robot model.

Each factory returns a ``better_robot.optim.TermFn``. Kinematic terms read a cached
``better_robot.kinematics.KinematicState`` (register it with ``kinematic_state``)
so one forward-kinematics pass serves every term.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..data_model.model import Model
from ..kinematics.state import KinematicState, forward_kinematics_state, frame_jacobian
from ..lie import se3
from ..lie.tangents import right_jacobian_inv_se3
from ..optim import CacheRef, TermFn


class RobotManifold:
    """Configuration manifold of ``model``: ``q: (B, nq)`` or ``(B, T, nq)``, tangent ``nv`` per configuration.

    ``retract`` is ``model.integrate``; ``project`` integrates a zero step, which renormalizes quaternion joints.
    Pass it as ``Problem(..., manifolds={"q": RobotManifold(model)})``.
    """

    def __init__(self, model: Model) -> None:
        self.model = model

    def tangent_dim(self, x: torch.Tensor) -> int:
        return x[0].numel() // self.model.nq * self.model.nv

    def retract(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return self.model.integrate(x, delta.reshape(*x.shape[:-1], self.model.nv))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.integrate(x, x.new_zeros(*x.shape[:-1], self.model.nv))

    def tangent_bounds(
        self, x: torch.Tensor, *, lower: torch.Tensor, upper: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


def frame_pose(model: Model, frame: int | str, *, target: str, state: CacheRef | str = "state", q: str = "q") -> TermFn:
    """Pose error of ``frame`` against data ``target: (B, 7)``. Residual ``(B, 6)``, ``[linear, angular]``."""
    frame_id = model.frame_id(frame)

    def error(kinematics: KinematicState, goal: torch.Tensor) -> torch.Tensor:
        return se3.log(se3.compose(se3.inverse(goal), kinematics.frame_pose_of(frame_id)))

    def jac(kinematics: KinematicState, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        local = frame_jacobian(model, kinematics, frame=frame_id, reference="local")
        return {q: right_jacobian_inv_se3(error(kinematics, goal)) @ local}

    return TermFn(fn=error, reads=(str(state), target), jac=jac)


def frame_position(model: Model, frame: int | str, *, target: str, state: CacheRef | str = "state", q: str = "q") -> TermFn:
    """Position error of ``frame`` against data ``target: (B, 3)``. Residual ``(B, 3)``."""
    frame_id = model.frame_id(frame)

    def error(kinematics: KinematicState, goal: torch.Tensor) -> torch.Tensor:
        return kinematics.frame_translation[..., frame_id, :] - goal

    def jac(kinematics: KinematicState, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        return {q: frame_jacobian(model, kinematics, frame=frame_id)[..., :3, :]}

    return TermFn(fn=error, reads=(str(state), target), jac=jac)


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


def finite_difference(model: Model, *, order: int, dt: float, q: str = "q") -> TermFn:
    """Velocity (``order=1``) or acceleration (``order=2``) of a trajectory ``q: (B, T, nq)`` in tangent space.

    Residual ``(B, T - order, nv)``; a time-local term with ``window = order + 1``.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")

    def derivative(value: torch.Tensor) -> torch.Tensor:
        out = model.difference(value[:, :-1], value[:, 1:]) / dt
        return out if order == 1 else (out[:, 1:] - out[:, :-1]) / dt

    jac = None
    if model.nq == model.nv:
        stencil = torch.tensor([-1.0, 1.0] if order == 1 else [1.0, -2.0, 1.0]) / dt**order

        def jac(value: torch.Tensor) -> dict[str, torch.Tensor]:
            eye = torch.eye(model.nv, dtype=value.dtype, device=value.device)
            block = stencil.to(value)[:, None, None] * eye  # (window, nv, nv)
            return {q: block.permute(1, 0, 2).expand(value.shape[0], value.shape[1] - order, -1, -1, -1)}

    return TermFn(fn=derivative, reads=(q,), jac=jac, window=order + 1)


__all__ = [
    "RobotManifold",
    "finite_difference",
    "frame_pose",
    "frame_position",
    "joint_limits",
    "kinematic_state",
    "rest",
]
