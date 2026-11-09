"""Kinematic terms and the configuration manifold of a robot model.

Each factory returns a ``better_robot.optim.TermFn``. Kinematic terms read a cached
``better_robot.kinematics.KinematicState`` (register it with ``kinematic_state``)
so one forward-kinematics pass serves every term.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import final

import torch

from ..exceptions import ArgumentError
from ..kinematics import (
    KinematicState,
    difference,
    euclidean_columns,
    forward_kinematics,
    frame_jacobian,
    integrate,
)
from ..lie import se3, tangents
from ..model import JointFreeFlyer, Model
from ..optim import Manifold, TermFn


@final
@dataclass(frozen=True)
class RobotManifold(Manifold):
    """Configuration manifold of ``model``: ``q: (B, nq)`` or ``(B, T, nq)``, tangent ``nv`` per configuration.

    ``retract`` is ``kinematics.integrate``; ``project`` integrates a zero step, which renormalizes quaternion joints.
    Pass it as ``Problem(..., manifolds={"q": RobotManifold(model)})``. A frozen value: its model cannot be swapped.
    """

    model: Model

    def tangent_dim(self, x: torch.Tensor) -> int:
        return x.shape[1:].numel() // self.model.nq * self.model.nv

    def retract(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return integrate(self.model, x, delta.reshape(*x.shape[:-1], self.model.nv))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return integrate(self.model, x, x.new_zeros(*x.shape[:-1], self.model.nv))

    def tangent_bounds(
        self, x: torch.Tensor, *, lower: torch.Tensor, upper: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Step range ``(B, nv), (B, nv)`` at ``x: (B, nq)`` for bounds ``lower, upper: (nq,)`` or ``(B, nq)``.

        Only the coordinates on the ``EUCLIDEAN`` manifold are bounded; every other coordinate is unbounded.
        """
        q_columns, v_columns = euclidean_columns(self.model)
        low = x.new_full((*x.shape[:-1], self.model.nv), -float("inf"))
        high = x.new_full((*x.shape[:-1], self.model.nv), float("inf"))
        low[..., v_columns] = (lower - x)[..., q_columns]
        high[..., v_columns] = (upper - x)[..., q_columns]
        return low.flatten(1), high.flatten(1)


def kinematic_state(model: Model, *, q: str = "q") -> TermFn:
    """Cached value: forward kinematics of variable ``q``. Register with ``problem.cached(..., name="state")``."""
    return TermFn(fn=lambda value: forward_kinematics(model, value), reads=(q,))


def frame_pose(model: Model, *, frame: int | str, target: str, state: str = "state", q: str = "q") -> TermFn:
    """Pose error of ``frame`` against data ``target: (B, 7)``. Residual ``(B, 6)``, ``[linear, angular]``."""
    frame_id = model.frame_id(frame)

    def error(kinematics: KinematicState, goal: torch.Tensor) -> torch.Tensor:
        return se3.log(se3.compose(se3.inverse(goal), kinematics.frame_pose_of(frame_id)))

    def jacobian(kinematics: KinematicState, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        local = frame_jacobian(model, kinematics, frame=frame_id, reference="local")
        return {q: tangents.right_jacobian_inverse_se3(error(kinematics, goal)) @ local}

    return TermFn(fn=error, reads=(state, target), jacobian=jacobian)


def frame_position(model: Model, *, frame: int | str, target: str, state: str = "state", q: str = "q") -> TermFn:
    """Position error of ``frame`` against data ``target: (B, 3)``. Residual ``(B, 3)``."""
    frame_id = model.frame_id(frame)

    def error(kinematics: KinematicState, goal: torch.Tensor) -> torch.Tensor:
        return kinematics.frame_translations[..., frame_id, :] - goal

    def jacobian(kinematics: KinematicState, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        return {q: frame_jacobian(model, kinematics, frame=frame_id)[..., :3, :]}

    return TermFn(fn=error, reads=(state, target), jacobian=jacobian)


def _all_euclidean(model: Model) -> bool:
    """Whether every public coordinate is on the ``EUCLIDEAN`` manifold, so a tangent step adds to ``q`` itself."""
    return euclidean_columns(model)[0].numel() == model.nq


def _identity_jacobian(model: Model, q: str, sign_fn: Callable[[torch.Tensor], torch.Tensor]):
    """Diagonal Jacobian when every coordinate is Euclidean; else ``None``: autodiff, right on any manifold."""
    if not _all_euclidean(model):
        return None
    return lambda value: {q: torch.diag_embed(sign_fn(value))}


def joint_limits(model: Model, *, q: str = "q") -> TermFn:
    """Violation of the position limits, zero inside them. Residual ``(B, nq)``."""
    lower, upper = model.lower_position_limit, model.upper_position_limit

    def violation(value: torch.Tensor) -> torch.Tensor:
        return (value - upper).clamp_min(0.0) - (lower - value).clamp_min(0.0)

    def slope(value: torch.Tensor) -> torch.Tensor:
        return ((value > upper) | (value < lower)).to(value.dtype)

    return TermFn(fn=violation, reads=(q,), jacobian=_identity_jacobian(model, q, slope))


def rest(model: Model, q_rest: torch.Tensor, *, q: str = "q") -> TermFn:
    """Tangent difference to a rest configuration ``q_rest: (nq,)`` or ``(B, nq)``. Residual ``(B, nv)``."""

    def offset(value: torch.Tensor) -> torch.Tensor:
        return difference(model, q_rest.expand_as(value), value)

    return TermFn(fn=offset, reads=(q,), jacobian=_identity_jacobian(model, q, torch.ones_like))


def finite_difference(model: Model, *, order: int, dt: float, q: str = "q", world_translation: bool = False) -> TermFn:
    """Time derivative of order ``order`` of a trajectory ``q: (B, T, nq)`` in tangent space, by finite differences.

    The steps ``difference(model, q[t], q[t + 1]) / dt`` are differenced ``order - 1`` more times (the binomial
    stencil), each time divided by ``dt``: velocity (1), acceleration (2), jerk (3), snap (4).
    ``world_translation=True`` steps each free flyer's position in its parent frame (the world for a floating base)
    instead of taking the translation of the SE3 log, which is expressed in the turning base frame; its rotation step
    stays the body-frame log. Residual ``(B, T - order, nv)``; a time-local term with ``window = order + 1``.
    """
    if order < 1:
        raise ArgumentError(f"order must be at least 1, got {order}")
    # A free flyer's q and v both start with its position [tx, ty, tz].
    free_flyers = [joint for joint, kind in enumerate(model.joints) if isinstance(kind, JointFreeFlyer)]
    position_q = torch.tensor(
        [model.idx_qs[j] + k for j in free_flyers for k in range(3)], dtype=torch.long, device=model.device
    )
    position_v = torch.tensor(
        [model.idx_vs[j] + k for j in free_flyers for k in range(3)], dtype=torch.long, device=model.device
    )

    def derivative(value: torch.Tensor) -> torch.Tensor:
        out = difference(model, value[:, :-1], value[:, 1:])
        if world_translation:
            out = out.index_copy(-1, position_v, value[:, 1:, position_q] - value[:, :-1, position_q])
        out = out / dt
        for _ in range(order - 1):
            out = (out[:, 1:] - out[:, :-1]) / dt
        return out

    jacobian = None
    if _all_euclidean(model):
        # Built in float64 and rounded once to the working dtype by ``.to(value)``; a float32 stencil was off by
        # ~1e-7 (relative) in a float64 problem.
        coefficients = [(-1.0) ** (order - i) * math.comb(order, i) for i in range(order + 1)]
        stencil = torch.tensor(coefficients, dtype=torch.float64) / dt**order

        def jacobian(value: torch.Tensor) -> dict[str, torch.Tensor]:
            eye = torch.eye(model.nv, dtype=value.dtype, device=value.device)
            block = stencil.to(value)[:, None, None] * eye  # (window, nv, nv)
            return {q: block.permute(1, 0, 2).expand(value.shape[0], value.shape[1] - order, -1, -1, -1)}

    return TermFn(fn=derivative, reads=(q,), jacobian=jacobian, window=order + 1)


def velocity_limit(model: Model, *, dt: float, fraction: float = 1.0, q: str = "q") -> TermFn:
    """Excess of the step velocity of ``q: (B, T, nq)`` over ``fraction`` of ``model.velocity_limit``, zero inside it.

    Velocities are ``finite_difference(order=1)``, signed like them. A stored limit of 0 means unspecified (MJCF, URDF
    without ``velocity``, continuous / planar / translation joints) and does not bind, like an infinite one.
    Residual ``(B, T - 1, nv)``; ``window = 2``.
    """
    velocity = finite_difference(model, order=1, dt=dt, q=q).fn
    limit = torch.where(model.velocity_limit > 0, fraction * model.velocity_limit, torch.inf)

    def excess(value: torch.Tensor) -> torch.Tensor:
        speed = velocity(value)
        return (speed - limit).clamp_min(0.0) - (-limit - speed).clamp_min(0.0)

    return TermFn(fn=excess, reads=(q,), window=2)


__all__ = [
    "RobotManifold",
    "finite_difference",
    "frame_pose",
    "frame_position",
    "joint_limits",
    "kinematic_state",
    "rest",
    "velocity_limit",
]
