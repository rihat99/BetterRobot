"""Kinematic terms and the configuration manifold of a robot model.

Each factory returns a ``better_robot.optim.TermFn``. Kinematic terms read a cached
``better_robot.kinematics.KinematicState`` (register it with ``kinematic_state``)
so one forward-kinematics pass serves every term.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import final, override

import torch

from ..exceptions import ArgumentError, ShapeError
from ..kinematics import (
    KinematicState,
    difference,
    euclidean_columns,
    forward_kinematics,
    frame_jacobian,
    integrate,
    support,
)
from ..lie import se3, so3, tangents
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

    @override
    def tangent_dim(self, x: torch.Tensor) -> int:
        return x.shape[1:].numel() // self.model.nq * self.model.nv

    @override
    def retract(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return integrate(self.model, x, delta.reshape(*x.shape[:-1], self.model.nv))

    @override
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
    """Pose error of ``frame`` against data ``target: (B, 7)``. Residual ``(B, 6)``, ``[linear, angular]``: the
    ``se3.log`` of ``target⁻¹ ∘ pose``. At a rotation error of exactly pi the log's axis sign is a cut: either sign
    is a valid error, and which one comes out depends on rounding."""
    frame_id = model.frame_id(frame)

    # The solver calls these on every iteration with inputs ik_problem (or the caller) checked once: raw kernels (principle 2.8).
    def relative(kinematics: KinematicState, goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Translation ``(B, 3)`` and quaternion ``(B, 4)`` of ``goal⁻¹ ∘ pose``, from the state's ``[R | p]``."""
        transform = kinematics.frame_transforms[..., frame_id, :, :]
        inverse = so3.to_matrix_raw(goal[..., 3:]).mT
        translation = (inverse @ (transform[..., 3] - goal[..., :3]).unsqueeze(-1)).squeeze(-1)
        return translation, so3.from_matrix_raw(inverse @ transform[..., :3])

    def error(kinematics: KinematicState, goal: torch.Tensor) -> torch.Tensor:
        return se3.log_raw(torch.cat(relative(kinematics, goal), dim=-1))

    def jacobian(kinematics: KinematicState, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        # A body twist ξ of the frame moves the error [ρ, ω] = [V⁻¹(ω) p, ω] by dω = A ξ_ω, A = Jr⁻¹(ω), and dρ =
        # A ξ_v + M A ξ_ω (V⁻¹ R = Jr⁻¹), with V⁻¹(ω) = I - Ω/2 + c Ω² and M = ∂(V⁻¹(ω) p)/∂ω at fixed p.
        p, quaternion = relative(kinematics, goal)
        w = so3.log_raw(quaternion)
        theta2 = (w * w).sum(-1, keepdim=True)
        small = theta2 < 0.04  # below θ = 0.2 the closed forms cancel; the series' next terms are below 1e-16
        s = torch.where(small, torch.ones_like(theta2), theta2)
        theta = s.sqrt()
        cot = 1.0 / torch.tan(0.5 * theta)
        series = 1 / 12 + theta2 * (1 / 720 + theta2 * (1 / 30240 + theta2 * (1 / 1209600 + theta2 / 47900160)))
        c = torch.where(small, series, 1.0 / s - cot / (2.0 * theta))
        series = 1 / 720 + theta2 * (1 / 15120 + theta2 * (1 / 403200 + theta2 / 11975040))
        dc = torch.where(small, series, (cot / (4.0 * theta) + (1.0 + cot * cot) / 8.0 - 1.0 / s) / s)  # dc / dθ²
        skew = tangents.hat_so3_raw(w)
        eye = torch.eye(3, dtype=w.dtype, device=w.device)
        a = eye + 0.5 * skew + c.unsqueeze(-1) * (skew @ skew)  # A
        wp = (w * p).sum(-1, keepdim=True)
        # M = ½ P̂ + c ((ω·p) I + ω pᵀ - 2 p ωᵀ) + 2 c' (Ω² p) ωᵀ, Ω² p = ω (ω·p) - θ² p
        u = 2.0 * (dc * (w * wp - p * theta2) - c * p)
        m = 0.5 * tangents.hat_so3_raw(p) + (c * wp).unsqueeze(-1) * eye + (c * w).unsqueeze(-1) * p.unsqueeze(-2)
        m = m + u.unsqueeze(-1) * w.unsqueeze(-2)  # M
        local = frame_jacobian(model, kinematics, frame=frame_id, reference="local")  # (B, 6, nv), [linear, angular]
        angular = a @ local[..., 3:, :]
        return {q: torch.cat([a @ local[..., :3, :] + m @ angular, angular], dim=-2)}

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


def _band(width: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(1 / width, width / 2)``, the constants of `_hinge` for a band ``width >= 0``; a zero width is a plain hinge."""
    return 1.0 / width.clamp_min(torch.finfo(width.dtype).tiny), width / 2


def _hinge_slope(gap: torch.Tensor, band: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Slope of `_hinge`: 0 for ``gap <= 0``, rising linearly to 1 across the band, then 1. Same shape as ``gap``."""
    return (gap.clamp_min(0.0) * band[0]).clamp_max(1.0)


def _hinge(gap: torch.Tensor, band: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """C1 hinge of ``gap`` over a band of width ``w`` (`_band`): 0 for ``gap <= 0``, ``gap² / (2 w)`` up to ``w``, then
    ``gap - w / 2``.

    The plain hinge ``max(gap, 0)`` (``w = 0``) has a step in its slope, so the Gauss-Newton model of a term jumps when
    it starts to act; this slope ramps across the band. With ``s`` the slope the value is ``s (gap - s w / 2)``.
    """
    rise = gap.clamp_min(0.0)
    slope = (rise * band[0]).clamp_max(1.0)
    return slope * (rise - band[1] * slope)


def joint_limits(model: Model, *, margin: float | torch.Tensor = 0.0, q: str = "q") -> TermFn:
    """Push away from the position limits, starting ``margin`` inside them. Residual ``(B, nq)``, signed like ``q``.

    ``margin`` is a number or ``(nq,)`` in each coordinate's unit (radians or metres). It applies to the Euclidean
    coordinates only (`kinematics.euclidean_columns`): a quaternion or ``(cos, sin)`` coordinate keeps its plain
    limits. A coordinate whose range is narrower than twice ``margin`` gets half its range, so its residual is zero
    at mid-range only. The residual is zero while a coordinate stays ``margin`` inside its limits; past that point,
    by ``gap``, it is ``gap² / (2 margin)``, the C1 hinge of `sphere_clearance`, so it is ``margin / 2`` at the limit
    and ``gap - margin / 2`` beyond it. With the default 0 it is the violation of the limits.
    """
    band = torch.as_tensor(margin, dtype=model.dtype, device=model.device)
    if band.shape not in ((), (model.nq,)):
        raise ShapeError(f"margin must be a number or have shape ({model.nq},), got {tuple(band.shape)}")
    half_range = (model.upper_position_limit - model.lower_position_limit) / 2
    width = torch.zeros_like(half_range)
    euclidean = euclidean_columns(model)[0]
    width[euclidean] = torch.minimum(band, half_range)[euclidean]
    lower, upper = model.lower_position_limit + width, model.upper_position_limit - width
    band = _band(width.clamp_min(0.0))

    def violation(value: torch.Tensor) -> torch.Tensor:
        return _hinge(value - upper, band) - _hinge(lower - value, band)

    def slope(value: torch.Tensor) -> torch.Tensor:
        return _hinge_slope(value - upper, band) + _hinge_slope(lower - value, band)

    return TermFn(fn=violation, reads=(q,), jacobian=_identity_jacobian(model, q, slope))


def _yoshikawa(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjugate ``(B..., 3, 3)`` of ``rows rowsᵀ`` and Yoshikawa's ``sqrt(det(rows rowsᵀ))`` ``(B...)`` of ``rows (B..., 3, n)``."""
    gram = rows @ rows.mT
    # Row a of the adjugate of a symmetric 3x3 matrix is the cross product of its other two rows.
    adjugate = torch.linalg.cross(gram[..., [1, 2, 0], :], gram[..., [2, 0, 1], :])
    det = (gram[..., 0, :] * adjugate[..., 0, :]).sum(-1)
    # where, not clamp_min: at det = +0.0 clamp_min passes sqrt'(0) = inf on to a zero derivative, NaN in autodiff.
    return adjugate, torch.where(det > 0, det, 0.0).sqrt()


def manipulability(
    model: Model, *, frame: int | str, threshold: float, hinge: bool = True, state: str = "state", q: str = "q"
) -> TermFn:
    """Push ``frame`` away from singular configurations once its manipulability falls below ``threshold``.

    The manipulability is Yoshikawa's ``w = sqrt(det(J Jᵀ))`` of the position rows ``J (3, nv)`` of the frame's
    Jacobian without the columns of free-flyer joints, which move the frame in every direction. Residual ``(B, 1)``:
    the C1 hinge of ``threshold - w`` with band ``threshold`` (`joint_limits`), zero while ``w >= threshold``; with
    ``hinge=False`` the plain ``threshold - w``, the row of ``problem.inequality(..., upper=0.0)``. The Jacobian is
    analytic (each column of the world Jacobian moves by its Lie bracket with the columns above it), or autodiff when
    a mimic joint drives the frame. At ``w = 0`` it is zero, the smallest subgradient: a row there cannot hold and
    gives way (`optim.solve_qp`).
    """
    if threshold <= 0:
        raise ArgumentError(f"threshold must be positive, got {threshold}")
    frame_id = model.frame_id(frame)
    width = torch.tensor(threshold, dtype=model.dtype, device=model.device)
    band = _band(width)
    owner = [0] * model.nv  # the joint of each tangent column
    for joint in range(model.njoints):
        for column in range(model.nvs[joint]):
            owner[model.idx_vs[joint] + column] = joint
    columns = torch.tensor(
        [not isinstance(model.joints[joint], JointFreeFlyer) for joint in owner], dtype=model.dtype, device=model.device
    )

    def residual(kinematics: KinematicState) -> torch.Tensor:
        _, size = _yoshikawa(frame_jacobian(model, kinematics, frame=frame_id)[..., :3, :] * columns)
        return (_hinge(width - size, band) if hinge else width - size)[..., None]

    chain = support(model, joint=model.frame_parent_joint_ids[frame_id])
    if any(model.mimic_source_ids[joint] != joint for joint in chain):
        return TermFn(fn=residual, reads=(state,))
    chains = {joint: support(model, joint=joint) for joint in set(owner)}
    # above[i, k]: column i belongs to joint k's support, so moving it carries column k (pairs within one joint too).
    above = torch.tensor(
        [[owner[i] in chains[owner[k]] for k in range(model.nv)] for i in range(model.nv)],
        dtype=model.dtype,
        device=model.device,
    )

    def jacobian(kinematics: KinematicState) -> dict[str, torch.Tensor]:
        world = frame_jacobian(model, kinematics, frame=frame_id, reference="world")  # (B..., 6, nv)
        linear, angular = world[..., :3, :], world[..., 3:, :]
        origin = kinematics.frame_translations[..., frame_id, :, None].expand_as(angular)
        rows = linear + torch.linalg.cross(angular, origin, dim=-2)  # velocity of the frame origin
        adjugate, size = _yoshikawa(rows * columns)
        # dw = <P, dJ> with P = w (J Jᵀ)⁻¹ J = adj(J Jᵀ) J / w, finite as w -> 0. At w = 0, the tip of a cone where
        # adj(J Jᵀ) J is rounding noise, P is 0: the smallest subgradient, which autodiff of `_yoshikawa` gives.
        weights = adjugate @ (rows * columns) / torch.where(size > 0, size, torch.inf)[..., None, None]
        # dJ_k/dq_i = above[i, k] [J_i, J_k] (world columns) plus the moving origin; triple products fold the sum over k.
        first = torch.linalg.cross(linear, weights, dim=-2) + torch.linalg.cross(
            angular, torch.linalg.cross(origin, weights, dim=-2), dim=-2
        )
        second = torch.linalg.cross(angular, weights, dim=-2)
        third = torch.linalg.cross(weights, angular, dim=-2).sum(-1, keepdim=True)
        gradient = (
            (angular * (first @ above.mT)).sum(-2) + (linear * (second @ above.mT)).sum(-2) + (rows * third).sum(-2)
        )
        if hinge:
            gradient = _hinge_slope(width - size, band)[..., None] * gradient
        return {q: -gradient[..., None, :]}

    return TermFn(fn=residual, reads=(state,), jacobian=jacobian)


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

    jacobian: Callable[[torch.Tensor], dict[str, torch.Tensor]] | None = None
    if _all_euclidean(model):
        # Built in float64 and rounded once to the working dtype by ``.to(value)``; a float32 stencil was off by
        # ~1e-7 (relative) in a float64 problem.
        coefficients = [(-1.0) ** (order - i) * math.comb(order, i) for i in range(order + 1)]
        stencil = torch.tensor(coefficients, dtype=torch.float64) / dt**order

        def stencil_jacobian(value: torch.Tensor) -> dict[str, torch.Tensor]:
            eye = torch.eye(model.nv, dtype=value.dtype, device=value.device)
            block = stencil.to(value)[:, None, None] * eye  # (window, nv, nv)
            return {q: block.permute(1, 0, 2).expand(value.shape[0], value.shape[1] - order, -1, -1, -1)}

        jacobian = stencil_jacobian

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
    "manipulability",
    "rest",
    "velocity_limit",
]
