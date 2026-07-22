"""Inverse contact-force fitting with object-referenced residuals."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .._validation import check_tensor
from ..data_model.model import Model
from ..data_model.model_values import ModelValues
from ..dynamics.rnea import rnea_raw
from ..kinematics.forward import forward_kinematics
from ..lie import so3
from ..optim import LevenbergMarquardt, Problem, Residual, Variable
from ..residuals.nodes import Node


@dataclass(frozen=True)
class ContactForceWeights:
    """Scalar objective coefficients for inverse contact-force terms."""

    base_wrench: float = 1.0
    force_magnitude: float = 1e-4
    force_smooth: float = 0.0
    torque_smooth: float = 0.0

    def __post_init__(self) -> None:
        for name in ("base_wrench", "force_magnitude", "force_smooth", "torque_smooth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"ContactForceWeights.{name} must be a real number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"ContactForceWeights.{name} must be finite and >= 0")


@dataclass(frozen=True)
class ContactForceResult:
    """Result of fitting world-frame point forces to floating-base dynamics."""

    forces_world: torch.Tensor
    fext_local: torch.Tensor
    generalized_force: torch.Tensor
    residual: torch.Tensor
    """Final whitened rows; outer term coefficients are excluded."""
    cost: torch.Tensor
    iters: int | torch.Tensor
    converged: bool | torch.Tensor
    status: torch.Tensor
    model: Model


def _trajectory_derivatives(model: Model, q: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return q-aligned central-difference velocity and acceleration."""

    time = q.shape[-2]
    if time == 1:
        zeros = q.new_zeros((*q.shape[:-2], 1, model.nv))
        return zeros, zeros

    adjacent = model.difference(q[..., :-1, :], q[..., 1:, :]) / dt
    if time == 2:
        velocity = torch.cat((adjacent[..., :1, :], adjacent[..., -1:, :]), dim=-2)
    else:
        central = model.difference(q[..., :-2, :], q[..., 2:, :]) / (2.0 * dt)
        velocity = torch.cat((adjacent[..., :1, :], central, adjacent[..., -1:, :]), dim=-2)

    delta_v = (velocity[..., 1:, :] - velocity[..., :-1, :]) / dt
    if time == 2:
        acceleration = torch.cat((delta_v[..., :1, :], delta_v[..., -1:, :]), dim=-2)
    else:
        central_a = (velocity[..., 2:, :] - velocity[..., :-2, :]) / (2.0 * dt)
        acceleration = torch.cat((delta_v[..., :1, :], central_a, delta_v[..., -1:, :]), dim=-2)
    return velocity, acceleration


class _ContactDynamicsNode(Node):
    def __init__(
        self,
        forces: Variable,
        *,
        model: Model,
        q: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        world_to_local: torch.Tensor,
        active: torch.Tensor,
        contact_to_joint: torch.Tensor,
        values: ModelValues,
    ) -> None:
        self.forces = forces
        self.model = model
        self.q = q
        self.velocity = velocity
        self.acceleration = acceleration
        self.world_to_local = world_to_local
        self.active = active
        self.contact_to_joint = contact_to_joint
        self.values = values
        super().__init__(forces)

    def compute(self) -> dict[str, torch.Tensor]:
        forces_world = self.forces.tensor * self.active[..., None]
        forces_local = (self.world_to_local @ forces_world.unsqueeze(-1)).squeeze(-1)
        by_joint = torch.einsum("...tci,cj->...tji", forces_local, self.contact_to_joint)
        fext_local = torch.cat((by_joint, torch.zeros_like(by_joint)), dim=-1)
        result = rnea_raw(
            self.model.structure,
            self.values,
            self.q,
            self.velocity,
            self.acceleration,
            fext=fext_local,
        )
        return {
            "generalized_force": result.tau,
            "fext_local": fext_local,
        }


class _BaseWrenchResidual(Residual):
    def __init__(self, dynamics: _ContactDynamicsNode, time: int, *, weight: float) -> None:
        self.dynamics = dynamics
        self.nodes = (dynamics,)
        super().__init__(dim=time * 6, weight=weight, group_size=6, name="base_wrench")

    def error(self) -> torch.Tensor:
        tau = self.dynamics.value()["generalized_force"]
        return tau[..., :6].reshape(*tau.shape[:-2], self.dim)


class _ForceMagnitudeResidual(Residual):
    def __init__(self, forces: Variable, time: int, contacts: int, *, weight: float) -> None:
        self.forces = forces
        super().__init__(
            forces,
            dim=time * contacts * 3,
            weight=weight,
            group_size=3,
            name="force_magnitude",
        )

    def error(self) -> torch.Tensor:
        forces = self.forces.tensor
        return forces.reshape(*forces.shape[:-3], self.dim)


class _ForceSmoothResidual(Residual):
    def __init__(self, forces: Variable, time: int, contacts: int, *, weight: float = 1.0) -> None:
        self.forces = forces
        super().__init__(
            forces,
            dim=(time - 1) * contacts * 3,
            weight=weight,
            group_size=3,
            name="force_smooth",
        )

    def error(self) -> torch.Tensor:
        forces = self.forces.tensor
        delta = forces[..., 1:, :, :] - forces[..., :-1, :, :]
        return delta.reshape(*forces.shape[:-3], self.dim)


class _TorqueSmoothResidual(Residual):
    def __init__(
        self,
        dynamics: _ContactDynamicsNode,
        time: int,
        actuated: int,
        *,
        weight: float = 1.0,
    ) -> None:
        self.dynamics = dynamics
        self.nodes = (dynamics,)
        super().__init__(dim=(time - 1) * actuated, weight=weight, name="torque_smooth")

    def error(self) -> torch.Tensor:
        tau = self.dynamics.value()["generalized_force"][..., 6:]
        delta = tau[..., 1:, :] - tau[..., :-1, :]
        return delta.reshape(*tau.shape[:-2], self.dim)


def _gravity_values(
    model: Model,
    gravity: torch.Tensor | None,
    batch_shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> ModelValues:
    if gravity is None:
        return model.values
    gravity = check_tensor("gravity", gravity, dtype=dtype, device=device)
    if gravity.shape[-1:] == (3,):
        gravity = torch.cat((gravity, torch.zeros_like(gravity)), dim=-1)
    if gravity.shape[-1:] != (6,):
        raise ValueError(f"gravity must end in 3 or 6, got {tuple(gravity.shape)}")
    try:
        gravity = torch.broadcast_to(gravity, (*batch_shape, 6))
    except RuntimeError as error:
        raise ValueError(f"gravity is not broadcastable to {(*batch_shape, 6)}") from error
    # RNEA sees time as the final execution-batch axis. A singleton time
    # dimension lets per-clip gravity broadcast over every knot.
    if batch_shape:
        gravity = gravity.unsqueeze(-2)
    return dataclasses.replace(model.values, gravity=gravity)


def _contact_joint_ids(
    value: torch.Tensor | Sequence[int],
    *,
    device: torch.device,
    count: int,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("contact_joint_ids tensor must use an integer dtype")
        result = value.to(device=device, dtype=torch.long)
    else:
        ids = list(value)
        if any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
            raise TypeError("contact_joint_ids must contain integers")
        result = torch.tensor(ids, dtype=torch.long, device=device)
    if result.ndim != 1 or result.numel() == 0:
        raise ValueError("contact_joint_ids must be a non-empty one-dimensional sequence")
    if bool(((result < 0) | (result >= count)).any()):
        raise ValueError(f"contact_joint_ids must lie in [0, {count})")
    return result


def _contact_activity(value: torch.Tensor, shape: tuple[int, ...], exemplar: torch.Tensor) -> torch.Tensor:
    value = check_tensor("active_mask", value, device=exemplar.device)
    if value.dtype != torch.bool and not value.is_floating_point():
        raise TypeError("active_mask must be boolean or floating point")
    if value.is_floating_point() and (
        not bool(torch.isfinite(value).all()) or bool(((value < 0.0) | (value > 1.0)).any())
    ):
        raise ValueError("floating active_mask values must be finite and lie in [0, 1]")
    try:
        return torch.broadcast_to(value, shape).to(dtype=exemplar.dtype)
    except RuntimeError as error:
        raise ValueError(f"active_mask is not broadcastable to {shape}") from error


def _initial_forces(value: torch.Tensor | None, shape: tuple[int, ...], exemplar: torch.Tensor) -> torch.Tensor:
    if value is None:
        return exemplar.new_zeros(shape)
    value = check_tensor("initial_forces", value, dtype=exemplar.dtype, device=exemplar.device)
    if tuple(value.shape) != shape:
        raise ValueError(f"initial_forces must have shape {shape}, got {tuple(value.shape)}")
    return value.detach().clone()


def solve_contact_forces(  # noqa: PLR0912, PLR0915 - one complete public task boundary
    model: Model,
    q_traj: torch.Tensor,
    contact_joint_ids: torch.Tensor | Sequence[int],
    active_mask: torch.Tensor,
    *,
    dt: float,
    gravity: torch.Tensor | None = None,
    initial_forces: torch.Tensor | None = None,
    weights: ContactForceWeights | None = None,
    max_iter: int = 50,
    damping_parameter: float = 1e-3,
    tolerance: float = 1e-6,
) -> ContactForceResult:
    """Fit world-frame contact forces for a frozen floating-base trajectory.

    ``q_traj`` has event shape ``(T, model.nq)`` and may carry arbitrary
    independent leading batch axes. ``active_mask`` is broadcast to
    ``(B..., T, C)``. Forces are optimized as one Euclidean ``Variable``;
    a lazy ``Node`` scatters them into local external wrenches and evaluates
    RNEA once per residual/Jacobian evaluation.

    Each ``contact_joint_ids`` entry places its force at that joint's origin.
    The resulting local wrench is ``[force, torque=0]``. Arbitrary contact
    offsets are not represented: a force applied at offset ``r`` would also
    contribute the moment ``r × force``.
    """

    if model.nv < 6 or len(model.joint_models) < 2 or model.joint_models[1].kind != "free_flyer":
        raise ValueError("solve_contact_forces requires a floating-base model with a six-dimensional base tangent")
    q_traj = check_tensor("q_traj", q_traj)
    if q_traj.ndim < 2 or q_traj.shape[-2] < 1:
        raise ValueError(f"q_traj must contain at least one timestep, got shape {tuple(q_traj.shape)}")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(float(dt)) or dt <= 0.0:
        raise ValueError("dt must be a finite positive number")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError("max_iter must be a non-negative integer")
    weights = ContactForceWeights() if weights is None else weights
    if not isinstance(weights, ContactForceWeights):
        raise TypeError("weights must be ContactForceWeights or None")

    data = forward_kinematics(model, q_traj)
    q = data.q.detach()
    batch_shape = tuple(q.shape[:-2])
    time = q.shape[-2]
    velocity, acceleration = _trajectory_derivatives(model, q, float(dt))

    ids = _contact_joint_ids(contact_joint_ids, device=q.device, count=model.njoints)
    contacts = ids.numel()
    active = _contact_activity(active_mask, (*batch_shape, time, contacts), q)

    joint_pose = data.joint_pose_world.index_select(-2, ids)
    world_to_local = so3.to_matrix(joint_pose[..., 3:]).mT.detach()
    contact_to_joint = torch.nn.functional.one_hot(ids, num_classes=model.njoints).to(dtype=q.dtype)
    values = _gravity_values(
        model,
        gravity,
        batch_shape,
        dtype=q.dtype,
        device=q.device,
    )

    force_shape = (*batch_shape, time, contacts, 3)
    forces0 = _initial_forces(initial_forces, force_shape, q)

    force_variable = Variable(
        forces0,
        name="forces",
        batch_ndim=len(batch_shape),
    )
    dynamics = _ContactDynamicsNode(
        force_variable,
        model=model,
        q=q,
        velocity=velocity,
        acceleration=acceleration,
        world_to_local=world_to_local,
        active=active,
        contact_to_joint=contact_to_joint,
        values=values,
    )
    residuals = [
        _BaseWrenchResidual(
            dynamics,
            time,
            weight=weights.base_wrench,
        ),
        _ForceMagnitudeResidual(
            force_variable,
            time,
            contacts,
            weight=weights.force_magnitude,
        ),
    ]
    if time > 1:
        residuals.append(
            _ForceSmoothResidual(
                force_variable,
                time,
                contacts,
                weight=weights.force_smooth,
            )
        )
    if time > 1 and model.nv > 6:
        residuals.append(
            _TorqueSmoothResidual(
                dynamics,
                time,
                model.nv - 6,
                weight=weights.torque_smooth,
            )
        )

    problem = Problem(residuals)
    solver = LevenbergMarquardt(
        problem,
        max_iterations=max_iter,
        damping=damping_parameter,
        tolerance=tolerance,
    )
    info = solver.optimize()
    forces = force_variable.tensor
    diagnostics = dynamics.compute()
    residual = problem.error().detach()
    iters: int | torch.Tensor = info.iterations
    converged: bool | torch.Tensor = info.converged
    if not batch_shape:
        iters = int(info.iterations)
        converged = bool(info.converged)
    return ContactForceResult(
        forces_world=forces,
        fext_local=diagnostics["fext_local"],
        generalized_force=diagnostics["generalized_force"],
        residual=residual,
        cost=info.cost,
        iters=iters,
        converged=converged,
        status=info.status,
        model=model,
    )


__all__ = ["ContactForceResult", "ContactForceWeights", "solve_contact_forces"]
