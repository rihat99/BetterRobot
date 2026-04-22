"""``solve_trajopt`` — kinematic trajectory optimisation facade.

Thin wrapper that flattens a ``(T, nq)`` trajectory into the standard
``LeastSquaresProblem`` vector, installs a custom retraction using
``Model.integrate`` per-timestep, and delegates the inner loop to any
BetterRobot :class:`~better_robot.optim.optimizers.base.Optimizer`.

The user assembles their own :class:`CostStack` — ``solve_trajopt`` takes
no pose-target / keyframe / smoothness-weight kwargs of its own. Keyframes
are expressed as ``TimeIndexedResidual(PoseResidual(...), t_idx=i)`` in
the cost stack; this matches ``solve_ik``'s "target is a cost" convention.

Future expansion path (``docs/08_TASKS.md §3`` — dynamics milestone):
dynamics residuals slot into the same ``CostStack``. Torque-as-variable
scenarios will warrant a sibling ``solve_dyn_trajopt`` task.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..costs.stack import CostStack
from ..data_model.model import Model
from ..kinematics.forward import forward_kinematics
from ..kinematics.jacobian_strategy import JacobianStrategy
from ..optim.optimizers.base import Optimizer
from ..optim.problem import LeastSquaresProblem
from ..residuals.base import ResidualState
from .trajectory import Trajectory


@dataclass
class TrajOptResult:
    """Return type of ``solve_trajopt``.

    The optimised trajectory is packed into :class:`Trajectory` with a
    leading batch of 1 — use ``result.trajectory.q[0]`` to recover
    ``(T, nq)``.
    """

    trajectory: Trajectory
    residual: torch.Tensor
    iters: int
    converged: bool
    model: Model


def solve_trajopt(
    model: Model,
    *,
    horizon: int,
    dt: float,
    initial_q_traj: torch.Tensor,
    cost_stack: CostStack,
    optimizer: Optimizer,
    max_iter: int = 50,
    jacobian_strategy: JacobianStrategy = JacobianStrategy.AUTO,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
) -> TrajOptResult:
    """Kinematic trajectory optimisation.

    Parameters
    ----------
    model : Model
    horizon : int
        ``T`` — number of timesteps. Must match ``initial_q_traj.shape[0]``.
    dt : float
        Timestep in seconds — stored on the returned :class:`Trajectory`.
    initial_q_traj : Tensor
        Shape ``(T, nq)``. For floating-base robots the first 7 components
        of each row are the base pose.
    cost_stack : CostStack
        User-built cost composition. Residuals are expected to consume
        ``state.variables`` of shape ``(T, nq)`` and return flat vectors
        with dense ``(dim, T*nv)`` Jacobians (see
        :class:`~better_robot.residuals.smoothness.AccelerationResidual`).
    optimizer : Optimizer
        BetterRobot optimizer instance — :class:`LevenbergMarquardt` for
        small ``T`` problems where a dense Jacobian fits in memory;
        :class:`LBFGS` for larger ones. The human-motion pipeline in
        ``BetterHumanForce`` bypasses this and uses ``torch.optim.LBFGS``
        directly to avoid materialising a huge Jacobian.

    Returns
    -------
    TrajOptResult

    See ``docs/08_TASKS.md §3``.
    """
    if initial_q_traj.dim() != 2:
        raise ValueError(
            f"initial_q_traj must be (T, nq); got {tuple(initial_q_traj.shape)}"
        )
    T, nq = initial_q_traj.shape
    if T != horizon:
        raise ValueError(
            f"horizon={horizon} inconsistent with initial_q_traj.shape[0]={T}"
        )
    if nq != model.nq:
        raise ValueError(
            f"initial_q_traj.shape[1]={nq} != model.nq={model.nq}"
        )

    nv = model.nv
    device, dtype = initial_q_traj.device, initial_q_traj.dtype

    # Flat x vector: (T * nq,); tangent step: (T * nv,).
    x0 = initial_q_traj.detach().clone().reshape(-1)

    def _state_factory(x: torch.Tensor) -> ResidualState:
        q_traj = x.reshape(T, nq)
        data = forward_kinematics(model, q_traj, compute_frames=True)
        return ResidualState(model=model, data=data, variables=q_traj)

    def _retract(x: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
        q_flat = x.reshape(T, nq)
        dv_flat = dv.reshape(T, nv)
        return model.integrate(q_flat, dv_flat).reshape(-1)

    # Per-timestep box constraints, tiled across T.
    def _tile_limit(lim: torch.Tensor | None) -> torch.Tensor | None:
        if lim is None:
            return None
        if lim.shape != (nq,):
            raise ValueError(f"expected limit of shape ({nq},); got {tuple(lim.shape)}")
        return lim.repeat(T)

    problem = LeastSquaresProblem(
        cost_stack=cost_stack,
        state_factory=_state_factory,
        x0=x0,
        lower=_tile_limit(lower),
        upper=_tile_limit(upper),
        jacobian_strategy=jacobian_strategy,
        nv=T * nv,
        retract=_retract,
    )

    solver_state = optimizer.minimize(problem, max_iter=max_iter)

    q_opt = solver_state.x.detach().reshape(T, nq)
    t_axis = torch.linspace(0.0, (T - 1) * dt, T, dtype=dtype, device=device)
    trajectory = Trajectory(
        t=t_axis.unsqueeze(0),                          # (1, T)
        q=q_opt.unsqueeze(0),                           # (1, T, nq)
        model_id=getattr(model, "id", -1),
    )

    return TrajOptResult(
        trajectory=trajectory,
        residual=solver_state.residual,
        iters=int(solver_state.iters),
        converged=(solver_state.status == "converged"),
        model=model,
    )
