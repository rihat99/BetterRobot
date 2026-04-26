"""``solve_trajopt`` — kinematic trajectory optimisation facade.

Thin wrapper that flattens a ``(T, nq)`` trajectory into the standard
``LeastSquaresProblem`` vector, installs a custom retraction using
``Model.integrate`` per-timestep, and delegates the inner loop to any
BetterRobot :class:`~better_robot.optim.optimizers.base.Optimizer`.

The user assembles their own :class:`CostStack` — ``solve_trajopt`` takes
no pose-target / keyframe / smoothness-weight kwargs of its own. Keyframes
are expressed as ``TimeIndexedResidual(PoseResidual(...), t_idx=i)`` in
the cost stack; this matches ``solve_ik``'s "target is a cost" convention.

Future expansion path (``docs/design/08_TASKS.md §3`` — dynamics milestone):
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
from .parameterization import (
    BSplineTrajectory,
    KnotTrajectory,
    TrajectoryParameterization,
)
from .trajectory import Trajectory


class _ChainRuleProblem(LeastSquaresProblem):
    """Wraps a parameterised trajopt problem in a chain-ruled Jacobian.

    ``state.variables`` holds the *expanded* ``q_traj`` (so existing
    residuals are unmodified). The optimisation variable is the smaller
    ``z``. We keep the standard ``residual`` definition and override
    ``jacobian`` to post-multiply by ``dq_traj/dz`` — turning a
    ``(dim, T·nq)`` residual Jacobian into ``(dim, C·nq)`` in z-space.
    """

    def __init__(self, *, dq_dz: torch.Tensor, **kwargs) -> None:
        # ``dq_dz`` has shape ``(T·nq, C·nq)`` — Kronecker block diag of
        # the per-feature basis with the identity. Stored as a tensor for
        # straight matmul.
        super().__init__(**kwargs)
        self._dq_dz = dq_dz

    def jacobian(self, x):
        from ..residuals.base import ResidualState

        # Build state in z-space: variables=z (so residuals operating on
        # parameter-space see the right shape) — but FK uses expanded q_traj.
        state = self.state_factory(x)
        from ..costs.stack import CostStack
        cs: CostStack = self.cost_stack
        J_q = cs.jacobian(state, strategy=self.jacobian_strategy)  # (dim, T*nq)
        return J_q @ self._dq_dz  # (dim, C*nq)


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
    parameterization: TrajectoryParameterization | None = None,
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

    See ``docs/design/08_TASKS.md §3``.
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

    if parameterization is None:
        parameterization = KnotTrajectory()

    # ``z`` is the optimisation variable; ``q_traj = parameterization.expand(z)``.
    z0 = parameterization.init(initial_q_traj)
    z_shape = tuple(z0.shape)
    x0 = z0.detach().clone().reshape(-1)

    def _state_factory(x: torch.Tensor) -> ResidualState:
        z = x.reshape(z_shape)
        q_traj = parameterization.expand(z, T=T, nq=nq)
        data = forward_kinematics(model, q_traj, compute_frames=True)
        return ResidualState(model=model, data=data, variables=q_traj)

    def _retract(x: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
        # When the parameterisation is identity (KnotTrajectory) we still
        # apply the per-knot ``Model.integrate`` retraction. Otherwise we
        # use Euclidean addition on the control points — the spline basis
        # carries the smoothness, and the SE3 residuals see the expanded
        # trajectory through ``state_factory``.
        if isinstance(parameterization, KnotTrajectory):
            q_flat = x.reshape(T, nq)
            dv_flat = dv.reshape(T, nv)
            return model.integrate(q_flat, dv_flat).reshape(-1)
        return x + dv

    # Per-timestep box constraints, tiled across T.
    def _tile_limit(lim: torch.Tensor | None) -> torch.Tensor | None:
        if lim is None:
            return None
        if lim.shape != (nq,):
            raise ValueError(f"expected limit of shape ({nq},); got {tuple(lim.shape)}")
        return lim.repeat(T)

    if isinstance(parameterization, KnotTrajectory):
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
    else:
        # Build dq_traj / dz Jacobian: q_flat = B_block @ z_flat where
        # B_block = kron(B, I_nq). For BSplineTrajectory the basis is
        # cached after .init().
        if not hasattr(parameterization, "_basis") or parameterization._basis is None:
            raise RuntimeError(
                f"{type(parameterization).__name__} does not expose a basis "
                f"matrix needed for the Jacobian chain rule"
            )
        B = parameterization._basis  # (T, C)
        I_nq = torch.eye(nq, dtype=dtype, device=device)
        dq_dz = torch.kron(B, I_nq)  # (T*nq, C*nq)
        problem = _ChainRuleProblem(
            dq_dz=dq_dz,
            cost_stack=cost_stack,
            state_factory=_state_factory,
            x0=x0,
            lower=None, upper=None,
            jacobian_strategy=jacobian_strategy,
            nv=int(z0.numel()),
            retract=_retract,
        )

    solver_state = optimizer.minimize(problem, max_iter=max_iter)

    z_opt = solver_state.x.detach().reshape(z_shape)
    q_opt = parameterization.expand(z_opt, T=T, nq=nq)
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
