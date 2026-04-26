"""``TimeIndexedResidual`` — wrap a point-in-time residual for trajectory state.

Most residuals (``PoseResidual``, ``JointPositionLimit``) operate on a
single configuration. When the optimization variable is a whole trajectory
``(T, nq)``, we need to slice out a single timestep, call the inner
residual, and — for the analytic Jacobian — scatter the resulting block
into the correct columns of a ``(dim_inner, T*nv)`` matrix.

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §2``.
"""

from __future__ import annotations

import torch

from ..data_model.data import Data
from .base import Residual, ResidualState


class TimeIndexedResidual:
    """Evaluate an inner residual at a single timestep of a trajectory state.

    Parameters
    ----------
    inner : Residual
        Single-configuration residual (e.g. ``PoseResidual``,
        ``JointPositionLimit``).
    t_idx : int
        Timestep to evaluate at.
    name : str | None
        Override the default name ``f"{inner.name}_t{t_idx}"``.
    """

    def __init__(self, inner, t_idx: int, *, name: str | None = None) -> None:
        self.inner = inner
        self.t_idx = int(t_idx)
        self.name = name if name is not None else f"{inner.name}_t{t_idx}"
        self.dim = int(inner.dim)

    def _slice_state(self, state: ResidualState) -> ResidualState:
        q = state.variables
        if q.dim() != 2:
            raise ValueError(
                f"TimeIndexedResidual expects (T, nq); got {tuple(q.shape)}"
            )
        if not (0 <= self.t_idx < q.shape[0]):
            raise IndexError(
                f"t_idx={self.t_idx} out of range for T={q.shape[0]}"
            )

        # Slice the single-timestep view of Data. frame_pose_world is (T, nframes, 7);
        # joint_pose_world is (T, njoints, 7). Any fields that are None stay None.
        from ..data_model._kinematics_level import KinematicsLevel

        data_sliced = Data(
            _model_id=state.data._model_id,
            q=q[self.t_idx],
            joint_pose_world=(
                state.data.joint_pose_world[self.t_idx]
                if state.data.joint_pose_world is not None
                else None
            ),
            frame_pose_world=(
                state.data.frame_pose_world[self.t_idx]
                if state.data.frame_pose_world is not None
                else None
            ),
            joint_jacobians=None,  # recompute if needed at this timestep
        )
        # The slice carries already-populated FK fields, so promote the
        # cache-level so downstream residuals (PoseResidual, frame Jacobians)
        # don't trip the StaleCacheError guard.
        if state.data.joint_pose_world is not None:
            object.__setattr__(data_sliced, "_kinematics_level", KinematicsLevel.PLACEMENTS)
        return ResidualState(model=state.model, data=data_sliced, variables=q[self.t_idx])

    def __call__(self, state: ResidualState) -> torch.Tensor:
        sub = self._slice_state(state)
        r = self.inner(sub)  # (dim,)
        self.dim = int(r.numel())
        return r

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        sub = self._slice_state(state)
        J_inner = self.inner.jacobian(sub)  # (dim, nv) or None
        if J_inner is None:
            return None
        q = state.variables
        T, _ = q.shape
        nv = state.model.nv
        dim = int(J_inner.shape[0])

        J = torch.zeros(dim, T * nv, device=q.device, dtype=q.dtype)
        J[:, self.t_idx * nv:(self.t_idx + 1) * nv] = J_inner
        return J

    def apply_jac_transpose(
        self, state: ResidualState, vec: torch.Tensor
    ) -> torch.Tensor:
        """Sparse ``J^T @ vec`` — only ``t_idx`` knot is non-zero.

        Avoids allocating the dense ``(dim, T·nv)`` Jacobian — important
        for long trajectories where ``T·nv`` can be tens of thousands.
        """
        sub = self._slice_state(state)
        J_inner = self.inner.jacobian(sub)
        if J_inner is None:
            from .base import default_apply_jac_transpose

            return default_apply_jac_transpose(self, state, vec)
        T = state.variables.shape[0]
        nv = state.model.nv
        out = torch.zeros(T * nv, device=vec.device, dtype=vec.dtype)
        out[self.t_idx * nv:(self.t_idx + 1) * nv] = J_inner.mT @ vec
        return out

    @property
    def spec(self):
        from ..optim.jacobian_spec import ResidualSpec

        return ResidualSpec(
            dim=self.dim,
            structure="block",
            time_coupling="single",
            affected_knots=(self.t_idx,),
        )
