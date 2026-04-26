"""Contact-consistency residual: penalise cartesian motion of contact frames.

Given a contact mask ``c ∈ (T, K)`` over ``K`` target frames, the residual
penalises the per-frame linear velocity of each contact frame — measured
as the world-frame displacement of the frame origin between consecutive
timesteps. This is exactly the linear part of the frame's
``LOCAL_WORLD_ALIGNED`` spatial velocity (see ``docs/design/05_KINEMATICS.md §3``):

    r_{t,k,:3} = c_{t,k} * (p_k(q_{t+1}) - p_k(q_t)) / dt

where ``p_k(q) = data.frame_pose_world[t, frame_ids[k], :3]``.

Output dim: ``3 * K * (T - 1)`` — linear only in v1. The ``angular`` flag
is reserved as an expansion hook.

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §2``.
"""

from __future__ import annotations

import torch

from ..data_model.model import Model
from .base import Residual, ResidualState
from .registry import register_residual


@register_residual("contact_consistency")
class ContactConsistencyResidual:
    """Linear cartesian velocity penalty on tagged contact frames."""

    name: str = "contact_consistency"

    def __init__(
        self,
        model: Model,
        frame_ids: tuple[int, ...],
        contact_weights: torch.Tensor,
        *,
        dt: float,
        weight: float = 1.0,
        angular: bool = False,
    ) -> None:
        if angular:
            raise NotImplementedError(
                "angular contact-consistency is an expansion hook; not implemented in v1"
            )
        if contact_weights.dim() != 2 or contact_weights.shape[1] != len(frame_ids):
            raise ValueError(
                f"contact_weights must be (T, {len(frame_ids)}); got {tuple(contact_weights.shape)}"
            )
        self.model = model
        self.frame_ids = tuple(int(i) for i in frame_ids)
        self.contact_weights = contact_weights
        self.dt = float(dt)
        self.weight = float(weight)
        T = int(contact_weights.shape[0])
        K = len(self.frame_ids)
        self.dim = 3 * K * (T - 1)

    def _frame_positions(self, state: ResidualState) -> torch.Tensor:
        """Return ``(T, K, 3)`` frame-origin positions in world frame."""
        if state.data.frame_pose_world is None:
            raise RuntimeError(
                "ContactConsistencyResidual: state.data.frame_pose_world is None; "
                "state_factory must call forward_kinematics with compute_frames=True"
            )
        frame_idx = torch.as_tensor(self.frame_ids, device=state.data.frame_pose_world.device)
        p = state.data.frame_pose_world[..., :, :3]  # (..., nframes, 3)
        return p[..., frame_idx, :]                   # (..., K, 3)

    def __call__(self, state: ResidualState) -> torch.Tensor:
        q = state.variables
        if q.dim() != 2:
            raise ValueError(
                f"ContactConsistencyResidual expects (T, nq); got {tuple(q.shape)}"
            )
        T = q.shape[0]
        if T != self.contact_weights.shape[0]:
            raise ValueError(
                f"trajectory length {T} != contact_weights length {self.contact_weights.shape[0]}"
            )

        p = self._frame_positions(state)  # (T, K, 3)
        dp = (p[1:] - p[:-1]) / self.dt   # (T-1, K, 3)

        w = self.contact_weights.to(device=q.device, dtype=q.dtype)
        # Use the average of the endpoint masks — so transitions do not
        # weight the displacement asymmetrically.
        w_pair = 0.5 * (w[:-1] + w[1:]).unsqueeze(-1)  # (T-1, K, 1)
        r = self.weight * w_pair * dp                   # (T-1, K, 3)
        return r.reshape(-1)

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        """Analytic Jacobian via ``get_frame_jacobian`` (LOCAL_WORLD_ALIGNED).

        The linear rows of the LWA frame Jacobian at ``q_t`` exactly equal
        ``d p_k(q_t) / d q_t`` (as a tangent-space derivative), so the
        residual Jacobian is tridiagonal per contact frame:

            ∂r_{t,k} / ∂q_t     = -w_pair_{t,k} * J_k_LWA_lin(q_t) / dt
            ∂r_{t,k} / ∂q_{t+1} = +w_pair_{t,k} * J_k_LWA_lin(q_{t+1}) / dt
        """
        from ..kinematics import ReferenceFrame
        from ..kinematics.jacobian import get_frame_jacobian

        q = state.variables
        T, _ = q.shape
        nv = state.model.nv
        K = len(self.frame_ids)
        device, dtype = q.device, q.dtype

        w = self.contact_weights.to(device=device, dtype=dtype)  # (T, K)
        w_pair = 0.5 * (w[:-1] + w[1:])                           # (T-1, K)

        J = torch.zeros(self.dim, T * nv, device=device, dtype=dtype)
        # Pre-compute LWA Jacobians at each timestep for each contact frame.
        # state.data is batched over T — index into the leading dim.
        for k, fid in enumerate(self.frame_ids):
            # get_frame_jacobian reads data.q and joint_pose_world at the
            # whole batch; it returns (T, 6, nv).
            J_lwa_all = get_frame_jacobian(
                state.model, state.data, fid, reference=ReferenceFrame.LOCAL_WORLD_ALIGNED
            )  # (T, 6, nv)
            J_lin_all = J_lwa_all[..., :3, :]  # (T, 3, nv)

            for t in range(T - 1):
                # rows correspond to timestep t, frame k, 3 linear components.
                row_offset = (t * K + k) * 3
                r0, r1 = row_offset, row_offset + 3
                scale = self.weight * float(w_pair[t, k]) / self.dt
                J[r0:r1, t * nv:(t + 1) * nv] = -scale * J_lin_all[t]
                J[r0:r1, (t + 1) * nv:(t + 2) * nv] = scale * J_lin_all[t + 1]
        return J

    def apply_jac_transpose(self, state: ResidualState, r: torch.Tensor) -> torch.Tensor:
        """``J^T @ r`` without materialising the dense Jacobian — O(K·T·nv²).

        For each frame ``k`` and timestep ``t``, applies the transpose of
        the 3×nv LWA linear Jacobian block to the corresponding 3-vector
        slice of ``r``.
        """
        from ..kinematics import ReferenceFrame
        from ..kinematics.jacobian import get_frame_jacobian

        q = state.variables
        T, _ = q.shape
        nv = state.model.nv
        K = len(self.frame_ids)
        device, dtype = q.device, q.dtype

        r_mat = r.reshape(T - 1, K, 3)
        w = self.contact_weights.to(device=device, dtype=dtype)
        w_pair = 0.5 * (w[:-1] + w[1:])                           # (T-1, K)
        scale = self.weight / self.dt

        g = torch.zeros(T, nv, dtype=dtype, device=device)
        for k, fid in enumerate(self.frame_ids):
            J_lwa_all = get_frame_jacobian(
                state.model, state.data, fid, reference=ReferenceFrame.LOCAL_WORLD_ALIGNED
            )  # (T, 6, nv)
            J_lin_all = J_lwa_all[..., :3, :]                         # (T, 3, nv)
            wk = (scale * w_pair[:, k]).unsqueeze(-1)                 # (T-1, 1)

            r_tk = r_mat[:, k, :]                                      # (T-1, 3)
            # ∂r_{t,k} / ∂q_t     = −wk * J_lin[t];   so g[t]   += −wk · J_lin[t]^T · r
            # ∂r_{t,k} / ∂q_{t+1} = +wk * J_lin[t+1]; so g[t+1] += +wk · J_lin[t+1]^T · r
            J_t = J_lin_all[:T - 1].transpose(-2, -1)                  # (T-1, nv, 3)
            J_tp1 = J_lin_all[1:].transpose(-2, -1)                    # (T-1, nv, 3)
            contrib_t = (J_t @ r_tk.unsqueeze(-1)).squeeze(-1)         # (T-1, nv)
            contrib_tp1 = (J_tp1 @ r_tk.unsqueeze(-1)).squeeze(-1)     # (T-1, nv)

            g[:T - 1] += -wk * contrib_t
            g[1:T] += wk * contrib_tp1
        return g.reshape(-1)
