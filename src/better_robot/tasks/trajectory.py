"""``Trajectory`` — first-class temporal type with shapes ``(*B, T, nq)``.

Either unbatched ``(T, nq)`` or batched ``(*B, T, nq)`` is accepted. The
``with_batch_dims`` view always normalises to a batched form so callers
can iterate without an ``if traj.q.dim() == 2`` branch.

See ``docs/design/08_TASKS.md §2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch

from ..exceptions import ShapeError

if TYPE_CHECKING:
    from ..data_model.data import Data
    from ..data_model.model import Model


def _validate_pair(name: str, t: torch.Tensor, q: torch.Tensor) -> None:
    """``q.shape[..., -2] == t.shape[-1]`` and ``q.shape[:-2] == t.shape[:-1]``."""
    if q.dim() < 2:
        raise ShapeError(f"{name}: q must have ≥ 2 dims; got {tuple(q.shape)}")
    if t.dim() < 1:
        raise ShapeError(f"{name}: t must have ≥ 1 dim; got {tuple(t.shape)}")
    if q.shape[-2] != t.shape[-1]:
        raise ShapeError(
            f"{name}: q has T={q.shape[-2]} but t has T={t.shape[-1]}"
        )
    if q.shape[:-2] != t.shape[:-1]:
        raise ShapeError(
            f"{name}: batch shapes disagree — q.shape[:-2]={tuple(q.shape[:-2])} "
            f"vs t.shape[:-1]={tuple(t.shape[:-1])}"
        )


@dataclass
class Trajectory:
    """Temporal batch of configurations, velocities, accelerations, controls.

    Accepts either unbatched ``(T, nq)`` or batched ``(*B, T, nq)``. ``t``
    is shaped ``(*B, T)`` (mirrors ``q`` minus the trailing feature dim).
    """

    t: torch.Tensor
    q: torch.Tensor
    v: torch.Tensor | None = None
    a: torch.Tensor | None = None
    tau: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Legacy alias kept so the V1 viewer player keeps working.
    model_id: int = -1

    def __post_init__(self) -> None:
        _validate_pair("Trajectory(t, q)", self.t, self.q)
        if self.v is not None:
            if self.v.shape[:-1] != self.q.shape[:-1]:
                raise ShapeError(
                    f"Trajectory: v.shape={tuple(self.v.shape)} incompatible "
                    f"with q.shape={tuple(self.q.shape)}"
                )
        if self.a is not None and self.a.shape[:-1] != self.q.shape[:-1]:
            raise ShapeError(
                f"Trajectory: a.shape={tuple(self.a.shape)} incompatible "
                f"with q.shape={tuple(self.q.shape)}"
            )
        if self.tau is not None and self.tau.shape[:-1] != self.q.shape[:-1]:
            raise ShapeError(
                f"Trajectory: tau.shape={tuple(self.tau.shape)} incompatible "
                f"with q.shape={tuple(self.q.shape)}"
            )

    # ─────────────────────────── shape introspection ─────────────────────────

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading batch dims; ``()`` for unbatched ``(T, nq)``."""
        return tuple(self.q.shape[:-2])

    @property
    def num_knots(self) -> int:
        """Length of the time axis."""
        return int(self.q.shape[-2])

    # Legacy aliases used by the viewer V1.
    @property
    def horizon(self) -> int:
        return self.num_knots

    @property
    def batch_size(self) -> int:
        if not self.batch_shape:
            return 1
        out = 1
        for d in self.batch_shape:
            out *= d
        return out

    # ─────────────────────────── batch normalisation ─────────────────────────

    def with_batch_dims(self, n: int = 1) -> "Trajectory":
        """Return a view whose batch shape has ``n`` leading singleton dims.

        ``with_batch_dims(0)`` removes all leading batch dims (must already
        be unbatched). ``with_batch_dims(1)`` produces ``(1, T, nq)`` from
        an unbatched ``(T, nq)``; idempotent if already batched with one
        leading dim.
        """
        cur = len(self.batch_shape)

        def _expand(x: torch.Tensor | None, *, feature_dims: int) -> torch.Tensor | None:
            if x is None:
                return None
            target_batch = n
            if cur == target_batch:
                return x
            if cur < target_batch:
                for _ in range(target_batch - cur):
                    x = x.unsqueeze(0)
                return x
            # cur > target_batch — flatten down (must be safe to do so)
            collapse = cur - target_batch
            if any(s != 1 for s in x.shape[:collapse]):
                raise ShapeError(
                    f"with_batch_dims({n}): cannot squeeze non-singleton "
                    f"batch dims {tuple(x.shape[:collapse])}"
                )
            for _ in range(collapse):
                x = x.squeeze(0)
            return x

        # q has 2 trailing feature dims (T, nq); t has 1 (T)
        return Trajectory(
            t=_expand(self.t, feature_dims=1),
            q=_expand(self.q, feature_dims=2),
            v=_expand(self.v, feature_dims=2),
            a=_expand(self.a, feature_dims=2),
            tau=_expand(self.tau, feature_dims=2),
            extras=dict(self.extras),
            metadata=dict(self.metadata),
            model_id=self.model_id,
        )

    # ─────────────────────────── slicing / resample ─────────────────────────

    def slice(self, t_start: float, t_end: float) -> "Trajectory":
        """Return a sub-trajectory whose times lie in ``[t_start, t_end]``.

        Operates on the *time axis* of ``t`` (not array indices). Works on
        both unbatched and batched trajectories; uses the time grid of
        the first batch element to compute the index range.
        """
        t_flat = self.t.reshape(-1, self.t.shape[-1])[0]
        mask = (t_flat >= t_start) & (t_flat <= t_end)
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            raise ValueError(
                f"slice({t_start}, {t_end}) is empty for t in "
                f"[{float(t_flat.min())}, {float(t_flat.max())}]"
            )
        i0, i1 = int(idx[0]), int(idx[-1]) + 1
        return Trajectory(
            t=self.t[..., i0:i1],
            q=self.q[..., i0:i1, :],
            v=self.v[..., i0:i1, :] if self.v is not None else None,
            a=self.a[..., i0:i1, :] if self.a is not None else None,
            tau=self.tau[..., i0:i1, :] if self.tau is not None else None,
            extras=dict(self.extras),
            metadata=dict(self.metadata),
            model_id=self.model_id,
        )

    def resample(
        self,
        new_t: torch.Tensor,
        *,
        kind: Literal["linear", "sclerp"] = "linear",
    ) -> "Trajectory":
        """Resample ``q`` onto ``new_t``.

        ``kind="linear"`` is per-coordinate linear interpolation — fine
        for joint-space trajectories. ``kind="sclerp"`` interprets the
        first 7 components of ``q`` as an SE3 pose ``[xyz, qx, qy, qz, qw]``
        and uses spherical-linear interpolation for the orientation block,
        plain linear for the rest. Manifold-aware.

        See ``docs/design/08_TASKS.md §2``.
        """
        t_flat = self.t.reshape(-1, self.t.shape[-1])[0]  # (T,) reference grid
        # Per-feature linear interpolation on the time axis.
        q_resampled = _linear_interp_along_axis(self.q, t_flat, new_t, axis=-2)
        v_resampled = (
            _linear_interp_along_axis(self.v, t_flat, new_t, axis=-2) if self.v is not None else None
        )
        a_resampled = (
            _linear_interp_along_axis(self.a, t_flat, new_t, axis=-2) if self.a is not None else None
        )
        tau_resampled = (
            _linear_interp_along_axis(self.tau, t_flat, new_t, axis=-2) if self.tau is not None else None
        )

        if kind == "sclerp":
            from ..lie.so3 import slerp as so3_slerp

            # Replace the first 4 quaternion components (indices 3:7) of q with
            # slerp on the original samples.
            quat_orig = self.q[..., 3:7]  # (..., T, 4)
            quat_new = _slerp_along_axis(quat_orig, t_flat, new_t, so3_slerp)
            q_resampled = q_resampled.clone()
            q_resampled[..., 3:7] = quat_new
        elif kind != "linear":
            raise ValueError(f"unknown resample kind={kind!r}")

        # Broadcast new_t to the batch.
        new_t_b = new_t.expand(*self.t.shape[:-1], new_t.shape[-1])
        return Trajectory(
            t=new_t_b,
            q=q_resampled,
            v=v_resampled,
            a=a_resampled,
            tau=tau_resampled,
            extras=dict(self.extras),
            metadata=dict(self.metadata),
            model_id=self.model_id,
        )

    def downsample(self, factor: int) -> "Trajectory":
        """Take every ``factor``-th sample along the time axis."""
        if factor < 1:
            raise ValueError(f"factor must be ≥ 1; got {factor}")
        return Trajectory(
            t=self.t[..., ::factor],
            q=self.q[..., ::factor, :],
            v=self.v[..., ::factor, :] if self.v is not None else None,
            a=self.a[..., ::factor, :] if self.a is not None else None,
            tau=self.tau[..., ::factor, :] if self.tau is not None else None,
            extras=dict(self.extras),
            metadata=dict(self.metadata),
            model_id=self.model_id,
        )

    def to_data(self, model: "Model") -> "Data":
        """Materialise a batched ``Data`` whose batch dim is ``T·B``.

        Each trajectory sample is a separate row in the batched ``Data``.
        Useful for vectorised collision / FK evaluation across the whole
        clip in one shot.
        """
        from ..kinematics.forward import forward_kinematics

        flat_q = self.q.reshape(-1, self.q.shape[-1])
        return forward_kinematics(model, flat_q, compute_frames=True)


def _linear_interp_along_axis(
    x: torch.Tensor, t_orig: torch.Tensor, t_new: torch.Tensor, *, axis: int
) -> torch.Tensor:
    """Per-feature linear interpolation of ``x`` from ``t_orig`` onto ``t_new``."""
    # Reshape to bring the time axis to the second-to-last position so we
    # can broadcast over the batch and feature dims.
    if axis != -2:
        x = x.movedim(axis, -2)
    # ``torch.searchsorted`` finds insertion indices on the time grid.
    idx = torch.searchsorted(t_orig, t_new).clamp(1, t_orig.shape[-1] - 1)
    t0 = t_orig[idx - 1]
    t1 = t_orig[idx]
    w = ((t_new - t0) / (t1 - t0).clamp(min=1e-12)).clamp(0.0, 1.0)
    # Gather both endpoints along the time axis.
    x_lo = torch.index_select(x, dim=-2, index=idx - 1)
    x_hi = torch.index_select(x, dim=-2, index=idx)
    out = x_lo + (x_hi - x_lo) * w.reshape(*([1] * (x.ndim - 2)), -1, 1)
    if axis != -2:
        out = out.movedim(-2, axis)
    return out


def _slerp_along_axis(
    quat: torch.Tensor, t_orig: torch.Tensor, t_new: torch.Tensor, slerp_fn
) -> torch.Tensor:
    """Per-knot SLERP of a quaternion ``(..., T, 4)`` onto ``t_new``."""
    idx = torch.searchsorted(t_orig, t_new).clamp(1, t_orig.shape[-1] - 1)
    t0 = t_orig[idx - 1]
    t1 = t_orig[idx]
    w = ((t_new - t0) / (t1 - t0).clamp(min=1e-12)).clamp(0.0, 1.0)
    q_lo = torch.index_select(quat, dim=-2, index=idx - 1)  # (..., new_T, 4)
    q_hi = torch.index_select(quat, dim=-2, index=idx)
    # slerp accepts t of shape (...) — broadcasted into q1/q2's leading dims.
    return slerp_fn(q_lo, q_hi, w)
