"""``Motion`` — 6D spatial twist value type.

Stored as a ``(..., 6)`` tensor ``[vx, vy, vz, wx, wy, wz]``. Methods return
new ``Motion`` instances (value-typed dataclass).

See ``docs/concepts/lie_and_spatial.md §7``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..lie import tangents as _tan


@dataclass(frozen=True)
class Motion:
    """6D spatial twist (linear + angular velocity).

    Stored as ``(..., 6)`` ``[vx, vy, vz, wx, wy, wz]``.
    """

    data: torch.Tensor

    @classmethod
    def zero(
        cls,
        *,
        batch_shape: tuple[int, ...] = (),
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "Motion":
        return cls(torch.zeros((*batch_shape, 6), device=device, dtype=dtype))

    @property
    def linear(self) -> torch.Tensor:
        """``(..., 3)`` linear velocity component."""
        return self.data[..., :3]

    @property
    def angular(self) -> torch.Tensor:
        """``(..., 3)`` angular velocity component."""
        return self.data[..., 3:]

    # ---- algebraic ops (value-type; return a new Motion) ----

    def cross_motion(self, other: "Motion") -> "Motion":
        """Motion × Motion spatial cross product (``ad`` operator).

        ad(v) * u = [[hat(w_v),  hat(v_v)],  * [v_u]
                      [0,         hat(w_v)]]    [w_u]

        Result: [hat(w_v)@v_u + hat(v_v)@w_u,  hat(w_v)@w_u]
        """
        vv = self.linear    # (..., 3)
        wv = self.angular   # (..., 3)
        vu = other.linear
        wu = other.angular
        H_wv = _tan.hat_so3(wv)
        H_vv = _tan.hat_so3(vv)
        lin = (H_wv @ vu.unsqueeze(-1)).squeeze(-1) + (H_vv @ wu.unsqueeze(-1)).squeeze(-1)
        ang = (H_wv @ wu.unsqueeze(-1)).squeeze(-1)
        return Motion(torch.cat([lin, ang], dim=-1))

    def cross_force(self, other) -> "Force":  # other: Force
        """Motion × Force spatial cross (``ad*`` operator).

        ad*(v) * f = -ad(v)^T * f
                   = [hat(w_v)^T @ f_lin,
                      hat(v_v)^T @ f_lin + hat(w_v)^T @ f_ang]
        Note: hat^T = -hat, so hat(w)^T @ x = -hat(w) @ x = hat(-w) @ x
        """
        from .force import Force
        vv = self.linear
        wv = self.angular
        fl = other.linear
        fa = other.angular
        H_wv = _tan.hat_so3(wv)
        H_vv = _tan.hat_so3(vv)
        # ad*(v)*f: lin = -hat(wv)^T @ fl ... using hat^T = -hat:
        # lin_out = hat(wv)@fl, ang_out = hat(wv)@fa + hat(vv)@fl
        lin_out = (H_wv @ fl.unsqueeze(-1)).squeeze(-1)
        ang_out = (H_wv @ fa.unsqueeze(-1)).squeeze(-1) + (H_vv @ fl.unsqueeze(-1)).squeeze(-1)
        return Force(torch.cat([lin_out, ang_out], dim=-1))

    def se3_action(self, T) -> "Motion":
        """Apply an SE3 transform via the adjoint: ``Ad(T) * v``.

        ``T`` may be a raw ``(..., 7)`` tensor or an
        :class:`~better_robot.lie.types.SE3` value (its ``.tensor`` is
        unwrapped).
        """
        from ..lie import se3 as _se3
        from ..lie.types import SE3 as _SE3
        if isinstance(T, _SE3):
            T = T.tensor
        Ad = _se3.adjoint(T)   # (..., 6, 6)
        return Motion((Ad @ self.data.unsqueeze(-1)).squeeze(-1))

    def compose(self, other: "Motion") -> "Motion":
        return Motion(self.data + other.data)

    # ---- vector-space operators (safe) ----

    def __neg__(self) -> "Motion":
        return Motion(-self.data)

    def __add__(self, other: "Motion") -> "Motion":
        return Motion(self.data + other.data)

    def __sub__(self, other: "Motion") -> "Motion":
        return Motion(self.data - other.data)
    # NB: no __mul__ — scale vs. cross is ambiguous. Use named methods.
