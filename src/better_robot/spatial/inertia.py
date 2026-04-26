"""``Inertia`` — packed spatial inertia of a rigid body.

Stored as a ``(..., 10)`` tensor
``[mass, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]``.

See ``docs/concepts/lie_and_spatial.md §7``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Inertia:
    """Spatial inertia of a rigid body in a single ``(..., 10)`` packed tensor."""

    data: torch.Tensor  # (..., 10)

    # ---- accessors ----

    @property
    def mass(self) -> torch.Tensor:
        return self.data[..., 0]

    @property
    def com(self) -> torch.Tensor:
        return self.data[..., 1:4]

    @property
    def inertia_matrix(self) -> torch.Tensor:
        """Expand the packed Symmetric3 portion to ``(..., 3, 3)``."""
        from .symmetric3 import Symmetric3
        return Symmetric3(self.data[..., 4:10]).to_matrix()

    # ---- named-constructor factories (Pinocchio style) ----

    @classmethod
    def zero(
        cls,
        *,
        batch_shape: tuple[int, ...] = (),
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "Inertia":
        return cls(torch.zeros((*batch_shape, 10), device=device, dtype=dtype))

    @classmethod
    def from_sphere(cls, mass: float, radius: float) -> "Inertia":
        """Solid sphere inertia. I = 2/5 * m * r^2 on diagonal, com at origin."""
        I_diag = (2.0 / 5.0) * mass * radius ** 2
        # packing: [mass, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
        data = torch.tensor([mass, 0.0, 0.0, 0.0,
                             I_diag, I_diag, I_diag, 0.0, 0.0, 0.0],
                            dtype=torch.float32)
        return cls(data)

    @classmethod
    def from_box(cls, mass: float, size: torch.Tensor) -> "Inertia":
        """Box inertia. ``size`` = full edge lengths (lx, ly, lz).

        Ixx = m/12*(ly²+lz²), Iyy = m/12*(lx²+lz²), Izz = m/12*(lx²+ly²).
        """
        lx, ly, lz = float(size[0]), float(size[1]), float(size[2])
        k = mass / 12.0
        Ixx = k * (ly ** 2 + lz ** 2)
        Iyy = k * (lx ** 2 + lz ** 2)
        Izz = k * (lx ** 2 + ly ** 2)
        data = torch.tensor([mass, 0.0, 0.0, 0.0,
                             Ixx, Iyy, Izz, 0.0, 0.0, 0.0],
                            dtype=torch.float32)
        return cls(data)

    @classmethod
    def from_capsule(cls, mass: float, radius: float, length: float) -> "Inertia":
        """Capsule (cylinder + 2 hemispheres) inertia along the Z axis.

        Uses Pinocchio's formula:
          m_cyl  = pi*r²*l*rho,  m_hemi = (2/3)*pi*r³*rho
          total  = mass
          I_axial (z): m_cyl/2 * r² + 2*m_hemi*(2/5*r²)
          I_lateral (x,y): m_cyl*(l²/12 + r²/4) + m_hemi*(2/5*r² + l/2*(3/4*l + r))
        """
        vol_cyl  = math.pi * radius ** 2 * length
        vol_hemi = (2.0 / 3.0) * math.pi * radius ** 3
        vol_total = vol_cyl + 2.0 * vol_hemi
        m_cyl  = mass * vol_cyl  / vol_total
        m_hemi = mass * vol_hemi / vol_total

        I_zz = 0.5 * m_cyl * radius ** 2 + 2 * m_hemi * 0.4 * radius ** 2
        I_xx = (m_cyl * (length ** 2 / 12.0 + radius ** 2 / 4.0) +
                2 * m_hemi * (0.4 * radius ** 2 + (length / 2.0) * (0.75 * length + radius)))
        I_yy = I_xx
        data = torch.tensor([mass, 0.0, 0.0, 0.0,
                             I_xx, I_yy, I_zz, 0.0, 0.0, 0.0],
                            dtype=torch.float32)
        return cls(data)

    @classmethod
    def from_ellipsoid(cls, mass: float, radii: torch.Tensor) -> "Inertia":
        """Ellipsoid inertia with principal radii (a, b, c).

        I = diag(2/5*m*(b²+c²), 2/5*m*(a²+c²), 2/5*m*(a²+b²)).
        """
        a, b, c = float(radii[0]), float(radii[1]), float(radii[2])
        k = 0.4 * mass
        Ixx = k * (b ** 2 + c ** 2)
        Iyy = k * (a ** 2 + c ** 2)
        Izz = k * (a ** 2 + b ** 2)
        data = torch.tensor([mass, 0.0, 0.0, 0.0,
                             Ixx, Iyy, Izz, 0.0, 0.0, 0.0],
                            dtype=torch.float32)
        return cls(data)

    @classmethod
    def from_mass_com_sym3(
        cls,
        mass: torch.Tensor,
        com: torch.Tensor,
        sym3: torch.Tensor,
    ) -> "Inertia":
        """Construct from raw ``(mass, com, symmetric3)`` arrays.

        ``mass``: scalar or (B...,)
        ``com``: (..., 3)
        ``sym3``: (..., 6)  [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
        """
        m = mass.unsqueeze(-1) if mass.dim() == 0 else mass[..., None]
        data = torch.cat([m, com, sym3], dim=-1)
        return cls(data)

    @classmethod
    def from_mass_com_matrix(
        cls,
        mass: torch.Tensor,
        com: torch.Tensor,
        I: torch.Tensor,
    ) -> "Inertia":
        """Construct from ``(mass, com, 3×3 inertia matrix)``.

        ``mass`` is a scalar or ``(B...,)``; ``com`` is ``(..., 3)``;
        ``I`` is ``(..., 3, 3)`` and is assumed symmetric (the upper
        triangle is read).
        """
        from .symmetric3 import Symmetric3

        sym3 = Symmetric3.from_matrix(I).data
        return cls.from_mass_com_sym3(mass, com, sym3)

    # ---- algebra ----

    def _to_6x6(self) -> torch.Tensor:
        """Expand to a full 6×6 spatial inertia matrix — Pinocchio linear-first.

        Motion/Force store 6-vectors as ``[v_lin, ω]`` / ``[f_lin, τ]``, so the
        block layout is::

            M = [[ m·I3,       −m·hat(c) ],
                 [ m·hat(c),    I_o      ]]

        where ``c = com`` and ``I_o = I_c − m·hat(c)²`` is the inertia about
        the origin of the body frame (parallel-axis theorem applied to the
        CoM inertia ``I_c``). Verifies against Pinocchio's explicit formula
        ``f = m·(v − c×ω)``, ``τ = I_c·ω + c×f``.
        """
        from ..lie.tangents import hat_so3
        m = self.mass                  # (...)
        c = self.com                   # (..., 3)
        I3_body = self.inertia_matrix  # (..., 3, 3)  about COM

        hatc = hat_so3(c)              # (..., 3, 3)

        # I_o = I_c − m · hat(c)² (= I_c + m·(|c|²I − c cᵀ), parallel-axis shift)
        I_o = I3_body - m[..., None, None] * (hatc @ hatc)  # (..., 3, 3)

        m_I3 = m[..., None, None] * torch.eye(
            3, dtype=self.data.dtype, device=self.data.device
        )
        m_hatc = m[..., None, None] * hatc

        top    = torch.cat([m_I3,       -m_hatc], dim=-1)
        bottom = torch.cat([m_hatc,      I_o  ], dim=-1)
        return torch.cat([top, bottom], dim=-2)   # (..., 6, 6)

    def se3_action(self, T) -> "Inertia":
        """Transform the inertia by an SE3 pose.

        ``T`` may be either a raw ``(..., 7)`` tensor or an :class:`SE3`
        value-class instance (the ``.tensor`` attribute is unwrapped).

        ``I_new = Ad(T)^{-T} · I_6×6 · Ad(T)^{-1}``, then repack. Blocks are
        extracted under the linear-first layout set by :meth:`_to_6x6`:
        ``M = [[m·I3, −m·hat(c)], [m·hat(c), I_o]]``.
        """
        from ..lie import se3 as _se3
        from ..lie.tangents import hat_so3, vee_so3
        from ..lie.types import SE3
        from .symmetric3 import Symmetric3

        if isinstance(T, SE3):
            T = T.tensor

        Ad_inv = _se3.adjoint_inv(T)          # (..., 6, 6)
        M = self._to_6x6()                    # (..., 6, 6)
        M_new = Ad_inv.transpose(-1, -2) @ M @ Ad_inv

        m = self.mass
        m_safe = m.clamp(min=1e-12)
        # Top-right block of M_new equals −m·hat(c_new), so c_new = vee(−top_right / m).
        hatc_new = -M_new[..., :3, 3:6] / m_safe[..., None, None]
        c_new = vee_so3(hatc_new)             # (..., 3)

        # Bottom-right block of M_new is I_o (inertia about origin). Shift back to
        # inertia-about-COM via the parallel-axis theorem: I_c = I_o + m·hat(c)².
        hatc_n = hat_so3(c_new)
        I_o_new = M_new[..., 3:, 3:]
        I_c_new = I_o_new + m[..., None, None] * (hatc_n @ hatc_n)

        sym3 = Symmetric3.from_matrix(I_c_new).data

        m_vec = m.unsqueeze(-1)
        data = torch.cat([m_vec, c_new, sym3], dim=-1)
        return Inertia(data)

    def apply(self, v) -> "Force":  # v: Motion
        """``I * v`` — spatial inertia times twist = spatial momentum."""
        from .force import Force
        M = self._to_6x6()   # (..., 6, 6)
        momentum = (M @ v.data.unsqueeze(-1)).squeeze(-1)
        return Force(momentum)

    def add(self, other: "Inertia") -> "Inertia":
        """Composite-rigid-body inertia addition (element-wise on packed 10-vector)."""
        return Inertia(self.data + other.data)
