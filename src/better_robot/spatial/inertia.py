"""``Inertia`` — packed spatial inertia of a rigid body.

Stored as a ``(..., 10)`` tensor
``[mass, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]``.

See ``docs/concepts/lie_and_spatial.md §7``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

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
        from .symmetric3 import Symmetric3  # noqa: PLC0415 - keep value-type imports cycle-safe

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
        I_diag = (2.0 / 5.0) * mass * radius**2
        # packing: [mass, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
        data = torch.tensor([mass, 0.0, 0.0, 0.0, I_diag, I_diag, I_diag, 0.0, 0.0, 0.0], dtype=torch.float32)
        return cls(data)

    @classmethod
    def from_box(cls, mass: float, size: torch.Tensor) -> "Inertia":
        """Box inertia. ``size`` = full edge lengths (lx, ly, lz).

        Ixx = m/12*(ly²+lz²), Iyy = m/12*(lx²+lz²), Izz = m/12*(lx²+ly²).
        """
        lx, ly, lz = float(size[0]), float(size[1]), float(size[2])
        k = mass / 12.0
        Ixx = k * (ly**2 + lz**2)
        Iyy = k * (lx**2 + lz**2)
        Izz = k * (lx**2 + ly**2)
        data = torch.tensor([mass, 0.0, 0.0, 0.0, Ixx, Iyy, Izz, 0.0, 0.0, 0.0], dtype=torch.float32)
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
        vol_cyl = math.pi * radius**2 * length
        vol_hemi = (2.0 / 3.0) * math.pi * radius**3
        vol_total = vol_cyl + 2.0 * vol_hemi
        m_cyl = mass * vol_cyl / vol_total
        m_hemi = mass * vol_hemi / vol_total

        I_zz = 0.5 * m_cyl * radius**2 + 2 * m_hemi * 0.4 * radius**2
        I_xx = m_cyl * (length**2 / 12.0 + radius**2 / 4.0) + 2 * m_hemi * (
            0.4 * radius**2 + (length / 2.0) * (0.75 * length + radius)
        )
        I_yy = I_xx
        data = torch.tensor([mass, 0.0, 0.0, 0.0, I_xx, I_yy, I_zz, 0.0, 0.0, 0.0], dtype=torch.float32)
        return cls(data)

    @classmethod
    def from_ellipsoid(cls, mass: float, radii: torch.Tensor) -> "Inertia":
        """Ellipsoid inertia with principal radii (a, b, c).

        I = diag(2/5*m*(b²+c²), 2/5*m*(a²+c²), 2/5*m*(a²+b²)).
        """
        a, b, c = float(radii[0]), float(radii[1]), float(radii[2])
        k = 0.4 * mass
        Ixx = k * (b**2 + c**2)
        Iyy = k * (a**2 + c**2)
        Izz = k * (a**2 + b**2)
        data = torch.tensor([mass, 0.0, 0.0, 0.0, Ixx, Iyy, Izz, 0.0, 0.0, 0.0], dtype=torch.float32)
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
        I: torch.Tensor,  # noqa: E741 - conventional inertia-matrix symbol
    ) -> "Inertia":
        """Construct from ``(mass, com, 3×3 inertia matrix)``.

        ``mass`` is a scalar or ``(B...,)``; ``com`` is ``(..., 3)``;
        ``I`` is ``(..., 3, 3)`` and is assumed symmetric (the upper
        triangle is read).
        """
        from .symmetric3 import Symmetric3  # noqa: PLC0415 - keep value-type imports cycle-safe

        sym3 = Symmetric3.from_matrix(I).data
        return cls.from_mass_com_sym3(mass, com, sym3)

    @classmethod
    def from_mesh(  # noqa: PLR0912, PLR0915 - validates and integrates the full mesh contract
        cls,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        density: Real | torch.Tensor = 1.0,
    ) -> "Inertia":
        """Integrate uniform-density inertia over a closed triangle mesh.

        Parameters
        ----------
        vertices
            Floating tensor shaped ``(..., V, 3)``. Leading dimensions are
            arbitrary model-value batch dimensions and remain differentiable.
        faces
            Shared integer triangle table shaped ``(F, 3)``. The surface must
            be closed, non-self-intersecting, and consistently wound. Either
            global winding orientation is accepted; mixed component winding
            is not.
        density
            Positive scalar or tensor with exactly ``vertices.shape[:-2]``.
            Tensor density remains differentiable.

        Notes
        -----
        Each oriented face and the origin define a signed tetrahedron. Closed
        forms for its volume, first moment, and second moment are summed using
        Torch tensor operations only. The integration origin is shifted near
        the mesh first to reduce cancellation for meshes far from the body
        origin; the returned COM is shifted back. No ``trimesh`` or host array
        conversion is used.
        """
        if not isinstance(vertices, torch.Tensor) or not vertices.is_floating_point():
            raise TypeError("vertices must be a floating torch.Tensor")
        if vertices.ndim < 2 or vertices.shape[-1] != 3:
            raise ValueError(f"vertices must have shape (..., V, 3), got {tuple(vertices.shape)}")
        if vertices.shape[-2] < 4:
            raise ValueError("vertices must contain at least four points")
        if not bool(torch.isfinite(vertices).all()):
            raise ValueError("vertices must contain only finite values")
        if not isinstance(faces, torch.Tensor) or faces.dtype not in (torch.int32, torch.int64):
            raise TypeError("faces must be a torch.int32 or torch.int64 tensor")
        if faces.ndim != 2 or faces.shape[-1] != 3 or faces.shape[0] == 0:
            raise ValueError(f"faces must have non-empty shape (F, 3), got {tuple(faces.shape)}")
        if bool(torch.any(faces < 0)) or bool(torch.any(faces >= vertices.shape[-2])):
            raise ValueError("faces contains a vertex index outside the vertices table")

        if isinstance(density, Real) and not isinstance(density, bool):
            rho = vertices.new_tensor(float(density))
        elif isinstance(density, torch.Tensor) and density.is_floating_point():
            rho = density.to(device=vertices.device, dtype=vertices.dtype)
        else:
            raise TypeError("density must be a positive real number or floating torch.Tensor")
        batch_shape = tuple(vertices.shape[:-2])
        if tuple(rho.shape) not in ((), batch_shape):
            raise ValueError(
                f"density must be scalar or have exact mesh batch shape {batch_shape}, got {tuple(rho.shape)}"
            )
        if not bool(torch.isfinite(rho).all()) or bool(torch.any(rho <= 0.0)):
            raise ValueError("density must contain only finite positive values")

        # Signed-tetrahedron integration is translation invariant for a closed
        # surface. Centering first avoids catastrophic cancellation when the
        # body-frame origin is far from a small mesh.
        integration_origin = vertices.mean(dim=-2)
        centered = vertices - integration_origin.unsqueeze(-2)
        face_indices = faces.to(device=vertices.device, dtype=torch.long)
        triangles = centered[..., face_indices, :]
        a, b, c = triangles.unbind(dim=-2)

        signed_face_volume = (a * torch.cross(b, c, dim=-1)).sum(dim=-1) / 6.0
        face_sum = a + b + c
        signed_volume = signed_face_volume.sum(dim=-1)
        signed_first = (signed_face_volume.unsqueeze(-1) * face_sum / 4.0).sum(dim=-2)

        outer_sum = face_sum.unsqueeze(-1) * face_sum.unsqueeze(-2)
        outer_sum = outer_sum + a.unsqueeze(-1) * a.unsqueeze(-2)
        outer_sum = outer_sum + b.unsqueeze(-1) * b.unsqueeze(-2)
        outer_sum = outer_sum + c.unsqueeze(-1) * c.unsqueeze(-2)
        signed_second = (signed_face_volume[..., None, None] * outer_sum / 20.0).sum(dim=-3)

        # Reversing every face reverses all three signed integrals. Fold this
        # global orientation so physical mass and inertia remain unchanged.
        orientation = torch.where(
            signed_volume < 0.0,
            -torch.ones_like(signed_volume),
            torch.ones_like(signed_volume),
        )
        volume = signed_volume * orientation
        first = signed_first * orientation.unsqueeze(-1)
        second = signed_second * orientation[..., None, None]

        scale = torch.linalg.vector_norm(centered, dim=-1).amax(dim=-1)
        minimum_volume = torch.finfo(vertices.dtype).eps * scale.pow(3)
        if bool(torch.any(volume <= minimum_volume)):
            raise ValueError(
                "mesh has zero or numerically degenerate enclosed volume; "
                "faces must define a closed, consistently wound surface"
            )

        com_local = first / volume.unsqueeze(-1)
        com = com_local + integration_origin
        central_second_per_density = second - volume[..., None, None] * (
            com_local.unsqueeze(-1) * com_local.unsqueeze(-2)
        )
        eye = torch.eye(3, dtype=vertices.dtype, device=vertices.device)
        inertia_per_density = (
            central_second_per_density.diagonal(dim1=-2, dim2=-1).sum(dim=-1)[..., None, None] * eye
            - central_second_per_density
        )
        mass = rho * volume
        inertia_com = rho[..., None, None] * inertia_per_density
        return cls.from_mass_com_matrix(mass, com, inertia_com)

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
        from ..lie.tangents import hat_so3  # noqa: PLC0415 - keep spatial/lie imports cycle-safe

        m = self.mass  # (...)
        c = self.com  # (..., 3)
        I3_body = self.inertia_matrix  # (..., 3, 3)  about COM

        hatc = hat_so3(c)  # (..., 3, 3)

        # I_o = I_c − m · hat(c)² (= I_c + m·(|c|²I − c cᵀ), parallel-axis shift)
        I_o = I3_body - m[..., None, None] * (hatc @ hatc)  # (..., 3, 3)

        m_I3 = m[..., None, None] * torch.eye(3, dtype=self.data.dtype, device=self.data.device)
        m_hatc = m[..., None, None] * hatc

        top = torch.cat([m_I3, -m_hatc], dim=-1)
        bottom = torch.cat([m_hatc, I_o], dim=-1)
        return torch.cat([top, bottom], dim=-2)  # (..., 6, 6)

    def se3_action(self, T) -> "Inertia":
        """Transform the inertia by an SE3 pose.

        ``T`` may be either a raw ``(..., 7)`` tensor or an :class:`SE3`
        value-class instance (the ``.tensor`` attribute is unwrapped).

        ``I_new = Ad(T)^{-T} · I_6×6 · Ad(T)^{-1}``, then repack. Blocks are
        extracted under the linear-first layout set by :meth:`_to_6x6`:
        ``M = [[m·I3, −m·hat(c)], [m·hat(c), I_o]]``.
        """
        from ..lie import se3 as _se3  # noqa: PLC0415 - keep spatial/lie imports cycle-safe
        from ..lie.tangents import hat_so3, vee_so3  # noqa: PLC0415
        from ..lie.types import SE3  # noqa: PLC0415
        from .symmetric3 import Symmetric3  # noqa: PLC0415

        if isinstance(T, SE3):
            T = T.tensor

        Ad_inv = _se3.adjoint_inv(T)  # (..., 6, 6)
        M = self._to_6x6()  # (..., 6, 6)
        M_new = Ad_inv.transpose(-1, -2) @ M @ Ad_inv

        m = self.mass
        m_safe = m.clamp(min=1e-12)
        # Top-right block of M_new equals −m·hat(c_new), so c_new = vee(−top_right / m).
        hatc_new = -M_new[..., :3, 3:6] / m_safe[..., None, None]
        c_new = vee_so3(hatc_new)  # (..., 3)

        # Bottom-right block of M_new is I_o (inertia about origin). Shift back to
        # inertia-about-COM via the parallel-axis theorem: I_c = I_o + m·hat(c)².
        hatc_n = hat_so3(c_new)
        I_o_new = M_new[..., 3:, 3:]
        I_c_new = I_o_new + m[..., None, None] * (hatc_n @ hatc_n)

        sym3 = Symmetric3.from_matrix(I_c_new).data

        m_vec = m.unsqueeze(-1)
        data = torch.cat([m_vec, c_new, sym3], dim=-1)
        return Inertia(data)

    def apply(self, v) -> "Force":  # noqa: F821 - imported lazily below; v: Motion
        """``I * v`` — spatial inertia times twist = spatial momentum."""
        from .force import Force  # noqa: PLC0415 - avoid eager spatial value-type cycle

        M = self._to_6x6()  # (..., 6, 6)
        momentum = (M @ v.data.unsqueeze(-1)).squeeze(-1)
        return Force(momentum)

    def add(self, other: "Inertia") -> "Inertia":
        """Composite-rigid-body inertia addition (element-wise on packed 10-vector)."""
        return Inertia(self.data + other.data)
