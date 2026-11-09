"""Rigid-body inertia packed as a ``(..., 10)`` tensor: its spatial matrix and its constructors.

Layout ``[mass, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]``: mass, centre of mass ``c`` in the body frame, and the
symmetric rotational inertia about the centre of mass (not the frame origin), in body-frame axes, its six entries in
MuJoCo's ``fullinertia`` order. It is not Pinocchio's ``Inertia.toDynamicParameters``, which holds ``m c`` and the
inertia about the frame origin as ``[Ixx, Ixy, Iyy, Ixz, Iyz, Izz]``. The 6x6 matrix acting on linear-first twists
is ``[[m I3, -m hat(c)], [m hat(c), I_c - m hat(c)^2]]``. Every function is differentiable in its tensor inputs and
broadcasts leading axes. Every public function checks types, trailing shapes, dtypes and devices and reads no tensor
value, except ``inertia_of_mesh(..., check_mesh=True)``; `inertia_matrix`, `rotational_inertia` and
`inertia_from_mass_com` have unchecked ``*_raw`` twins for callers whose inputs are already checked (dynamics).
"""

from __future__ import annotations

import math
from numbers import Real

import torch

from ..exceptions import ArgumentError, DeviceMismatchError, DtypeMismatchError, ShapeError
from ..lie.tangents import hat_so3_raw
from ..lie._checks import check_tensors


def inertia_matrix(inertia: torch.Tensor) -> torch.Tensor:
    """Spatial inertia matrix ``(..., 6, 6)`` of packed inertias ``inertia: (..., 10)``, about the frame origin.

    It acts on linear-first twists: ``inertia_matrix(inertia) @ twist`` is the spatial momentum, a wrench with the
    linear part first. The blocks are ``[[m I3, -m hat(c)], [m hat(c), I_c - m hat(c)^2]]``. Raises ``TypeError``
    for a non-tensor, ``DtypeMismatchError`` for a non-floating dtype and ``ShapeError`` for a trailing size other
    than 10.
    """
    check_tensors(inertia=(inertia, (10,)))
    return inertia_matrix_raw(inertia)


def inertia_matrix_raw(inertia: torch.Tensor) -> torch.Tensor:
    """`inertia_matrix` without input checks, for checked inputs: ``(..., 10) → (..., 6, 6)``."""
    mass, com = inertia[..., 0], inertia[..., 1:4]
    hat_com = hat_so3_raw(com)
    inertia_origin = rotational_inertia_raw(inertia) - mass[..., None, None] * (hat_com @ hat_com)
    eye = torch.eye(3, dtype=inertia.dtype, device=inertia.device)
    mass_eye = mass[..., None, None] * eye
    mass_hat = mass[..., None, None] * hat_com
    top = torch.cat((mass_eye, -mass_hat), dim=-1)
    bottom = torch.cat((mass_hat, inertia_origin), dim=-1)
    return torch.cat((top, bottom), dim=-2)


def rotational_inertia(inertia: torch.Tensor) -> torch.Tensor:
    """Rotational inertia ``(..., 3, 3)`` of packed inertias ``inertia: (..., 10)``, about the centre of mass.

    The symmetric matrix of the six stored entries, in body-frame axes; `inertia_matrix` adds the parallel-axis shift
    to the frame origin. Raises as `inertia_matrix`.
    """
    check_tensors(inertia=(inertia, (10,)))
    return rotational_inertia_raw(inertia)


def rotational_inertia_raw(inertia: torch.Tensor) -> torch.Tensor:
    """`rotational_inertia` without input checks, for checked inputs: ``(..., 10) → (..., 3, 3)``."""
    ixx, iyy, izz, ixy, ixz, iyz = inertia[..., 4:10].unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((ixx, ixy, ixz), dim=-1),
            torch.stack((ixy, iyy, iyz), dim=-1),
            torch.stack((ixz, iyz, izz), dim=-1),
        ),
        dim=-2,
    )


def inertia_from_mass_com(mass: torch.Tensor, com: torch.Tensor, *, rotational_inertia: torch.Tensor) -> torch.Tensor:
    """Packed inertia ``(..., 10)`` from ``mass (...)``, ``com (..., 3)`` and ``rotational_inertia (..., 3, 3)``.

    ``rotational_inertia`` is taken about the centre of mass in body-frame axes, and its symmetric part is stored.
    Leading axes broadcast. Raises ``TypeError`` for an input that is not a tensor, ``ShapeError`` for a wrong
    trailing shape, ``DtypeMismatchError`` for a non-floating dtype or one other than ``mass``'s and
    ``DeviceMismatchError`` for a device other than ``mass``'s.
    """
    check_tensors(mass=(mass, ()), com=(com, (3,)), rotational_inertia=(rotational_inertia, (3, 3)))
    return inertia_from_mass_com_raw(mass, com, rotational_inertia=rotational_inertia)


def inertia_from_mass_com_raw(
    mass: torch.Tensor, com: torch.Tensor, *, rotational_inertia: torch.Tensor
) -> torch.Tensor:
    """`inertia_from_mass_com` without input checks, for checked inputs.

    ``mass (...)``, ``com (..., 3)``, ``rotational_inertia (..., 3, 3)`` → ``(..., 10)``.
    """
    batch = torch.broadcast_shapes(mass.shape, com.shape[:-1], rotational_inertia.shape[:-2])
    symmetric = (rotational_inertia + rotational_inertia.mT) * 0.5
    entries = torch.stack(
        (
            symmetric[..., 0, 0],
            symmetric[..., 1, 1],
            symmetric[..., 2, 2],
            symmetric[..., 0, 1],
            symmetric[..., 0, 2],
            symmetric[..., 1, 2],
        ),
        dim=-1,
    )
    return torch.cat((mass.expand(batch).unsqueeze(-1), com.expand(*batch, 3), entries.expand(*batch, 6)), dim=-1)


def inertia_of_sphere(mass: torch.Tensor, *, radius: torch.Tensor) -> torch.Tensor:
    """Packed inertia ``(..., 10)`` of a solid sphere of ``mass (...)`` and ``radius (...)`` centred at the origin.

    The rotational inertia is ``2/5 m r^2`` about every axis. Raises as `inertia_from_mass_com`.
    """
    check_tensors(mass=(mass, ()), radius=(radius, ()))
    moment = (2.0 / 5.0) * mass * radius**2
    return _principal(mass, moment, moment, moment)


def inertia_of_box(mass: torch.Tensor, *, size: torch.Tensor) -> torch.Tensor:
    """Packed inertia ``(..., 10)`` of a solid box of ``mass (...)`` centred at the origin, edges along the axes.

    ``size (..., 3)`` holds the full edge lengths ``[lx, ly, lz]``; ``Ixx = m (ly^2 + lz^2) / 12`` and likewise.
    Raises as `inertia_from_mass_com`.
    """
    check_tensors(mass=(mass, ()), size=(size, (3,)))
    lx, ly, lz = size.unbind(dim=-1)
    k = mass / 12.0
    return _principal(mass, k * (ly**2 + lz**2), k * (lx**2 + lz**2), k * (lx**2 + ly**2))


def inertia_of_capsule(mass: torch.Tensor, *, radius: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """Packed inertia ``(..., 10)`` of a solid capsule of ``mass (...)`` centred at the origin, axis along z.

    The capsule is a cylinder of ``length (...)`` and ``radius (...)`` capped by two hemispheres of ``radius``; the
    mass splits between the parts by volume, as in Pinocchio's ``Inertia.FromCapsule``. Raises as
    `inertia_from_mass_com`.
    """
    check_tensors(mass=(mass, ()), radius=(radius, ()), length=(length, ()))
    cylinder_volume = math.pi * radius**2 * length
    hemisphere_volume = (2.0 / 3.0) * math.pi * radius**3
    total_volume = cylinder_volume + 2.0 * hemisphere_volume
    cylinder_mass = mass * cylinder_volume / total_volume
    hemisphere_mass = mass * hemisphere_volume / total_volume
    # A hemisphere's centroid sits 3r/8 from its flat face, and its moment about a transverse axis through the
    # centroid is 83/320 m r^2; the parallel-axis shift then runs from that centroid to the capsule centre.
    centroid_distance = 0.5 * length + 0.375 * radius
    transverse = cylinder_mass * (length**2 / 12.0 + radius**2 / 4.0) + 2.0 * hemisphere_mass * (
        (83.0 / 320.0) * radius**2 + centroid_distance**2
    )
    axial = 0.5 * cylinder_mass * radius**2 + 2.0 * hemisphere_mass * 0.4 * radius**2
    return _principal(mass, transverse, transverse, axial)


def inertia_of_ellipsoid(mass: torch.Tensor, *, semi_axes: torch.Tensor) -> torch.Tensor:
    """Packed inertia ``(..., 10)`` of a solid ellipsoid of ``mass (...)`` centred at the origin, axes along x, y, z.

    ``semi_axes (..., 3)`` holds ``[a, b, c]``; ``Ixx = m (b^2 + c^2) / 5`` and likewise. Raises as
    `inertia_from_mass_com`.
    """
    check_tensors(mass=(mass, ()), semi_axes=(semi_axes, (3,)))
    a, b, c = semi_axes.unbind(dim=-1)
    k = mass / 5.0
    return _principal(mass, k * (b**2 + c**2), k * (a**2 + c**2), k * (a**2 + b**2))


def inertia_of_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    density: float | torch.Tensor = 1.0,
    check_mesh: bool = False,
) -> torch.Tensor:
    """Packed inertia ``(..., 10)`` of a solid of uniform ``density`` bounded by a closed triangle mesh.

    ``vertices: (..., V, 3)``, with leading axes a batch that stays differentiable; ``faces: (F, 3)`` int32 or int64
    on the device of ``vertices``, shared by the batch, bounding a closed, non-self-intersecting, consistently wound
    surface (either global winding; `orient_faces_consistently` repairs mixed winding). ``density`` is a number or a
    tensor of shape ``()`` or ``vertices.shape[:-2]`` on the dtype and device of ``vertices``; the default 1 gives
    the inertia per unit density, whose mass is the volume. Each face and the origin span a signed tetrahedron whose
    volume, first and second moments are summed in closed form, after the origin moves to the vertex mean to limit
    cancellation.

    Raises ``TypeError`` for a non-tensor, ``ShapeError``, ``DtypeMismatchError`` or ``DeviceMismatchError`` for a
    wrong shape, dtype or device. ``check_mesh=True``
    also reads the values, one host sync per check, and raises ``ArgumentError`` for non-finite vertices, a face id
    outside ``[0, V)``, a density that is not finite and positive, or a degenerate enclosed volume; without it such
    input gives a meaningless or non-finite result.
    """
    _check_mesh_structure(vertices, faces)
    rho = _density_tensor(density, vertices)
    if check_mesh:
        _check_mesh_values(vertices, faces, rho)

    # The signed-tetrahedron sums are translation invariant for a closed surface, so centring first avoids
    # catastrophic cancellation when the body-frame origin is far from a small mesh.
    integration_origin = vertices.mean(dim=-2)
    centered = vertices - integration_origin.unsqueeze(-2)
    triangles = centered[..., faces, :]
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

    # Reversing every face reverses all three signed integrals; folding the global orientation keeps mass and
    # inertia physical for either winding.
    orientation = torch.where(signed_volume < 0.0, -torch.ones_like(signed_volume), torch.ones_like(signed_volume))
    volume = signed_volume * orientation
    first = signed_first * orientation.unsqueeze(-1)
    second = signed_second * orientation[..., None, None]
    if check_mesh:
        scale = torch.linalg.vector_norm(centered, dim=-1).amax(dim=-1)
        if bool(torch.any(volume <= torch.finfo(vertices.dtype).eps * scale.pow(3))):
            raise ArgumentError(
                "mesh has zero or numerically degenerate enclosed volume; "
                "faces must define a closed, consistently wound surface"
            )

    com_local = first / volume.unsqueeze(-1)
    central_second_per_density = second - volume[..., None, None] * (com_local.unsqueeze(-1) * com_local.unsqueeze(-2))
    eye = torch.eye(3, dtype=vertices.dtype, device=vertices.device)
    inertia_per_density = (
        central_second_per_density.diagonal(dim1=-2, dim2=-1).sum(dim=-1)[..., None, None] * eye
        - central_second_per_density
    )
    return inertia_from_mass_com_raw(
        rho * volume, com_local + integration_origin, rotational_inertia=rho[..., None, None] * inertia_per_density
    )


def _check_mesh_structure(vertices: torch.Tensor, faces: torch.Tensor) -> None:
    """Raise unless ``vertices`` is floating ``(..., V >= 4, 3)``, ``faces`` integer ``(F >= 1, 3)``, on one device."""
    check_tensors(vertices=(vertices, ()))
    if vertices.ndim < 2 or vertices.shape[-1] != 3:
        raise ShapeError(f"vertices must have shape (..., V, 3), got {tuple(vertices.shape)}")
    if vertices.shape[-2] < 4:
        raise ShapeError("vertices must contain at least four points")
    if not isinstance(faces, torch.Tensor):
        raise TypeError(f"faces must be a torch.Tensor, got {type(faces).__name__}")
    if faces.dtype not in (torch.int32, torch.int64):
        raise DtypeMismatchError(f"faces.dtype={faces.dtype} is not torch.int32 or torch.int64; pass faces.long()")
    if faces.ndim != 2 or faces.shape[-1] != 3 or faces.shape[0] == 0:
        raise ShapeError(f"faces must have non-empty shape (F, 3), got {tuple(faces.shape)}")
    if faces.device != vertices.device:
        raise DeviceMismatchError(
            f"faces.device={faces.device} != vertices.device={vertices.device}; pass faces.to({vertices.device})"
        )


def _density_tensor(density: float | torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """``density`` as a tensor of shape ``()`` or ``vertices.shape[:-2]`` on the dtype and device of ``vertices``."""
    if isinstance(density, torch.Tensor):
        check_tensors(vertices=(vertices, ()), density=(density, ()))
        rho = density
    elif isinstance(density, Real) and not isinstance(density, bool):
        rho = vertices.new_tensor(float(density))
    else:
        raise TypeError("density must be a real number or a floating torch.Tensor")
    batch_shape = tuple(vertices.shape[:-2])
    if tuple(rho.shape) not in ((), batch_shape):
        raise ShapeError(f"density must be scalar or have the mesh batch shape {batch_shape}, got {tuple(rho.shape)}")
    return rho


def _check_mesh_values(vertices: torch.Tensor, faces: torch.Tensor, density: torch.Tensor) -> None:
    """Raise ``ArgumentError`` for non-finite vertices, out-of-range face ids or a density not finite positive."""
    if not bool(torch.isfinite(vertices).all()):
        raise ArgumentError("vertices must contain only finite values")
    if bool(torch.any(faces < 0)) or bool(torch.any(faces >= vertices.shape[-2])):
        raise ArgumentError("faces contains a vertex index outside the vertices table")
    if not bool(torch.isfinite(density).all()) or bool(torch.any(density <= 0.0)):
        raise ArgumentError("density must contain only finite positive values")


def _principal(mass: torch.Tensor, ixx: torch.Tensor, iyy: torch.Tensor, izz: torch.Tensor) -> torch.Tensor:
    """Packed inertia ``(..., 10)`` with the centre of mass at the origin and principal axes along x, y, z."""
    mass, ixx, iyy, izz = torch.broadcast_tensors(mass, ixx, iyy, izz)
    zero = torch.zeros_like(mass)
    return torch.stack((mass, zero, zero, zero, ixx, iyy, izz, zero, zero, zero), dim=-1)
