"""Face-table topology helpers for closed triangle meshes.

Companions to :meth:`better_robot.spatial.Inertia.from_mesh`. That integrator
folds a single *global* winding sign but trusts every face to already agree on
orientation; it silently returns a wrong result for a surface whose faces are
inconsistently wound. These helpers operate on the **integer face table only**
— they never touch vertex values, so reorientation is a one-time spawn-time
precompute and the downstream vertex integration stays fully differentiable.

- :func:`validate_closed_manifold` rejects open and non-manifold surfaces.
- :func:`orient_faces_consistently` rewinds each connected component so all its
  faces agree, replacing the historical ``trimesh.fix_normals`` step.

See ``docs/concepts/lie_and_spatial.md`` and ``spatial/inertia.py``.
"""

from __future__ import annotations

import torch

_MAX_REPORTED_EDGES = 5


def _check_face_table(faces: torch.Tensor) -> None:
    """Validate that ``faces`` is a non-empty integer ``(F, 3)`` triangle table."""
    if not isinstance(faces, torch.Tensor) or faces.dtype not in (torch.int32, torch.int64):
        raise TypeError("faces must be a torch.int32 or torch.int64 tensor")
    if faces.ndim != 2 or faces.shape[-1] != 3 or faces.shape[0] == 0:
        raise ValueError(f"faces must have non-empty shape (F, 3), got {tuple(faces.shape)}")


def _undirected_edges(faces: torch.Tensor) -> torch.Tensor:
    """Return the ``(3F, 2)`` sorted undirected edges of every triangle."""
    faces_long = faces.to(torch.long)
    directed = torch.stack(
        (faces_long[:, [0, 1]], faces_long[:, [1, 2]], faces_long[:, [2, 0]]),
        dim=1,
    ).reshape(-1, 2)
    return directed.sort(dim=-1).values


def validate_closed_manifold(faces: torch.Tensor) -> None:
    """Raise unless ``faces`` bound a closed, 2-manifold triangle surface.

    Every undirected edge of a closed 2-manifold is shared by exactly two
    triangles. This checks that incidence and reports the offending edges.
    Multiple disjoint closed components are accepted; each must be closed.

    Parameters
    ----------
    faces
        Integer triangle table shaped ``(F, 3)`` indexing a vertex array. Any
        device is accepted; the edge-incidence reduction runs in place on it.

    Raises
    ------
    TypeError
        If ``faces`` is not a ``torch.int32`` or ``torch.int64`` tensor.
    ValueError
        If ``faces`` is not a non-empty ``(F, 3)`` table, or if any undirected
        edge has an incidence other than two: incidence one marks a boundary
        edge (open surface) and incidence above two marks a non-manifold edge.
        The message names a few offending edges.
    """
    _check_face_table(faces)
    edges = _undirected_edges(faces)
    unique_edges, counts = torch.unique(edges, dim=0, return_counts=True)
    boundary = counts == 1
    non_manifold = counts > 2
    if not bool(boundary.any()) and not bool(non_manifold.any()):
        return

    problems: list[str] = []
    if bool(boundary.any()):
        sample = unique_edges[boundary][:_MAX_REPORTED_EDGES].tolist()
        problems.append(f"{int(boundary.sum())} boundary edge(s) with one incident face (open surface), e.g. {sample}")
    if bool(non_manifold.any()):
        sample_edges = unique_edges[non_manifold][:_MAX_REPORTED_EDGES].tolist()
        sample_counts = counts[non_manifold][:_MAX_REPORTED_EDGES].tolist()
        sample = list(zip(sample_edges, sample_counts))
        problems.append(
            f"{int(non_manifold.sum())} non-manifold edge(s) shared by more than two faces, "
            f"e.g. (edge, incidence) {sample}"
        )
    raise ValueError("faces do not form a closed 2-manifold surface: " + "; ".join(problems))


def _build_face_adjacency(face_list: list[list[int]]) -> list[list[tuple[int, int]]]:
    """Face-adjacency graph over edges shared by exactly two faces.

    Each undirected edge is grouped; only edges with incidence two produce a
    link. Boundary (incidence 1) and non-manifold (incidence > 2) edges are
    left out, so the graph is well defined for both the strict manifold path
    and the best-effort :func:`orient_faces_by_component` path. ``weight`` is
    the product of the two faces' traversal directions on the shared edge
    (+1 for ``min -> max``, -1 for ``max -> min``).
    """
    grouped: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_index, (i, j, k) in enumerate(face_list):
        for u, v in ((i, j), (j, k), (k, i)):
            key = (u, v) if u < v else (v, u)
            direction = 1 if u < v else -1
            grouped.setdefault(key, []).append((face_index, direction))

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(len(face_list))]
    for entries in grouped.values():
        if len(entries) == 2:
            (first_face, first_dir), (second_face, second_dir) = entries
            weight = first_dir * second_dir
            adjacency[first_face].append((second_face, weight))
            adjacency[second_face].append((first_face, weight))
    return adjacency


def _orient_components(
    adjacency: list[list[tuple[int, int]]],
    face_count: int,
    *,
    allow_non_orientable: bool,
) -> tuple[list[int], list[int]]:
    """Assign each face a winding sign and a connected-component label.

    ``sign[f]`` is +1 to keep face ``f`` and -1 to reverse it. Consistency
    across a shared edge requires ``sign[g] == -sign[f] * weight``. A closed
    2-manifold can still be *non-orientable* (for example the real projective
    plane), so the walk can meet an already-signed neighbour whose shared edge
    implies the opposite sign. When ``allow_non_orientable`` is false that
    contradiction raises; otherwise it is ignored (best effort).
    """
    sign = [0] * face_count
    component = [-1] * face_count
    next_component = 0
    for seed in range(face_count):
        if sign[seed] != 0:
            continue
        sign[seed] = 1
        component[seed] = next_component
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbour, weight in adjacency[current]:
                expected = -sign[current] * weight
                if sign[neighbour] == 0:
                    sign[neighbour] = expected
                    component[neighbour] = next_component
                    stack.append(neighbour)
                elif sign[neighbour] != expected and not allow_non_orientable:
                    raise ValueError(
                        f"surface is non-orientable: faces {current} and {neighbour} impose "
                        "contradictory winding across their shared edge"
                    )
        next_component += 1
    return sign, component


def _apply_reversal(faces: torch.Tensor, reverse_mask: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``faces`` with the rows selected by ``reverse_mask`` reversed."""
    oriented = faces.clone()
    reverse_mask = reverse_mask.to(faces.device)
    oriented[reverse_mask] = oriented[reverse_mask].flip(-1)
    return oriented


def orient_faces_consistently(faces: torch.Tensor) -> torch.Tensor:
    """Return a face table whose winding is consistent within each component.

    Two triangles sharing an edge are consistently wound when they traverse
    that shared edge in opposite directions. Starting from one seed face per
    connected component, this walks the face-adjacency graph over shared edges
    and reverses any neighbour that would otherwise traverse the shared edge in
    the same direction. Only the integer vertex order of each row changes, so
    the result plugs directly into :meth:`Inertia.from_mesh`, which then folds
    the remaining single global winding sign from the enclosed volume.

    The graph walk runs on the host, which is appropriate for a spawn-time
    precompute; the returned table keeps the input dtype and device.

    Parameters
    ----------
    faces
        Integer triangle table shaped ``(F, 3)`` of a closed 2-manifold
        surface, as accepted by :func:`validate_closed_manifold`.

    Returns
    -------
    torch.Tensor
        A ``(F, 3)`` table on the input device and dtype. Each connected
        component is internally consistent (all faces agree on winding). A
        single closed surface is exactly what :meth:`Inertia.from_mesh`
        expects; integrate disjoint components separately, because their
        relative orientation is undetermined by topology alone.

    Raises
    ------
    TypeError
        If ``faces`` is not an integer tensor.
    ValueError
        If ``faces`` is not a closed 2-manifold surface (see
        :func:`validate_closed_manifold`), or if that surface is non-orientable
        (a consistent winding then does not exist).
    """
    _check_face_table(faces)
    validate_closed_manifold(faces)

    face_list: list[list[int]] = faces.to(torch.long).cpu().tolist()
    adjacency = _build_face_adjacency(face_list)
    sign, _ = _orient_components(adjacency, len(face_list), allow_non_orientable=False)

    reverse_mask = torch.tensor([value < 0 for value in sign], device=faces.device)
    return _apply_reversal(faces, reverse_mask)


def orient_faces_by_component(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Best-effort outward reorientation that tolerates imperfect topology.

    A companion to :func:`orient_faces_consistently` for authored meshes that
    are *not* strict closed 2-manifolds — parts stitched at a shared seam with
    boundary edges, non-manifold T-junctions, or several disconnected shells.
    Adjacency is built only over edges shared by exactly two faces (boundary
    and non-manifold edges are ignored, never raised on), so each maximal
    two-manifold patch becomes a connected component. Every component is first
    made internally consistent and then flipped, if needed, so its mean-centered
    signed volume is non-negative — i.e. every component faces outward. Passing
    the result to :meth:`Inertia.from_mesh` (with the same ``vertices``) then
    needs no global sign fold and the component volumes add constructively.

    Only the integer face table changes, so the orientation is a one-time
    precompute from rest geometry and downstream vertex integration stays
    differentiable.

    Parameters
    ----------
    vertices
        Floating tensor shaped ``(V, 3)`` giving the reference (rest) geometry
        the face table indexes. It is used only to sign each component's
        volume; it is mean-centered exactly as :meth:`Inertia.from_mesh`
        centers, so pass the same vertex array to both.
    faces
        Integer triangle table shaped ``(F, 3)``.

    Returns
    -------
    torch.Tensor
        A ``(F, 3)`` table on the input ``faces`` device and dtype, with every
        connected component consistently wound outward.

    Raises
    ------
    TypeError
        If ``faces`` is not an integer tensor or ``vertices`` is not floating.
    ValueError
        If ``faces`` is not a non-empty ``(F, 3)`` table or ``vertices`` is not
        ``(V, 3)``. Boundary and non-manifold edges never raise here.
    """
    _check_face_table(faces)
    if not isinstance(vertices, torch.Tensor) or not vertices.is_floating_point():
        raise TypeError("vertices must be a floating torch.Tensor")
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError(f"vertices must have shape (V, 3), got {tuple(vertices.shape)}")

    face_list: list[list[int]] = faces.to(torch.long).cpu().tolist()
    face_count = len(face_list)
    adjacency = _build_face_adjacency(face_list)
    sign, component = _orient_components(adjacency, face_count, allow_non_orientable=True)

    topo_sign = torch.tensor(sign, device=vertices.device, dtype=vertices.dtype)  # (F,)
    component_id = torch.tensor(component, device=vertices.device)  # (F,)

    # Signed tetrahedron volume per face after the topological reorientation,
    # centered like ``Inertia.from_mesh`` so an open component's sign is stable.
    face_indices = faces.to(device=vertices.device, dtype=torch.long)
    centered = vertices - vertices.mean(dim=0)
    a, b, c = centered[face_indices].unbind(dim=1)
    face_signed_volume = topo_sign * (a * torch.cross(b, c, dim=-1)).sum(dim=-1) / 6.0

    component_count = int(component_id.amax()) + 1
    component_volume = torch.zeros(component_count, device=vertices.device, dtype=vertices.dtype)
    component_volume.scatter_add_(0, component_id, face_signed_volume)
    component_reversed = (component_volume < 0.0)[component_id]  # flip inward components

    reverse_mask = (topo_sign < 0) ^ component_reversed  # reversed relative to the input rows
    return _apply_reversal(faces, reverse_mask)
