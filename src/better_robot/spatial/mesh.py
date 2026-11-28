"""Face-table orientation for triangle meshes, the companion of ``inertia_of_mesh``.

``inertia_of_mesh`` folds one global winding sign but trusts every face to agree on orientation, so a surface whose
faces are wound inconsistently gives a wrong result without an error. ``orient_faces_by_component`` rewinds each
connected component so its faces agree and turns it outward. It changes only the integer face table (it reads the
vertices to sign each component's volume), so reorientation is a one-time precompute and the vertex integration
downstream stays differentiable.
"""

from __future__ import annotations

import torch

from ..exceptions import ShapeError
from ..lie.checks import check_index_tensor, check_tensors


def _check_face_table(faces: torch.Tensor, *, device: torch.device | None = None) -> None:
    """Raise unless ``faces`` is a non-empty integer ``(F, 3)`` triangle table, on ``device`` (the vertices') if given."""
    check_index_tensor(
        "faces", faces, shape=("F", 3), dtypes=(torch.int32, torch.int64), device=device, owner="vertices"
    )
    if faces.shape[0] == 0:
        raise ShapeError("faces must have non-empty shape (F, 3), got (0, 3)")


def _build_face_adjacency(face_list: list[list[int]]) -> list[list[tuple[int, int]]]:
    """Face-adjacency graph over edges shared by exactly two faces.

    Each undirected edge is grouped; only edges with incidence two produce a
    link. Boundary (incidence 1) and non-manifold (incidence > 2) edges are
    left out, so the graph is well defined on imperfect topology. ``weight`` is
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


def _orient_components(adjacency: list[list[tuple[int, int]]], face_count: int) -> tuple[list[int], list[int]]:
    """Assign each face a winding sign and a connected-component label.

    ``sign[f]`` is +1 to keep face ``f`` and -1 to reverse it. Consistency
    across a shared edge requires ``sign[g] == -sign[f] * weight``. A closed
    2-manifold can still be *non-orientable* (for example the real projective
    plane), so the walk can meet an already-signed neighbour whose shared edge
    implies the opposite sign; it keeps the first sign (best effort).
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
                if sign[neighbour] == 0:
                    sign[neighbour] = -sign[current] * weight
                    component[neighbour] = next_component
                    stack.append(neighbour)
        next_component += 1
    return sign, component


def _apply_reversal(faces: torch.Tensor, reverse_mask: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``faces`` with the rows selected by ``reverse_mask`` reversed."""
    oriented = faces.clone()
    reverse_mask = reverse_mask.to(faces.device)
    oriented[reverse_mask] = oriented[reverse_mask].flip(-1)
    return oriented


def orient_faces_by_component(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """The face table ``faces: (F, 3)`` rewound so every connected component is consistent and faces outward, on the
    device and in the dtype of ``faces``; a best effort that tolerates imperfect topology.

    ``vertices: (V, 3)`` is the rest geometry the table indexes, read only to sign each component's volume and
    mean-centred as ``inertia_of_mesh`` centres it, so pass the same vertices to both; the result then needs no
    global sign fold there, and the component volumes add up. Two triangles that share an edge agree when they
    traverse it in opposite directions. Authored meshes are often not closed 2-manifolds (parts stitched at a seam
    with boundary edges, non-manifold T-junctions, several shells), so faces are linked only across edges that
    exactly two faces share; each maximal two-manifold patch is a component, made consistent and then flipped if its
    signed volume is negative. Only the integer table changes, so this is a one-time precompute and the vertex
    integration downstream stays differentiable. A ``faces`` that is not a non-empty integer ``(F, 3)`` tensor, or
    ``vertices`` that are not floating ``(V, 3)``, raises ``TypeError``, ``DtypeMismatchError`` or ``ShapeError``;
    boundary and non-manifold edges never raise.
    """
    _check_face_table(faces)
    check_tensors(vertices=(vertices, (3,)))
    if vertices.ndim != 2:
        raise ShapeError(f"vertices must have shape (V, 3), got {tuple(vertices.shape)}")

    face_list: list[list[int]] = faces.to(torch.long).cpu().tolist()  # bench-ok: face topology, read once per mesh
    face_count = len(face_list)
    adjacency = _build_face_adjacency(face_list)
    sign, component = _orient_components(adjacency, face_count)

    topo_sign = torch.tensor(sign, device=vertices.device, dtype=vertices.dtype)  # (F,)
    component_id = torch.tensor(component, device=vertices.device)  # (F,)

    # Signed tetrahedron volume per face after the topological reorientation,
    # centered like ``inertia_of_mesh`` so an open component's sign is stable.
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
