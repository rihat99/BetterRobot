"""Face-table reorientation and closed-manifold validation."""

from __future__ import annotations

import pytest
import torch

from better_robot.spatial import (
    Inertia,
    orient_faces_by_component,
    orient_faces_consistently,
    validate_closed_manifold,
)

_DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "cuda",
        id="cuda",
        marks=[
            pytest.mark.cuda,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
        ],
    ),
]


def _box_mesh(
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a closed, consistently wound ``(2, 3, 4)`` box mesh."""
    vertices = torch.tensor(
        [
            [-1.0, -1.5, -2.0],
            [1.0, -1.5, -2.0],
            [1.0, 1.5, -2.0],
            [-1.0, 1.5, -2.0],
            [-1.0, -1.5, 2.0],
            [1.0, -1.5, 2.0],
            [1.0, 1.5, 2.0],
            [-1.0, 1.5, 2.0],
        ],
        dtype=dtype,
        device=device,
    )
    faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=torch.int64,
        device=device,
    )
    return vertices, faces


def _mix_winding(faces: torch.Tensor, flipped: tuple[int, ...]) -> torch.Tensor:
    """Reverse the winding of ``flipped`` rows, leaving a mixed-winding table."""
    mixed = faces.clone()
    index = torch.tensor(flipped, device=faces.device)
    mixed[index] = mixed[index].flip(-1)
    return mixed


@pytest.mark.parametrize("device", _DEVICES)
def test_reorient_matches_reference_inertia_and_beats_the_mixed_input(device: str) -> None:
    vertices, faces = _box_mesh(device=device)
    reference = Inertia.from_mesh(vertices, faces, density=2.5)

    mixed = _mix_winding(faces, (0, 3, 5, 8, 11))
    # The mixed table is still a closed manifold, so ``from_mesh`` accepts it and
    # returns a plausible but wrong result — the bug this tool exists to repair.
    wrong = Inertia.from_mesh(vertices, mixed, density=2.5)
    assert not torch.allclose(wrong.data, reference.data, atol=1e-3)

    oriented = orient_faces_consistently(mixed)
    repaired = Inertia.from_mesh(vertices, oriented, density=2.5)
    torch.testing.assert_close(repaired.data, reference.data, atol=1e-5, rtol=1e-5)
    assert repaired.mass > 0.0


@pytest.mark.parametrize("device", _DEVICES)
def test_reorient_preserves_dtype_and_device_and_is_idempotent(device: str) -> None:
    _, faces = _box_mesh(device=device, dtype=torch.float32)
    faces32 = faces.to(torch.int32)
    mixed = _mix_winding(faces32, (1, 4, 9))

    oriented = orient_faces_consistently(mixed)
    assert oriented.dtype == torch.int32
    assert oriented.device == mixed.device
    assert oriented.shape == mixed.shape
    # A second pass changes nothing: the table is already consistent.
    torch.testing.assert_close(orient_faces_consistently(oriented), oriented)


def test_reorient_yields_opposite_edge_traversal_everywhere() -> None:
    _, faces = _box_mesh()
    oriented = orient_faces_consistently(_mix_winding(faces, (2, 6, 7, 10)))

    directed: dict[tuple[int, int], int] = {}
    for i, j, k in oriented.tolist():
        for a, b in ((i, j), (j, k), (k, i)):
            directed[(a, b)] = directed.get((a, b), 0) + 1
    # Consistent winding means each half-edge appears once and its reverse once.
    for (a, b), count in directed.items():
        assert count == 1
        assert directed.get((b, a), 0) == 1


@pytest.mark.parametrize("device", _DEVICES)
def test_two_disjoint_closed_boxes_validate_and_orient(device: str) -> None:
    vertices, faces = _box_mesh(device=device)
    two_faces = torch.cat([faces, faces + vertices.shape[0]], dim=0)

    validate_closed_manifold(two_faces)  # each component is closed
    oriented = orient_faces_consistently(_mix_winding(two_faces, (0, 13, 20)))
    validate_closed_manifold(oriented)


def test_open_surface_raises_naming_boundary_edges() -> None:
    _, faces = _box_mesh()
    open_faces = torch.cat([faces[:2], faces[4:]], dim=0)  # drop the +z cap
    with pytest.raises(ValueError, match="boundary edge") as excinfo:
        validate_closed_manifold(open_faces)
    assert "[" in str(excinfo.value)  # the message names offending edges


def test_open_surface_also_rejected_by_reorientation() -> None:
    _, faces = _box_mesh()
    open_faces = torch.cat([faces[:2], faces[4:]], dim=0)
    with pytest.raises(ValueError, match="boundary edge"):
        orient_faces_consistently(open_faces)


def test_non_manifold_edge_raises_naming_edges() -> None:
    _, faces = _box_mesh()
    non_manifold = torch.cat([faces, faces[:1]], dim=0)  # duplicate one face
    with pytest.raises(ValueError, match="non-manifold edge") as excinfo:
        validate_closed_manifold(non_manifold)
    assert "incidence" in str(excinfo.value)


def test_non_integer_faces_raise_type_error() -> None:
    with pytest.raises(TypeError, match="int32 or torch.int64"):
        validate_closed_manifold(torch.zeros((4, 3), dtype=torch.float32))


@pytest.mark.parametrize("bad", [torch.zeros((4, 4), dtype=torch.int64), torch.zeros((0, 3), dtype=torch.int64)])
def test_wrong_shape_faces_raise_value_error(bad: torch.Tensor) -> None:
    with pytest.raises(ValueError, match=r"shape \(F, 3\)"):
        orient_faces_consistently(bad)


# --- non-orientable input (bug fix) ------------------------------------------


def _projective_plane_faces() -> torch.Tensor:
    """Minimal RP2 triangulation: the antipodal quotient of the icosahedron.

    Ten faces over six vertices; every edge has incidence two (a closed
    2-manifold) yet the surface is non-orientable, so no consistent winding
    exists.
    """
    icosahedron = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    antipodal = {0: 0, 3: 0, 1: 1, 2: 1, 4: 2, 7: 2, 5: 3, 6: 3, 8: 4, 11: 4, 9: 5, 10: 5}
    seen: set[frozenset[int]] = set()
    faces: list[tuple[int, int, int]] = []
    for triangle in icosahedron:
        quotient = tuple(antipodal[v] for v in triangle)
        key = frozenset(quotient)
        if key not in seen:
            seen.add(key)
            faces.append(quotient)
    return torch.tensor(faces, dtype=torch.int64)


def test_non_orientable_surface_is_rejected_by_the_strict_orienter() -> None:
    faces = _projective_plane_faces()
    validate_closed_manifold(faces)  # closed 2-manifold: incidence is exactly two everywhere
    with pytest.raises(ValueError, match="non-orientable"):
        orient_faces_consistently(faces)


# --- best-effort component orientation ---------------------------------------


def _perturbed_icosahedron(
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """An irregular (symmetry-broken) closed convex mesh: 12 verts, 20 faces."""
    t = (1.0 + 5.0**0.5) / 2.0
    vertices = torch.tensor(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],  # fmt: skip
        dtype=dtype,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(0)
    vertices = vertices + 0.15 * torch.randn(vertices.shape, generator=generator, dtype=dtype, device=device)
    faces = torch.tensor(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],  # fmt: skip
        dtype=torch.int64,
        device=device,
    )
    return vertices, orient_faces_consistently(faces)


@pytest.mark.parametrize("device", _DEVICES)
def test_by_component_repairs_mixed_winding_on_an_irregular_mesh(device: str) -> None:
    # A non-symmetric mesh: unlike the box, summation order cannot mask a partial
    # inconsistency behind cancellation.
    vertices, faces = _perturbed_icosahedron(device=device)
    reference = Inertia.from_mesh(vertices, faces)

    mask = torch.rand(faces.shape[0], generator=torch.Generator().manual_seed(3)) < 0.4
    mixed = faces.clone()
    mixed[mask.to(device)] = mixed[mask.to(device)].flip(-1)

    repaired = Inertia.from_mesh(vertices, orient_faces_by_component(vertices, mixed))
    torch.testing.assert_close(repaired.data, reference.data, atol=1e-4, rtol=1e-4)
    assert repaired.mass > 0.0


@pytest.mark.parametrize("device", _DEVICES)
def test_by_component_matches_strict_orienter_on_a_clean_manifold(device: str) -> None:
    vertices, faces = _perturbed_icosahedron(device=device)
    mixed = _mix_winding(faces, (0, 4, 9, 15))
    by_component = Inertia.from_mesh(vertices, orient_faces_by_component(vertices, mixed))
    strict = Inertia.from_mesh(vertices, orient_faces_consistently(mixed))
    torch.testing.assert_close(by_component.data, strict.data, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES)
def test_by_component_orients_disjoint_shells_outward_and_conserves_volume(device: str) -> None:
    box_vertices, box_faces = _box_mesh(device=device)
    offset = torch.tensor([50.0, 0.0, 0.0], dtype=box_vertices.dtype, device=device)
    vertices = torch.cat([box_vertices, box_vertices + offset], dim=0)
    faces = torch.cat([box_faces, box_faces + box_vertices.shape[0]], dim=0)
    mixed = _mix_winding(faces, (0, 3, 5))  # break winding of one shell

    oriented = orient_faces_by_component(vertices, mixed)
    both_boxes = Inertia.from_mesh(vertices, oriented)
    torch.testing.assert_close(
        both_boxes.mass, torch.tensor(48.0, dtype=vertices.dtype, device=device), atol=1e-3, rtol=1e-4
    )

    # Both shells outward means every half-edge appears once with its reverse once.
    directed: dict[tuple[int, int], int] = {}
    for i, j, k in oriented.cpu().tolist():
        for a, b in ((i, j), (j, k), (k, i)):
            directed[(a, b)] = directed.get((a, b), 0) + 1
    for (a, b), count in directed.items():
        assert count == 1
        assert directed.get((b, a), 0) == 1


def test_by_component_degrades_gracefully_on_boundary_and_non_manifold_input() -> None:
    vertices, faces = _box_mesh()
    open_faces = torch.cat([faces[:2], faces[4:]], dim=0)  # boundary edges present
    non_manifold = torch.cat([faces, faces[:1]], dim=0)  # an incidence-3 edge
    # The strict path rejects both; the best-effort path must not raise.
    assert orient_faces_by_component(vertices, open_faces).shape == open_faces.shape
    assert orient_faces_by_component(vertices, non_manifold).shape == non_manifold.shape


def test_by_component_validates_its_vertices() -> None:
    _, faces = _box_mesh()
    with pytest.raises(TypeError, match="floating"):
        orient_faces_by_component(torch.zeros((8, 3), dtype=torch.int64), faces)
    with pytest.raises(ValueError, match=r"shape \(V, 3\)"):
        orient_faces_by_component(torch.zeros((2, 8, 3), dtype=torch.float32), faces)
