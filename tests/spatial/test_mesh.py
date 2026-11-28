"""Face-table reorientation, ``better_robot.spatial.orient_faces_by_component``.

Oracles: the inertia of the outward-wound mesh before its winding was broken, and the half-edge rule (a consistently
wound closed surface holds each half-edge once and its reverse once).
"""

from __future__ import annotations

import pytest
import torch

from better_robot.exceptions import DtypeMismatchError
from better_robot.spatial import inertia_of_mesh, orient_faces_by_component
from tests.support.meshes import box_mesh

_DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.cuda,
    ),
]


def _mix_winding(faces: torch.Tensor, flipped: tuple[int, ...]) -> torch.Tensor:
    """Reverse the winding of ``flipped`` rows, leaving a mixed-winding table."""
    mixed = faces.clone()
    index = torch.tensor(flipped, device=faces.device)
    mixed[index] = mixed[index].flip(-1)
    return mixed


@pytest.mark.parametrize("device", _DEVICES)
def test_by_component_preserves_dtype_and_device_and_is_idempotent(device: str) -> None:
    vertices, faces = box_mesh(torch.float32, device=device)
    faces32 = faces.to(torch.int32)
    mixed = _mix_winding(faces32, (1, 4, 9))

    oriented = orient_faces_by_component(vertices, mixed)
    assert oriented.dtype == torch.int32
    assert oriented.device == mixed.device
    assert oriented.shape == mixed.shape
    # A second pass changes nothing: the table is already consistent.
    torch.testing.assert_close(orient_faces_by_component(vertices, oriented), oriented)


def _perturbed_icosahedron(
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """An irregular (symmetry-broken) closed convex mesh, 12 vertices and 20 faces wound outward."""
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
    return vertices, faces


@pytest.mark.parametrize("device", _DEVICES)
def test_by_component_repairs_mixed_winding_on_an_irregular_mesh(device: str) -> None:
    # A non-symmetric mesh: unlike the box, summation order cannot mask a partial
    # inconsistency behind cancellation.
    vertices, faces = _perturbed_icosahedron(device=device)
    reference = inertia_of_mesh(vertices, faces)

    mask = torch.rand(faces.shape[0], generator=torch.Generator().manual_seed(3)) < 0.4
    mixed = faces.clone()
    mixed[mask.to(device)] = mixed[mask.to(device)].flip(-1)

    repaired = inertia_of_mesh(vertices, orient_faces_by_component(vertices, mixed))
    torch.testing.assert_close(repaired, reference, atol=1e-4, rtol=1e-4)
    assert repaired[0] > 0.0


@pytest.mark.parametrize("device", _DEVICES)
def test_by_component_orients_disjoint_shells_outward_and_conserves_volume(device: str) -> None:
    box_vertices, box_faces = box_mesh(torch.float32, device=device)
    offset = torch.tensor([50.0, 0.0, 0.0], dtype=box_vertices.dtype, device=device)
    vertices = torch.cat([box_vertices, box_vertices + offset], dim=0)
    faces = torch.cat([box_faces, box_faces + box_vertices.shape[0]], dim=0)
    mixed = _mix_winding(faces, (0, 3, 5))  # break winding of one shell

    oriented = orient_faces_by_component(vertices, mixed)
    both_boxes = inertia_of_mesh(vertices, oriented)
    torch.testing.assert_close(
        both_boxes[0], torch.tensor(48.0, dtype=vertices.dtype, device=device), atol=1e-3, rtol=1e-4
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
    vertices, faces = box_mesh(torch.float32)
    open_faces = torch.cat([faces[:2], faces[4:]], dim=0)  # boundary edges present
    non_manifold = torch.cat([faces, faces[:1]], dim=0)  # an incidence-3 edge
    assert orient_faces_by_component(vertices, open_faces).shape == open_faces.shape
    assert orient_faces_by_component(vertices, non_manifold).shape == non_manifold.shape


def test_by_component_validates_its_vertices() -> None:
    _, faces = box_mesh(torch.float32)
    with pytest.raises(DtypeMismatchError, match="not float32 or float64"):
        orient_faces_by_component(torch.zeros((8, 3), dtype=torch.int64), faces)
    with pytest.raises(ValueError, match=r"shape \(V, 3\)"):
        orient_faces_by_component(torch.zeros((2, 8, 3), dtype=torch.float32), faces)
