"""Differentiable signed-tetrahedron mesh inertia integration."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from better_robot.spatial import Inertia


def _box_mesh(
    size: tuple[float, float, float] = (2.0, 3.0, 4.0),
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    hx, hy, hz = (length / 2.0 for length in size)
    vertices = torch.tensor(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=dtype,
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
        dtype=torch.long,
    )
    return vertices, faces


def test_box_matches_closed_form_with_translation_and_density() -> None:
    size = (2.0, 3.0, 4.0)
    vertices, faces = _box_mesh(size)
    translation = torch.tensor([1000.0, -2000.0, 3000.0], dtype=torch.float64)
    density = 2.5
    inertia = Inertia.from_mesh(vertices + translation, faces, density)

    volume = size[0] * size[1] * size[2]
    mass = density * volume
    expected_diagonal = torch.tensor(
        [
            mass * (size[1] ** 2 + size[2] ** 2) / 12.0,
            mass * (size[0] ** 2 + size[2] ** 2) / 12.0,
            mass * (size[0] ** 2 + size[1] ** 2) / 12.0,
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(inertia.mass, torch.tensor(mass, dtype=torch.float64))
    torch.testing.assert_close(inertia.com, translation, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(inertia.inertia_matrix.diagonal(), expected_diagonal, atol=1e-10, rtol=1e-12)
    torch.testing.assert_close(
        inertia.inertia_matrix - torch.diag(inertia.inertia_matrix.diagonal()),
        torch.zeros(3, 3, dtype=torch.float64),
        atol=1e-10,
        rtol=0.0,
    )


def test_global_face_winding_is_orientation_invariant() -> None:
    vertices, faces = _box_mesh()
    forward = Inertia.from_mesh(vertices, faces, density=1.7)
    reversed_winding = Inertia.from_mesh(vertices, faces.flip(-1), density=1.7)
    torch.testing.assert_close(reversed_winding.data, forward.data, atol=1e-12, rtol=1e-12)


def test_multi_axis_batch_and_density_match_per_element_loop() -> None:
    vertices, faces = _box_mesh(dtype=torch.float32)
    scales = torch.tensor([[0.7, 1.1, 1.4], [0.9, 1.3, 1.8]], dtype=torch.float32)
    translations = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3) * 0.1
    batched_vertices = vertices.reshape(1, 1, 8, 3) * scales[..., None, None]
    batched_vertices = batched_vertices + translations[..., None, :]
    density = torch.linspace(0.8, 1.8, 6, dtype=torch.float32).reshape(2, 3)

    actual = Inertia.from_mesh(batched_vertices, faces, density).data
    expected = torch.stack(
        [Inertia.from_mesh(batched_vertices[i, j], faces, density[i, j]).data for i in range(2) for j in range(3)]
    ).reshape(2, 3, 10)
    assert actual.shape == (2, 3, 10)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_vertex_and_density_autograd() -> None:
    vertices, faces = _box_mesh()
    vertices = vertices.requires_grad_(True)
    density = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    inertia = Inertia.from_mesh(vertices, faces, density)
    loss = inertia.mass + inertia.com.square().sum() + inertia.inertia_matrix.diagonal().sum()
    vertex_grad, density_grad = torch.autograd.grad(loss, (vertices, density))

    assert torch.isfinite(vertex_grad).all()
    assert torch.count_nonzero(vertex_grad) > 0
    assert torch.isfinite(density_grad)
    volume = 24.0
    density_mass_grad = torch.autograd.grad(
        Inertia.from_mesh(vertices.detach(), faces, density).mass,
        density,
    )[0]
    torch.testing.assert_close(density_mass_grad, torch.tensor(volume, dtype=torch.float64))


def test_float64_vertex_gradcheck() -> None:
    vertices, faces = _box_mesh()
    vertices = vertices + torch.linspace(-0.01, 0.01, vertices.numel(), dtype=torch.float64).reshape_as(vertices)
    vertices.requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda value: Inertia.from_mesh(value, faces, density=1.2).data,
        (vertices,),
        eps=1e-6,
        atol=2e-6,
        rtol=2e-5,
        fast_mode=True,
    )


def test_matches_trimesh_reference() -> None:
    trimesh = pytest.importorskip("trimesh")
    vertices, faces = _box_mesh()
    reference = trimesh.Trimesh(vertices=vertices.numpy(), faces=faces.numpy(), process=False)
    actual = Inertia.from_mesh(vertices, faces)

    np.testing.assert_allclose(actual.mass.numpy(), reference.mass, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(actual.com.numpy(), reference.center_mass, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(actual.inertia_matrix.numpy(), reference.moment_inertia, atol=1e-12, rtol=1e-12)


def test_degenerate_mesh_fails_fast() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="degenerate enclosed volume"):
        Inertia.from_mesh(vertices, faces)
