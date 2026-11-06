"""Differentiable signed-tetrahedron mesh inertia, ``better_robot.spatial.inertia_of_mesh``.

Oracles: the closed-form inertia of a box, and ``trimesh.Trimesh`` mass properties on the same mesh.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import trimesh

from better_robot.spatial import inertia_of_mesh, rotational_inertia


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
    inertia = inertia_of_mesh(vertices + translation, faces, density=density)

    mass = density * size[0] * size[1] * size[2]
    expected_diagonal = torch.tensor(
        [
            mass * (size[1] ** 2 + size[2] ** 2) / 12.0,
            mass * (size[0] ** 2 + size[2] ** 2) / 12.0,
            mass * (size[0] ** 2 + size[1] ** 2) / 12.0,
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(inertia[0], torch.tensor(mass, dtype=torch.float64))
    torch.testing.assert_close(inertia[1:4], translation, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(inertia[4:7], expected_diagonal, atol=1e-10, rtol=1e-12)
    torch.testing.assert_close(inertia[7:10], torch.zeros(3, dtype=torch.float64), atol=1e-10, rtol=0.0)


def test_global_face_winding_is_orientation_invariant() -> None:
    vertices, faces = _box_mesh()
    forward = inertia_of_mesh(vertices, faces, density=1.7)
    reversed_winding = inertia_of_mesh(vertices, faces.flip(-1), density=1.7)
    torch.testing.assert_close(reversed_winding, forward, atol=1e-12, rtol=1e-12)


def test_multi_axis_batch_and_density_match_per_element_loop() -> None:
    vertices, faces = _box_mesh(dtype=torch.float32)
    scales = torch.tensor([[0.7, 1.1, 1.4], [0.9, 1.3, 1.8]], dtype=torch.float32)
    translations = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3) * 0.1
    batched_vertices = vertices.reshape(1, 1, 8, 3) * scales[..., None, None]
    batched_vertices = batched_vertices + translations[..., None, :]
    density = torch.linspace(0.8, 1.8, 6, dtype=torch.float32).reshape(2, 3)

    actual = inertia_of_mesh(batched_vertices, faces, density=density)
    expected = torch.stack(
        [inertia_of_mesh(batched_vertices[i, j], faces, density=density[i, j]) for i in range(2) for j in range(3)]
    ).reshape(2, 3, 10)
    assert actual.shape == (2, 3, 10)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_number_and_tensor_density_agree_and_default_to_unit_density() -> None:
    vertices, faces = _box_mesh()
    torch.testing.assert_close(
        inertia_of_mesh(vertices, faces, density=2),
        inertia_of_mesh(vertices, faces, density=torch.tensor(2.0, dtype=torch.float64)),
        atol=0.0,
        rtol=0.0,
    )
    assert float(inertia_of_mesh(vertices, faces)[0]) == pytest.approx(24.0)


def test_vertex_and_density_autograd() -> None:
    vertices, faces = _box_mesh()
    vertices = vertices.requires_grad_(True)
    density = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    inertia = inertia_of_mesh(vertices, faces, density=density)
    loss = inertia[0] + inertia[1:4].square().sum() + inertia[4:7].sum()
    vertex_grad, density_grad = torch.autograd.grad(loss, (vertices, density))

    assert torch.isfinite(vertex_grad).all()
    assert torch.count_nonzero(vertex_grad) > 0
    assert torch.isfinite(density_grad)
    (density_mass_grad,) = torch.autograd.grad(inertia_of_mesh(vertices.detach(), faces, density=density)[0], density)
    torch.testing.assert_close(density_mass_grad, torch.tensor(24.0, dtype=torch.float64))


def test_float64_vertex_gradcheck() -> None:
    vertices, faces = _box_mesh()
    vertices = vertices + torch.linspace(-0.01, 0.01, vertices.numel(), dtype=torch.float64).reshape_as(vertices)
    vertices.requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda value: inertia_of_mesh(value, faces, density=1.2),
        (vertices,),
        eps=1e-6,
        atol=2e-6,
        rtol=2e-5,
        fast_mode=True,
    )


def test_matches_trimesh_reference() -> None:
    vertices, faces = _box_mesh()
    reference = trimesh.Trimesh(vertices=vertices.numpy(), faces=faces.numpy(), process=False)
    actual = inertia_of_mesh(vertices, faces)

    np.testing.assert_allclose(actual[0].numpy(), reference.mass, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(actual[1:4].numpy(), reference.center_mass, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(rotational_inertia(actual).numpy(), reference.moment_inertia, atol=1e-12, rtol=1e-12)


def _flat_mesh() -> tuple[torch.Tensor, torch.Tensor]:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    return vertices, torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long)


def test_degenerate_mesh_raises_only_when_checked() -> None:
    vertices, faces = _flat_mesh()
    with pytest.raises(ValueError, match="degenerate enclosed volume"):
        inertia_of_mesh(vertices, faces, check_mesh=True)
    assert not torch.isfinite(inertia_of_mesh(vertices, faces)).all()


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        pytest.param(
            lambda v, f, d: (v.index_fill(0, torch.tensor([0]), torch.nan), f, d), "finite values", id="nan-vertex"
        ),
        pytest.param(
            lambda v, f, d: (v, f.index_fill(0, torch.tensor([0]), 8), d), "outside the vertices", id="face-high"
        ),
        pytest.param(
            lambda v, f, d: (v, f.index_fill(0, torch.tensor([0]), -1), d), "outside the vertices", id="face-low"
        ),
        pytest.param(lambda v, f, d: (v, f, -d), "finite positive", id="negative-density"),
    ],
)
def test_check_mesh_reads_the_values(edit, match: str) -> None:
    vertices, faces, density = edit(*_box_mesh(), torch.tensor(1.0, dtype=torch.float64))
    with pytest.raises(ValueError, match=match):
        inertia_of_mesh(vertices, faces, density=density, check_mesh=True)


def test_unchecked_call_reads_no_values() -> None:
    """A NaN vertex and a negative density pass through without a host-side check."""
    vertices, faces = _box_mesh()
    vertices = vertices.index_fill(0, torch.tensor([0]), torch.nan)
    assert torch.isnan(inertia_of_mesh(vertices, faces, density=-1.0)).any()


@pytest.mark.parametrize(
    ("vertices", "faces", "density", "error", "match"),
    [
        pytest.param(
            torch.zeros(8, 3, dtype=torch.int64),
            None,
            1.0,
            ValueError,
            "vertices.dtype=torch.int64 is not floating",
            id="int-vertices",
        ),
        pytest.param(
            [[0.0] * 3] * 8, None, 1.0, TypeError, "vertices must be a floating torch.Tensor", id="list-vertices"
        ),
        pytest.param(torch.zeros(8, 2), None, 1.0, ValueError, r"shape \(\.\.\., V, 3\)", id="vertex-shape"),
        pytest.param(torch.zeros(3, 3), None, 1.0, ValueError, "at least four points", id="three-vertices"),
        pytest.param(
            None, torch.zeros(12, 3), 1.0, ValueError, "faces.dtype=torch.float32 is not torch.int32", id="float-faces"
        ),
        pytest.param(None, [[0, 1, 2]] * 12, 1.0, TypeError, "faces must be a torch.Tensor", id="list-faces"),
        pytest.param(None, torch.zeros(12, 4, dtype=torch.long), 1.0, ValueError, r"shape \(F, 3\)", id="face-shape"),
        pytest.param(None, torch.zeros(0, 3, dtype=torch.long), 1.0, ValueError, r"shape \(F, 3\)", id="no-faces"),
        pytest.param(
            None,
            torch.zeros(12, 3, dtype=torch.long, device="meta"),
            1.0,
            ValueError,
            r"faces is on meta but vertices is on cpu; pass faces.to\(cpu\)",
            id="faces-device",
        ),
        pytest.param(None, None, True, TypeError, "density must be a real number", id="bool-density"),
        pytest.param(None, None, torch.tensor(1.0), ValueError, "density.dtype=torch.float32", id="density-dtype"),
        pytest.param(
            None, None, torch.ones(2, dtype=torch.float64), ValueError, "mesh batch shape", id="density-shape"
        ),
    ],
)
def test_structure_is_checked(vertices, faces, density, error: type, match: str) -> None:
    box_vertices, box_faces = _box_mesh()
    with pytest.raises(error, match=match):
        inertia_of_mesh(
            box_vertices if vertices is None else vertices, box_faces if faces is None else faces, density=density
        )
