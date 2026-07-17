"""Weighted batched Umeyama similarity alignment."""

from __future__ import annotations

import pytest
import torch

from better_robot.lie import so3, umeyama


def _transform(
    source: torch.Tensor,
    scale: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    return scale[..., None, None] * torch.einsum("...ij,...nj->...ni", rotation, source) + translation[..., None, :]


def test_umeyama_recovers_known_batched_similarity() -> None:
    generator = torch.Generator().manual_seed(17)
    source = torch.randn((2, 3, 12, 3), generator=generator, dtype=torch.float64)
    euler = torch.randn((2, 3, 3), generator=generator, dtype=torch.float64) * 0.4
    rotation = so3.to_matrix(so3.from_euler(euler))
    scale = torch.rand((2, 3), generator=generator, dtype=torch.float64) + 0.5
    translation = torch.randn((2, 3, 3), generator=generator, dtype=torch.float64)
    target = _transform(source, scale, rotation, translation)

    fitted_scale, fitted_rotation, fitted_translation = umeyama(
        source, target, weights=torch.linspace(0.2, 1.0, 12, dtype=torch.float64)
    )

    torch.testing.assert_close(fitted_scale, scale)
    torch.testing.assert_close(fitted_rotation, rotation)
    torch.testing.assert_close(fitted_translation, translation)


def test_umeyama_reflection_fix_returns_proper_rotation() -> None:
    generator = torch.Generator().manual_seed(5)
    source = torch.randn((15, 3), generator=generator, dtype=torch.float64)
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    target = source @ reflection.mT

    _, rotation, _ = umeyama(source, target)

    torch.testing.assert_close(torch.linalg.det(rotation), torch.tensor(1.0, dtype=rotation.dtype))


def test_umeyama_weights_downweight_outlier() -> None:
    generator = torch.Generator().manual_seed(8)
    source = torch.randn((10, 3), generator=generator, dtype=torch.float64)
    rotation = so3.to_matrix(so3.from_euler(torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64)))
    scale = torch.tensor(1.4, dtype=torch.float64)
    translation = torch.tensor([0.2, -0.5, 0.7], dtype=torch.float64)
    clean_target = _transform(source, scale, rotation, translation)
    target = clean_target.clone()
    target[-1] += torch.tensor([20.0, -15.0, 9.0], dtype=torch.float64)

    unweighted = umeyama(source, target)
    weights = torch.ones(10, dtype=torch.float64)
    weights[-1] = 0.0
    weighted = umeyama(source, target, weights=weights)

    unweighted_error = (
        (unweighted[0] - scale).abs() + (unweighted[1] - rotation).norm() + (unweighted[2] - translation).norm()
    )
    weighted_error = (weighted[0] - scale).abs() + (weighted[1] - rotation).norm() + (weighted[2] - translation).norm()
    assert weighted_error < unweighted_error * 1e-6
    torch.testing.assert_close(weighted[0], scale)
    torch.testing.assert_close(weighted[1], rotation)
    torch.testing.assert_close(weighted[2], translation)


def test_umeyama_rigid_fit_disables_scale() -> None:
    generator = torch.Generator().manual_seed(12)
    source = torch.randn((9, 3), generator=generator, dtype=torch.float64)
    rotation = so3.to_matrix(so3.from_euler(torch.tensor([-0.2, 0.1, 0.3], dtype=torch.float64)))
    translation = torch.tensor([-0.4, 0.6, 0.2], dtype=torch.float64)
    target = _transform(source, torch.tensor(1.0), rotation, translation)
    scale, fitted_rotation, fitted_translation = umeyama(source, target, estimate_scale=False)
    torch.testing.assert_close(scale, torch.tensor(1.0, dtype=source.dtype))
    torch.testing.assert_close(fitted_rotation, rotation)
    torch.testing.assert_close(fitted_translation, translation)


def test_umeyama_gradients_are_finite() -> None:
    generator = torch.Generator().manual_seed(21)
    source = torch.randn((11, 3), generator=generator, dtype=torch.float64, requires_grad=True)
    target = torch.randn((11, 3), generator=generator, dtype=torch.float64)
    scale, rotation, translation = umeyama(source, target)
    (scale.square() + rotation.square().sum() + translation.square().sum()).backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


def test_umeyama_rejects_mismatched_dtypes() -> None:
    source = torch.randn((5, 3), dtype=torch.float32)
    target = torch.randn((5, 3), dtype=torch.float64)

    with pytest.raises(ValueError, match="source and target must share dtype and device"):
        umeyama(source, target)

    with pytest.raises(ValueError, match="weights must share source dtype and device"):
        umeyama(source, source.clone(), weights=torch.ones(5, dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for device mismatch coverage")
def test_umeyama_rejects_mismatched_devices() -> None:
    source = torch.randn((5, 3), device="cpu")
    target = torch.randn((5, 3), device="cuda")

    with pytest.raises(ValueError, match="source and target must share dtype and device"):
        umeyama(source, target)

    with pytest.raises(ValueError, match="weights must share source dtype and device"):
        umeyama(source, source.clone(), weights=torch.ones(5, device="cuda"))


@pytest.mark.parametrize(("input_name", "nonfinite"), [("source", torch.nan), ("target", torch.inf)])
def test_umeyama_rejects_nonfinite_point_inputs(input_name: str, nonfinite: float) -> None:
    source = torch.randn((5, 3))
    target = torch.randn((5, 3))
    value = source if input_name == "source" else target
    value[0, 0] = nonfinite

    with pytest.raises(ValueError, match="source and target must contain only finite values"):
        umeyama(source, target)


@pytest.mark.parametrize("nonfinite", [torch.nan, torch.inf, -torch.inf])
def test_umeyama_rejects_nonfinite_weights(nonfinite: float) -> None:
    source = torch.randn((5, 3))
    weights = torch.ones(5)
    weights[0] = nonfinite

    with pytest.raises(ValueError, match="weights must be finite and non-negative"):
        umeyama(source, source.clone(), weights=weights)
