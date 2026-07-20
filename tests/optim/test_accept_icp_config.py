"""Public v2 acceptance: ICP workarounds are optimizer configuration."""

from __future__ import annotations

import torch

from better_robot.lie import so3
from better_robot.optim import (
    LevenbergMarquardt,
    Problem,
    Residual,
    SO3Variable,
    Variable,
)


class _PointToPlaneResidual(Residual):
    """Test-local fixed-correspondence Sim(3)-style point-to-plane residual."""

    def __init__(
        self,
        translation: Variable,
        rotation: SO3Variable,
        log_s: Variable,
        source: Variable,
        target: Variable,
        normals: Variable,
    ) -> None:
        self.translation = translation
        self.rotation = rotation
        self.log_s = log_s
        self.source = source
        self.target = target
        self.normals = normals
        super().__init__(
            translation,
            rotation,
            log_s,
            source,
            target,
            normals,
            dim=source.tensor.shape[0],
            name="point_to_plane",
        )

    def error(self) -> torch.Tensor:
        scaled = self.log_s.tensor.exp().unsqueeze(-2) * self.source.tensor
        rotated = so3.act(self.rotation.tensor.unsqueeze(-2), scaled)
        predicted = rotated + self.translation.tensor.unsqueeze(-2)
        return ((predicted - self.target.tensor) * self.normals.tensor).sum(dim=-1)


def _synthetic_observations(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(1207)
    source = torch.randn(40, 3, generator=generator, dtype=dtype)
    normals = torch.randn(40, 3, generator=generator, dtype=dtype)
    normals = normals / normals.norm(dim=-1, keepdim=True)
    translation = torch.tensor([0.48, -0.36, 0.27], dtype=dtype)
    rotation = so3.exp(torch.tensor([0.21, -0.14, 0.11], dtype=dtype))
    log_s = torch.tensor([0.18], dtype=dtype)
    target = so3.act(rotation.unsqueeze(-2), log_s.exp().unsqueeze(-2) * source)
    target = target + translation.unsqueeze(-2)
    return source, target, normals, translation, rotation, log_s


def test_icp_relative_damping_and_external_stop_are_configuration() -> None:
    dtype = torch.float64
    source_tensor, target_tensor, normals_tensor, translation_true, rotation_true, log_s_true = _synthetic_observations(
        dtype
    )
    translation = Variable(torch.tensor([-1.60, 1.25, -0.90], dtype=dtype), name="translation")
    rotation = SO3Variable(
        so3.exp(torch.tensor([-0.62, 0.45, -0.31], dtype=dtype)),
        name="rotation",
    )
    log_s = Variable(torch.tensor([-0.42], dtype=dtype), name="log_s")
    source = Variable(source_tensor, name="source", trainable=False)
    target = Variable(target_tensor, name="target", trainable=False)
    normals = Variable(normals_tensor, name="normals", trainable=False)
    problem = Problem([_PointToPlaneResidual(translation, rotation, log_s, source, target, normals)])
    initial = {
        "translation": translation.tensor,
        "rotation": rotation.tensor,
        "log_s": log_s.tensor,
    }
    relative_damping = 1e-3
    optimizer = LevenbergMarquardt(
        problem,
        max_iterations=0,
        tolerance=0.0,
        step_tolerance=0.0,
        relative_tolerance=0.0,
        damping=relative_damping,
    )

    state = optimizer._init_state(initial, problem)
    jacobian = problem.dense_jacobian()
    expected_mu = relative_damping * (jacobian.mT @ jacobian).diagonal().amax()
    torch.testing.assert_close(state.mu, expected_mu)

    values = initial
    external_tolerance = 1e-11
    stopped_externally = False
    for _outer_iteration in range(120):
        next_values, next_state = optimizer._update(values, state, problem)
        decrease = state.cost - next_state.cost
        values, state = next_values, next_state
        if bool((decrease > 0.0) & (decrease.abs() < external_tolerance)):
            stopped_externally = True
            break

    assert stopped_externally
    assert state.iterations > 0
    assert state.cost < 1e-13
    torch.testing.assert_close(values["translation"], translation_true, atol=2e-6, rtol=2e-6)
    rotation_error = so3.log(so3.compose(so3.inverse(rotation_true), values["rotation"]))
    torch.testing.assert_close(rotation_error, torch.zeros(3, dtype=dtype), atol=2e-6, rtol=0.0)
    torch.testing.assert_close(values["log_s"], log_s_true, atol=2e-6, rtol=2e-6)
