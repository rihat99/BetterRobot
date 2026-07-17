"""Public named-block acceptance: ICP workarounds are solver configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from better_robot.lie import so3
from better_robot.optim import (
    Euclidean,
    LevenbergMarquardt,
    Problem,
    ResidualItem,
    SO3Manifold,
    VarSpec,
)


class _PointToPlaneResidual:
    """Test-local fixed-correspondence Sim(3)-style point-to-plane residual."""

    name = "point_to_plane"
    reads = ("translation", "rotation", "log_s", "source", "target", "normals")

    def __init__(self, points: int) -> None:
        self.dim = points

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        scaled = ctx["log_s"].exp().unsqueeze(-2) * ctx["source"]
        rotated = so3.act(ctx["rotation"].unsqueeze(-2), scaled)
        predicted = rotated + ctx["translation"].unsqueeze(-2)
        return ((predicted - ctx["target"]) * ctx["normals"]).sum(dim=-1)


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


def test_icp_step_caps_relative_damping_and_external_stop_are_configuration() -> None:
    dtype = torch.float64
    source, target, normals, translation_true, rotation_true, log_s_true = _synthetic_observations(dtype)
    residual = _PointToPlaneResidual(source.shape[0])
    problem = Problem(
        vars=(
            VarSpec("translation", (3,), manifold=Euclidean()),
            VarSpec("rotation", (4,), manifold=SO3Manifold()),
            VarSpec("log_s", (1,), manifold=Euclidean()),
        ),
        residuals=(ResidualItem(residual.name, residual),),
        parameters={"source": source, "target": target, "normals": normals},
    )
    initial = {
        "translation": torch.tensor([-1.60, 1.25, -0.90], dtype=dtype),
        "rotation": so3.exp(torch.tensor([-0.62, 0.45, -0.31], dtype=dtype)),
        "log_s": torch.tensor([-0.42], dtype=dtype),
    }
    relative_damping = 1e-3
    limits = (("translation", 0.20), ("rotation", 0.50), ("log_s", 0.30))
    solver = LevenbergMarquardt(
        max_iter=0,
        gtol=0.0,
        xtol=0.0,
        ftol=0.0,
        damping_parameter=relative_damping,
        block_step_limits=limits,
    )

    state = solver.init_state(initial, problem)
    jacobian = problem.dense_jacobian(initial)
    expected_mu = relative_damping * (jacobian.mT @ jacobian).diagonal().amax()
    torch.testing.assert_close(state.mu, expected_mu)

    first_values, first_state = solver.update(initial, state, problem)
    translation_step = (first_values["translation"] - initial["translation"]).norm()
    rotation_step = so3.log(so3.compose(so3.inverse(initial["rotation"]), first_values["rotation"])).norm()
    scale_step = (first_values["log_s"] - initial["log_s"]).norm()
    assert 0.19 < translation_step <= 0.20 + 1e-12
    assert 0.49 < rotation_step <= 0.50 + 1e-12
    assert 0.29 < scale_step <= 0.30 + 1e-12

    # Consumer-owned association/stopping loop: max_iter=0 disables run(), but
    # the public step API remains usable and the consumer chooses |delta cost|.
    values = first_values
    state = first_state
    external_tolerance = 1e-11
    stopped_externally = False
    for _outer_iteration in range(120):
        next_values, next_state = solver.update(values, state, problem)
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
