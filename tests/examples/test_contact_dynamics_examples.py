"""Headless smoke tests for the contact-force and dynamics examples."""

from __future__ import annotations

import runpy
from pathlib import Path

import torch


_EXAMPLES = Path(__file__).parents[2] / "examples"


def _load(filename: str) -> dict[str, object]:
    return runpy.run_path(str(_EXAMPLES / filename), run_name=f"example_{filename[:-3]}")


def test_contact_force_example_runs_headless() -> None:
    namespace = _load("07_contact_forces.py")
    result = namespace["run"](time_steps=1, max_iter=30)

    assert result.forces_world.shape == (1, 1, 3)
    assert result.fext_local.shape == (1, result.model.njoints, 6)
    assert result.generalized_force[..., :6].norm() < 1e-3


def test_dynamics_example_round_trips_gravity_compensation() -> None:
    namespace = _load("08_dynamics.py")
    torque, recovered_acceleration, error = namespace["run"]()

    assert torch.count_nonzero(torque) > 0
    torch.testing.assert_close(recovered_acceleration, torch.zeros_like(recovered_acceleration))
    assert error < 1e-12
