"""Jacobian throughput micro-benches."""

from __future__ import annotations

import pytest

pytest.importorskip("pytest_benchmark")
import better_robot as br

pytestmark = pytest.mark.bench


def test_compute_joint_jacobians_panda(benchmark, panda, panda_data) -> None:
    benchmark(lambda: br.compute_joint_jacobians(panda, panda_data))
