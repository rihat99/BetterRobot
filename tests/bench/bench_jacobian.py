"""Jacobian throughput micro-benches."""

from __future__ import annotations

import pytest

pytest.importorskip("pytest_benchmark")
import better_robot as br

pytestmark = pytest.mark.bench


def test_joint_jacobian_panda(benchmark, panda, panda_state) -> None:
    joint_id = panda.njoints - 1
    benchmark(lambda: br.joint_jacobian(panda, panda_state, joint=joint_id, reference="world"))
