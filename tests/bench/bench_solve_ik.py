"""solve_ik throughput micro-bench (single Panda target)."""

from __future__ import annotations

import pytest

pytest.importorskip("pytest_benchmark")
from better_robot.terms import solve_ik

pytestmark = pytest.mark.bench


def test_solve_ik_panda(benchmark, panda, panda_state) -> None:
    frame_id = panda.frame_id("body_panda_hand") if "body_panda_hand" in panda.frame_name_to_id else panda.frame_id(panda.frame_names[-1])
    target = panda_state.frame_pose[frame_id].clone()
    target[..., 0] += 0.02
    benchmark(lambda: solve_ik(panda, {panda.frame_names[-1]: target[None]}, q0=panda.q_neutral[None]))
