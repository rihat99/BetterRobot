"""solve_ik throughput micro-bench (single Panda target)."""

from __future__ import annotations

import pytest

from better_robot.terms import solve_ik

pytestmark = pytest.mark.bench


def test_solve_ik_panda(benchmark, panda, panda_state) -> None:
    frame_id = (
        panda.frame_id("panda_hand")
        if "panda_hand" in panda.frame_names
        else panda.frame_id(panda.frame_names[-1])
    )
    target = panda_state.frame_poses[frame_id].clone()
    target[..., 0] += 0.02
    benchmark(lambda: solve_ik(panda, {panda.frame_names[-1]: target[None]}, q0=panda.q_neutral[None]))
