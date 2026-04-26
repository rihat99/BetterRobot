"""Nightly-only: track peak memory of a Panda IK solve.

Flagged with the ``slow`` marker so day-to-day CI doesn't run it. The
nightly bench job pulls these numbers into a memory dashboard.
"""

from __future__ import annotations

import resource

import pytest

import better_robot as br

pytestmark = [pytest.mark.bench, pytest.mark.slow]


def test_solve_ik_peak_memory(panda, panda_data) -> None:
    target = panda_data.frame_pose_world[panda.frame_id(panda.frame_names[-1])].clone()
    target[..., 0] += 0.02
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _ = br.solve_ik(panda, {panda.frame_names[-1]: target})
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"\nsolve_ik peak RSS delta: {(after - before) / 1024:.1f} MiB")
