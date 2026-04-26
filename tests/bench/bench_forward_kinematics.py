"""Forward-kinematics throughput micro-benches."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("pytest_benchmark")
import better_robot as br

pytestmark = pytest.mark.bench


def test_fk_panda_unbatched(benchmark, panda) -> None:
    q = panda.q_neutral.clone()
    benchmark(lambda: br.forward_kinematics(panda, q, compute_frames=True))


@pytest.mark.parametrize("batch", [16, 256])
def test_fk_panda_batched(benchmark, panda, batch) -> None:
    q = panda.q_neutral.expand(batch, -1).contiguous()
    benchmark(lambda: br.forward_kinematics(panda, q, compute_frames=True))
