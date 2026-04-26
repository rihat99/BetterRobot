"""SE3 / SO3 op throughput micro-benches.

Skipped automatically when ``pytest-benchmark`` is not installed; the
results when present feed into the advisory bench-cpu CI job.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("pytest_benchmark")
from better_robot.lie import se3, so3

pytestmark = pytest.mark.bench


@pytest.mark.parametrize("size", [1, 1024])
def test_se3_compose(benchmark, size) -> None:
    a = torch.randn(size, 7); a[..., 3:7] /= a[..., 3:7].norm(dim=-1, keepdim=True)
    b = torch.randn(size, 7); b[..., 3:7] /= b[..., 3:7].norm(dim=-1, keepdim=True)
    benchmark(lambda: se3.compose(a, b))


@pytest.mark.parametrize("size", [1, 1024])
def test_so3_exp(benchmark, size) -> None:
    omega = torch.randn(size, 3) * 0.1
    benchmark(lambda: so3.exp(omega))
