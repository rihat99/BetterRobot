"""Reproducible eager-CPU FK/RNEA benchmark for the M1 seam.

Run this file against both the pre-M1 source tree and the M1 source tree on
the same host.  It deliberately uses only public APIs shared by both trees.
The report records the source path, hardware, torch version, dtype, robot,
batch shapes, warmup count, sample count, median, and interquartile range.

Example::

    PYTHONPATH=/path/to/source/src python benchmarks/m1_eager_cpu.py \
        --label pre-m1 --output /tmp/pre-m1.json
    python benchmarks/m1_eager_cpu.py \
        --label m1 --output /tmp/m1.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

import better_robot as br


DEFAULT_BATCHES = (1, 64)
DEFAULT_WARMUP = 20
DEFAULT_SAMPLES = 100


def _percentiles(samples: list[float]) -> tuple[float, float]:
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return quartiles[0], quartiles[2]


def _measure(function: Callable[[], object], warmup: int, samples: int) -> dict[str, float]:
    for _ in range(warmup):
        function()
    timings_ms: list[float] = []
    for _ in range(samples):
        start_ns = time.perf_counter_ns()
        function()
        timings_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000.0)
    q1_ms, q3_ms = _percentiles(timings_ms)
    return {
        "median_ms": statistics.median(timings_ms),
        "q1_ms": q1_ms,
        "q3_ms": q3_ms,
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
    }


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def run(label: str, batches: tuple[int, ...], warmup: int, samples: int) -> dict[str, object]:
    try:
        from robot_descriptions import panda_description  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - command-line guard
        raise SystemExit("install the 'demos' extra to run the Panda M1 benchmark") from exc

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dtype = torch.float32
    model = br.load(panda_description.URDF_PATH, dtype=dtype)
    measurements: list[dict[str, object]] = []

    with torch.inference_mode():
        for batch in batches:
            q = model.q_neutral.unsqueeze(0).expand(batch, -1).contiguous()
            v = torch.zeros((batch, model.nv), dtype=dtype)
            a = torch.zeros_like(v)
            data = model.create_data(batch_shape=(batch,), device="cpu", dtype=dtype)

            cases: tuple[tuple[str, Callable[[], object]], ...] = (
                ("forward_kinematics", lambda: br.forward_kinematics(model, q, compute_frames=True)),
                ("rnea", lambda: br.rnea(model, q, v, a, data=data)),
            )
            for operation, function in cases:
                measurements.append(
                    {
                        "operation": operation,
                        "batch_shape": [batch],
                        **_measure(function, warmup, samples),
                    }
                )

    return {
        "schema_version": 1,
        "label": label,
        "source": str(Path(br.__file__).resolve()),
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "dtype": str(dtype),
        "robot": "Franka Panda (robot_descriptions)",
        "batch_shapes": [[batch] for batch in batches],
        "warmup_iterations": warmup,
        "samples": samples,
        "statistic": "median with inclusive Q1/Q3 over wall-clock samples",
        "measurements": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    args = parser.parse_args()
    report = run(args.label, tuple(args.batches), args.warmup, args.samples)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
