"""Benchmark the opt-in fused Warp FK against full-graph compiled Torch FK.

This is the M1 standing-rule benchmark for the prototype Warp lane.  The two
lanes compute the same joint and frame poses, and parity is checked before
steady-state timings are accepted.  CPU is the default so the definition runs
on an ordinary Warp installation; pass ``--device cuda`` for an explicit GPU
run.

Examples::

    python benchmarks/m1_warp_fk_vs_compiled_torch.py \
        --label local-cpu --output /tmp/m1-warp-fk-cpu.json
    python benchmarks/m1_warp_fk_vs_compiled_torch.py \
        --label rtx6000 --device cuda --dtype float32 \
        --output /tmp/m1-warp-fk-cuda.json

Cold-start measurements include the first invocation after constructing each
lane, but not Python imports or model construction.  Persistent Torch/Warp
code caches may still make that invocation cache-warm; their configuration is
recorded in the report.  Steady-state CUDA measurements synchronize after
every invocation and therefore report end-to-end host-observed latency.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

import better_robot as br
from better_robot.data_model.model_structure import ModelStructure
from better_robot.data_model.model_values import ModelValues
from better_robot.io.builders.smpl_like import make_smpl_like_model
from better_robot.kinematics.forward import (
    forward_kinematics_raw,
    frame_placements_raw,
)

try:
    import warp as wp
    from better_robot.kinematics._warp_bridge import try_warp_forward_kinematics
except ImportError:  # pragma: no cover - command-line dependency guard
    wp = None
    try_warp_forward_kinematics = None


DEFAULT_BATCHES = (1, 64)
DEFAULT_WARMUP = 20
DEFAULT_SAMPLES = 100

FKOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _percentiles(samples: list[float]) -> tuple[float, float]:
    if len(samples) == 1:
        return samples[0], samples[0]
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return quartiles[0], quartiles[2]


def _summary(samples_ms: list[float]) -> dict[str, float]:
    q1_ms, q3_ms = _percentiles(samples_ms)
    return {
        "median_ms": statistics.median(samples_ms),
        "q1_ms": q1_ms,
        "q3_ms": q3_ms,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_once(function: Callable[[], FKOutputs], device: torch.device) -> tuple[FKOutputs, float]:
    _synchronize(device)
    start_ns = time.perf_counter_ns()
    output = function()
    _synchronize(device)
    return output, (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _measure(
    function: Callable[[], FKOutputs],
    device: torch.device,
    warmup: int,
    samples: int,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    _synchronize(device)

    timings_ms: list[float] = []
    for _ in range(samples):
        _, elapsed_ms = _time_once(function, device)
        timings_ms.append(elapsed_ms)
    return _summary(timings_ms)


def _torch_lane(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
) -> FKOutputs:
    world, local = forward_kinematics_raw(structure, values, q)
    frames = frame_placements_raw(structure, values, world)
    return world, local, frames


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _device_metadata(device: torch.device) -> dict[str, Any]:
    if device.type == "cpu":
        return {"type": "cpu", "name": _cpu_name()}
    properties = torch.cuda.get_device_properties(device)
    return {
        "type": "cuda",
        "index": device.index,
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "torch_cuda_version": torch.version.cuda,
    }


def _max_abs_error(actual: FKOutputs, expected: FKOutputs) -> float:
    return max(
        float((actual_tensor - expected_tensor).abs().max().detach().cpu())
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True)
    )


def _assert_parity(actual: FKOutputs, expected: FKOutputs, dtype: torch.dtype) -> float:
    if dtype == torch.float32:
        rtol, atol = 2e-5, 2e-6
    else:
        rtol, atol = 2e-9, 2e-10
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=rtol, atol=atol)
    return _max_abs_error(actual, expected)


def _warp_version() -> str:
    try:
        return metadata.version("warp-lang")
    except metadata.PackageNotFoundError:
        return "unknown"


def run(
    *,
    label: str,
    device: torch.device,
    dtype: torch.dtype,
    batches: tuple[int, ...],
    warmup: int,
    samples: int,
    compile_backend: str,
) -> dict[str, Any]:
    if wp is None or try_warp_forward_kinematics is None:
        raise SystemExit("install the 'warp' extra to run this benchmark")

    model = make_smpl_like_model(device=device, dtype=dtype)
    measurements: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch in batches:
            q = model.q_neutral.unsqueeze(0).expand(batch, -1).contiguous()

            compiled_torch = torch.compile(
                _torch_lane,
                fullgraph=True,
                backend=compile_backend,
            )

            def compiled_call() -> FKOutputs:
                return compiled_torch(model.structure, model.values, q)

            def warp_call() -> FKOutputs:
                result = try_warp_forward_kinematics(model.structure, model.values, q)
                if result is None:
                    raise RuntimeError(
                        "supported benchmark input fell back to Torch; "
                        "the Warp lane was not measured"
                    )
                return result.world, result.local, result.frames

            torch_output, torch_cold_ms = _time_once(compiled_call, device)
            warp_output, warp_cold_ms = _time_once(warp_call, device)
            max_abs_error = _assert_parity(warp_output, torch_output, dtype)

            torch_steady = _measure(compiled_call, device, warmup, samples)
            warp_steady = _measure(warp_call, device, warmup, samples)
            measurements.append(
                {
                    "batch_shape": [batch],
                    "max_abs_error": max_abs_error,
                    "torch_compile_fullgraph": {
                        "cold_first_call_ms": torch_cold_ms,
                        **torch_steady,
                    },
                    "warp_fused_fk": {
                        "cold_first_call_ms": warp_cold_ms,
                        **warp_steady,
                    },
                    "steady_state_speedup_warp_over_compiled_torch": (
                        torch_steady["median_ms"] / warp_steady["median_ms"]
                    ),
                }
            )

    return {
        "schema_version": 1,
        "benchmark": "m1_warp_fk_vs_compiled_torch",
        "label": label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(Path(br.__file__).resolve()),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "warp_version": _warp_version(),
        "device": _device_metadata(device),
        "dtype": str(dtype),
        "model": {
            "name": "SMPL-like 24-joint tree",
            "nq": model.nq,
            "nv": model.nv,
            "njoints": model.njoints,
            "nframes": model.nframes,
        },
        "batch_shapes": [[batch] for batch in batches],
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "compile": {"fullgraph": True, "backend": compile_backend},
        "warp_cache_path": os.environ.get(
            "WARP_CACHE_PATH",
            str(getattr(wp.config, "kernel_cache_dir", "unknown")),
        ),
        "warmup_iterations": warmup,
        "samples": samples,
        "statistic": "median with inclusive Q1/Q3 over host wall-clock samples",
        "timing_scope": (
            "first-call cold timings exclude imports/model construction; "
            "CUDA steady-state samples include completion synchronization"
        ),
        "measurements": measurements,
    }


def _parse_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but torch.cuda.is_available() is false")
    if any(batch < 1 for batch in args.batches):
        parser.error("every batch size must be positive")
    if args.warmup < 0 or args.samples < 1:
        parser.error("--warmup must be non-negative and --samples must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    device = torch.device(args.device)
    report = run(
        label=args.label,
        device=device,
        dtype=_parse_dtype(args.dtype),
        batches=tuple(args.batches),
        warmup=args.warmup,
        samples=args.samples,
        compile_backend=args.compile_backend,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
