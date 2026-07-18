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

For a cache-cold measurement, invoke one batch shape per fresh process and
provide new, empty cache directories::

    TORCHINDUCTOR_CACHE_DIR=/tmp/torch-case-b1 \
    WARP_CACHE_PATH=/tmp/warp-case-b1 \
    python benchmarks/m1_warp_fk_vs_compiled_torch.py \
        --label rtx6000-b1 --device cuda --batches 1 \
        --require-fresh-case-caches

Cold-start measurements include the first invocation after constructing each
lane, but not Python imports or model construction.  Persistent Torch/Warp
code caches may still make that invocation cache-warm; their configuration is
recorded in the report.  Steady-state CUDA measurements synchronize after
every invocation and therefore report end-to-end host-observed latency.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
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
except ImportError:  # pragma: no cover - command-line dependency guard
    wp = None


DEFAULT_BATCHES = (1, 64)
DEFAULT_WARMUP = 20
DEFAULT_SAMPLES = 100
_REPO_ROOT = Path(__file__).resolve().parents[1]

FKOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _percentiles(samples: list[float]) -> tuple[float, float]:
    if len(samples) == 1:
        return samples[0], samples[0]
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return quartiles[0], quartiles[2]


def _summary(samples_ms: list[float]) -> dict[str, Any]:
    q1_ms, q3_ms = _percentiles(samples_ms)
    return {
        "median_ms": statistics.median(samples_ms),
        "q1_ms": q1_ms,
        "q3_ms": q3_ms,
        "iqr_ms": q3_ms - q1_ms,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": list(samples_ms),
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


def _start_cuda_memory_measurement(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    _synchronize(device)
    baseline = {
        "baseline_allocated_bytes": torch.cuda.memory_allocated(device),
        "baseline_reserved_bytes": torch.cuda.memory_reserved(device),
    }
    torch.cuda.reset_peak_memory_stats(device)
    return baseline


def _finish_cuda_memory_measurement(
    device: torch.device,
    baseline: dict[str, int] | None,
) -> dict[str, Any] | None:
    if baseline is None:
        return None
    _synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        **baseline,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": max(
            0,
            peak_allocated - baseline["baseline_allocated_bytes"],
        ),
        "incremental_peak_reserved_bytes": max(
            0,
            peak_reserved - baseline["baseline_reserved_bytes"],
        ),
        "ending_allocated_bytes": torch.cuda.memory_allocated(device),
        "ending_reserved_bytes": torch.cuda.memory_reserved(device),
        "scope": (
            "PyTorch CUDA caching allocator; allocations owned directly by Warp "
            "may not be visible"
        ),
    }


def _time_once_with_memory(
    function: Callable[[], FKOutputs],
    device: torch.device,
) -> tuple[FKOutputs, float, dict[str, Any] | None]:
    baseline = _start_cuda_memory_measurement(device)
    output, elapsed_ms = _time_once(function, device)
    memory = _finish_cuda_memory_measurement(device, baseline)
    return output, elapsed_ms, memory


def _measure(
    function: Callable[[], FKOutputs],
    device: torch.device,
    warmup: int,
    samples: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        function()
    _synchronize(device)

    baseline = _start_cuda_memory_measurement(device)
    timings_ms: list[float] = []
    for _ in range(samples):
        output, elapsed_ms = _time_once(function, device)
        timings_ms.append(elapsed_ms)
        del output
    return {
        **_summary(timings_ms),
        "cuda_memory": _finish_cuda_memory_measurement(device, baseline),
    }


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


def _git_metadata() -> dict[str, str | bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=_REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit": commit, "dirty": dirty}


def _nvidia_smi_inventory() -> tuple[list[dict[str, Any]], str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return [], f"{type(error).__name__}: {error}"
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        return [], detail

    devices: list[dict[str, Any]] = []
    for fields in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
        if len(fields) != 5:
            continue
        index, uuid, pci_bus_id, name, driver_version = fields
        try:
            physical_index = int(index)
        except ValueError:
            continue
        devices.append(
            {
                "physical_index": physical_index,
                "uuid": uuid,
                "pci_bus_id": pci_bus_id,
                "name": name,
                "driver_version": driver_version,
            }
        )
    if not devices:
        return [], "nvidia-smi returned no parseable GPU rows"
    return devices, None


def _visible_device_token(logical_index: int) -> str | None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return None
    tokens = [token.strip() for token in visible_devices.split(",")]
    if logical_index >= len(tokens):
        return None
    return tokens[logical_index]


def _match_physical_device(
    *,
    logical_index: int,
    visible_token: str | None,
    device_uuid: str | None,
    inventory: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    if visible_token is not None and visible_token.isdecimal():
        physical_index = int(visible_token)
        return (
            next(
                (
                    candidate
                    for candidate in inventory
                    if candidate["physical_index"] == physical_index
                ),
                None,
            ),
            "numeric CUDA_VISIBLE_DEVICES token",
        )
    if visible_token is not None and visible_token.startswith("GPU-"):
        return (
            next(
                (
                    candidate
                    for candidate in inventory
                    if candidate["uuid"].startswith(visible_token)
                    or visible_token.startswith(candidate["uuid"])
                ),
                None,
            ),
            "UUID CUDA_VISIBLE_DEVICES token",
        )
    if device_uuid is not None:
        match = next(
            (
                candidate
                for candidate in inventory
                if candidate["uuid"] == device_uuid
            ),
            None,
        )
        if match is not None:
            return match, "CUDA device UUID"
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        return (
            next(
                (
                    candidate
                    for candidate in inventory
                    if candidate["physical_index"] == logical_index
                ),
                None,
            ),
            "default CUDA ordinal",
        )
    return None, "unresolved CUDA_VISIBLE_DEVICES mapping"


def _device_metadata(device: torch.device) -> dict[str, Any]:
    if device.type == "cpu":
        return {"type": "cpu", "name": _cpu_name()}
    logical_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    properties = torch.cuda.get_device_properties(device)
    inventory, inventory_error = _nvidia_smi_inventory()
    visible_token = _visible_device_token(logical_index)
    raw_uuid = getattr(properties, "uuid", None)
    device_uuid = str(raw_uuid) if raw_uuid is not None else None
    physical_device, mapping_source = _match_physical_device(
        logical_index=logical_index,
        visible_token=visible_token,
        device_uuid=device_uuid,
        inventory=inventory,
    )
    return {
        "type": "cuda",
        "index": device.index,
        "logical_index": logical_index,
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "torch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_device_token": visible_token,
        "physical_index": (
            physical_device["physical_index"] if physical_device is not None else None
        ),
        "uuid": (
            physical_device["uuid"] if physical_device is not None else device_uuid
        ),
        "torch_device_uuid": device_uuid,
        "pci_bus_id": (
            physical_device["pci_bus_id"] if physical_device is not None else None
        ),
        "physical_mapping_source": mapping_source,
        "nvidia_driver_version": (
            physical_device["driver_version"]
            if physical_device is not None
            else (inventory[0]["driver_version"] if inventory else None)
        ),
        "nvidia_smi_inventory_error": inventory_error,
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


def _environment_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off"
    )


def _validate_fresh_case_caches(
    *,
    required: bool,
    batches: tuple[int, ...],
) -> bool:
    if not required:
        return False
    if len(batches) != 1:
        raise ValueError(
            "--require-fresh-case-caches requires exactly one --batches value; "
            "run each shape in a new process"
        )

    cache_variables = ("TORCHINDUCTOR_CACHE_DIR", "WARP_CACHE_PATH")
    cache_paths: dict[str, Path] = {}
    for variable in cache_variables:
        raw_path = os.environ.get(variable)
        if not raw_path:
            raise ValueError(
                f"--require-fresh-case-caches requires {variable} to be set"
            )
        cache_path = Path(raw_path).expanduser().resolve()
        if cache_path.exists() and not cache_path.is_dir():
            raise ValueError(f"{variable} is not a directory: {cache_path}")
        if cache_path.exists() and next(cache_path.iterdir(), None) is not None:
            raise ValueError(f"{variable} must be absent or empty: {cache_path}")
        cache_paths[variable] = cache_path

    if len(set(cache_paths.values())) != len(cache_paths):
        raise ValueError("TorchInductor and Warp cache directories must be distinct")
    return True


def _cold_timing_metadata(
    *,
    batches: tuple[int, ...],
    fresh_case_caches_verified: bool,
) -> dict[str, Any]:
    if fresh_case_caches_verified:
        cache_scope = "fresh-process_single-case_empty-persistent-caches"
        limitation = (
            "Cache directories were absent or empty before the single case; timing "
            "still excludes Python imports and model construction."
        )
    else:
        cache_scope = "first-call_in-process_cache-state-not-guaranteed-fresh"
        limitation = (
            "Persistent and in-process compiler/kernel caches may be warm; with "
            "multiple batch shapes, dynamic-shape cache reuse is also possible."
        )
    return {
        "definition": "first invocation after constructing each lane",
        "cache_scope": cache_scope,
        "verified_cache_cold": fresh_case_caches_verified,
        "single_case_process": len(batches) == 1,
        "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "warp_cache_path": os.environ.get("WARP_CACHE_PATH"),
        "limitation": limitation,
    }


def run(
    *,
    label: str,
    device: torch.device,
    dtype: torch.dtype,
    batches: tuple[int, ...],
    warmup: int,
    samples: int,
    compile_backend: str,
    fresh_case_caches_verified: bool = False,
) -> dict[str, Any]:
    if wp is None:
        raise SystemExit("install the 'warp' extra to run this benchmark")
    if fresh_case_caches_verified:
        _validate_fresh_case_caches(required=True, batches=batches)

    # Importing the bridge calls wp.init(), so keep it after the strict cache
    # preflight. This makes absent/empty WARP_CACHE_PATH a measured invariant,
    # rather than rejecting scaffolding that Warp itself creates at import time.
    try:
        from better_robot.kinematics._warp_bridge import (  # noqa: PLC0415
            try_warp_forward_kinematics,
        )
    except ImportError as error:  # pragma: no cover - command-line dependency guard
        raise SystemExit("install the 'warp' extra to run this benchmark") from error

    model = make_smpl_like_model(device=device, dtype=dtype)
    measurements: list[dict[str, Any]] = []
    cold_timing = _cold_timing_metadata(
        batches=batches,
        fresh_case_caches_verified=fresh_case_caches_verified,
    )

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

            torch_output, torch_cold_ms, torch_cold_memory = _time_once_with_memory(
                compiled_call,
                device,
            )
            warp_output, warp_cold_ms, warp_cold_memory = _time_once_with_memory(
                warp_call,
                device,
            )
            max_abs_error = _assert_parity(warp_output, torch_output, dtype)

            torch_steady = _measure(compiled_call, device, warmup, samples)
            warp_steady = _measure(warp_call, device, warmup, samples)
            measurements.append(
                {
                    "batch_shape": [batch],
                    "max_abs_error": max_abs_error,
                    "torch_compile_fullgraph": {
                        "cold_first_call_ms": torch_cold_ms,
                        "cold_first_call_cache_scope": cold_timing["cache_scope"],
                        "cold_first_call_verified_cache_cold": cold_timing[
                            "verified_cache_cold"
                        ],
                        "cold_cuda_memory": torch_cold_memory,
                        **torch_steady,
                    },
                    "warp_fused_fk": {
                        "cold_first_call_ms": warp_cold_ms,
                        "cold_first_call_cache_scope": cold_timing["cache_scope"],
                        "cold_first_call_verified_cache_cold": cold_timing[
                            "verified_cache_cold"
                        ],
                        "cold_cuda_memory": warp_cold_memory,
                        **warp_steady,
                    },
                    "steady_state_speedup_warp_over_compiled_torch": (
                        torch_steady["median_ms"] / warp_steady["median_ms"]
                    ),
                }
            )

    return {
        "schema_version": 2,
        "benchmark": "m1_warp_fk_vs_compiled_torch",
        "label": label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(Path(br.__file__).resolve()),
        "git": _git_metadata(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "hostname": platform.node(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
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
        "statistic": (
            "median with inclusive Q1/Q3 and IQR over raw host wall-clock samples"
        ),
        "timing_scope": (
            "first-call cold timings exclude imports/model construction; "
            "CUDA steady-state samples include completion synchronization"
        ),
        "cold_timing": cold_timing,
        "measurements": measurements,
    }


def _parse_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    try:
        require_fresh_case_caches = _environment_flag(
            "BETTERROBOT_BENCH_REQUIRE_FRESH_CASE_CACHES"
        )
    except ValueError as error:
        parser.error(str(error))
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--require-fresh-case-caches",
        action="store_true",
        default=require_fresh_case_caches,
        help=(
            "require one batch shape plus absent/empty TORCHINDUCTOR_CACHE_DIR "
            "and WARP_CACHE_PATH (also enabled by "
            "BETTERROBOT_BENCH_REQUIRE_FRESH_CASE_CACHES=1)"
        ),
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but torch.cuda.is_available() is false")
    if any(batch < 1 for batch in args.batches):
        parser.error("every batch size must be positive")
    if args.warmup < 0 or args.samples < 1:
        parser.error("--warmup must be non-negative and --samples must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        fresh_case_caches_verified = _validate_fresh_case_caches(
            required=args.require_fresh_case_caches,
            batches=tuple(args.batches),
        )
    except ValueError as error:
        parser.error(str(error))

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
        fresh_case_caches_verified=fresh_case_caches_verified,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
