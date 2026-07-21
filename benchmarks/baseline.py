"""Definition-first Torch baseline harness.

The canonical matrix covers fixed-base Panda, free-flyer Panda, and the
SMPL-like 25-joint tree; FK, RNEA, and public fixed-budget IK; CPU and CUDA;
eager and compiled lanes; and ``B in {1, 16, 256, 4096}``.

Canonical runs are intentionally expensive.  Select one or more cases with
the matrix filters or an exact ``--case`` identifier while developing::

    python benchmarks/baseline.py --label cpu-smoke --smoke \
        --models smpl --operations fk --devices cpu --lanes eager
    python benchmarks/baseline.py --label one-case \
        --case panda_fixed/rnea/cuda/compiled/b16 --output /tmp/baseline.json

Every selected combination produces a result row.  Public ``solve_ik`` is a
Python problem-construction and host-controlled solve facade, so its compiled
rows are explicitly ``UNSUPPORTED``; the harness never substitutes a lower
level update or a different workload.  Successful eager IK rows assert that
every element consumed the fixed iteration budget.

Each selected case runs in a fresh child process.  This makes per-case timeout
reporting reliable and prevents process-local Dynamo graphs from turning a
later shape's first call into a warm call.  Each child also receives a fresh
TorchInductor cache directory.  The report still labels the compiled first
invocation honestly: it contains lazy compilation *and* execution, which
Torch does not expose as separable wall-clock intervals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch

import better_robot as br
from better_robot.data_model.model import Model
from better_robot.data_model.model_structure import ModelStructure
from better_robot.data_model.model_values import ModelValues
from better_robot.dynamics.rnea import rnea_raw
from better_robot.kinematics.forward import forward_kinematics_raw, frame_placements_raw
from better_robot.tasks import IKCostConfig, OptimizerConfig, solve_ik


SCHEMA_VERSION = 1
SEED = 20260718
CANONICAL_BATCHES = (1, 16, 256, 4096)
MODEL_NAMES = ("panda_fixed", "panda_free", "smpl")
OPERATIONS = ("fk", "rnea", "ik")
DEVICES = ("cpu", "cuda")
LANES = ("eager", "compiled")
DEFAULT_WARMUP = 20
DEFAULT_SAMPLES = 100
DEFAULT_IK_ITERATIONS = 5
DEFAULT_CASE_TIMEOUT_SECONDS = 900.0
CANONICAL_DTYPE = "float32"
CANONICAL_COMPILE_BACKEND = "inductor"
CANONICAL_THREADS = 1
CANONICAL_CUDA_INDEX = 0
CANONICAL_CUDA_VISIBLE_DEVICES = "4"

ModelName = Literal["panda_fixed", "panda_free", "smpl"]
Operation = Literal["fk", "rnea", "ik"]
DeviceName = Literal["cpu", "cuda"]
Lane = Literal["eager", "compiled"]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = Path(__file__).resolve()


@dataclass(frozen=True)
class CaseSpec:
    """One exact selector in the canonical matrix."""

    model: ModelName
    operation: Operation
    device: DeviceName
    lane: Lane
    batch: int

    @property
    def case_id(self) -> str:
        return f"{self.model}/{self.operation}/{self.device}/{self.lane}/b{self.batch}"

    @property
    def ik_mode(self) -> str | None:
        if self.operation != "ik":
            return None
        return "single_unbatched" if self.batch == 1 else "independently_batched"


@dataclass(frozen=True)
class Workload:
    """Prepared callable and immutable invocation arguments."""

    function: Callable[..., object]
    args: tuple[object, ...]
    model: Model
    details: dict[str, object]
    fixed_iteration_budget: int | None = None


def _fk_workload(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fk = forward_kinematics_raw(structure, values, q)
    frames = frame_placements_raw(structure, values, fk.joint_pose_world)
    return fk.joint_pose_world, fk.joint_pose_local, frames.frame_pose_world


def _rnea_workload(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    return rnea_raw(structure, values, q, v, a).tau


def _public_ik_workload(
    model: Model,
    frame_name: str,
    target: torch.Tensor,
    initial_q: torch.Tensor,
    cost_config: IKCostConfig,
    optimizer_config: OptimizerConfig,
) -> object:
    return solve_ik(
        model,
        {frame_name: target},
        initial_q=initial_q,
        cost_cfg=cost_config,
        optimizer_cfg=optimizer_config,
    )


def _all_cases() -> tuple[CaseSpec, ...]:
    return tuple(
        CaseSpec(model, operation, device, lane, batch)
        for model in MODEL_NAMES
        for operation in OPERATIONS
        for device in DEVICES
        for lane in LANES
        for batch in CANONICAL_BATCHES
    )


def _parse_case_id(case_id: str) -> CaseSpec:
    fields = case_id.split("/")
    if len(fields) != 5 or not fields[4].startswith("b"):
        raise ValueError(
            f"invalid case id {case_id!r}; expected model/operation/device/lane/batch, for example smpl/fk/cpu/eager/b1"
        )
    try:
        batch = int(fields[4][1:])
    except ValueError as exc:
        raise ValueError(f"invalid batch component in case id {case_id!r}") from exc
    candidate = CaseSpec(  # type: ignore[arg-type]
        model=fields[0],
        operation=fields[1],
        device=fields[2],
        lane=fields[3],
        batch=batch,
    )
    known = {case.case_id for case in _all_cases()}
    if candidate.case_id not in known:
        raise ValueError(f"unknown canonical case id {case_id!r}")
    return candidate


def _select_cases(args: argparse.Namespace) -> tuple[CaseSpec, ...]:
    all_cases = _all_cases()
    if args.case:
        requested = tuple(dict.fromkeys(args.case))
        known = {case.case_id: case for case in all_cases}
        unknown = [case_id for case_id in requested if case_id not in known]
        if unknown:
            raise ValueError(f"unknown --case selector(s): {unknown}")
        return tuple(known[case_id] for case_id in requested)
    batches = (1,) if args.smoke else tuple(args.batches)
    return tuple(
        case
        for case in all_cases
        if case.model in args.models
        and case.operation in args.operations
        and case.device in args.devices
        and case.lane in args.lanes
        and case.batch in batches
    )


def _dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    try:
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _git_metadata() -> dict[str, object]:
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


def _nvidia_smi_inventory() -> tuple[list[dict[str, object]], str | None]:
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

    devices: list[dict[str, object]] = []
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


def _physical_cuda_device(
    cuda_index: int,
    *,
    device_uuid: str | None,
    inventory: list[dict[str, object]],
) -> tuple[dict[str, object] | None, str | None, str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = [] if visible_devices is None else [token.strip() for token in visible_devices.split(",")]
    visible_token = tokens[cuda_index] if cuda_index < len(tokens) else None
    if visible_token is not None and visible_token.isdecimal():
        physical_index = int(visible_token)
        return (
            next(
                (candidate for candidate in inventory if candidate["physical_index"] == physical_index),
                None,
            ),
            visible_token,
            "numeric CUDA_VISIBLE_DEVICES token",
        )
    if visible_token is not None and visible_token.startswith("GPU-"):
        return (
            next(
                (
                    candidate
                    for candidate in inventory
                    if str(candidate["uuid"]).startswith(visible_token)
                    or visible_token.startswith(str(candidate["uuid"]))
                ),
                None,
            ),
            visible_token,
            "UUID CUDA_VISIBLE_DEVICES token",
        )
    if device_uuid is not None:
        match = next(
            (candidate for candidate in inventory if candidate["uuid"] == device_uuid),
            None,
        )
        if match is not None:
            return match, visible_token, "CUDA device UUID"
    if visible_devices is None:
        match = next(
            (candidate for candidate in inventory if candidate["physical_index"] == cuda_index),
            None,
        )
        return match, visible_token, "default CUDA ordinal"
    return None, visible_token, "unresolved CUDA_VISIBLE_DEVICES mapping"


def _cuda_metadata(cuda_index: int) -> dict[str, object]:
    if not torch.cuda.is_available() or cuda_index >= torch.cuda.device_count():
        return {
            "available": False,
            "requested_index": cuda_index,
            "device_count": torch.cuda.device_count(),
        }
    device = torch.device("cuda", cuda_index)
    properties = torch.cuda.get_device_properties(device)
    inventory, inventory_error = _nvidia_smi_inventory()
    raw_uuid = getattr(properties, "uuid", None)
    device_uuid = str(raw_uuid) if raw_uuid is not None else None
    physical_device, visible_token, mapping_source = _physical_cuda_device(
        cuda_index,
        device_uuid=device_uuid,
        inventory=inventory,
    )
    return {
        "available": True,
        "requested_index": cuda_index,
        "logical_index": cuda_index,
        "device_count": torch.cuda.device_count(),
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "torch_cuda_build": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_device_token": visible_token,
        "physical_index": physical_device["physical_index"] if physical_device else None,
        "uuid": physical_device["uuid"] if physical_device else device_uuid,
        "torch_device_uuid": device_uuid,
        "pci_bus_id": physical_device["pci_bus_id"] if physical_device else None,
        "physical_mapping_source": mapping_source,
        "nvidia_driver_version": (
            physical_device["driver_version"]
            if physical_device is not None
            else (inventory[0]["driver_version"] if inventory else None)
        ),
        "nvidia_smi_inventory_error": inventory_error,
    }


def _environment_metadata(cuda_index: int, *, probe_cuda: bool) -> dict[str, object]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "better_robot_source": str(Path(br.__file__).resolve()),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": affinity,
        "cuda": _cuda_metadata(cuda_index) if probe_cuda else {"probed": False},
        "git": _git_metadata(),
    }


def _case_seed(base_seed: int, case: CaseSpec) -> int:
    # Lane and device deliberately do not participate: every comparison row
    # receives byte-identical persisted inputs.
    workload_id = f"{case.model}/{case.operation}/b{case.batch}"
    digest = hashlib.sha256(workload_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "little")
    return (base_seed + offset) % (2**63 - 1)


def _load_model(name: ModelName, device: torch.device, dtype: torch.dtype) -> Model:
    if name == "smpl":
        _repo_root = Path(__file__).resolve().parents[1]
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))
        from tests.support.branching_tree import make_branching_tree_model  # noqa: PLC0415

        return make_branching_tree_model(device=device, dtype=dtype)
    try:
        from robot_descriptions import panda_description  # noqa: PLC0415
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Panda benchmark cases require the optional 'demos' dependency robot_descriptions"
        ) from exc
    return br.load(
        panda_description.URDF_PATH,
        free_flyer=name == "panda_free",
        device=device,
        dtype=dtype,
    )


def _feasible_neutral(model: Model) -> torch.Tensor:
    return torch.maximum(
        torch.minimum(model.q_neutral.clone(), model.upper_pos_limit),
        model.lower_pos_limit,
    )


def _cpu_random(
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randn(tuple(shape), dtype=dtype, generator=generator, device="cpu")


def _tensor_fingerprint(tensor: torch.Tensor) -> dict[str, object]:
    canonical = tensor.detach().to(device="cpu").contiguous()
    payload = canonical.numpy().tobytes(order="C")
    return {
        "shape": list(canonical.shape),
        "dtype": str(canonical.dtype),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "float64_sum_checksum": float(canonical.to(dtype=torch.float64).sum()),
    }


def _input_fingerprints(**inputs: torch.Tensor) -> dict[str, dict[str, object]]:
    return {name: _tensor_fingerprint(tensor) for name, tensor in inputs.items()}


def _configuration_batch(
    model: Model,
    batch: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    neutral = _feasible_neutral(model).expand(batch, -1)
    tangent = _cpu_random(
        (batch, model.nv),
        dtype=neutral.dtype,
        generator=generator,
    ).to(neutral.device)
    return model.integrate(neutral, tangent * 0.05).contiguous()


def _ik_frame(model: Model, name: ModelName) -> str:
    candidates = (
        ("body_panda_hand", "body_panda_link8")
        if name != "smpl"
        else ("body_right_hand", "body_left_hand", "body_head")
    )
    return next((candidate for candidate in candidates if candidate in model.frame_name_to_id), model.frame_names[-1])


def _prepare_workload(
    case: CaseSpec,
    *,
    dtype: torch.dtype,
    cuda_index: int,
    seed: int,
    ik_iterations: int,
) -> Workload:
    device = torch.device("cpu") if case.device == "cpu" else torch.device("cuda", cuda_index)
    # Construct every fixture on CPU, fingerprint it, then transfer it.  This
    # prevents CPU/CUDA math in fixture generation from changing the workload.
    host_model = _load_model(case.model, torch.device("cpu"), dtype)
    generator = torch.Generator(device="cpu")
    input_seed = _case_seed(seed, case)
    generator.manual_seed(input_seed)

    if case.operation == "fk":
        q_host = _configuration_batch(host_model, case.batch, generator=generator)
        model = host_model.to(device=device)
        q = q_host.to(device=device)
        return Workload(
            function=_fk_workload,
            args=(model.structure, model.values, q),
            model=model,
            details={
                "input_shapes": {"q": list(q.shape)},
                "input_seed": input_seed,
                "inputs": _input_fingerprints(q=q_host),
                "compute_frames": True,
                "api": "forward_kinematics_raw + frame_placements_raw",
                "autograd_mode": "torch.inference_mode",
            },
        )

    if case.operation == "rnea":
        q_host = _configuration_batch(host_model, case.batch, generator=generator)
        v_host = _cpu_random((case.batch, host_model.nv), dtype=dtype, generator=generator) * 0.1
        a_host = _cpu_random((case.batch, host_model.nv), dtype=dtype, generator=generator) * 0.1
        model = host_model.to(device=device)
        q = q_host.to(device=device)
        v = v_host.to(device=device)
        a = a_host.to(device=device)
        return Workload(
            function=_rnea_workload,
            args=(model.structure, model.values, q, v, a),
            model=model,
            details={
                "input_shapes": {"q": list(q.shape), "v": list(v.shape), "a": list(a.shape)},
                "input_seed": input_seed,
                "inputs": _input_fingerprints(q=q_host, v=v_host, a=a_host),
                "api": "rnea_raw(...).tau",
                "autograd_mode": "torch.inference_mode",
            },
        )

    initial_host = _feasible_neutral(host_model)
    if case.batch > 1:
        initial_host = initial_host.expand(case.batch, -1).clone()
    target_delta_host = torch.linspace(
        -0.15,
        0.15,
        host_model.nv,
        device="cpu",
        dtype=dtype,
    )
    if case.batch > 1:
        target_delta_host = target_delta_host.expand(case.batch, -1).clone()
    target_q_host = host_model.integrate(initial_host, target_delta_host)
    frame_name = _ik_frame(host_model, case.model)
    target_data = br.forward_kinematics(host_model, target_q_host, compute_frames=True)
    target_host = target_data.frame_pose_world[..., host_model.frame_id(frame_name), :].clone()
    model = host_model.to(device=device)
    initial = initial_host.to(device=device)
    target = target_host.to(device=device)
    optimizer_config = OptimizerConfig(max_iter=ik_iterations, tol=0.0)
    return Workload(
        function=_public_ik_workload,
        args=(model, frame_name, target, initial, IKCostConfig(), optimizer_config),
        model=model,
        details={
            "input_shapes": {"initial_q": list(initial.shape), "target": list(target.shape)},
            "input_seed": input_seed,
            "inputs": _input_fingerprints(initial_q=initial_host, target=target_host),
            "api": "solve_ik",
            "autograd_mode": "normal grad mode (public facade default)",
            "frame": frame_name,
            "mode": case.ik_mode,
            "target_definition": "FK(integrate(feasible_neutral, linspace(-0.15, 0.15, nv)))",
            "optimizer": "LevenbergMarquardt",
            "iteration_budget": ik_iterations,
            "tol": 0.0,
        },
        fixed_iteration_budget=ik_iterations,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_once(function: Callable[[], object], device: torch.device) -> tuple[object, float]:
    _synchronize(device)
    start_ns = time.perf_counter_ns()
    output = function()
    _synchronize(device)
    return output, (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _quartiles(samples: list[float]) -> tuple[float, float]:
    if len(samples) == 1:
        return samples[0], samples[0]
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return quartiles[0], quartiles[2]


def _timing_summary(samples: list[float]) -> dict[str, object]:
    q1_ms, q3_ms = _quartiles(samples)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "q1_ms": q1_ms,
        "q3_ms": q3_ms,
        "iqr_ms": q3_ms - q1_ms,
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _iteration_tensor(output: object) -> torch.Tensor | None:
    iterations = getattr(output, "iters", None)
    if iterations is None:
        return None
    if isinstance(iterations, torch.Tensor):
        return iterations
    return torch.tensor(iterations, dtype=torch.int64)


def _assert_fixed_iterations(output: object, budget: int) -> None:
    iterations = _iteration_tensor(output)
    if iterations is None:
        raise RuntimeError("public IK result did not expose iteration diagnostics")
    if not bool((iterations == budget).all()):
        minimum = int(iterations.min().detach().cpu())
        maximum = int(iterations.max().detach().cpu())
        raise RuntimeError(
            "fixed-budget IK case terminated early; refusing to record a variable-work timing "
            f"(budget={budget}, observed range=[{minimum}, {maximum}])"
        )


def _output_tensors(output: object) -> tuple[torch.Tensor, ...]:
    if isinstance(output, torch.Tensor):
        return (output,)
    if isinstance(output, tuple):
        tensors: list[torch.Tensor] = []
        for item in output:
            tensors.extend(_output_tensors(item))
        return tuple(tensors)
    q = getattr(output, "q", None)
    residual = getattr(output, "residual", None)
    return tuple(value for value in (q, residual) if isinstance(value, torch.Tensor))


def _output_summary(output: object) -> dict[str, object]:
    tensors = _output_tensors(output)
    if not tensors:
        raise RuntimeError(f"workload returned no tensor outputs ({type(output).__name__})")
    finite = all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
    checksum = sum(float(tensor.detach().to(dtype=torch.float64).sum().cpu()) for tensor in tensors)
    result: dict[str, object] = {
        "tensor_shapes": [list(tensor.shape) for tensor in tensors],
        "all_finite": finite,
        "float64_sum_checksum": checksum,
        "tensor_fingerprints": [_tensor_fingerprint(tensor) for tensor in tensors],
    }
    iterations = _iteration_tensor(output)
    converged = getattr(output, "converged", None)
    if iterations is not None:
        flat = iterations.detach().reshape(-1).cpu()
        result["iterations"] = {
            "minimum": int(flat.min()),
            "maximum": int(flat.max()),
            "unique": [int(value) for value in torch.unique(flat)],
        }
    if isinstance(converged, torch.Tensor):
        result["converged_elements"] = int(converged.detach().sum().cpu())
        result["total_elements"] = converged.numel()
    elif isinstance(converged, bool):
        result["converged_elements"] = int(converged)
        result["total_elements"] = 1
    return result


def _memory_start(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    _synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
    }


def _memory_finish(device: torch.device, start: dict[str, int] | None) -> dict[str, int] | None:
    if device.type != "cuda" or start is None:
        return None
    _synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "start_allocated_bytes": start["allocated_bytes"],
        "start_reserved_bytes": start["reserved_bytes"],
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": max(0, peak_allocated - start["allocated_bytes"]),
        "incremental_peak_reserved_bytes": max(0, peak_reserved - start["reserved_bytes"]),
    }


def _measure_case(  # noqa: PLR0915 - benchmark phases stay explicit in the artifact
    case: CaseSpec,
    workload: Workload,
    *,
    warmup: int,
    samples: int,
    compile_backend: str,
    output_artifact: Path | None,
) -> dict[str, object]:
    device = workload.model.q_neutral.device
    memory_start = _memory_start(device)
    compile_wrapper_creation_ms: float | None = None

    if case.lane == "compiled":
        torch.compiler.reset()
        compile_start_ns = time.perf_counter_ns()
        compiled_function = torch.compile(
            workload.function,
            fullgraph=True,
            dynamic=False,
            backend=compile_backend,
        )
        compile_wrapper_creation_ms = (time.perf_counter_ns() - compile_start_ns) / 1_000_000.0

        def invoke() -> object:
            return compiled_function(*workload.args)

    else:

        def invoke() -> object:
            return workload.function(*workload.args)

    evaluation_context = torch.inference_mode() if case.operation != "ik" else nullcontext()
    with evaluation_context:
        first_output, first_call_ms = _time_once(invoke, device)
        if workload.fixed_iteration_budget is not None:
            _assert_fixed_iterations(first_output, workload.fixed_iteration_budget)
        for _ in range(warmup):
            invoke()
        _synchronize(device)
        timings_ms: list[float] = []
        final_output = first_output
        for _ in range(samples):
            final_output, elapsed_ms = _time_once(invoke, device)
            timings_ms.append(elapsed_ms)
        if workload.fixed_iteration_budget is not None:
            _assert_fixed_iterations(final_output, workload.fixed_iteration_budget)

    memory = _memory_finish(device, memory_start)
    output = _output_summary(final_output)
    if not output["all_finite"]:
        raise RuntimeError("workload produced a non-finite tensor output")
    if output_artifact is not None:
        torch.save(
            tuple(tensor.detach().to(device="cpu") for tensor in _output_tensors(final_output)),
            output_artifact,
        )
    return {
        "status": "SUCCESS",
        "case_id": case.case_id,
        "selector": _selector(case),
        "model": {
            "name": workload.model.name,
            "nq": workload.model.nq,
            "nv": workload.model.nv,
            "njoints": workload.model.njoints,
            "nframes": workload.model.nframes,
            "base": "free_flyer" if case.model in {"panda_free", "smpl"} else "fixed",
        },
        "workload": workload.details,
        "cold": {
            "compile_wrapper_creation_ms": compile_wrapper_creation_ms,
            "first_call_ms": first_call_ms,
            "first_call_includes_lazy_graph_compilation": case.lane == "compiled",
            "note": (
                "torch.compile wrapper creation is separate; the compiled first call combines lazy compilation and execution"
                if case.lane == "compiled"
                else "first eager invocation after model and input preparation"
            ),
        },
        "steady_state": _timing_summary(timings_ms),
        "cuda_memory": memory,
        "output": output,
    }


def _unsupported_result(case: CaseSpec, reason: str) -> dict[str, object]:
    return {
        "status": "UNSUPPORTED",
        "case_id": case.case_id,
        "selector": _selector(case),
        "reason": reason,
    }


def _selector(case: CaseSpec) -> dict[str, object]:
    return {
        "model": case.model,
        "operation": case.operation,
        "device": case.device,
        "lane": case.lane,
        "batch": case.batch,
        "ik_mode": case.ik_mode,
    }


def _run_worker(case: CaseSpec, args: argparse.Namespace) -> dict[str, object]:
    if case.operation == "ik" and case.lane == "compiled":
        return _unsupported_result(
            case,
            "the public solve_ik facade constructs Python Problem/solver objects and uses host-controlled termination; "
            "no honest fullgraph compiled public-IK workload exists, and no lower-level substitute is measured",
        )
    if case.device == "cuda" and (not torch.cuda.is_available() or args.cuda_index >= torch.cuda.device_count()):
        return _unsupported_result(
            case,
            f"CUDA device index {args.cuda_index} is unavailable (torch sees {torch.cuda.device_count()} device(s))",
        )

    dtype = _dtype_from_name(args.dtype)
    try:
        workload = _prepare_workload(
            case,
            dtype=dtype,
            cuda_index=args.cuda_index,
            seed=args.seed,
            ik_iterations=args.ik_iterations,
        )
        return _measure_case(
            case,
            workload,
            warmup=args.warmup,
            samples=args.samples,
            compile_backend=args.compile_backend,
            output_artifact=args._worker_tensor_output,
        )
    except ModuleNotFoundError as exc:
        return _unsupported_result(case, str(exc))
    except torch.OutOfMemoryError as exc:
        return {
            "status": "OOM",
            "case_id": case.case_id,
            "selector": _selector(case),
            "reason": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 - child must preserve an explicit case result
        return {
            "status": "ERROR",
            "case_id": case.case_id,
            "selector": _selector(case),
            "error_type": type(exc).__name__,
            "reason": str(exc),
        }


def _worker_command(
    case: CaseSpec,
    args: argparse.Namespace,
    output: Path,
    tensor_output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(_SCRIPT_PATH),
        "--_worker-case",
        case.case_id,
        "--_worker-output",
        str(output),
        "--_worker-tensor-output",
        str(tensor_output),
        "--label",
        args.label,
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--samples",
        str(args.samples),
        "--ik-iterations",
        str(args.ik_iterations),
        "--compile-backend",
        args.compile_backend,
        "--cuda-index",
        str(args.cuda_index),
        "--threads",
        str(args.threads),
        "--seed",
        str(args.seed),
    ]


def _run_child_case(  # noqa: PLR0913 - explicit child protocol is easier to audit
    case: CaseSpec,
    args: argparse.Namespace,
    *,
    scratch: Path,
) -> dict[str, object]:
    case_token = case.case_id.replace("/", "__")
    output = scratch / f"{case_token}.json"
    tensor_output = scratch / f"{case_token}.pt"
    cache = scratch / f"inductor__{case_token}"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": str(args.seed),
            "OMP_NUM_THREADS": str(args.threads),
            "MKL_NUM_THREADS": str(args.threads),
            "OPENBLAS_NUM_THREADS": str(args.threads),
            "TORCHINDUCTOR_CACHE_DIR": str(cache),
        }
    )
    command = _worker_command(case, args, output, tensor_output)
    try:
        completed = subprocess.run(
            command,
            cwd=_REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=args.case_timeout_seconds if args.case_timeout_seconds > 0.0 else None,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "TIMEOUT",
            "case_id": case.case_id,
            "selector": _selector(case),
            "timeout_seconds": args.case_timeout_seconds,
            "reason": "fresh worker exceeded the configured whole-case wall-clock deadline",
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else None,
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else None,
        }

    if not output.exists():
        return {
            "status": "ERROR",
            "case_id": case.case_id,
            "selector": _selector(case),
            "error_type": "WorkerFailure",
            "reason": f"worker exited {completed.returncode} without a result artifact",
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    try:
        result = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "ERROR",
            "case_id": case.case_id,
            "selector": _selector(case),
            "error_type": type(exc).__name__,
            "reason": f"could not read worker result: {exc}",
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    if completed.returncode != 0 and result.get("status") != "ERROR":
        result = {
            "status": "ERROR",
            "case_id": case.case_id,
            "selector": _selector(case),
            "error_type": "WorkerFailure",
            "reason": f"worker exited {completed.returncode}",
            "worker_result": result,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    if result.get("status") == "SUCCESS" and tensor_output.exists():
        result["_output_artifact"] = str(tensor_output)
    return result


def _canonical_protocol(args: argparse.Namespace, cases: Sequence[CaseSpec]) -> bool:
    cpu_affinity_is_pinned = hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) == 1
    return (
        not args.smoke
        and not args.case
        and tuple(args.models) == MODEL_NAMES
        and tuple(args.operations) == OPERATIONS
        and tuple(args.devices) == DEVICES
        and tuple(args.lanes) == LANES
        and tuple(args.batches) == CANONICAL_BATCHES
        and args.warmup == DEFAULT_WARMUP
        and args.samples == DEFAULT_SAMPLES
        and args.ik_iterations == DEFAULT_IK_ITERATIONS
        and args.seed == SEED
        and args.dtype == CANONICAL_DTYPE
        and args.compile_backend == CANONICAL_COMPILE_BACKEND
        and args.threads == CANONICAL_THREADS
        and args.cuda_index == CANONICAL_CUDA_INDEX
        and os.environ.get("CUDA_VISIBLE_DEVICES") == CANONICAL_CUDA_VISIBLE_DEVICES
        and cpu_affinity_is_pinned
        and len(cases) == len(_all_cases())
    )


def _definition(args: argparse.Namespace) -> dict[str, object]:
    return {
        "seed": args.seed,
        "dtype": f"torch.{args.dtype}",
        "models": {
            "panda_fixed": "Franka Panda URDF, fixed base",
            "panda_free": "Franka Panda URDF with JointFreeFlyer root",
            "smpl": "make_smpl_like_model, 25 joints including universe, free-flyer + spherical joints",
        },
        "operations": {
            "fk": "tensor-only FK plus frame placements",
            "rnea": "tensor-only RNEA torque output",
            "ik": "public solve_ik; B=1 is unbatched, B>1 independently batched",
        },
        "canonical_batches": list(CANONICAL_BATCHES),
        "canonical_warmup": DEFAULT_WARMUP,
        "canonical_samples": DEFAULT_SAMPLES,
        "warmup": args.warmup,
        "samples": args.samples,
        "ik_iteration_budget": args.ik_iterations,
        "torch_intra_and_interop_threads": args.threads,
        "compiled": {
            "api": "torch.compile",
            "backend": args.compile_backend,
            "fullgraph": True,
            "dynamic": False,
            "public_ik": "UNSUPPORTED",
        },
        "timer": "time.perf_counter_ns wall clock; CUDA synchronized before and after every timed invocation",
        "statistics": "raw samples plus median, inclusive Q1/Q3, IQR, min, and max",
        "cuda_memory": "torch.cuda.max_memory_allocated/reserved after reset; start and incremental peaks retained",
        "cold_scope": (
            "one fresh process and fresh TorchInductor cache directory per case; model/input construction excluded; "
            "compile wrapper creation and lazy compile-plus-first-execution are separate fields"
        ),
        "deferred_cold_costs": {
            "warp_first_launch": "measured by the separate Warp lane harness",
            "cuda_graph_record": "not implemented by this harness",
        },
    }


def _result_inputs(result: dict[str, object]) -> object:
    workload = result.get("workload")
    return workload.get("inputs") if isinstance(workload, dict) else None


def _input_identity(case_results: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for result in case_results:
        if result.get("status") != "SUCCESS":
            continue
        selector = result.get("selector")
        if not isinstance(selector, dict):
            continue
        key = (selector.get("model"), selector.get("operation"), selector.get("batch"))
        groups.setdefault(key, []).append(result)

    checks: list[dict[str, object]] = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        reference = _result_inputs(members[0])
        identical = all(_result_inputs(member) == reference for member in members[1:])
        checks.append(
            {
                "workload_id": f"{key[0]}/{key[1]}/b{key[2]}",
                "status": "PASS" if identical else "FAIL",
                "case_ids": [str(member["case_id"]) for member in members],
                "fingerprints_identical": identical,
            }
        )
    return checks


def _artifact_tensors(result: dict[str, object]) -> tuple[torch.Tensor, ...]:
    artifact = result.get("_output_artifact")
    if not isinstance(artifact, str):
        raise RuntimeError(f"successful case {result.get('case_id')} has no tensor artifact")
    loaded = torch.load(artifact, map_location="cpu", weights_only=True)
    if not isinstance(loaded, tuple) or not all(isinstance(tensor, torch.Tensor) for tensor in loaded):
        raise RuntimeError(f"invalid tensor artifact for {result.get('case_id')}")
    return loaded


def _compare_outputs(
    eager: dict[str, object],
    compiled: dict[str, object],
    *,
    dtype: torch.dtype,
) -> dict[str, object]:
    eager_tensors = _artifact_tensors(eager)
    compiled_tensors = _artifact_tensors(compiled)
    atol, rtol = (2e-5, 2e-5) if dtype == torch.float32 else (2e-9, 2e-9)
    shapes_match = len(eager_tensors) == len(compiled_tensors) and all(
        eager_tensor.shape == compiled_tensor.shape
        for eager_tensor, compiled_tensor in zip(eager_tensors, compiled_tensors, strict=True)
    )
    if not shapes_match:
        return {
            "passed": False,
            "shapes_match": False,
            "max_abs_error": None,
            "max_rel_error": None,
            "atol": atol,
            "rtol": rtol,
        }
    max_abs_error = 0.0
    max_rel_error = 0.0
    numeric_passed = True
    for eager_tensor, compiled_tensor in zip(eager_tensors, compiled_tensors, strict=True):
        difference = (compiled_tensor - eager_tensor).abs()
        max_abs_error = max(max_abs_error, float(difference.max()))
        denominator = eager_tensor.abs().clamp_min(torch.finfo(eager_tensor.dtype).tiny)
        max_rel_error = max(max_rel_error, float((difference / denominator).max()))
        numeric_passed = numeric_passed and torch.allclose(
            compiled_tensor,
            eager_tensor,
            atol=atol,
            rtol=rtol,
        )
    return {
        "passed": numeric_passed,
        "shapes_match": True,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "atol": atol,
        "rtol": rtol,
    }


def _eager_compiled_parity(
    case_results: Sequence[dict[str, object]],
    *,
    dtype: torch.dtype,
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], dict[object, dict[str, object]]] = {}
    for result in case_results:
        selector = result.get("selector")
        if not isinstance(selector, dict):
            continue
        key = (
            selector.get("model"),
            selector.get("operation"),
            selector.get("device"),
            selector.get("batch"),
        )
        groups.setdefault(key, {})[selector.get("lane")] = result

    comparisons: list[dict[str, object]] = []
    for key, lanes in groups.items():
        if "eager" not in lanes or "compiled" not in lanes:
            continue
        eager, compiled = lanes["eager"], lanes["compiled"]
        comparison: dict[str, object] = {
            "comparison_id": f"{key[0]}/{key[1]}/{key[2]}/b{key[3]}",
            "eager_case_id": eager["case_id"],
            "compiled_case_id": compiled["case_id"],
            "input_fingerprints_identical": _result_inputs(eager) == _result_inputs(compiled),
        }
        if eager.get("status") != "SUCCESS" or compiled.get("status") != "SUCCESS":
            comparison.update(
                {
                    "status": "NOT_EVALUATED",
                    "reason": "both lanes must be SUCCESS",
                    "lane_statuses": {
                        "eager": eager.get("status"),
                        "compiled": compiled.get("status"),
                    },
                }
            )
        else:
            numeric = _compare_outputs(eager, compiled, dtype=dtype)
            passed = bool(comparison["input_fingerprints_identical"]) and bool(numeric["passed"])
            comparison.update({"status": "PASS" if passed else "FAIL", "numeric": numeric})
        comparisons.append(comparison)
    return comparisons


def run(args: argparse.Namespace, cases: Sequence[CaseSpec]) -> dict[str, object]:
    protocol = "smoke" if args.smoke else ("canonical" if _canonical_protocol(args, cases) else "filtered")
    case_results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="betterrobot-baseline-") as scratch_name:
        scratch = Path(scratch_name)
        for case in cases:
            case_results.append(_run_child_case(case, args, scratch=scratch))
        input_identity = _input_identity(case_results)
        eager_compiled_parity = _eager_compiled_parity(
            case_results,
            dtype=_dtype_from_name(args.dtype),
        )
        for result in case_results:
            result.pop("_output_artifact", None)
    counts = {
        status: sum(result.get("status") == status for result in case_results)
        for status in ("SUCCESS", "OOM", "TIMEOUT", "UNSUPPORTED", "ERROR")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "torch_baseline",
        "label": args.label,
        "protocol": protocol,
        "canonical_protocol": protocol == "canonical",
        "advisory_only": True,
        "environment": _environment_metadata(
            args.cuda_index,
            probe_cuda=any(case.device == "cuda" for case in cases),
        ),
        "definition": _definition(args),
        "selection": {
            "case_ids": [case.case_id for case in cases],
            "selected_count": len(cases),
            "status_counts": counts,
            "case_timeout_seconds": args.case_timeout_seconds,
            "worker_isolation": "fresh Python process and fresh temporary TorchInductor cache per case",
        },
        "validation": {
            "input_identity": input_identity,
            "eager_vs_compiled_parity": eager_compiled_parity,
            "all_evaluated_checks_passed": all(
                check["status"] != "FAIL" for check in (*input_identity, *eager_compiled_parity)
            ),
        },
        "cases": case_results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", choices=MODEL_NAMES, nargs="+", default=MODEL_NAMES)
    parser.add_argument("--operations", choices=OPERATIONS, nargs="+", default=OPERATIONS)
    parser.add_argument("--devices", choices=DEVICES, nargs="+", default=DEVICES)
    parser.add_argument("--lanes", choices=LANES, nargs="+", default=LANES)
    parser.add_argument("--batches", choices=CANONICAL_BATCHES, type=int, nargs="+", default=CANONICAL_BATCHES)
    parser.add_argument(
        "--case",
        action="append",
        help="exact model/operation/device/lane/bN selector; repeatable and overrides matrix filters",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--ik-iterations", type=int, default=DEFAULT_IK_ITERATIONS)
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--case-timeout-seconds", type=float, default=DEFAULT_CASE_TIMEOUT_SECONDS)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="non-canonical protocol: force B=1, warmup=0, samples=1, and IK budget=1",
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--_worker-case", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-tensor-output", type=Path, help=argparse.SUPPRESS)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.warmup < 0 or args.samples < 1:
        parser.error("--warmup must be non-negative and --samples must be positive")
    if args.ik_iterations < 1:
        parser.error("--ik-iterations must be positive")
    if args.cuda_index < 0 or args.threads < 1:
        parser.error("--cuda-index must be non-negative and --threads must be positive")
    if args.case_timeout_seconds < 0.0:
        parser.error("--case-timeout-seconds must be non-negative (zero disables it)")
    if args.smoke and args.case:
        parser.error("--smoke cannot be combined with exact --case selectors; use the matrix filters")
    if args.smoke:
        args.warmup = 0
        args.samples = 1
        args.ik_iterations = 1


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    if args._worker_case is not None:
        if args._worker_output is None or args._worker_tensor_output is None:
            parser.error("internal worker requires both worker output paths")
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)
        try:
            case = _parse_case_id(args._worker_case)
        except ValueError as exc:
            parser.error(str(exc))
        result = _run_worker(case, args)
        args._worker_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit(1 if result.get("status") == "ERROR" else 0)

    try:
        cases = _select_cases(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not cases:
        parser.error("the filters selected no cases")
    if args.list_cases:
        for case in cases:
            support = "UNSUPPORTED" if case.operation == "ik" and case.lane == "compiled" else "SELECTED"
            print(f"{case.case_id}\t{support}")
        return

    report = run(args, cases)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["selection"]["status_counts"]["ERROR"]:  # type: ignore[index]
        raise SystemExit("one or more benchmark workers failed unexpectedly")
    if not report["validation"]["all_evaluated_checks_passed"]:  # type: ignore[index]
        raise SystemExit("benchmark input identity or eager-vs-compiled parity failed")


if __name__ == "__main__":
    main()
