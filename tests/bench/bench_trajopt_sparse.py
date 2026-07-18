"""M5 Phase-C CPU benchmark for dense and block-banded trajectory LM.

This file is an executable benchmark harness rather than a pytest-benchmark
micro-benchmark.  Every ``(route, horizon)`` case runs in a fresh child
process so Linux peak RSS is independent.  The canonical invocation is::

    uv run python tests/bench/bench_trajopt_sparse.py

For a short harness check that does not overwrite the committed baseline::

    uv run python tests/bench/bench_trajopt_sparse.py --quick --allow-unpinned

The exact definition is frozen in
``plan/design/m5_sparse_trajectory_structure_design.md §11``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
import json
import math
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Literal

import torch


SCHEMA_VERSION = 1
SEED = 20260717
DT = 1.0 / 30.0
HORIZONS = (50, 125, 250, 500)
PATHS = ("dense", "structured")
EXPECTED_ACTIVE_ENVELOPE = {50: 127, 125: 323, 250: 652, 500: 1305}
EXPECTED_HOST = "robotics2-ESC8000-E11"
EXPECTED_CPU_MODEL = "INTEL(R) XEON(R) PLATINUM 8570"
CPU_AFFINITY = 0
ADDRESS_LIMIT_BYTES = 16 * 2**30
CASE_TIMEOUT_SECONDS = 600
CANONICAL_UPDATES = 5
CANONICAL_WARMUPS = 2
CANONICAL_MEASUREMENTS = 7
FRAME_NAMES = (
    "body_head",
    "body_left_wrist",
    "body_right_wrist",
    "body_left_ankle",
    "body_right_ankle",
)
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": str(SEED),
}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BASELINE = Path(__file__).resolve().parent / "baselines" / "trajopt_sparse_cpu.json"


class TrajectoryTangentEnvelopeResidual:
    """Benchmark-only tangent-space envelope hinge with diagonal blocks."""

    reads = ("q",)

    def __init__(
        self,
        model: Any,
        *,
        horizon: int,
        half_width: torch.Tensor,
        name: str = "tangent_reference_envelope",
    ) -> None:
        if tuple(half_width.shape) != (model.nv,):
            raise ValueError(f"half_width must have shape ({model.nv},), got {tuple(half_width.shape)}")
        self.model = model
        self.horizon = horizon
        self.half_width = half_width
        self.name = name
        self.dim = 2 * horizon * model.nv

    def _trajectory(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        q = ctx["q"]
        if not isinstance(q, torch.Tensor):
            raise TypeError("TrajectoryTangentEnvelopeResidual q must be a tensor")
        expected = (self.horizon, self.model.nq)
        if tuple(q.shape[-2:]) != expected:
            raise ValueError(f"q must end in {expected}, got {tuple(q.shape)}")
        return q

    def _delta(self, ctx: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._trajectory(ctx)
        neutral = self.model.q_neutral.to(dtype=q.dtype, device=q.device).expand(self.horizon, -1)
        delta = self.model.difference(neutral, q)
        half_width = self.half_width.to(dtype=q.dtype, device=q.device)
        return delta, half_width

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        delta, half_width = self._delta(ctx)
        lower_hinge = torch.clamp(-half_width - delta, min=0.0)
        upper_hinge = torch.clamp(delta - half_width, min=0.0)
        return torch.cat((lower_hinge, upper_hinge), dim=-1).reshape(
            *delta.shape[:-2],
            self.dim,
        )

    def temporal_structure(self, variable_name: str):
        from better_robot.residuals.structure import TemporalPattern  # noqa: PLC0415

        if variable_name != "q":
            return None
        return TemporalPattern(
            rows=self.horizon,
            row_width=2 * self.model.nv,
            row_origin=0,
            offsets=(0,),
        )

    def temporal_jacobian_blocks(
        self,
        ctx: Mapping[str, Any],
        variable_name: str,
    ) -> Mapping[int, torch.Tensor]:
        from better_robot.residuals._temporal_jacobian import temporal_free_indices  # noqa: PLC0415

        if variable_name != "q":
            return {}
        q = self._trajectory(ctx)
        delta, half_width = self._delta(ctx)
        indices = temporal_free_indices(ctx, "q", device=q.device)
        lower = -torch.diag_embed((delta < -half_width).to(dtype=q.dtype)).index_select(-1, indices)
        upper = torch.diag_embed((delta > half_width).to(dtype=q.dtype)).index_select(-1, indices)
        block = torch.cat((lower, upper), dim=-2)
        # Preserve the explicit-unroll graph contract with a mathematically
        # zero anchor; the active-set indicators remain piecewise constant.
        anchor = q.sum(dim=(-2, -1)) * 0.0
        return {0: block + anchor[..., None, None, None]}

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        from better_robot.residuals._temporal_jacobian import dense_temporal_jacobian  # noqa: PLC0415

        pattern = self.temporal_structure("q")
        assert pattern is not None
        return {
            "q": dense_temporal_jacobian(
                pattern,
                self.temporal_jacobian_blocks(ctx, "q"),
                horizon=self.horizon,
            )
        }


def _hemisphere_align(q: torch.Tensor, manifold: Any) -> torch.Tensor:
    """Apply the approved ingress representative policy on a clone."""
    aligned = q.clone()
    for coordinate_slice in manifold.unit_coordinate_slices:
        start = 0 if coordinate_slice.start is None else coordinate_slice.start
        stop = manifold.model.nq if coordinate_slice.stop is None else coordinate_slice.stop
        if stop - start != 4:
            continue
        quaternion = aligned[..., coordinate_slice]
        adjacent_dot = (quaternion[..., 1:, :] * quaternion[..., :-1, :]).sum(dim=-1)
        step_sign = torch.where(
            adjacent_dot < 0.0,
            -torch.ones_like(adjacent_dot),
            torch.ones_like(adjacent_dot),
        )
        first = torch.ones_like(quaternion[..., :1, 0])
        signs = torch.cat((first, step_sign), dim=-1).cumprod(dim=-1).unsqueeze(-1)
        aligned[..., coordinate_slice] = quaternion * signs
    return aligned


def _build_problem(path: Literal["dense", "structured"], horizon: int) -> dict[str, Any]:  # noqa: PLR0915
    """Construct the exact deterministic §11 model, Problem, and optimizer."""
    from better_robot.io.builders.smpl_like import make_smpl_like_model  # noqa: PLC0415
    from better_robot.kinematics.forward import forward_kinematics  # noqa: PLC0415
    from better_robot.optim import (  # noqa: PLC0415
        Bounds,
        LevenbergMarquardt,
        Problem,
        ResidualItem,
        RobotConfig,
        RobotStateProvider,
        VarSpec,
    )
    from better_robot.optim.kernels import L2  # noqa: PLC0415
    from better_robot.residuals.pose import PoseResidual  # noqa: PLC0415
    from better_robot.residuals.regularization import ReferenceTrajectoryResidual  # noqa: PLC0415
    from better_robot.residuals.smoothness import AccelerationResidual, VelocityResidual  # noqa: PLC0415
    from better_robot.residuals.temporal import TimeIndexedResidual  # noqa: PLC0415

    if path not in PATHS:
        raise ValueError(f"path must be one of {PATHS}, got {path!r}")
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon}")

    model = make_smpl_like_model(
        height=1.75,
        mass=70.0,
        preserve_joint_order=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    if (model.nq, model.nv, model.njoints) != (99, 75, 25):
        raise AssertionError(
            "SMPL benchmark model changed: expected (nq,nv,njoints)=(99,75,25), "
            f"got {(model.nq, model.nv, model.njoints)}"
        )

    cpu_generator = torch.Generator(device="cpu").manual_seed(SEED)
    directions = torch.randn(3, 75, generator=cpu_generator, dtype=torch.float32)
    directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    u = torch.linspace(0.0, 1.0, horizon, dtype=torch.float32)
    tangent_amplitude = torch.tensor((0.04, 0.02, 0.01), dtype=torch.float32)
    tangent_phase = torch.tensor((0.0, 0.4, 0.8), dtype=torch.float32)
    tangent = sum(
        tangent_amplitude[index]
        * torch.sin(2 * torch.pi * (index + 1) * u + tangent_phase[index])[:, None]
        * directions[index]
        for index in range(3)
    )
    q_neutral = model.q_neutral.expand(horizon, -1)
    q_ground_truth = model.integrate(q_neutral, tangent)
    q_initial = q_neutral.clone()

    manifold = RobotConfig(model)
    q_ground_truth = _hemisphere_align(q_ground_truth, manifold)
    q_initial = _hemisphere_align(q_initial, manifold)
    box = manifold.box_mask
    raw_lower = model.lower_pos_limit.to(dtype=torch.float32, device="cpu")
    raw_upper = model.upper_pos_limit.to(dtype=torch.float32, device="cpu")
    robot_bounds = Bounds(
        lower=torch.where(box, raw_lower, torch.full_like(raw_lower, -torch.inf)),
        upper=torch.where(box, raw_upper, torch.full_like(raw_upper, torch.inf)),
    )
    if not bool(torch.isneginf(robot_bounds.lower[~box]).all()):
        raise AssertionError("non-box lower bounds must be -inf")
    if not bool(torch.isposinf(robot_bounds.upper[~box]).all()):
        raise AssertionError("non-box upper bounds must be +inf")
    q_spec = VarSpec(
        "q",
        (horizon, model.nq),
        manifold=manifold,
        bounds=robot_bounds,
        time_axis=0,
    )

    half_width = torch.full((75,), 0.010, dtype=torch.float32)
    half_width[:3] = 0.003
    active_envelope = int((model.difference(q_neutral, q_ground_truth).abs() > half_width).sum())
    expected_active = EXPECTED_ACTIVE_ENVELOPE[horizon]
    if active_envelope != expected_active:
        raise AssertionError(
            f"active envelope count changed at T={horizon}: expected {expected_active}, got {active_envelope}"
        )

    keyframes = torch.linspace(0, horizon - 1, 8, dtype=torch.float64).round().to(torch.int64).tolist()
    if len(set(keyframes)) != 8:
        raise AssertionError(f"keyframe indices must be unique, got {keyframes}")
    frame_ids = {name: model.frame_id(name) for name in FRAME_NAMES}
    target_data = forward_kinematics(model, q_ground_truth, compute_frames=True)
    if target_data.frame_pose_world is None:
        raise AssertionError("benchmark target FK did not produce frame poses")

    l2 = L2()
    residual_items: list[Any] = []
    for t_idx in keyframes:
        for frame_name in FRAME_NAMES:
            item_name = f"pose_t{t_idx}_{frame_name}"
            inner = PoseResidual(
                frame_id=frame_ids[frame_name],
                target=target_data.frame_pose_world[t_idx, frame_ids[frame_name]].clone(),
                pos_weight=10.0,
                ori_weight=2.0,
                model=model,
                name=item_name,
            )
            temporal = TimeIndexedResidual(inner, t_idx, horizon=horizon, name=item_name)
            residual_items.append(ResidualItem(item_name, temporal, weight=1.0, kernel=l2, group_size=6))

    velocity = VelocityResidual(model, dt=DT, horizon=horizon, name="central_velocity")
    acceleration = AccelerationResidual(model, dt=DT, horizon=horizon, name="acceleration")
    reference = ReferenceTrajectoryResidual(
        model,
        q_neutral.clone(),
        name="reference_to_neutral",
    )
    envelope = TrajectoryTangentEnvelopeResidual(
        model,
        horizon=horizon,
        half_width=half_width,
    )
    residual_items.extend(
        (
            ResidualItem("central_velocity", velocity, weight=0.05, kernel=l2, group_size=75),
            ResidualItem("acceleration", acceleration, weight=0.005, kernel=l2, group_size=75),
            ResidualItem("reference_to_neutral", reference, weight=0.01, kernel=l2, group_size=75),
            ResidualItem(
                "tangent_reference_envelope",
                envelope,
                weight=0.10,
                kernel=l2,
                group_size=1,
            ),
        )
    )
    problem = Problem(
        vars=(q_spec,),
        residuals=tuple(residual_items),
        providers=(RobotStateProvider(model),),
    )
    optimizer = LevenbergMarquardt(
        max_iter=CANONICAL_UPDATES,
        damping_parameter=1e-4,
        gtol=0.0,
        xtol=0.0,
        ftol=0.0,
        linear_solver=None,
        linearization=path,
    )
    decision = optimizer.resolve_linearization(problem)
    expected_route = "dense" if path == "dense" else "banded"
    if decision.used != expected_route:
        raise AssertionError(f"requested {path!r} but optimizer selected {decision.used!r}")
    return {
        "model": model,
        "problem": problem,
        "optimizer": optimizer,
        "q_initial": q_initial,
        "active_envelope_coordinates": active_envelope,
        "keyframes": keyframes,
        "route": decision.used,
        "route_reason": decision.reason.value,
        "route_detail": decision.detail,
    }


def _enum_name(enum_type: Any, value: int) -> str:
    try:
        return enum_type(value).name
    except ValueError:
        return f"UNKNOWN_{value}"


def _run_one_solve(case: Mapping[str, Any], *, updates: int) -> dict[str, Any]:
    """Run exactly one fresh fixed-budget solve and retain small diagnostics."""
    from better_robot.optim import LMStatus  # noqa: PLC0415

    problem = case["problem"]
    optimizer = case["optimizer"]
    gc.collect()
    values = {"q": case["q_initial"].clone()}
    factorization_diagnostics: list[torch.Tensor] = []
    started = time.perf_counter()
    state = optimizer.init_state(values, problem)
    for _ in range(updates):
        values, state = optimizer.update(values, state, problem)
        # References are safe: LMState is immutable and each update creates
        # fresh scalar tensors. Keeping only these tensors avoids retaining a
        # prior residual/Jacobian while adding no timed tensor operation.
        factorization_diagnostics.append(state.factorization_ok)
    values, state = optimizer.finalize(values, state, problem)
    elapsed_seconds = time.perf_counter() - started

    lm_status = int(state.status)
    allowed_lm = {
        LMStatus.RUNNING.value,
        LMStatus.CONVERGED.value,
        LMStatus.STALLED_AT_BOUNDS.value,
    }
    finite_final = bool(
        torch.isfinite(state.cost)
        & torch.isfinite(state.residual).all()
        & torch.isfinite(state.gradient).all()
        & torch.isfinite(state.grad_norm)
    )
    linear_success = all(bool(ok) for ok in factorization_diagnostics)
    expected_route = "dense" if case["optimizer"].linearization == "dense" else "banded"
    route_success = case["route"] == expected_route
    failure_reasons: list[str] = []
    if not finite_final:
        failure_reasons.append("nonfinite_final_metrics")
    if lm_status not in allowed_lm:
        failure_reasons.append(f"disallowed_lm_status:{_enum_name(LMStatus, lm_status)}")
    if not linear_success:
        failure_reasons.append("linear_solve_not_successful")
    if not route_success:
        failure_reasons.append(f"unexpected_route:{case['route']}")

    return {
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_update": elapsed_seconds / updates,
        "success": not failure_reasons,
        "failure_reasons": failure_reasons,
        "route": case["route"],
        "final_lm_status": lm_status,
        "final_lm_status_name": _enum_name(LMStatus, lm_status),
        "final_cost": float(state.cost),
        "final_residual_norm": float(torch.linalg.vector_norm(state.residual)),
        "final_gradient_norm": float(torch.linalg.vector_norm(state.gradient)),
        "linear_solves": [{"factorization_ok": bool(ok)} for ok in factorization_diagnostics],
    }


def _current_rss_bytes() -> int:
    fields = Path("/proc/self/statm").read_text(encoding="utf-8").split()
    return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))


def _virtual_size_bytes() -> int:
    fields = Path("/proc/self/statm").read_text(encoding="utf-8").split()
    return int(fields[0]) * int(os.sysconf("SC_PAGE_SIZE"))


def _peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _quartiles(samples: Sequence[float]) -> tuple[float, float]:
    if len(samples) == 1:
        return samples[0], samples[0]
    q1, _median, q3 = statistics.quantiles(samples, n=4, method="inclusive")
    return q1, q3


def _run_child_case(  # noqa: PLR0915
    *,
    path: Literal["dense", "structured"],
    horizon: int,
    updates: int,
    warmups: int,
    measurements: int,
    skip_cold: bool,
    address_limit_bytes: int,
    virtual_size_before_limit: int,
) -> dict[str, Any]:
    construction_started = time.perf_counter()
    case = _build_problem(path, horizon)
    construction_seconds = time.perf_counter() - construction_started
    construction_rss_bytes = _current_rss_bytes()

    cold = None if skip_cold else _run_one_solve(case, updates=updates)
    warmup_records = [_run_one_solve(case, updates=updates) for _ in range(warmups)]
    measured_records = [_run_one_solve(case, updates=updates) for _ in range(measurements)]
    peak_rss_bytes = _peak_rss_bytes()
    incremental_peak_rss_bytes = peak_rss_bytes - construction_rss_bytes
    seconds_per_update = [record["seconds_per_update"] for record in measured_records]
    elapsed_seconds = [record["elapsed_seconds"] for record in measured_records]
    q1, q3 = _quartiles(seconds_per_update)
    all_records = ([cold] if cold is not None else []) + warmup_records + measured_records
    successful = all(record["success"] for record in all_records)
    return {
        "status": "SUCCESS" if successful else "FAILED_NUMERICS",
        "path": path,
        "horizon": horizon,
        "route": case["route"],
        "route_reason": case["route_reason"],
        "route_detail": case["route_detail"],
        "active_envelope_coordinates": case["active_envelope_coordinates"],
        "keyframes": case["keyframes"],
        "updates_per_solve": updates,
        "warmup_solve_count": warmups,
        "measurement_solve_count": measurements,
        "cold_solve": cold,
        "warmup_solves": warmup_records,
        "measured_solves": measured_records,
        "timing": {
            "construction_seconds": construction_seconds,
            "raw_elapsed_seconds": elapsed_seconds,
            "raw_seconds_per_update": seconds_per_update,
            "median_seconds_per_update": statistics.median(seconds_per_update),
            "q1_seconds_per_update": q1,
            "q3_seconds_per_update": q3,
        },
        "memory": {
            "construction_rss_bytes": construction_rss_bytes,
            "peak_rss_bytes": peak_rss_bytes,
            "incremental_peak_rss_bytes": incremental_peak_rss_bytes,
            "construction_rss_mib": construction_rss_bytes / 2**20,
            "peak_rss_mib": peak_rss_bytes / 2**20,
            "incremental_peak_rss_mib": incremental_peak_rss_bytes / 2**20,
        },
        "limits": {
            "address_space_bytes": address_limit_bytes,
            "virtual_size_before_limit_bytes": virtual_size_before_limit,
        },
        "process": _runtime_metadata(),
    }


def _read_cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _git_metadata() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = (
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=_REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            != ""
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _runtime_metadata() -> dict[str, Any]:
    commit, dirty = _git_metadata()
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "hostname": platform.node(),
        "expected_hostname": EXPECTED_HOST,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "better_robot_commit": commit,
        "better_robot_dirty": dirty,
        "kernel": platform.release(),
        "cpu_model": _read_cpu_model(),
        "cpu_affinity": affinity,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
    }


def _definition() -> dict[str, Any]:
    return {
        "design_reference": "plan/design/m5_sparse_trajectory_structure_design.md §11",
        "host": EXPECTED_HOST,
        "cpu": "Intel Xeon Platinum 8570, 2 sockets, 56 cores/socket, 2 threads/core",
        "affinity": f"taskset -c {CPU_AFFINITY}",
        "device": "cpu",
        "dtype": "torch.float32",
        "seed": SEED,
        "dt": DT,
        "horizons": list(HORIZONS),
        "paths": list(PATHS),
        "model": {
            "builder": "make_smpl_like_model",
            "height": 1.75,
            "mass": 70.0,
            "preserve_joint_order": False,
            "nq": 99,
            "nv": 75,
            "njoints": 25,
        },
        "keyframes": 8,
        "frames": list(FRAME_NAMES),
        "residuals": [
            {
                "name": "40 time-indexed pose targets",
                "weight": 1.0,
                "position_scale": 10.0,
                "orientation_scale": 2.0,
                "group_size": 6,
            },
            {"name": "central velocity", "weight": 0.05, "group_size": 75},
            {"name": "acceleration", "weight": 0.005, "group_size": 75},
            {"name": "reference-to-neutral", "weight": 0.01, "group_size": 75},
            {"name": "tangent-reference envelope hinge", "weight": 0.10, "group_size": 1},
        ],
        "optimizer": {
            "type": "LevenbergMarquardt",
            "max_iter": CANONICAL_UPDATES,
            "damping_parameter": 1e-4,
            "gtol": 0.0,
            "xtol": 0.0,
            "ftol": 0.0,
            "linear_solver": None,
        },
        "measurement": {
            "cold_solves": 1,
            "warmup_solves": CANONICAL_WARMUPS,
            "measured_solves": CANONICAL_MEASUREMENTS,
            "updates_per_solve": CANONICAL_UPDATES,
            "statistic": "median and inclusive quartiles",
            "case_timeout_seconds": CASE_TIMEOUT_SECONDS,
            "address_space_limit_bytes": ADDRESS_LIMIT_BYTES,
        },
        "acceptance": {
            "minimum_successful_dense_points": 3,
            "slope_margin": 0.15,
            "structured_t500_required": True,
            "maximum_structured_slope": 1.35,
        },
    }


def _ols_log_slope(points: Sequence[tuple[int, float]]) -> float | None:
    usable = [(float(horizon), float(metric)) for horizon, metric in points if horizon > 0 and metric > 0]
    if len(usable) < 2:
        return None
    xs = [math.log(horizon) for horizon, _metric in usable]
    ys = [math.log(metric) for _horizon, metric in usable]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0.0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / denominator


def _compute_slopes(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in PATHS:
        successful = [case for case in cases if case.get("path") == path and case.get("status") == "SUCCESS"]
        time_points = [
            (int(case["horizon"]), float(case["timing"]["median_seconds_per_update"])) for case in successful
        ]
        memory_points = [
            (int(case["horizon"]), float(case["memory"]["incremental_peak_rss_bytes"])) for case in successful
        ]
        result[path] = {
            "successful_horizons": [horizon for horizon, _value in time_points],
            "time_seconds_per_update": _ols_log_slope(time_points),
            "incremental_peak_rss_bytes": _ols_log_slope(memory_points),
        }
    return result


def _evaluate_acceptance(
    cases: Sequence[Mapping[str, Any]],
    slopes: Mapping[str, Any],
    *,
    canonical_protocol: bool,
    complete_matrix: bool,
) -> dict[str, Any]:
    if not canonical_protocol or not complete_matrix:
        return {
            "status": "NOT_EVALUATED",
            "reason": "acceptance requires the complete pinned canonical protocol",
        }
    dense_count = len(slopes["dense"]["successful_horizons"])
    if dense_count < 3:
        return {
            "status": "INCONCLUSIVE",
            "reason": f"only {dense_count} dense points completed; at least 3 are required",
        }
    structured_t500 = next(
        (case for case in cases if case.get("path") == "structured" and case.get("horizon") == 500),
        None,
    )
    checks: dict[str, bool] = {
        "structured_t500_success": structured_t500 is not None and structured_t500.get("status") == "SUCCESS",
    }
    for metric in ("time_seconds_per_update", "incremental_peak_rss_bytes"):
        dense_slope = slopes["dense"][metric]
        structured_slope = slopes["structured"][metric]
        checks[f"{metric}_available"] = dense_slope is not None and structured_slope is not None
        checks[f"{metric}_margin"] = bool(
            dense_slope is not None and structured_slope is not None and structured_slope + 0.15 <= dense_slope
        )
        checks[f"{metric}_structured_cap"] = bool(structured_slope is not None and structured_slope <= 1.35)
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def _failure_status(returncode: int, stderr: str) -> str:
    del returncode
    lowered = stderr.lower()
    oom_markers = (
        "out of memory",
        "cannot allocate memory",
        "can't allocate memory",
        "defaultcpuallocator",
        "std::bad_alloc",
        "memoryerror",
    )
    # A signal alone is not proof of OOM: SIGKILL/SIGSEGV can also mean an
    # operator or harness failure. Only explicit allocator evidence earns the
    # DNF_OOM label; unknown deaths stay actionable HARNESS_ERROR results.
    if any(marker in lowered for marker in oom_markers):
        return "DNF_OOM"
    return "HARNESS_ERROR"


def _run_case_subprocess(  # noqa: PLR0913
    *,
    path: Literal["dense", "structured"],
    horizon: int,
    updates: int,
    warmups: int,
    measurements: int,
    skip_cold: bool,
    timeout_seconds: int,
    address_limit_bytes: int,
    allow_unpinned: bool,
) -> dict[str, Any]:
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="betterrobot-m5-bench-") as temp_dir:
        case_output = Path(temp_dir) / "case.json"
        command = [
            sys.executable,
            str(script),
            "--child",
            "--path",
            path,
            "--horizon",
            str(horizon),
            "--updates",
            str(updates),
            "--warmups",
            str(warmups),
            "--measurements",
            str(measurements),
            "--address-limit-bytes",
            str(address_limit_bytes),
            "--case-output",
            str(case_output),
        ]
        if skip_cold:
            command.append("--skip-cold")
        if not allow_unpinned:
            taskset = shutil.which("taskset")
            if taskset is None:
                return {
                    "status": "HARNESS_ERROR",
                    "path": path,
                    "horizon": horizon,
                    "stderr": "taskset is required by the canonical benchmark but was not found",
                    "command": shlex.join(command),
                }
            command = [taskset, "-c", str(CPU_AFFINITY), *command]

        environment = os.environ.copy()
        environment.update(THREAD_ENVIRONMENT)
        command_text = shlex.join(command)
        try:
            completed = subprocess.run(
                command,
                cwd=_REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "DNF_TIMEOUT",
                "path": path,
                "horizon": horizon,
                "timeout_seconds": timeout_seconds,
                "address_space_limit_bytes": address_limit_bytes,
                "stderr": (exc.stderr or "")[-20_000:] if isinstance(exc.stderr, str) else "",
                "command": command_text,
            }

        stderr = completed.stderr[-20_000:]
        if not case_output.exists():
            return {
                "status": _failure_status(completed.returncode, stderr),
                "path": path,
                "horizon": horizon,
                "returncode": completed.returncode,
                "timeout_seconds": timeout_seconds,
                "address_space_limit_bytes": address_limit_bytes,
                "stderr": stderr,
                "stdout": completed.stdout[-20_000:],
                "command": command_text,
            }
        try:
            result = json.loads(case_output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "status": "HARNESS_ERROR",
                "path": path,
                "horizon": horizon,
                "stderr": f"invalid child JSON: {exc}\n{stderr}",
                "command": command_text,
            }
        result["command"] = command_text
        result["stderr"] = stderr
        if completed.returncode != 0:
            result["returncode"] = completed.returncode
        return result


def _run_parent(args: argparse.Namespace) -> int:
    quick = bool(args.quick)
    selected_paths = tuple(args.path) if args.path else (("structured",) if quick else PATHS)
    selected_horizons = tuple(args.horizon) if args.horizon else ((50,) if quick else HORIZONS)
    if len(set(selected_paths)) != len(selected_paths):
        raise ValueError("--path selectors must not contain duplicates")
    if len(set(selected_horizons)) != len(selected_horizons):
        raise ValueError("--horizon selectors must not contain duplicates")
    updates = 1 if quick else CANONICAL_UPDATES
    warmups = 0 if quick else CANONICAL_WARMUPS
    measurements = 1 if quick else CANONICAL_MEASUREMENTS
    skip_cold = quick
    cases: list[dict[str, Any]] = []
    for path in selected_paths:
        for horizon in selected_horizons:
            print(f"running {path} T={horizon} ...", file=sys.stderr, flush=True)
            cases.append(
                _run_case_subprocess(
                    path=path,
                    horizon=horizon,
                    updates=updates,
                    warmups=warmups,
                    measurements=measurements,
                    skip_cold=skip_cold,
                    timeout_seconds=args.timeout_seconds,
                    address_limit_bytes=args.address_limit_bytes,
                    allow_unpinned=args.allow_unpinned,
                )
            )

    actual_host = platform.node()
    actual_cpu_model = _read_cpu_model()
    host_matches = actual_host == EXPECTED_HOST
    cpu_matches = actual_cpu_model == EXPECTED_CPU_MODEL
    canonical_protocol = (
        not quick
        and not args.allow_unpinned
        and host_matches
        and cpu_matches
        and updates == CANONICAL_UPDATES
        and warmups == CANONICAL_WARMUPS
        and measurements == CANONICAL_MEASUREMENTS
        and args.timeout_seconds == CASE_TIMEOUT_SECONDS
        and args.address_limit_bytes == ADDRESS_LIMIT_BYTES
    )
    complete_matrix = selected_paths == PATHS and selected_horizons == HORIZONS
    slopes = _compute_slopes(cases)
    output = {
        "_schema_version": SCHEMA_VERSION,
        "_status": "MEASURED" if canonical_protocol and complete_matrix else "NONCANONICAL_PARTIAL",
        "benchmark": "m5_sparse_trajectory_cpu",
        "definition": _definition(),
        "run": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "canonical_protocol": canonical_protocol,
            "complete_matrix": complete_matrix,
            "quick": quick,
            "allow_unpinned": bool(args.allow_unpinned),
            "hostname": actual_host,
            "expected_hostname": EXPECTED_HOST,
            "host_matches": host_matches,
            "cpu_model": actual_cpu_model,
            "expected_cpu_model": EXPECTED_CPU_MODEL,
            "cpu_matches": cpu_matches,
        },
        "cases": cases,
        "slopes": slopes,
        "acceptance": _evaluate_acceptance(
            cases,
            slopes,
            canonical_protocol=canonical_protocol,
            complete_matrix=complete_matrix,
        ),
        "gpu": {
            "status": "pending_m6",
            "device": None,
            "peak_memory_bytes": None,
            "cases": None,
        },
    }
    output_path = (
        Path(args.output).resolve()
        if args.output
        else (_DEFAULT_BASELINE if canonical_protocol and complete_matrix else None)
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {output_path}", file=sys.stderr)
    print(json.dumps(output, indent=2, sort_keys=True))
    if any(case["status"] == "HARNESS_ERROR" for case in cases):
        return 2
    if canonical_protocol and complete_matrix:
        return 0 if output["acceptance"]["status"] == "PASS" else 1
    return 0 if all(case["status"] == "SUCCESS" for case in cases) else 1


def _write_case_output(path: str | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        raise ValueError("--case-output is required in --child mode")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_child(args: argparse.Namespace) -> int:
    if len(args.path) != 1 or len(args.horizon) != 1:
        raise ValueError("--child requires exactly one --path and one --horizon")
    virtual_size = _virtual_size_bytes()
    if virtual_size >= args.address_limit_bytes:
        payload = {
            "status": "HARNESS_ERROR",
            "path": args.path[0],
            "horizon": args.horizon[0],
            "stderr": (
                f"current virtual size {virtual_size} already meets/exceeds address cap {args.address_limit_bytes}"
            ),
        }
        _write_case_output(args.case_output, payload)
        return 2
    resource.setrlimit(resource.RLIMIT_AS, (args.address_limit_bytes, args.address_limit_bytes))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        payload = _run_child_case(
            path=args.path[0],
            horizon=args.horizon[0],
            updates=args.updates,
            warmups=args.warmups,
            measurements=args.measurements,
            skip_cold=args.skip_cold,
            address_limit_bytes=args.address_limit_bytes,
            virtual_size_before_limit=virtual_size,
        )
    except MemoryError as exc:
        payload = {
            "status": "DNF_OOM",
            "path": args.path[0],
            "horizon": args.horizon[0],
            "address_space_limit_bytes": args.address_limit_bytes,
            "stderr": repr(exc),
        }
    except RuntimeError as exc:
        status = _failure_status(1, str(exc))
        payload = {
            "status": status,
            "path": args.path[0],
            "horizon": args.horizon[0],
            "address_space_limit_bytes": args.address_limit_bytes,
            "stderr": repr(exc),
        }
    except Exception as exc:  # noqa: BLE001 - child must persist an honest DNF record
        message = f"{type(exc).__name__}: {exc}"
        payload = {
            "status": _failure_status(1, message),
            "path": args.path[0],
            "horizon": args.horizon[0],
            "address_space_limit_bytes": args.address_limit_bytes,
            "stderr": message,
        }
    _write_case_output(args.case_output, payload)
    return 0 if payload["status"] in {"SUCCESS", "FAILED_NUMERICS", "DNF_OOM"} else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--quick", action="store_true", help="run one unpinned-friendly T=50 structured smoke case")
    parser.add_argument("--path", action="append", choices=PATHS, help="select one or more routes")
    parser.add_argument("--horizon", action="append", type=int, choices=HORIZONS, help="select one or more horizons")
    parser.add_argument("--output", help="parent JSON output path; full runs default to the committed baseline")
    parser.add_argument("--allow-unpinned", action="store_true", help="omit taskset; makes the run noncanonical")
    parser.add_argument("--timeout-seconds", type=int, default=CASE_TIMEOUT_SECONDS)
    parser.add_argument("--address-limit-bytes", type=int, default=ADDRESS_LIMIT_BYTES)
    parser.add_argument("--updates", type=int, default=CANONICAL_UPDATES, help=argparse.SUPPRESS)
    parser.add_argument("--warmups", type=int, default=CANONICAL_WARMUPS, help=argparse.SUPPRESS)
    parser.add_argument("--measurements", type=int, default=CANONICAL_MEASUREMENTS, help=argparse.SUPPRESS)
    parser.add_argument("--skip-cold", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-output", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name in ("timeout_seconds", "address_limit_bytes", "updates", "measurements"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    return _run_child(args) if args.child else _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
