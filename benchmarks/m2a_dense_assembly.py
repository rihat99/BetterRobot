"""Reproducible Panda pose-IK benchmark for M2a dense Jacobian assembly.

This is an advisory evidence script, not a CI performance gate.  It compares
the named-block :class:`better_robot.optim.Problem` dense assembly with today's
``LeastSquaresProblem.jacobian`` using the same model, configuration, target,
pose residual, and weights.  The legacy one-entry state cache is reported as a
separate case so it cannot silently turn an assembly comparison into a cached
FK comparison.

Forced ``jacrev`` and ``jacfwd`` timings are calibration data, not the draft
analytic-path headline.  An unsupported transform is recorded as an error in
the JSON report (not as a timing).  In particular, Torch forward AD may expose
dtype incompatibilities that ordinary eager evaluation does not.

Examples::

    python benchmarks/m2a_dense_assembly.py \
        --label workstation --output /tmp/m2a-dense.json
    python benchmarks/m2a_dense_assembly.py --label smoke --smoke
    python benchmarks/m2a_dense_assembly.py \
        --label float64-calibration --dtype float64 --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import better_robot as br
from better_robot.costs import CostStack
from better_robot.data_model.model import Model
from better_robot.optim import Problem, ResidualItem, RobotConfig, RobotStateProvider, VarSpec
from better_robot.optim.problem import LeastSquaresProblem
from better_robot.residuals.base import ResidualState
from better_robot.residuals.pose import PoseResidual


DEFAULT_BATCHES = (1, 16)
DEFAULT_WARMUP = 10
DEFAULT_SAMPLES = 50
DEFAULT_AD_WARMUP = 1
DEFAULT_AD_SAMPLES = 5
POSE_WEIGHT = 1.7
POSITION_WEIGHT = 1.25
ORIENTATION_WEIGHT = 0.75


@dataclass(frozen=True)
class _DraftPoseResidual:
    """Expose the exact legacy pose residual through the draft protocol."""

    model: Model
    legacy: PoseResidual
    name: str = "pose"
    reads: tuple[str, ...] = ("q", "data")
    dim: int = 6

    def _state(self, ctx: Mapping[str, Any]) -> ResidualState:
        return ResidualState(model=self.model, data=ctx["data"], variables=ctx["q"])

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return self.legacy(self._state(ctx))

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        block = self.legacy.jacobian(self._state(ctx))
        if block is None:  # pragma: no cover - PoseResidual is analytic
            raise RuntimeError("PoseResidual unexpectedly returned no analytic Jacobian")
        return {"q": block}


@dataclass(frozen=True)
class _Systems:
    draft: Problem
    legacy_fresh: LeastSquaresProblem
    legacy_cached: LeastSquaresProblem
    q: torch.Tensor
    target: torch.Tensor
    frame_name: str


def _percentiles(samples: list[float]) -> tuple[float, float]:
    if len(samples) == 1:
        return samples[0], samples[0]
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return quartiles[0], quartiles[2]


def _summarize(samples: list[float]) -> dict[str, float]:
    q1_ms, q3_ms = _percentiles(samples)
    return {
        "median_ms": statistics.median(samples),
        "q1_ms": q1_ms,
        "q3_ms": q3_ms,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "mean_ms": statistics.fmean(samples),
    }


def _measure(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    samples: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    start_ns = time.perf_counter_ns()
    first_value = function()
    first_call_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

    for _ in range(warmup):
        function()
    timings_ms: list[float] = []
    for _ in range(samples):
        start_ns = time.perf_counter_ns()
        function()
        timings_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000.0)
    return first_value, {
        "first_call_ms": first_call_ms,
        "steady_state_ms": _summarize(timings_ms),
    }


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    return (2e-4, 2e-3) if dtype == torch.float32 else (2e-9, 2e-7)


def _ad_tolerances() -> tuple[float, float]:
    # This is an analytic-vs-AD algorithm check, not merely a roundoff check.
    # Match the plan's fp32 Jacobian tolerance for both working dtypes while
    # retaining max_abs_error in the report for higher-precision inspection.
    return 2e-4, 2e-3


def _parity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    shape_match = tuple(actual.shape) == tuple(expected.shape)
    dtype_match = actual.dtype == expected.dtype
    device_match = actual.device == expected.device
    if shape_match:
        actual64 = actual.detach().to(dtype=torch.float64)
        expected64 = expected.detach().to(dtype=torch.float64)
        difference = (actual64 - expected64).abs()
        threshold = atol + rtol * expected64.abs()
        numeric_passed = bool(torch.all(difference <= threshold))
        max_abs_error: float | None = float(difference.max())
    else:
        numeric_passed = False
        max_abs_error = None
    return {
        "passed": numeric_passed and shape_match and dtype_match and device_match,
        "numeric_passed": numeric_passed,
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "device_match": device_match,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "actual_device": str(actual.device),
        "expected_device": str(expected.device),
        "max_abs_error": max_abs_error,
        "atol": atol,
        "rtol": rtol,
    }


def _build_systems(model: Model, batch: int) -> _Systems:
    frame_name = "body_panda_hand" if "body_panda_hand" in model.frame_name_to_id else model.frame_names[-1]
    frame_id = model.frame_id(frame_name)

    target_q = model.q_neutral.clone()
    target = br.forward_kinematics(model, target_q, compute_frames=True).frame_pose_world[frame_id].clone()
    perturbation = torch.linspace(
        -0.04,
        0.04,
        model.nv,
        dtype=target_q.dtype,
        device=target_q.device,
    )
    query = model.integrate(target_q, perturbation)
    q = query.unsqueeze(0).expand(batch, -1).contiguous()

    pose = PoseResidual(
        frame_id=frame_id,
        target=target,
        pos_weight=POSITION_WEIGHT,
        ori_weight=ORIENTATION_WEIGHT,
    )
    stack = CostStack()
    stack.add("pose", pose, weight=POSE_WEIGHT)

    def state_factory(value: torch.Tensor) -> ResidualState:
        data = br.forward_kinematics(model, value, compute_frames=True)
        return ResidualState(model=model, data=data, variables=value)

    legacy_kwargs = {
        "cost_stack": stack,
        "state_factory": state_factory,
        "x0": q,
        "nv": model.nv,
        "retract": model.integrate,
    }
    draft = Problem(
        vars=(VarSpec(name="q", shape=(model.nq,), manifold=RobotConfig(model)),),
        residuals=(
            ResidualItem(
                name="pose",
                residual=_DraftPoseResidual(model=model, legacy=pose),
                weight=POSE_WEIGHT,
            ),
        ),
        providers=(RobotStateProvider(model),),
    )
    return _Systems(
        draft=draft,
        legacy_fresh=LeastSquaresProblem(**legacy_kwargs),
        legacy_cached=LeastSquaresProblem(**legacy_kwargs),
        q=q,
        target=target,
        frame_name=frame_name,
    )


def _ad_calibration(
    systems: _Systems,
    *,
    strategy: str,
    warmup: int,
    samples: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    # AD calibration is deliberately unbatched. Whole-batch AD materializes
    # output-batch × input-batch axes before extracting their diagonal, so a
    # batched run would confound transform calibration with O(B²) workspace.
    q = systems.q[0].clone()
    expected = systems.legacy_fresh.jacobian(q.clone())
    function = lambda: systems.draft.dense_jacobian({"q": q.clone()}, strategy=strategy)  # noqa: E731
    try:
        actual, timing = _measure(function, warmup=warmup, samples=samples)
    except (RuntimeError, TypeError, ValueError) as exc:
        return {
            "strategy": strategy,
            "status": "error",
            "batch_shape": [],
            "error_type": type(exc).__name__,
            "error": " ".join(str(exc).splitlines()),
        }
    return {
        "strategy": strategy,
        "status": "measured",
        "batch_shape": [],
        "input_policy": "fresh q.clone() per invocation; clone cost is included",
        "provider_policy": "evaluation-local RobotStateProvider; FK runs per invocation",
        "parity_vs_legacy_analytic": _parity(actual, expected, atol=atol, rtol=rtol),
        **timing,
    }


def _validate_settings(
    batches: tuple[int, ...],
    warmup: int,
    samples: int,
    ad_warmup: int,
    ad_samples: int,
) -> None:
    if not batches or any(batch <= 0 for batch in batches):
        raise ValueError("batches must contain positive integers")
    if warmup < 0 or ad_warmup < 0:
        raise ValueError("warmup counts must be non-negative")
    if samples <= 0 or ad_samples <= 0:
        raise ValueError("sample counts must be positive")


def run(  # noqa: PLR0913, PLR0915 - one report owns the complete reproducibility metadata
    *,
    label: str,
    batches: tuple[int, ...],
    warmup: int,
    samples: int,
    dtype: torch.dtype,
    calibrate_ad: bool,
    ad_warmup: int,
    ad_samples: int,
) -> dict[str, object]:
    _validate_settings(batches, warmup, samples, ad_warmup, ad_samples)
    try:
        from robot_descriptions import panda_description  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - command-line guard
        raise SystemExit("install the 'demos' extra to run the Panda M2a benchmark") from exc

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = br.load(panda_description.URDF_PATH, dtype=dtype)
    atol, rtol = _tolerances(dtype)
    ad_atol, ad_rtol = _ad_tolerances()
    measurements: list[dict[str, object]] = []
    parity_checks: list[dict[str, object]] = []
    ratios: list[dict[str, object]] = []
    calibration_systems: _Systems | None = None

    for batch in batches:
        systems = _build_systems(model, batch)
        if calibration_systems is None:
            calibration_systems = systems

        # Both full-evaluation cases clone q inside the timed call.  This
        # prevents the legacy identity cache from skipping FK and charges the
        # same input-allocation overhead to the draft call.
        legacy_value, legacy_timing = _measure(
            lambda: systems.legacy_fresh.jacobian(systems.q.clone()),
            warmup=warmup,
            samples=samples,
        )
        legacy_measurement = {
            "case": "legacy_fresh_input",
            "batch_shape": [batch],
            "input_policy": "fresh q.clone() per invocation; clone cost is included",
            "provider_policy": "LeastSquaresProblem state cache misses; FK runs per invocation",
            **legacy_timing,
        }
        measurements.append(legacy_measurement)

        cached_value, cached_timing = _measure(
            lambda: systems.legacy_cached.jacobian(systems.q),
            warmup=warmup,
            samples=samples,
        )
        cached_measurement = {
            "case": "legacy_same_tensor_cached",
            "batch_shape": [batch],
            "input_policy": "same tensor identity and mutation version per invocation",
            "provider_policy": "first call runs FK; steady-state calls reuse the one-entry state cache",
            **cached_timing,
        }
        measurements.append(cached_measurement)

        draft_value, draft_timing = _measure(
            lambda: systems.draft.dense_jacobian({"q": systems.q.clone()}, strategy="auto"),
            warmup=warmup,
            samples=samples,
        )
        draft_measurement = {
            "case": "draft_dense_auto_analytic",
            "batch_shape": [batch],
            "input_policy": "fresh q.clone() per invocation; clone cost is included",
            "provider_policy": (
                "one evaluation-local RobotStateProvider context; the analytic "
                "Jacobian path runs one vectorized FK for the full batch"
            ),
            "assembly": "preallocated dense fill from the residual's analytic q block",
            **draft_timing,
        }
        measurements.append(draft_measurement)

        analytic_parity = _parity(draft_value, legacy_value, atol=atol, rtol=rtol)
        cached_parity = _parity(cached_value, legacy_value, atol=atol, rtol=rtol)
        parity_checks.append(
            {
                "batch_shape": [batch],
                "draft_vs_legacy_fresh": analytic_parity,
                "legacy_cached_vs_legacy_fresh": cached_parity,
            }
        )

        draft_median = draft_timing["steady_state_ms"]["median_ms"]
        legacy_median = legacy_timing["steady_state_ms"]["median_ms"]
        cached_median = cached_timing["steady_state_ms"]["median_ms"]
        ratios.append(
            {
                "batch_shape": [batch],
                "draft_over_legacy_fresh_median": draft_median / legacy_median,
                "draft_over_legacy_cached_median": draft_median / cached_median,
            }
        )

    ad_calibration: list[dict[str, object]] = []
    if calibrate_ad:
        if calibration_systems is None:  # pragma: no cover - settings reject empty batches
            raise RuntimeError("no system available for AD calibration")
        for strategy in ("jacrev", "jacfwd"):
            ad_calibration.append(
                _ad_calibration(
                    calibration_systems,
                    strategy=strategy,
                    warmup=ad_warmup,
                    samples=ad_samples,
                    atol=ad_atol,
                    rtol=ad_rtol,
                )
            )

    required_parity_passed = all(check["draft_vs_legacy_fresh"]["passed"] for check in parity_checks)
    findings = [
        "Legacy same-tensor steady state reuses FK; compare draft against legacy_fresh_input for full-evaluation cost.",
        "Ratios are same-host observations from this run, not acceptance thresholds or CI gates.",
    ]
    for item in ratios:
        batch = item["batch_shape"][0]
        ratio = item["draft_over_legacy_fresh_median"]
        findings.append(f"Observed draft/legacy-fresh median ratio at batch {batch}: {ratio:.3f}x.")
        if batch > 1 and ratio > 1.0:
            findings.append(
                "The analytic draft and legacy paths both evaluate the batch "
                f"vectorized at batch {batch}; this ratio includes protocol, "
                "context, and dense-fill overhead rather than a Python batch loop."
            )
    for item in ad_calibration:
        if item["status"] == "error":
            findings.append(
                f"Forced {item['strategy']} was not measurable for {dtype}: {item['error_type']}: {item['error']}"
            )
        elif not item["parity_vs_legacy_analytic"]["passed"]:
            findings.append(f"Forced {item['strategy']} did not meet numeric parity tolerance.")

    first_system = calibration_systems
    if first_system is None:  # pragma: no cover - settings reject empty batches
        raise RuntimeError("no benchmark system was built")
    return {
        "schema_version": 1,
        "label": label,
        "advisory_only": True,
        "source": str(Path(br.__file__).resolve()),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": _cpu_name(),
            "logical_cpus": os.cpu_count(),
        },
        "software": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
        },
        "device": "cpu",
        "dtype": str(dtype),
        "torch_threads": torch.get_num_threads(),
        "model": {
            "name": "Franka Panda (robot_descriptions)",
            "urdf": str(panda_description.URDF_PATH),
            "nq": model.nq,
            "nv": model.nv,
            "frame": first_system.frame_name,
        },
        "problem": {
            "residual": "PoseResidual, target pose from q_neutral",
            "query": "model.integrate(q_neutral, linspace(-0.04, 0.04, nv))",
            "pose_position_weight": POSITION_WEIGHT,
            "pose_orientation_weight": ORIENTATION_WEIGHT,
            "cost_stack_or_residual_item_weight": POSE_WEIGHT,
            "same_model_q_target_and_weights": True,
        },
        "batch_shapes": [[batch] for batch in batches],
        "timing_protocol": {
            "clock": "time.perf_counter_ns wall clock",
            "case_order": [
                "legacy_fresh_input",
                "legacy_same_tensor_cached",
                "draft_dense_auto_analytic",
            ],
            "first_call_definition": (
                "first timed invocation of each case in one process; later cases may benefit "
                "from globally warmed Torch/Python kernels"
            ),
            "steady_state_warmup_iterations": warmup,
            "steady_state_samples": samples,
            "ad_batch_shape": [],
            "ad_warmup_iterations": ad_warmup if calibrate_ad else None,
            "ad_samples": ad_samples if calibrate_ad else None,
            "forced_ad_parity_tolerance": {
                "atol": ad_atol,
                "rtol": ad_rtol,
                "reason": "M2a plan fp32 analytic-vs-autodiff Jacobian tolerance",
            },
            "statistics": "median, inclusive Q1/Q3, min, max, and mean over wall-clock samples",
        },
        "required_parity_passed": required_parity_passed,
        "parity": parity_checks,
        "measurements": measurements,
        "ratios": ratios,
        "forced_ad_calibration": ad_calibration,
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--calibrate-ad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="measure forced unbatched jacrev/jacfwd; errors are recorded in JSON",
    )
    parser.add_argument("--ad-warmup", type=int, default=DEFAULT_AD_WARMUP)
    parser.add_argument("--ad-samples", type=int, default=DEFAULT_AD_SAMPLES)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="bound work to batch=1, warmup=1, samples=2, AD warmup=0, AD samples=1",
    )
    args = parser.parse_args()
    if args.smoke:
        batches = (1,)
        warmup = 1
        samples = 2
        ad_warmup = 0
        ad_samples = 1
    else:
        batches = tuple(args.batches)
        warmup = args.warmup
        samples = args.samples
        ad_warmup = args.ad_warmup
        ad_samples = args.ad_samples

    report = run(
        label=args.label,
        batches=batches,
        warmup=warmup,
        samples=samples,
        dtype=_dtype_from_name(args.dtype),
        calibrate_ad=args.calibrate_ad,
        ad_warmup=ad_warmup,
        ad_samples=ad_samples,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["required_parity_passed"]:
        raise SystemExit("draft analytic Jacobian failed required legacy parity")


if __name__ == "__main__":
    main()
