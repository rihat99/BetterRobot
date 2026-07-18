"""Fast normal-suite checks for the definition-first M6 baseline harness."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "benchmarks" / "m6_baseline.py"
_FILTERED_RESULT = (
    _REPO_ROOT
    / "tests"
    / "bench"
    / "baselines"
    / "m6_torch_filtered_smpl_b1_rtx6000_ada.json"
)
_SPEC = importlib.util.spec_from_file_location("better_robot_m6_baseline", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
baseline = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = baseline
_SPEC.loader.exec_module(baseline)


def test_canonical_selector_matrix_is_complete_and_unique() -> None:
    cases = baseline._all_cases()
    case_ids = [case.case_id for case in cases]

    assert len(cases) == 3 * 3 * 2 * 2 * 4 == 144
    assert len(case_ids) == len(set(case_ids))
    assert {case.model for case in cases} == set(baseline.MODEL_NAMES)
    assert {case.operation for case in cases} == set(baseline.OPERATIONS)
    assert {case.device for case in cases} == set(baseline.DEVICES)
    assert {case.lane for case in cases} == set(baseline.LANES)
    assert {case.batch for case in cases} == set(baseline.CANONICAL_BATCHES)


@pytest.mark.parametrize(
    ("field", "noncanonical_value"),
    (
        ("seed", baseline.SEED + 1),
        ("dtype", "float64"),
        ("compile_backend", "eager"),
        ("threads", 2),
        ("cuda_index", 1),
    ),
)
def test_canonical_protocol_requires_every_measurement_policy_field(
    field: str,
    noncanonical_value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES",
        baseline.CANONICAL_CUDA_VISIBLE_DEVICES,
    )
    args = baseline._parser().parse_args(["--label", "canonical-check"])
    cases = baseline._all_cases()
    assert baseline._canonical_protocol(args, cases)

    setattr(args, field, noncanonical_value)
    assert not baseline._canonical_protocol(args, cases)


def test_canonical_protocol_requires_the_documented_visible_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = baseline._parser().parse_args(["--label", "canonical-check"])
    cases = baseline._all_cases()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", baseline.CANONICAL_CUDA_VISIBLE_DEVICES)
    assert baseline._canonical_protocol(args, cases)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert not baseline._canonical_protocol(args, cases)


def test_fixture_seed_and_inputs_exclude_lane_and_device() -> None:
    selectors = (
        baseline.CaseSpec("smpl", "fk", "cpu", "eager", 16),
        baseline.CaseSpec("smpl", "fk", "cpu", "compiled", 16),
        baseline.CaseSpec("smpl", "fk", "cuda", "eager", 16),
        baseline.CaseSpec("smpl", "fk", "cuda", "compiled", 16),
    )
    seeds = {baseline._case_seed(baseline.SEED, case) for case in selectors}
    assert len(seeds) == 1

    # Fixture generation is deliberately CPU-first.  Preparing both CPU
    # lanes proves that changing the lane cannot perturb persisted inputs;
    # the CUDA worker transfers this same host fixture after fingerprinting.
    eager = baseline._prepare_workload(
        selectors[0],
        dtype=torch.float32,
        cuda_index=0,
        seed=baseline.SEED,
        ik_iterations=1,
    )
    compiled = baseline._prepare_workload(
        selectors[1],
        dtype=torch.float32,
        cuda_index=0,
        seed=baseline.SEED,
        ik_iterations=1,
    )
    assert eager.details["input_seed"] == compiled.details["input_seed"]
    assert eager.details["inputs"] == compiled.details["inputs"]


def test_compiled_public_ik_is_explicitly_unsupported() -> None:
    case = baseline.CaseSpec("smpl", "ik", "cpu", "compiled", 16)
    result = baseline._run_worker(case, SimpleNamespace())

    assert result["status"] == "UNSUPPORTED"
    assert result["case_id"] == case.case_id
    assert result["selector"]["ik_mode"] == "independently_batched"
    assert "public solve_ik" in result["reason"]
    assert "no lower-level substitute" in result["reason"]


def test_numeric_visible_device_maps_to_physical_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    inventory: list[dict[str, object]] = [
        {
            "physical_index": 4,
            "uuid": "GPU-example",
            "pci_bus_id": "00000000:98:00.0",
            "name": "Example GPU",
            "driver_version": "560.0",
        }
    ]

    device, token, source = baseline._physical_cuda_device(
        0,
        device_uuid=None,
        inventory=inventory,
    )

    assert device == inventory[0]
    assert token == "4"
    assert source == "numeric CUDA_VISIBLE_DEVICES token"


def test_statistics_and_status_schema_helpers() -> None:
    summary = baseline._timing_summary([1.0, 2.0, 3.0, 4.0])
    assert summary == {
        "samples_ms": [1.0, 2.0, 3.0, 4.0],
        "median_ms": 2.5,
        "q1_ms": 1.75,
        "q3_ms": 3.25,
        "iqr_ms": 1.5,
        "min_ms": 1.0,
        "max_ms": 4.0,
    }

    case = baseline.CaseSpec("panda_fixed", "ik", "cuda", "compiled", 1)
    unsupported = baseline._unsupported_result(case, "test reason")
    assert unsupported == {
        "status": "UNSUPPORTED",
        "case_id": "panda_fixed/ik/cuda/compiled/b1",
        "selector": {
            "model": "panda_fixed",
            "operation": "ik",
            "device": "cuda",
            "lane": "compiled",
            "batch": 1,
            "ik_mode": "single_unbatched",
        },
        "reason": "test reason",
    }
    assert baseline.SCHEMA_VERSION == 1


def test_smpl_cpu_eager_fk_smoke_subprocess(tmp_path: Path) -> None:
    output = tmp_path / "m6-smoke.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--label",
            "pytest-smoke",
            "--smoke",
            "--models",
            "smpl",
            "--operations",
            "fk",
            "--devices",
            "cpu",
            "--lanes",
            "eager",
            "--output",
            str(output),
        ],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == baseline.SCHEMA_VERSION
    assert report["canonical_protocol"] is False
    assert report["protocol"] == "smoke"
    assert report["selection"]["selected_count"] == 1
    assert report["selection"]["status_counts"]["SUCCESS"] == 1
    assert report["validation"]["all_evaluated_checks_passed"] is True

    case = report["cases"][0]
    assert case["case_id"] == "smpl/fk/cpu/eager/b1"
    assert case["status"] == "SUCCESS"
    assert case["output"]["all_finite"] is True
    assert math.isfinite(case["cold"]["first_call_ms"])
    assert case["cold"]["first_call_ms"] >= 0.0
    assert len(case["steady_state"]["samples_ms"]) == 1
    assert math.isfinite(case["steady_state"]["median_ms"])
    assert case["steady_state"]["median_ms"] >= 0.0

    input_fingerprint = case["workload"]["inputs"]["q"]
    assert len(input_fingerprint["sha256"]) == 64
    assert math.isfinite(input_fingerprint["float64_sum_checksum"])
    assert case["output"]["tensor_fingerprints"]
    for fingerprint in case["output"]["tensor_fingerprints"]:
        assert len(fingerprint["sha256"]) == 64
        assert math.isfinite(fingerprint["float64_sum_checksum"])


def test_committed_filtered_result_is_measured_and_self_consistent() -> None:
    report = json.loads(_FILTERED_RESULT.read_text(encoding="utf-8"))

    assert report["schema_version"] == baseline.SCHEMA_VERSION
    assert report["protocol"] == "filtered"
    assert report["canonical_protocol"] is False
    cuda = report["environment"]["cuda"]
    assert cuda["available"] is True
    assert cuda["name"] == "NVIDIA RTX 6000 Ada Generation"
    assert cuda["physical_index"] == 4
    assert cuda["uuid"] == "GPU-ba51bb8b-da02-ea99-e99e-350952268322"
    assert cuda["nvidia_driver_version"] == "560.35.03"
    assert cuda["cuda_visible_devices"] == "4"
    assert report["selection"]["selected_count"] == 12
    assert report["selection"]["status_counts"] == {
        "ERROR": 0,
        "OOM": 0,
        "SUCCESS": 10,
        "TIMEOUT": 0,
        "UNSUPPORTED": 2,
    }
    assert report["validation"]["all_evaluated_checks_passed"] is True

    successes = [case for case in report["cases"] if case["status"] == "SUCCESS"]
    unsupported = [case for case in report["cases"] if case["status"] == "UNSUPPORTED"]
    assert len(successes) == 10
    assert len(unsupported) == 2
    assert all(len(case["steady_state"]["samples_ms"]) == 100 for case in successes)
    assert all(case["output"]["all_finite"] for case in successes)
    assert {case["selector"]["operation"] for case in unsupported} == {"ik"}
    assert {case["selector"]["lane"] for case in unsupported} == {"compiled"}
