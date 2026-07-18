from __future__ import annotations

import importlib.util
import json
import math
import re
import statistics
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_benchmark_module() -> ModuleType:
    benchmark_path = (
        Path(__file__).parents[2] / "benchmarks" / "m1_warp_fk_vs_compiled_torch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "m1_warp_fk_vs_compiled_torch_benchmark",
        benchmark_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark_module()
_MEASUREMENT_COMMIT = "c0560e3c16ee974a2bf6a8d09c618b45a5311163"
_BASELINES = tuple(
    sorted(
        (Path(__file__).parent / "baselines").glob(
            "warp_fk_cuda_rtx6000_ada_b*.json"
        )
    )
)


def test_summary_retains_raw_samples_and_quartiles() -> None:
    samples = [1.0, 2.0, 3.0, 4.0]

    summary = BENCHMARK._summary(samples)

    assert summary == {
        "median_ms": 2.5,
        "q1_ms": 1.75,
        "q3_ms": 3.25,
        "iqr_ms": 1.5,
        "min_ms": 1.0,
        "max_ms": 4.0,
        "samples_ms": samples,
    }
    assert summary["samples_ms"] is not samples


def test_strict_cache_validation_requires_one_case_and_empty_distinct_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_cache = tmp_path / "torch-cache"
    warp_cache = tmp_path / "warp-cache"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(torch_cache))
    monkeypatch.setenv("WARP_CACHE_PATH", str(warp_cache))

    assert BENCHMARK._validate_fresh_case_caches(
        required=True,
        batches=(1,),
    )

    torch_cache.mkdir()
    (torch_cache / "compiled-artifact").write_text("cached")
    with pytest.raises(ValueError, match="must be absent or empty"):
        BENCHMARK._validate_fresh_case_caches(required=True, batches=(1,))

    with pytest.raises(ValueError, match="exactly one"):
        BENCHMARK._validate_fresh_case_caches(required=True, batches=(1, 64))


def test_unverified_first_call_is_not_reported_as_cache_cold() -> None:
    metadata = BENCHMARK._cold_timing_metadata(
        batches=(1, 64),
        fresh_case_caches_verified=False,
    )

    assert metadata["verified_cache_cold"] is False
    assert metadata["single_case_process"] is False
    assert "not-guaranteed-fresh" in metadata["cache_scope"]
    assert "dynamic-shape cache reuse" in metadata["limitation"]


def test_numeric_visible_device_token_maps_to_physical_gpu() -> None:
    inventory = [
        {
            "physical_index": 5,
            "uuid": "GPU-example",
            "pci_bus_id": "00000000:01:00.0",
            "name": "Example GPU",
            "driver_version": "999.0",
        }
    ]

    device, source = BENCHMARK._match_physical_device(
        logical_index=0,
        visible_token="5",
        device_uuid=None,
        inventory=inventory,
    )

    assert device == inventory[0]
    assert source == "numeric CUDA_VISIBLE_DEVICES token"


def test_committed_cuda_artifacts_have_clean_source_and_recomputable_statistics() -> None:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in _BASELINES]

    assert len(reports) == 4
    assert {report["batch_shapes"][0][0] for report in reports} == {1, 16, 256, 4096}
    commits = {report["git"]["commit"] for report in reports}
    assert commits == {_MEASUREMENT_COMMIT}
    assert re.fullmatch(r"[0-9a-f]{40}", _MEASUREMENT_COMMIT)

    for report in reports:
        assert report["schema_version"] == 2
        assert report["benchmark"] == "m1_warp_fk_vs_compiled_torch"
        assert report["git"]["dirty"] is False
        assert report["cold_timing"]["verified_cache_cold"] is True
        assert report["cold_timing"]["single_case_process"] is True
        assert report["device"]["physical_index"] == 4
        assert report["device"]["uuid"] == "GPU-ba51bb8b-da02-ea99-e99e-350952268322"
        assert report["device"]["nvidia_driver_version"] == "560.35.03"
        assert report["samples"] == 100
        assert len(report["measurements"]) == 1

        measurement = report["measurements"][0]
        assert measurement["max_abs_error"] == 0.0
        for lane in ("torch_compile_fullgraph", "warp_fused_fk"):
            summary = measurement[lane]
            samples = summary["samples_ms"]
            assert len(samples) == 100
            assert all(math.isfinite(sample) and sample >= 0.0 for sample in samples)
            q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive")
            assert summary["median_ms"] == statistics.median(samples)
            assert summary["q1_ms"] == q1
            assert summary["q3_ms"] == q3
            assert summary["iqr_ms"] == q3 - q1
