from __future__ import annotations

import importlib.util
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
