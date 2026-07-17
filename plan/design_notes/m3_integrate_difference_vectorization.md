# M3 integrate/difference vectorization evidence

Date: 2026-07-17

## Decision

`Model.integrate` and `Model.difference` execute one tensor operation per
built-in manifold family: Euclidean, spherical, free-flyer, continuous
revolute, and planar. `ModelStructure` owns the device-local gather tables.
Only exact built-in classes enter those groups; composite and custom joints
retain per-joint dispatch. A reduced mimic target owns no public q/v slice and
is therefore absent from every group.

Leading dimensions use strict torch right-aligned broadcasting. The public
methods validate trailing q/v widths and model/input devices before gathering.
Inputs of different floating dtypes are promoted with `torch.promote_types`.

## Parity evidence

The oracle is the pre-M3 per-joint loop copied into
`tests/data_model/test_integrate_vectorized.py`. Coverage includes Panda, G1,
the 24-joint SMPL-like model, mixed scalar/translation/manifold joints,
composite/custom fallback, reduced mimic coordinates, `(B, T, *)` broadcasting,
and the theta-zero gradient path.

Grouped contiguous reductions are not promised to be bit-identical to strided
per-joint kernels. On the benchmark input below, maximum absolute differences
were:

| dtype | integrate | difference |
|---|---:|---:|
| float32 | 0 | 1.49e-8 |
| float64 | 0 | 2.78e-17 |

Tests therefore use tight dtype-calibrated `assert_close`, not `torch.equal`.
This is the only deviation from T3.7's literal “bit-for-parity” wording; the
right/local perturbation formulas are unchanged.

## CPU benchmark

Harness: `tests/bench/bench_integrate_difference.py`

Command:

```bash
.venv/bin/pytest tests/bench/bench_integrate_difference.py \
  -m bench --benchmark-only -q
```

Environment and protocol:

- Intel Xeon Platinum 8570, x86_64
- PyTorch 2.13.0+cu126, CPU, one torch thread
- float32 SMPL-like model: one free-flyer plus 23 spherical joints
- q shape `(200, 99)`, v shape `(200, 75)`
- 10 warmup rounds, 50 measured rounds, one call per round
- statistic: pytest-benchmark median

| operation | loop median | grouped median | speedup |
|---|---:|---:|---:|
| integrate | 4.8074 ms | 1.3391 ms | 3.59x |
| difference | 5.3164 ms | 1.4981 ms | 3.55x |

The benchmark is advisory and records both implementations as separate
distributions. It has no bare ratio gate; M6 should compare any proposed Warp
kernel against this grouped implementation, not the retired loop.
