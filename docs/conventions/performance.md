# Performance

Performance claims need a workload, hardware, dtype, batch shape, and raw
measurements. The tables below are engineering targets, not promises that
every release or machine already meets them. Current benchmarks are advisory.

## Tracked targets

### Panda-class arm, CUDA fp32

| Operation | Target |
|---|---|
| FK, `B=1` | at most 150 microseconds |
| FK, `B=1024` | at most 250 microseconds |
| joint Jacobians, `B=1` | at most 300 microseconds |
| cached frame Jacobian | at most 50 microseconds |
| pose-only IK, 30 iterations, `B=1` | at most 8 milliseconds |

### Floating-base humanoid, CUDA fp32

| Operation | Target |
|---|---|
| FK, `B=1` | at most 400 microseconds |
| whole-body IK, 60 iterations, `B=1` | at most 25 milliseconds |

### CPU and memory

| Quantity | Target |
|---|---|
| Panda FK, one CPU thread, `B=1` | at most 1 millisecond |
| Panda IK, one CPU thread, 30 iterations | at most 50 milliseconds |
| Panda model tensors | at most 50 KiB |
| FK data for `B=1024`, 14 joints | at most 5 MiB |
| batched IK working set for `B=1024` | at most 200 MiB |

The CPU timing target uses the documented measurement host. First-call
compilation, file parsing, mesh loading, and viewer drawing are measured
separately.

## Where speed comes from

### Leading batches

Tensor functions accept `(B..., event)` and loop over robot topology rather
than batch elements. One configuration and thousands of configurations use
the same source. See {doc}`/concepts/the_compute_seam` and
{doc}`/getting_started/05_batched_gpu`.

### Static topology and tensor values

`ModelStructure` stores topology in Python tuples and flat device tables.
`ModelValues` stores differentiable tensors. Raw passes receive both
explicitly, which makes tensor inputs visible to compilation and fused
implementations.

### Derivatives chosen for the path

Analytic Jacobians are valuable when a formula is central, stable, and cheaper
than building an automatic-differentiation graph. Automatic differentiation
is the default tool for less common residuals and for derivative helpers built
from a differentiable forward pass. Finite differences remain a debugging
comparison, not a production speed path.

Every speed claim about one choice needs a committed benchmark. “Analytic” or
“compiled” by itself is not evidence.

### Dense and temporal least squares

Dense Cholesky is the correctness path for LM and Gauss--Newton. A problem
with one declared temporal variable and compatible temporal residuals can use
block-banded normal storage. The optimizer never infers structure from numerical
zeros, and an ineligible problem falls back to dense work.

`TorchOptimizer` differentiates the scalar objective and lets `torch.optim`
update persistent tangent buffers; it does not assemble a dense Jacobian.

## Compilation

BetterRobot does not decorate public functions with `torch.compile`.
`forward_kinematics_raw` is tested with an explicit caller-side
`torch.compile(..., fullgraph=True)` wrapper. Model creation, `Data`
allocation, and public optimizer loops remain eager.

PyTorch may specialize on shape, dtype, device, layout, topology, and compiler
options. Cold compilation belongs in a cold-start measurement; do not hide it
inside a warmed timing.

Set `TORCHINDUCTOR_CACHE_DIR` when a measurement needs an explicit compiler
cache. BetterRobot does not choose a cache directory for the application.

## Optional Warp FK

Warp replaces one complete FK pass, not individual Lie operations. The public
boundary still accepts and returns Torch tensors. The Torch pass remains the
default and numerical reference.

The fused FK implementation has CUDA forward and gradient parity tests and is
selected explicitly. It is not the default because its backward recomputes
the Torch reference and that complete forward-plus-backward cost has not been
measured. Every future fused pass needs its own eligibility, parity, gradient,
stream, and benchmark evidence.

## CUDA graphs

No public captured optimizer loop ships. LM's private fixed-shape iteration is
not enough by itself: capture also needs stable storage, warmup, invalidation,
replay parity, and a forward/backward lifetime. An experimental caller owns
those pieces until a complete path is implemented and measured.

## Hot-path source rules

`tests/contract/test_hot_path_lint.py` checks selected kinematics, dynamics,
Lie, residual, and optimization files.

| Pattern | Why it is risky | Preferred response |
|---|---|---|
| `.item()` or `.cpu()` | synchronizes device work | keep the decision in tensors or move it to an eager boundary |
| `float(tensor)`, `bool(tensor)`, or selected `int(tensor)` | turns device state into Python control flow | keep per-element state tensorized |
| `.new_tensor(...)` inside a call | repeats allocation and conversion | reuse or hoist static data |
| allocation inside a topology loop | repeats work for every joint | preallocate or form the batched tensor once |
| rank-specific `if x.dim()` branches | creates separate scalar and batched paths | use the leading-batch convention |

A necessary eager decision may use `# bench-ok: <reason>`. The comment is an
explanation for review, not a blanket exemption.

## Measuring a change

The benchmark definitions live under `tests/bench/`. Before quoting a
number:

1. use the committed workload without making it easier;
2. record the source commit and dependency versions;
3. record CPU/GPU identity, dtype, robot, and batch;
4. synchronize accelerator work correctly;
5. state warmup and cold-cache policy; and
6. keep raw samples, not only the fastest value.

The current baseline scaffold is not a blocking comparison. Hardware-specific
artifacts prove only the exact case they record. A comparison becomes required
only after stable runner variance and an explicit repository change enable it.

## Ownership by package

| Area | Main performance responsibility |
|---|---|
| `lie/` | closed-form batched tensor math |
| `data_model/` | static topology, tensor values, and broadcast maps |
| `kinematics/` | FK and analytic spatial Jacobians |
| `dynamics/` | differentiable rigid-body recursions |
| `residuals/` | fixed output rows and analytic blocks where useful |
| `optim/` | dense or declared temporal linearization |
| `collision/` | no computation claim; current operations are unfinished |
| `io/` | readability; parsing is not a hot loop |
| `viewer/` | responsive scene updates |

## Review checklist

A performance change is ready for review when it answers:

- Which benchmark and number changed?
- Is numerical accuracy unchanged within the stated tolerance?
- Does the result cover both forward and backward work when gradients matter?
- What happens when batch, dtype, device, or topology changes?
- Is memory unchanged or better?
- Which test prevents the optimization from drifting away?
