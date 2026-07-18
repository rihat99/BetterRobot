# M2a Results — Variable Blocks and Vertical Slice

**Status:** complete on branch `dev` (2026-07-17).

## Outcome

M2a now provides a frozen named-block evaluation API under
`better_robot.optim`: state manifolds and feasible retraction, reduced tangent
blocks, static residual/provider dependencies, scalar first-order objectives,
per-block AD, dense assembly, and explicit external tensor parameters. The
permanent BVR-shaped synthetic slice runs end to end with all seven required
ingredients before the public freeze.

The owner confirmed both evidence-gated decisions:

- Keep scalar `ObjectiveItem`s for first-order phases. `Problem.require_least_squares`
  and the legacy `optim.solve` boundary strictly reject them for GN/LM.
- Adopt the recommended differentiation contract: detached solves by default,
  explicit future implicit differentiation to declared external tensors,
  no silent gradients for invalid terminal states, and only named/tested
  higher-order guarantees.

## Delivered

- `Bounds`, `Euclidean`, `SO3Manifold`, `SE3Manifold`, `RobotConfig`,
  `VarSpec`, `Values`, `Problem`, residual/objective items, provider DAGs,
  `RobotStateProvider`, tangent autograd, and `detach_values`.
- Strict shape, dtype, device, feasibility, dependency, tensor-weight, and
  graph-lifetime validation. Fixed coordinates are eliminated, not zeroed.
- Analytic / `jacrev` / `jacfwd` decision table, explicit graph-free central
  finite differences, dense deterministic `J` and `J.T @ J` assembly.
- Executable custom-residual guide, provider/invalidity conventions, dense-v1
  sparsity hand-off note, differentiation contract, and M2b/M5 cross-links.
- Reproducible Panda benchmark definition and a synthetic vertical slice with
  two blocks, shared NN provider, scalar term, masks, batching, and a manual
  root-to-full phase transition.

## Verification

| Check | Result |
|---|---|
| Final independent M2a audit | PASS; 141 focused tests and no blocker |
| Full non-Warp suite | 1,099 passed, 1 skipped, 43 warnings in 57.88 s |
| Fresh-cache Warp/layout suite | 12 passed, 2 CUDA skips, 1 warning |
| Public API contracts | 44 passed |
| M0 prerequisites | θ=0 gradient finite (`[0.5, 0.5, 0.5]`); `AUTODIFF` grep clean |
| M1 seam | no `backends/`; `ModelStructure` present |
| Lock and source hygiene | `uv lock --check`, scoped Ruff, and `git diff --check` pass |
| Documentation | offline warnings-as-errors HTML build passes; no generated `optim.blocks` pages |

No GPU result is claimed for this historical run: the default agent sandbox
did not expose the host's NVIDIA device nodes, so both CUDA tests skipped.
Later approved host-context M6 validation proved the GPU stack itself was
healthy. Online nitpicky docs remain blocked by four unavailable
external inventories and the pre-existing unresolved external-reference
baseline; the new guide and ordinary strict offline build are clean.

## Benchmark evidence

Command:

```bash
.venv/bin/python benchmarks/m2a_dense_assembly.py \
  --label m2a-final --batches 1 16 --warmup 10 --samples 30 \
  --ad-warmup 1 --ad-samples 3 --output /tmp/m2a-final.json
```

On one Intel Xeon Platinum 8570 CPU thread with Torch 2.13.0, fp32 analytic
assembly matched the legacy Jacobian exactly at B=1 and B=16. Median draft /
fresh-legacy ratios were 1.023x and 1.026x. Forced unbatched AD also passed:
`jacrev` max error 6.32e-6 (23.70 ms median) and `jacfwd` max error 6.68e-6
(129.63 ms median). These are advisory same-host observations, not CI gates.

## Deviations and deferred work

1. Public names are `SO3Manifold` / `SE3Manifold`, not the sketch's `SO3` /
   `SE3`, avoiding collision with the root Lie types. The 12 new symbols are
   qualified under `better_robot.optim`; the root API remains unchanged.
2. The `Manifold` protocol adds `project` and `validate_bounds`; trajectory
   event prefixes multiply tangent size. Both are required to make feasible
   retraction and `(T, nq)` events explicit.
3. Phase masks are static `VarSpec` structure. The slice performs the planned
   cheap `Problem` rebuild and uses tangent Adam buffers; the phase engine and
   matrix-free Adam remain M2c work.
4. Analytic Jacobians are vectorized once. Batched AD differentiates the full
   output batch against the full input batch and extracts its diagonal. This is
   correct for shared parameters and tensor weights but has an O(B²)
   intermediate; per-element transforms/vectorization are future optimization.
5. Provider caching is evaluation-local. A residual/objective/analytic-J call
   shares one context, while each missing AD `(residual, variable)` block owns
   a transform-local context and may rerun providers.
6. `ResidualSpec` was retained because legacy residual modules still expose it
   and `optim` exports it; its sparse/time semantics were copied to the M5 design note.
   Robust kernels/grouping are metadata only until M2b. Per-element invalidity
   is represented by NaN rows, not a separate validity mask.
7. Public value/manifold/bounds validation still uses tensor-to-Python
   predicates (`bool`, `allclose`). M2b must introduce a prevalidated,
   sync-free internal update path before claiming CUDA-graph readiness.
8. Torch 2.13 forward AD required two correctness repairs outside the new
   package: a typed half-angle constant in the Lie implementation and a
   neutral-safe planar angle reconstruction.
9. BVR and BHF were never modified. Their pre-existing dirty states remained
   (`experiments/` and a modified submodule in BVR; `scratch/` in BHF). No M2a
   symbol was removed. BHF still contains known pre-redesign `Data.oMi` imports
   from M1 migration debt; standing rule 1 defers consumer compatibility and
   migration on this dedicated branch.
10. CI was intentionally stopped at the owner's request: the committed CI
    workflow is manual-only, the remote workflow is `disabled_manually`, and
    queued/in-progress run counts were both zero. Remote CI is therefore not
    cited as verification evidence.
