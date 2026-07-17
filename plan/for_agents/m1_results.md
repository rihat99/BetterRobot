# M1 Results — Two-Lane Seam and Hygiene

- Date: 2026-07-17
- Branch: `dev`
- Pre-M1 commit: `649ab9f`
- Recovery branch: `experimental/dynamics-action` at `649ab9f`

## Outcome

M1 is implemented. BetterRobot now has a structure/value compute seam, pure
FK and dynamics pass boundaries, an opt-in fused Warp FK prototype, explicit
engineering and packaging contracts, a manual-only CI workflow, and
evaluation-local solver caching. The owner accepted the eager-CPU evidence
and selected Apache-2.0.

The ordinary Torch lane remains the default. Warp remains opt-in until the M6
GPU performance and default-on decision.

## Delivered

- Added frozen `ModelStructure`, pytree `ModelValues`, `ExecutionBatch`, joint
  kind dispatch tables, frame placement tables, and dual static/device
  topology consistency checks.
- Re-signed FK, RNEA, ABA, CRBA, and centroidal passes around pure raw results;
  public wrappers populate `Data` and preserve the existing user surface.
- Removed the 527-line backend registry/stub tree and renamed the direct Lie
  implementation to `lie/_impl.py`. A contract now pins the retired package's
  absence.
- Added zero-copy fp32/fp64 Warp pose-layout contracts, eligibility fallback,
  a functional custom forward op, FakeTensor/full-graph support, current-stream
  launches, and a fused FK+frames kernel.
- Added Warp parity and differentiation coverage for branches, free-flyer and
  spherical joints, shared values, singular configurations, and chains deeper
  than 16 joints. CUDA tests cover forward/q-gradient parity, non-default
  stream ordering, and graph replay.
- Added reproducible eager-CPU and Warp-vs-compiled-Torch benchmark definitions;
  CI uploads their JSON evidence.
- Enforced fp16/bf16 rejection and dtype/device mismatch errors at the FK pass
  boundary; fp32 and fp64 remain supported.
- Hoisted hot-path constants, broadened the AST hot-loop contract, and removed
  the remaining zero-DoF tensor construction from RNEA/ABA loops.
- Fixed programmatic builder support for helical joints, `JointModel`
  instances, and composites. Direct mimic remains rejected, with the
  owner-approved exact-identity exemption.
- Removed dead utilities, registries, stubs, action models, schema handshake,
  factory/legacy aliases, and stale contracts. Removed consumer-facing symbols
  are recorded in `docs/reference/m1_removed_symbols.md` for M4.
- Split optional extras (`viewer`, `io-mjcf`, `meshes`, `demos`, `warp`), added
  five CI jobs, regenerated `uv.lock`, and verified a fresh CPU-only core
  install.
- Added evaluation-local FK reuse in optimization; the five-iteration probe
  fell from 11 FK calls to 6 with invalidation coverage.
- Added the engineering, source-provenance, threading, serialization,
  differentiation, dtype, and compile-lifecycle contracts.

## Verification evidence

- Full suite: **992 passed, 3 skipped, 20 warnings in 54.75 s**.
- Focused parity/contracts: **306 passed**.
- Dynamics and Pinocchio parity after final hoisting: **118 passed**.
- Warp CPU after the final long-chain gradcheck: **12 passed, 2 CUDA-skipped**.
- CUDA validation on an RTX 6000 Ada: **2 passed in 6.38 s** for
  `tests/warp/test_fk_cuda.py`.
- Sphinx HTML build succeeded with four offline intersphinx inventory warnings.
- Targeted Ruff, workflow YAML parsing, `git diff --check`, and
  `uv lock --check` passed.
- The built wheel declares `License-Expression: Apache-2.0` and includes the
  complete license at `dist-info/licenses/LICENSE`.
- A fresh CPU-only environment installed `torch==2.13.0+cpu`; BetterRobot had
  exactly `torch`, `numpy`, and `yourdfpy` as direct core requirements and did
  not import optional modules eagerly.
- The real Inductor/Warp benchmark smoke produced exact parity
  (`max_abs_error == 0.0`). Its one-sample latency is smoke evidence only, not
  a production performance decision.

## Owner-reviewed eager-CPU benchmark

Same-host Intel Xeon Platinum 8570, Panda, fp32, one Torch thread, 20 warmups,
100 samples; values are medians.

| Operation | Batch | Pre-M1 | M1 | Change |
|---|---:|---:|---:|---:|
| FK | 1 | 4.573 ms | 3.242 ms | 29.1% faster |
| RNEA | 1 | 11.404 ms | 9.231 ms | 19.1% faster |
| FK | 64 | 4.805 ms | 3.511 ms | 26.9% faster |
| RNEA | 64 | 12.855 ms | 10.651 ms | 17.1% faster |

The owner accepted these results on 2026-07-17; no eager regression was
silently accepted.

## Deviations and open limitations

1. **The source-LOC target was not met.** Pre-M1 had 16,153 Python source
   lines; M1 has 16,627, a net increase of 474 rather than the requested
   reduction of at least 1,000. The tracked source diff removes 2,108 lines,
   but the required seam, bridge, Warp kernel, and renamed Lie implementation
   add 1,757 new-file lines plus pass changes. Meeting the numeric target would
   require deleting live M1 functionality.
2. **The historical baseline was stale.** The actual pre-M1 suite was
   940 passed and 1 skipped, not the plan's 897 passes.
3. **The paired backward custom op is not the active VJP.** Its schema and fake
   implementation exist, but Torch custom-op dispatch cannot record the Torch
   oracle inside that implementation. The registered forward autograd formula
   therefore recomputes directly through the Torch lane. Shared-value gradients
   reduce correctly through `index_select` autograd, not inside a Warp kernel.
   Immutable topology is read on the host in this prototype, so captured
   backward remains M6 work.
4. **The raw Jacobian surface still consumes `Model`/`Data`.** M1 re-signed the
   required FK and dynamics passes; the provider/Jacobian redesign remains M2a.
5. **Consumer import checks were intentionally not preserved.** The owner's
   branch-strategy override permits breaking legacy consumer imports until M4;
   the removed-symbol migration ledger replaces those checks.
6. **A literal no-`trimesh` core environment is impossible.** Required core
   parser `yourdfpy==0.0.60` declares `trimesh[easy]`. CI permits only that
   verified transitive edge and still proves BetterRobot does not import it.
7. **Executable documentation coverage is narrower than the broad wording in
   the plan.** CI executes the published front-page example and builds every
   Sphinx page, but it does not execute every Python fence.
8. **The CUDA caveat in the plan became stale.** The regenerated Torch/CUDA
   lock made the local RTX 6000 Ada runner usable, so M1 performed real CUDA
   validation instead of leaving T1.2 open. GPU benchmarking and default-on
   selection remain correctly deferred to M6.
9. **Warp CPU fixtures use an in-tree representative branched model rather
   than Panda** to keep the Warp-only CI job free of the demos extra. SMPL-like
   branching and the synthetic deep chain provide the remaining required
   topology coverage.
10. **Strict offline docs builds cannot resolve external intersphinx
    inventories.** The ordinary HTML build succeeds; the four warnings are
    network-resolution warnings, not source-document warnings.

## License and provenance decision

- License: Apache-2.0.
- Copyright: 2026 BetterRobot contributors.
- No voluntary `NOTICE` initially; required upstream notices will be added via
  the source ledger.
- Repository owner reviews provenance before the first licensed release.
- No candidate third-party source was ported during M1.
