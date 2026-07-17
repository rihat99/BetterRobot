# Audit: Cross-Cutting Quality — Viewer, Utils, Tests, Docs Drift, Perf Smells, Dead Code

Auditor: read-only quality subagent. Date: 2026-07-16.
Machine: Intel Xeon Platinum 8570 (224 threads), torch 2.11.0+cu130, **CUDA unusable on this box** (driver 12060 too old for the installed wheel) — all measurements CPU.

---

## 1. Scope & method

- Ran the full test suite once (`uv run pytest tests/ -q`).
- Grep-driven hunt across all of `src/` for hot-path smells (`.item()`, `.cpu()`, `float()/bool()` on tensors, per-call constant tensors, device transfers, Python loops over batch dims, `.clone()`, quaternion-normalize spam), then read the hot files end-to-end: `kinematics/forward.py`, `kinematics/jacobian.py`, `lie/_torch_native_backend.py`, `lie/se3.py`, `optim/optimizers/levenberg_marquardt.py`, `optim/problem.py`, `optim/state.py`, `costs/stack.py`, `residuals/pose.py`, `tasks/ik.py`, `data_model/model.py`, `dynamics/rnea.py`, `viewer/visualizer.py`, backends.
- Micro-benchmarked FK (Panda, batch 1024 and batch 1), FK+frames, joint Jacobians, one `solve_ik`, and instrumented FK-call counts inside a solve. Profiled batch-1 FK with cProfile.
- Empirically tested batched `solve_ik` and the finite-difference Jacobian fallback with batched inputs.
- Inventoried every `NotImplementedError` site (107 raise sites), cross-checked against `docs/reference/roadmap.md`'s completeness claim, and checked import reachability of `utils/`, registries, `ResidualSpec`, exceptions.
- Sampled 10 claims from `CLAUDE.md` / `docs/index.md` / `docs/getting_started/` / `docs/concepts/` / `docs/conventions/` and verified each against code or a live run.
- Reviewed all 15 contract tests in `tests/contract/`, `pyproject.toml`, `.pre-commit-config.yaml`.

**Raw numbers**

| Measurement | Result |
|---|---|
| Test suite | **897 passed, 1 skipped, 52.15 s** (wall 1m04s, `-q`) |
| FK, Panda (njoints=14, nq=9), B=1024, CPU | 6.79 ms/call (≈6.6 µs/sample) |
| FK, B=1, CPU | **3.96 ms/call** (per-call Python overhead dominates) |
| FK + frames, B=1024 | 13.0 ms/call (frames loop ≈ doubles cost) |
| FK + joint Jacobians, B=1024 | 16.2 ms/call |
| raw `se3_compose`, B=1024 | 209 µs; via `se3.py` facade: 211 µs (facade overhead ≈1 µs — negligible) |
| `solve_ik` Panda, 1 target, LM, converged in 5 iters | **108 ms/solve**; **11 `forward_kinematics_raw` calls** for those 5 iterations |
| cProfile FK+frames B=1 | 41 `se3_compose` calls/FK; `_quat_mul` alone = 45% of runtime; `update_frame_placements` = 33% |

For reference, the library's own budgets (`docs/conventions/performance.md §1.2`, CPU targets): FK B=1 ≤ 1 ms (measured 4 ms), `solve_ik` Panda 30 iters ≤ 50 ms (measured 108 ms for **5** iters). The budget hardware (Threadripper 5995WX) is the same class as this Xeon; the shortfall (≈4×–13×) is beyond hardware variance.

---

## 2. Confirmed problems

### 2.1 CRITICAL — Batched `solve_ik` crashes; the whole optim stack is single-problem-only

- Evidence: `optim/state.py:95` — `residual_norm=0.5 * (r0 @ r0)` assumes a 1-D residual. Live repro: `solve_ik(model, {...}, initial_q=q_batch_4)` → `RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x33 and 4x33)`.
- Same pattern throughout the optimizers: `levenberg_marquardt.py:110` `float(0.5 * (r_new @ r_new).sum())`, `:116` `float(-delta_v @ Jtr - …)`, `:126` `float(Jtr.norm())`; `gauss_newton.py:65,69`; `lbfgs.py:114,118,134,152`; `adam.py:96,100`. Even if `state.py:95` were fixed, LM's scalar `cost`, scalar `damping`, and single accept/reject decision would couple all batch items (a curobo/pyroki-style batched LM needs per-problem damping and per-problem accept masks).
- Why it matters: "Batched by default … no unbatched mode" is the project's headline claim (`CLAUDE.md` Batching Rules; `SolverState` docstring at `optim/state.py:36-40` explicitly documents `(B..., nx)` / `(B..., dim)` shapes). The flagship API violates the flagship claim, and **no test covers batched IK** (only `tests/tasks/test_trajectory.py` mentions batching under `tests/tasks/`).
- Severity: **Critical** (correctness + false advertising + the single biggest perf lever on GPU is batch IK).

### 2.2 HIGH — Finite-difference Jacobian fallback: not batch-aware and brutally expensive

- Evidence: `kinematics/jacobian.py:263-284`. The AUTO fallback runs `2·nv + 1` full FK-with-frames evaluations per Jacobian (`_fn` calls `forward_kinematics(..., compute_frames=True)` each time), in a Python loop with `v0.clone()` per column (`:281-282`).
- Batch bug: `dim = r0.numel()` (`:275`) flattens batch dims into the residual dimension, and `J[:, i] = (_fn(v_p) - _fn(v_m))/(2eps)` crashes for batched states — live repro: `residual_jacobian(NoJacResidual, batched_state)` → `RuntimeError: expand(torch.FloatTensor{[4, 3]}, size=[12])`.
- Why it matters: any user residual without an analytic Jacobian (exactly the "project-specific residual" use case the user cares about) silently falls into this path: for a G1 humanoid (nv=35+) that is ~70 FK calls per cost item per iteration, and it cannot run batched at all.
- Severity: **High**.

### 2.3 HIGH — Redundant FK per optimizer iteration; no state caching between residual and Jacobian

- Evidence: `optim/problem.py:42-60` — `problem.residual(x)` and `problem.jacobian(x)` each call `state_factory(x)`, and `tasks/ik.py:212-214` makes the state factory a full `forward_kinematics(..., compute_frames=True)`. Instrumented run: **11 FK calls for a 5-iteration LM solve** (~2.2/iter; jacobian-FK + trial-residual-FK, plus the accepted point's FK is recomputed next iteration for the Jacobian at the same `x`).
- Why it matters: FK is the dominant cost of an IK iteration; a memo of the last `(x → state)` pair (what Pinocchio's Model/Data split exists for!) halves the FK count for free. Ironically the library has the `Data`-cache machinery (`KinematicsLevel`) but the optimizer layer never reuses a `Data` across `residual`/`jacobian` at the same iterate.
- Severity: **High** (2× on the hot loop).

### 2.4 HIGH — Per-call constant tensor construction in pose residuals (the sibling of the fixed `so3_inverse` bug)

- Evidence: `residuals/pose.py:65` and `:102` — `weight = r.new_tensor([self.pos_weight] * 3 + [self.ori_weight] * 3)` builds a tensor **from a Python list on every residual and every Jacobian evaluation**. On CUDA this is a host→device copy with an implicit sync in the innermost loop — exactly the failure mode commit 93b8c03 just fixed in `so3_inverse` ("avoid per-call constant tensor (H2D copy + CUDA sync)").
- Also per-call: `self.target.to(device, dtype)` at `pose.py:59, 76, 129, 167, 180` (no-op after the first call only if target already lives on the right device, but still an op-dispatch per eval; for CPU targets + CUDA model it's an H2D copy **every iteration**).
- Severity: **High on GPU** (sync in inner loop), low on CPU.

### 2.5 MEDIUM — Host syncs and per-iteration constants in every optimizer loop

- `float(...)` / `bool(...)` on tensors are `.item()` in disguise; the hot-path lint (`tests/contract/test_hot_path_lint.py:66`) only bans literal `.item()`/`.cpu()`, so these pass:
  - LM: `levenberg_marquardt.py:86, 110, 116, 126` — ≥3 device syncs per iteration.
  - GN `gauss_newton.py:65,69`, LBFGS `lbfgs.py:83,87,114,118,134,152` (plus a Python line-search loop with a sync per trial), Adam `adam.py:96,100`.
- Per-iteration allocations the lint also misses (`torch.eye` is not in `ALLOC_FNS`, `test_hot_path_lint.py:27`):
  - `levenberg_marquardt.py:96` and `gauss_newton.py:53` — `torch.eye(nv)` rebuilt every iteration.
  - `levenberg_marquardt.py:103-107` — `problem.lower.to(x_new.device, x_new.dtype)` re-dispatched every accepted step.
- `_validate_q` (`kinematics/forward.py:64`) does `bool(((norm - 1.0).abs() > TOL).any())` — a guaranteed GPU→host sync per FK call for every free-flyer model. Worse, validation runs **twice per FK**: once in `forward_kinematics` (`forward.py:181`) and again inside `forward_kinematics_raw` (`forward.py:97`) which the backend calls.
- Lint watch-list blind spot: `WATCHED = ("kinematics", "dynamics", "optim/optimizers")` (`test_hot_path_lint.py:26`) — `residuals/`, `lie/`, `costs/`, `tasks/` are unwatched, which is where 2.4 lives.
- Severity: **Medium** (each sync is a pipeline stall on GPU; invisible on CPU).

### 2.6 HIGH — FK per-call Python overhead: 4 ms fixed cost, frames loop doubles it

- Evidence (profile, batch=1): 41 `se3_compose` calls per FK+frames; `_quat_mul` (`_torch_native_backend.py:44-52`: `unbind` + 16 scalar mults + `stack` ≈ 20 tiny kernels) is 45% of total time. FK B=1 = 3.96 ms vs the library's own ≤1 ms CPU budget; `solve_ik` = 108 ms/5 iters vs ≤50 ms/30 iters budget.
- `update_frame_placements` (`forward.py:211-216`): Python loop over all `nframes` with an in-place slice write and a per-frame `frame.joint_placement.to(device, dtype)` — this alone is one third of FK+frames time. All frame poses could be computed in **one** batched `se3_compose` (gather parent poses `(nframes,7)` × stacked local placements `(nframes,7)`), since frames depend only on joint poses, never on each other.
- Per-joint-per-call `.to()` spam inside the topo loop: `forward.py:112` (`model.joint_placements[j].to(...)`), `:116` (`T_j.to(dtype)`), `jacobian.py:66,187` (`S_local.to(...)`), and in dynamics `rnea.py:167` / `aba.py:121` / `crba.py:60` — `Inertia(model.body_inertias[i].to(device, dtype))._to_6x6()` rebuilds the 6×6 inertia from the 10-vector **per joint per call** (Pinocchio precomputes these once per Model/Data).
- The roadmap admits `torch.compile(fullgraph=True)` is "not yet applied" (`docs/reference/roadmap.md:84`), and no test ever compiles FK — so the "torch.compile friendliness" claims are untested.
- Severity: **High** relative to the library's stated budgets; the eager per-joint loop is the structural cause.

### 2.7 MEDIUM — `model.integrate` builds a Python list of per-joint slices + `torch.cat` per call

- Evidence: `data_model/model.py:160-174`. Called by `problem.step` every LM iteration and 2·nv times inside the FD fallback. For a revolute-only robot this could be a single `q + v`; the per-joint loop is only needed when a quaternion joint is present (and even then only for that slice).
- Severity: Medium.

### 2.8 Docs drift — 10 claims sampled, 6 materially wrong

| # | Claim | Reality | Evidence |
|---|---|---|---|
| 1 | `br.solve_ik(model, {"panda_hand": target})`, `result.frame_pose("panda_hand")` — front page, getting-started, CLAUDE.md, vision.md | **Raises `KeyError`.** Frames are body-prefixed: `body_panda_hand` (verified live: `model.frame_names`). Examples use the correct names (`examples/01_basic_ik.py:21`), docs don't. | `docs/index.md:36-39`, `docs/getting_started/inverse_kinematics.md:14,16,30,47`, `docs/getting_started/forward_kinematics.md:29`, `docs/concepts/vision.md:76-79`, `docs/concepts/batching_and_backends.md:75,260`, `docs/concepts/residuals_and_costs.md:325`, `CLAUDE.md:97-100` |
| 2 | `model.lower_pos_limit  # (nv,)` (CLAUDE.md:138-139) | Shape is **(nq,)** — verified live on free-flyer Panda: nq=16, nv=15, limits (16,). `model.py:65-66` comments agree with code, CLAUDE.md doesn't. | live run |
| 3 | Roadmap: "If a symbol is *not* on this page, it is implemented and tested" (`roadmap.md:8`) | False. All-stub symbols absent from the roadmap: `RobotCollision` (all 4 methods, `collision/robot_collision.py:48-73`), `collision/pairs.py:40`, `TrustRegion` (`optim/strategies/trust_region.py:16-22`), `SparseCholesky` (`optim/solvers/sparse_cholesky.py:15`), `costs/factory.factory` (`costs/factory.py:28`), `kinematics/chain.get_chain` (`chain.py:16`), `data_model/indexing.build_name_to_id` (`indexing.py:17`), `utils/broadcasting.flatten_batch`, `utils/testing.assert_close_manifold`, `spatial/force.py:47`. | grep of 107 `NotImplementedError` sites vs roadmap tables |
| 4 | "Batched by default … no unbatched mode" + `SolverState` documents `(B..., nx)` | Batched `solve_ik` crashes (§2.1). Also FK happily accepts unbatched `(nq,)` (bench `tests/bench/bench_forward_kinematics.py:15` relies on it) — the claim is wrong in both directions. | `optim/state.py:36-40,95` |
| 5 | CLAUDE.md:91 "`br.compute_joint_jacobians(model, data)` — fills `data.J`" | `data.J` is a *deprecated alias* the library's own naming contract (`tests/contract/test_naming.py`) bans from source; the field is `data.joint_jacobians`. | `data_model/data.py:54` |
| 6 | `tests/bench/README.md`: "the bench-cpu-advisory CI job records numbers", "self-hosted GPU runner", cites `docs/claude_plan/accepted/12_regression_and_benchmarks.md` | **There is no CI at all** — no `.github/` directory exists; the referenced `docs/claude_plan/` path doesn't exist. `baseline_cuda_l40.json` is checked in with no runner to produce it. | `ls -a .github` → absent |
| 7 | `data_model/indexing.py` docstring: "used by `Model` to build `joint_name_to_id`…" | Never imported; `io/build_model.py:473-475` builds the dicts inline. The function body raises `NotImplementedError`. | grep |
| 8 | `utils/testing.py` docstring: "helpers used across `tests_v2/`" | No `tests_v2/` exists anywhere; the helper raises `NotImplementedError` and nothing imports it. | grep |
| 9 | FD eps "1e-3 fp32 / 1e-7 fp64" (CLAUDE.md:74) | ✅ matches `jacobian.py:278`. | — |
| 10 | LM adaptive damping "starts 1e-4, doubles on reject, halves on accept" (CLAUDE.md:146); quaternion tol 10% (`contracts.md:74`); io depends only on data_model (CLAUDE.md:41) | ✅ all three verified (`adaptive.py`, `forward.py:32`, import grep). | — |

Bonus internal drift: `tests/contract/test_docstrings.py` module docstring says "25-symbol public API" while the test asserts 26 two functions later (`:3` vs `:30`).

- Severity: **High** for #1 (the first snippet a new user copies fails), Medium for the rest individually, High in aggregate: with ~7,200 lines of concepts+conventions docs against ~16 k LOC of src, the doc surface is large enough that drift is systemic, not incidental.

### 2.9 Dead / speculative code inventory

| Item | Evidence | Notes |
|---|---|---|
| **`utils/` package — 100% unreachable** | grep: zero imports of `better_robot.utils` anywhere in `src/`, `tests/`, `examples/` | 5 files, ~90 LOC; `broadcasting.flatten_batch` and `testing.assert_close_manifold` are stubs; `batching.batch_shape` and `logging.get_logger` are implemented but orphaned. CLAUDE.md lists `utils/ — batching, logging, broadcasting` as a real layer. |
| **`get_residual` registry lookup — never called** | exported at `residuals/__init__.py:20` and `br.register_residual` is in the frozen API; grep finds no caller of `get_residual` in src/tests/examples | Registry is write-only. Either it serves a planned serialization feature or it's speculative. |
| **`ResidualSpec` / `jacobian_blocks` — producers without consumers** | `optim/jacobian_spec.py`; `.spec` defined on `residuals/collision.py:61`, `temporal.py:119`; `problem.jacobian_blocks` (`problem.py:86-105`) called only by `tests/optim/test_matrix_free.py:92-101` | "Sparse-aware solvers can pre-build masks" — no solver reads `.spec` or calls `jacobian_blocks`. Speculative infrastructure with its own doc section. |
| **`collision/` — entire module is stubs, yet threaded through APIs** | `robot_collision.py:48,55,62,73`, `pairs.py:40`; `residuals/collision.py` (self/world collision residuals) also stubs; `solve_ik(robot_collision=...)` accepted and documented "unused in this version" (`ik.py:157,171`) | **`IKCostConfig.collision_margin` / `collision_weight` (`ik.py:46-47`) are never read anywhere** — dead config knobs. `residuals/collision.py:90` probes `robot_collision.link_indices`, an attribute `RobotCollision` doesn't even declare. |
| **`OptimizerConfig(damping="trust_region")` is a documented runtime trap** | `ik.py:67` offers it; `_make_damping_strategy` happily instantiates `TrustRegion`; first use raises `NotImplementedError` (`trust_region.py:16`) | A config enum value that always crashes. Same shape of problem: `retarget` sits in the **frozen 26-symbol public API** (`test_public_api.py`) while being a stub (`tasks/retarget.py:41`). |
| **Warp backend** | `backends/warp/` ~50 LOC of raise-only stubs + empty `kernels/`; `backends/__init__.py:55-62` can never reach a working path | Cost is low in LOC but it motivates the 3-Protocol indirection (see §2.10). |
| **Viewer stubs ≈ 420 LOC** | `recorder.py` (100), `offscreen_backend.py` (74), `overlays/{com,path_trace,residual_plot}.py` (122), `render_modes/collision.py` (45), 8 raise-only methods on `trajectory_player.py:96-121`, `camera.py` orbit/follow, `visualizer.record/add_robot/set_batch_index` | All raise with doc pointers. Roadmap does list these — consistent, but it's ~16% of the viewer package. |
| **5 of 12 exception classes never raised** | grep: `DtypeMismatchError`, `ConvergenceError`, `UnsupportedJointError`, `SingularityWarning` have zero raise sites (`BetterRobotError` base is fine) | `exceptions.py` is 154 lines for 12 classes; a third are aspirational. |
| **Deprecation shim for a library with zero released users** | `data_model/data.py:42-61` — 18 old→new aliases (`oMi`, `J`, `Ag`, …) installed via `setattr` (`:278-279`), plus `tests/contract/test_naming.py` (80 lines) and `test_deprecations.py` policing them | Pre-1.0, unreleased, no external consumers: the shim + two contract tests preserve compatibility with code that doesn't exist. |
| **`rich` dependency: required, never imported** | `pyproject.toml` deps; grep finds no `import rich` in `src/` | Dead dependency. |

### 2.10 Facade indirection: measurable? No. Justified? Questionable.

Every `se3.compose` travels `se3.py:_lie()` → `default_backend()` → dict lookup (`backends/__init__.py:47`) → `TorchNativeLieOps.se3_compose` → `_backend()` re-import (`lie_ops.py:19-21`, executes `from ...lie import _torch_native_backend` on **every op call**) → the real function. Measured overhead: ~1 µs/op — negligible against tensor-op cost, so this is **not** a perf problem today (4,100 facade transits in the 100-FK profile cost 19 ms cumulative ≈ 3%). It is, however, five layers to reach the only implementation that exists, and the only beneficiary (Warp) is an empty stub. Verdict: architecture-audit territory; from the perf side it's acquittal-with-a-warning.

### 2.11 Miscellaneous confirmed nits

- `solve_ik` force-casts to float32: `ik.py:186,188,221-222` (`.float()` on x0 and limits) — a float64 model silently gets optimized in fp32. Severity: Medium (silent precision change; contradicts the dtype-polymorphic `Model.to()` design).
- `tests/contract/test_docstrings.py:53` — `assert doc and "SolverState" not in doc.split("\n")[0] or "terminal" in (doc or "")` is trivially satisfiable due to `and`/`or` precedence; the assertion doesn't test what it thinks. Severity: Low.
- LM `state.history.append({...})` per iteration (`levenberg_marquardt.py:112`) — unbounded Python list with host-synced floats; fine for max_iter=100, wrong pattern for a compiled/batched future. Low.
- Quaternion-normalize spam: **not found** — normalization is confined to spherical/free-flyer joint models and `se3_normalize` call sites are appropriately sparse. (Good.)

---

## 3. Suspected problems needing verification

1. **Batched LM semantics post-fix.** Even after fixing `state.py:95`, LM's scalar damping/accept logic would silently couple batch items (one bad item rejects everyone's step). Needs a design decision (per-item damping masks à la curobo), not just a shape fix.
2. **torch.compile claims are untested.** CLAUDE.md advertises compile-friendliness; no test or benchmark ever calls `torch.compile` on FK/Jacobian/CostStack. The list-accumulation + `torch.stack` FK style *probably* compiles, but graph breaks from `_validate_q`'s `bool(.any())` sync are likely. Verify before advertising.
3. **`update_frame_placements` in-place writes vs autograd.** `forward_kinematics_raw` carefully avoids in-place ops "for autograd safety" (`forward.py:80-82`), yet `update_frame_placements` writes slices into a zeros tensor (`forward.py:211-216`). Index-put is autograd-legal, so this is probably fine, but the inconsistency suggests the autograd-safety comment or the frames code hasn't been stress-tested; a gradcheck through `frame_pose_world` would settle it.
4. **`CostStack.gradient` `hasattr` dispatch** (`stack.py:162`): every residual with the base-class default `apply_jac_transpose` passes the `hasattr` check, so the "sparse path" condition may be vacuously true for all residuals and the dense fallback unreachable — worth checking `residuals/base.py` inheritance vs duck-typed residuals like `PoseResidual` (which does *not* inherit a base and has no `apply_jac_transpose`, so dispatch happens to work — fragile).
5. **Viewer against a real viser server** is only exercised via `MockBackend` in tests; version drift with `viser>=0.2.0` (a fast-moving package) would go unnoticed (no CI, no pinned upper bound).
6. **`meta: dict` on frozen `Model`** (`model.py:88`) — a mutable dict on a frozen dataclass shared across workers is a foot-gun; didn't audit what gets stashed there.

---

## 4. What is actually good and should be kept

- **The math-verification test suite is genuinely strong.** `tests/test_pinocchio/` compares FK translations/rotations, frame Jacobians, RNEA/ABA/CRBA, centroidal quantities, and Lie log/exp against real Pinocchio at `atol=2e-6` (`test_fk_matches_pinocchio.py:37,56`); fp64 `gradcheck` covers the Lie backend (`tests/lie/test_torch_backend_gradcheck.py`) and RNEA/ABA (`tests/test_pinocchio/test_dynamics_derivatives.py:35-58`). 121 numeric-tolerance assertions across math test dirs. This is the opposite of shape-only theater and is the single most valuable asset for any rewrite.
- **The pure-PyTorch Lie backend** (`lie/_torch_native_backend.py`) is clean, correct, documented: Taylor-stitched `exp`/`log` via `torch.where` (differentiable at θ=0), Shepperd 4-branch matrix→quat, no data-dependent Python branching. Keep it verbatim.
- **Test suite health**: 897 passed / 1 skipped in 52 s; fast enough to run per-commit.
- **The layer-dependency contract works.** The DAG `backends → lie → … → viewer` is real (verified `io` imports only `data_model`/own modules), and `test_layer_dependencies.py` (249 lines, AST-based) is exactly the kind of contract that ages well pre-1.0.
- **The hot-path lint concept** (`test_hot_path_lint.py`) is the right idea — it just needs `float()/bool()/torch.eye/new_tensor` in its net and `residuals/`+`lie/` in its watch list (§2.5).
- **Optional heavy deps are lazily imported**: `import better_robot` does not pull mujoco/viser/trimesh (verified import sites + `test_optional_imports.py`). The install-time weight problem (§below) is purely a `pyproject` declaration issue.
- **Viewer V1 layering is proportionate where implemented**: `Visualizer` (224 LOC) → `Scene` → modes/overlays → `ViserBackend`/`MockBackend` with a testing backend that lets 10 viewer test files run headless. The implemented parts (skeleton, URDF mesh, targets gizmos with IK callback, force vectors) map 1:1 to the examples. It is *not* over-built — it is over-*stubbed* (§2.9).
- **Analytic pose/position/orientation Jacobians with the LWA-vs-adjoint pitfall correctly handled and documented** (`residuals/pose.py:87-95` matches the CLAUDE.md warning; parity-tested against Pinocchio).
- Examples use correct frame names and run against real robot models — they're truthier than the docs.

Contract-test verdict overall: layering, cache-invariant, optional-import, and backend-boundary contracts **help**. The frozen-26-symbol API test, the docstring-style tests, and the naming/deprecation shim tests **ossify** — they freeze a surface (including the stub `retarget`) that the owner explicitly wants license to break, and every rename now costs three test edits plus a shim entry for users who don't exist.

---

## 5. Recommendations

| # | Change | Effort | Risk |
|---|---|---|---|
| R1 | **Decide the batch story for optim, then enforce it.** Either implement batched LM (per-item cost `0.5*(r*r).sum(-1)`, per-item damping tensor, `torch.where` accept masks, on-device convergence with a periodic host check) or delete the batch claims from `SolverState`/CLAUDE.md and raise a clear error on batched `x0`. Add a batched-IK test either way. | M (batched LM) / S (honest error) | Medium: touches every optimizer; the pinocchio-parity and IK-regression tests fence it. |
| R2 | **Cache the `ResidualState` per iterate in `LeastSquaresProblem`** (memoize last `x` → state, or give optimizers a `residual_and_jacobian(x)` entry point). Halves FK count per iteration (measured 11 → ~6 per 5-iter solve). | S | Low. |
| R3 | **Hoist per-call constants in residuals**: precompute the 6-vector weight in `PoseResidual.__init__` (+ Position/Orientation), move `target` to model device once at construction or first call. Extend the hot-path lint to `residuals/` and to `new_tensor`/`torch.tensor`/`torch.eye`/`float(`/`bool(` patterns so the fixed `so3_inverse` bug class stays fixed. | S | Low. |
| R4 | **Vectorize `update_frame_placements`**: stack `frame.joint_placement` into a `(nframes, 7)` Model buffer at build time, gather parent joint poses, one batched `se3_compose`. Removes a third of FK+frames cost. Same trick for `model.joint_placements` `.to()` spam: assert device once (it's already validated at `forward.py:55`) instead of per-joint `.to()`. | S–M | Low; FK parity tests fence it. |
| R5 | **Fix or fence the FD fallback**: replace with `torch.func.jacrev` over `model.integrate` (autograd is clean per CLAUDE.md), or make FD batch-aware and warn loudly when it triggers (it's a 2·nv-FK-per-item cliff users will hit silently). | M | Medium: FD is currently the correctness backstop; keep it available for tests. |
| R6 | **Delete dead weight**: `utils/` (all of it), `kinematics/chain.py`, `data_model/indexing.py`, unused exception classes, `rich` dep, `IKCostConfig.collision_{margin,weight}` + `robot_collision` param (until collision exists), `damping="trust_region"` from the Literal (or implement it). Either implement `costs/factory.py` (it's the ad-hoc-residual story the user wants) or remove it. | S | Low — nothing imports any of it (verified). |
| R7 | **Drop the pre-1.0 compat theater**: remove the `Data` alias shim, `test_naming.py`, `test_deprecations.py`, and unfreeze the 26-symbol API test (replace with a "public API changed — was this intentional?" snapshot test that's easy to update). Remove stub `retarget` from `__all__`. | S | Low; it's self-imposed. |
| R8 | **Fix the doc lies with teeth**: (a) s/"panda_hand"/"body_panda_hand"/ everywhere or drop the `body_` prefix at build time (better: frame names should match URDF link names — check `io/build_model.py` naming policy); (b) make `docs/reference/roadmap.md` generated by the same grep this audit ran (`NotImplementedError` inventory) so the "single canonical list" claim becomes mechanically true; (c) run doctest/myst-nb execution over `docs/index.md` and `getting_started/` snippets in CI. | M | Low. |
| R9 | **Packaging**: move `mujoco`, `viser`, `trimesh`, `robot_descriptions` to extras (`[mjcf]`, `[viewer]`, `[examples]`) — imports are already lazy so this is a pyproject-only change; core install becomes torch+numpy+yourdfpy. Add a minimal GitHub Actions workflow (pytest + ruff) — pre-commit exists but nothing runs the suite automatically. | S | Low; document the extras. |
| R10 | **Sync-free FK validation**: make `_validate_q` run once (public boundary only), and make the quaternion-norm check lazy/debug-mode (env flag or `torch._assert_async`) so free-flyer FK doesn't sync every call. | S | Low. |
| R11 | **Apply `torch.compile` to FK and benchmark it** before any hand-vectorization of the joint loop; the 4 ms Python overhead at B=1 is exactly what compile removes, and the loop was designed for it. Gate with a `bench` test so the claim stops being aspirational. | M | Medium (compile flakiness across torch versions). |
| R12 | Precompute 6×6 body inertias (and `S` subspaces for fixed-axis joints) into `Model`/`Data` buffers instead of `Inertia(...)._to_6x6()` per joint per RNEA/ABA/CRBA call. | M | Low; pinocchio-parity tests fence it. |

Priority order for impact: R1 → R2/R3/R4 (cheap, immediate) → R8 (user trust) → R9 → R5 → R11/R12 → the rest.

---

## 6. Open questions

1. **Is batched IK a v1 requirement?** The consumer projects (BetterVideoReconstruction/BetterHumanForce) hand-roll Adam loops over batched losses — if BR is to replace them, batched first-order optimization (Adam/LBFGS over `(B, nq)`) may matter more than batched LM. The answer reorders R1.
2. **What is the `body_` frame-name prefix policy?** If intentional (disambiguating body frames from joint frames), the docs must follow; if accidental, fixing the builder is a breaking rename best done now, pre-release.
3. **Does anything planned actually need `get_residual`/`ResidualSpec`/`jacobian_blocks`?** If block-sparse trajopt solvers are near-term (roadmap is silent), keep `ResidualSpec`; otherwise delete all three and re-add with the first consumer.
4. **Which machine defines the perf budgets?** `performance.md` names RTX 4090/L40 and a Threadripper; there is no CI and no GPU here. Budgets without an enforcement machine are prose — where will they be measured?
5. **Float64 support: contract or accident?** `Model.to(dtype)` and the fp64 gradchecks suggest yes; `solve_ik`'s hard `.float()` says no. Pick one.
6. **Is the viewer's `Scene`/mode/overlay/backend 4-layer design earning its keep for V1's actual usage** (one robot, one viser server)? It's tested and not large, but a redesign pass should ask whether `MockBackend`+`ViserBackend` behind a Protocol is the right seam or whether the viewer should just be a viser adapter module.
