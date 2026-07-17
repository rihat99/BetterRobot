# Audit: Compute-Pass Inventory (Warp-first kernel-boundary planning)

Auditor: read-only research subagent (Claude). Date: 2026-07-16.
Scope: every compute pass in `src/better_robot/` — `lie`, `kinematics`, `dynamics`,
`data_model`, `residuals`, `costs`, `optim`, `collision`, `tasks`, `backends`, `spatial` —
so the owner can decide, function by function, what becomes a Warp kernel vs stays PyTorch.
Method: read every file in the listed packages in full (viewer/ and io/ skimmed — they are
render/build-time, not per-query compute). All `file:line` citations are relative to
`src/better_robot/`. No source was modified. No runtime measurements were taken here;
where timings are quoted they come from `audit_core_architecture.md` (same box, CPU-only —
CUDA broken, so GPU-sync claims are code-reading evidence).

Companion audits: `audit_core_architecture.md` (correctness/architecture findings),
`audit_optim_stack.md` (optim design findings). This file is deliberately a *neutral
inventory* — structure, shapes, metadata, sync points, backward story — not a verdict list.

---

## 1. Master table

Legend for **Structure**: `tree↓` = serial Python loop over `topo_order` root→leaf;
`tree↑` = leaf→root; `map/J` = independent per-joint loop; `map/F` = per-frame loop;
`map/B` = per-body loop; `batched` = pure batched tensor math, no Python loop;
`loop/T` = Python loop over trajectory timesteps; `loop/iter` = optimizer iteration loop.
**Sync** = `.item()`/`bool()`/`float()`/`new_tensor`/H2D copies/`try-except` on GPU op.
**Bwd** = `auto` (relies on autograd), `analytic` (hand-written), `n/a` (not differentiated),
`stub`.

| # | Pass | Location | Structure | Model metadata consumed | Sync / graph-break | Bwd |
|---|------|----------|-----------|------------------------|---------------------|-----|
| 1 | `so3_*` / `se3_*` primitives (exp, log, compose, inverse, act, adjoint, adjoint_inv, normalize, from/to matrix) | `lie/_torch_native_backend.py:128-367` | batched | none | `torch.eye` per call in `se3_exp`/`se3_log` (`:283,:308`); zeros allocs in adjoints (`:326,:340`) | auto (NaN grad at θ=0, see §2.1) |
| 2 | hat/vee, `right_jacobian_{so3,se3}` (+inv) | `lie/tangents.py:24-217` | batched (se3 Jr = 9-step matmul series + `linalg.inv`) | none | `torch.eye` per call (`:116,:143,:194`); `linalg.inv` (`:207`) | auto |
| 3 | `so3.slerp` / `se3.sclerp` | `lie/so3.py:104-138`, `lie/se3.py:125-143` | batched | none | `torch.as_tensor(t)` per call | auto |
| 4 | `forward_kinematics_raw` | `kinematics/forward.py:73-135` | tree↓ (1 `joint_transform` + 1–2 `se3.compose` per joint; list accumulate + `torch.stack`) | `topo_order`, `parents`, `nqs`, `idx_qs` (Python tuples); `joint_placements` (tensor, indexed per joint `:112`); `joint_models[j]` (Python object dispatch `:116`) | `_validate_q` `bool(...any())` `:64` (free-flyer, per call — GPU sync); per-joint `.to()` on placement `:112` (no-op if model moved) | auto (list+stack, no in-place) |
| 5 | `forward_kinematics` (public) | `kinematics/forward.py:138-191` | wrapper | as #4 + `Data` alloc | **re-runs `_validate_q`** `:181` — the quat-norm sync fires twice per call | auto |
| 6 | `update_frame_placements` | `kinematics/forward.py:194-219` | map/F (Python loop over `model.frames`, 1 `se3.compose` each; **in-place** writes into preallocated `(B...,nframes,7)`) | `frames` tuple of `Frame` objects; `frame.parent_joint`, `frame.joint_placement` (**CPU tensor — `Model.to()` does not move it**, `data_model/model.py:92-119`) | `frame.joint_placement.to(device)` per frame per call `:215` = real H2D copy on GPU | auto (in-place index writes) |
| 7 | `_compute_joint_jacobians_raw` | `kinematics/jacobian.py:30-79` | tree↓ (propagation `J[j]=J[parent]` + per-joint `to_matrix`/`hat`/3 matmuls; **in-place** slice writes into zeros `(B...,njoints,6,nv)`) | `topo_order`, `parents`, `nvs`, `idx_vs`, `nqs`, `idx_qs` (tuples); `joint_models[j].joint_motion_subspace` (Python dispatch) | none direct; per-joint `.to()` on S `:66` | auto |
| 8 | `get_frame_jacobian` | `kinematics/jacobian.py:129-227` | map over support chain (if `data.joint_jacobians` absent, loop `model.get_support(parent)` `:173-197`, in-place writes); then 1 compose + hat + matmuls | `frames[frame_id]` object, `supports` tuple, `nvs/idx_vs/nqs/idx_qs`, joint model dispatch | `frame.joint_placement.to()` per call `:200` (H2D on GPU); string branch on `reference` | auto |
| 9 | `get_joint_jacobian` | `kinematics/jacobian.py:100-126` | batched slice + optional `adjoint_inv` matmul | `data.joint_jacobians` | none | auto |
| 10 | `residual_jacobian` FD fallback | `kinematics/jacobian.py:230-284` | **`2·nv+1` full FK passes** + Python loop over `nv` columns `:280-283` | `model.integrate` (per-joint loop, see #22) | fresh `Data` per FK; dominates AUTO when a residual has no `.jacobian` | n/a (this *is* the derivative) |
| 11 | `rnea` (+ `bias_forces`, `compute_generalized_gravity` wrappers) | `dynamics/rnea.py:71-244` | FK (#4) then tree↓ forward (`adjoint_inv`, `S@a`, cross ops, `Inertia._to_6x6` per joint) + tree↑ backward (**recomputes `adjoint_inv` per joint** `:195`; per-DOF Python loop into `tau_slots` `:189-190`) | tuples `topo_order/parents/idx_qs/nqs/idx_vs/nvs`; `joint_models[i]` dispatch ×3 (`joint_motion_subspace`, `joint_velocity`, `joint_bias_acceleration` via `getattr` `data_model/joint_models/base.py:129-146`); `body_inertias` tensor (per-joint index + `_to_6x6` rebuild `:167`); `gravity` tensor | per-joint `.to()` on inertia; `torch.eye` inside `_to_6x6` (`spatial/inertia.py:179`) per joint per call | auto (documented: `derivatives.py:3-8`) |
| 12 | `aba` | `dynamics/aba.py:36-207` | FK; `Ad_inv` precompute loop `:74-75`; tree↓ pass 1; tree↑ pass 2 with **`torch.linalg.inv(D)` per joint** `:144` (usually 1×1!); tree↓ pass 3 with per-DOF `ddq_slots` Python writes `:193-194` | same as #11 | `linalg.inv` per joint; per-joint `_to_6x6` `:121` | auto |
| 13 | `crba` | `dynamics/crba.py:29-120` | FK; `Ad_inv` + `Y_c` init loops; tree↑ accumulate `:64-72`; forward assemble with **nested `while` ancestor walk per joint** `:103-117` → O(depth²) block writes into `M` (in-place) | tuples + joint dispatch + `body_inertias` | per-joint `_to_6x6`; double allocation of `Y_c` (zeros `:55-58` immediately overwritten `:59-61`) | auto |
| 14 | `ccrba` / `compute_centroidal_map` / `compute_centroidal_momentum` | `dynamics/centroidal.py:83-195` | FK; `_world_com` map/B loop `:40-47` (per body: `Inertia` wrap + `se3.act`); `Ad_inv` loop; tree↑ `Y_c` accumulate; map/J column loop with `adjoint_inv` per joint `:167-187` (in-place `A_g` writes) | tuples + joint dispatch + `body_inertias` | per-body `.to()`; same `_to_6x6` pattern | auto |
| 15 | `compute_rnea_derivatives` / `compute_aba_derivatives` / `compute_crba_derivatives` | `dynamics/derivatives.py:29-83` | `torch.autograd.functional.jacobian(..., vectorize=False)` → **one backward through the whole recursion per output element** (`nv` re-plays for RNEA, `nv²·nq` graph work for CRBA) | wraps #11–#13 | `.detach()` inputs; fresh `Data` per call | n/a (is the derivative; not itself differentiable — `create_graph=False`) |
| 16 | `StateMultibody.integrate/diff` (+ autograd `jacobian_*`) | `dynamics/state_manifold.py:42-72` | split + `Model.integrate/difference` (#22) + cat | `nq`, `nv` | none | auto |
| 17 | `integrate_q` | `dynamics/integrators.py:18-45` | delegates to #22 | — | none | auto |
| 18 | Dynamic integrators `semi_implicit_euler`/`symplectic_euler`/`rk4`; `compute_minverse`; `compute_coriolis_matrix`; centroidal derivatives | `dynamics/integrators.py:48-96`, `crba.py:123-132`, `rnea.py:247-262`, `derivatives.py:86-103` | — | — | — | **stub** |
| 19 | Action models `calc` (Euler / RK4) | `dynamics/action/integrated.py:26-63,66-121`, `differential.py:75-104` | wraps `aba` + `state.integrate`; RK4 = 4× aba per step | — | fresh `model.create_data()` per `forward_dynamics` call (`differential.py:104`) | auto |
| 20 | Action models `calc_diff` | `integrated.py:42-63`, `differential.py:82-96` | `autograd.functional.jacobian(vectorize=False)` around #19 | — | `.detach()` | n/a |
| 21 | `JointModel.joint_transform / joint_motion_subspace / joint_velocity` (per kind) | `data_model/joint_models/{revolute,prismatic,spherical,free_flyer,planar,translation,helical,composite}.py` | batched per slice; subspace builders allocate zeros + scalar in-place writes per call | axis constants are **module-level CPU tensors** (`revolute.py:63-65`, `prismatic.py:45-47`) or per-instance CPU tensors (`revolute.py:162`) moved with `.to(q.device)` **per call** — H2D copy per joint per FK/Jac/dyn pass on GPU | `float(self.pitch)` in helical `:35,:46-48`; fixed/universe/mimic `joint_transform` returns fresh CPU `torch.tensor` (`fixed.py:21-25`, `mimic.py:33-35`) — unreached in FK (nq=0 branch `forward.py:117-119`) | auto |
| 22 | `Model.integrate` / `Model.difference` | `data_model/model.py:160-190` | map/J Python loop over `njoints`, per-joint `jm.integrate/difference` dispatch + `torch.cat` of per-joint parts | `joint_models`, `idx_qs`, `idx_vs`, `nqs/nvs` (all Python) | none | auto |
| 23 | `Model.random_configuration` / `neutral` per joint | `model.py:192-205` | map/J loop (cold path) | limits tensors | CPU-only `torch.rand` | n/a |
| 24 | Mimic gather (`mimic_multiplier/offset/source`) | `data_model/model.py:76-78` | **declared, never consumed** — no compute pass reads them (grep: only `model.py`, `mimic.py` docstring, `io/build_model.py:486-504`); FK treats mimic joints as nq=0 identity | flat tensors + tuple, ready for a gather | — | — |
| 25 | `PoseResidual.__call__` / `.jacobian` | `residuals/pose.py:58-103` | batched: inverse+compose+log; jacobian = `get_frame_jacobian` (#8) + rotate-by-`R.mT` + `right_jacobian_inv_se3` (#2, incl. `linalg.inv`) + matmul | `frames` (via `_get_frame_pose` `:21-31` — per-call `frame.joint_placement.to()` H2D if `frame_pose_world` absent) | **`r.new_tensor([...])` per eval** `:65,:102` — H2D copy each call; `self.target.to()` per call | analytic J; residual itself auto |
| 26 | `PositionResidual` / `OrientationResidual` | `pose.py:106-142,145-195` | batched; jacobians slice frame Jacobian (+ `right_jacobian_inv_so3` for orientation) | as #25 | `target.to()` per call | analytic |
| 27 | `JointPositionLimit` | `residuals/limits.py:18-86` | batched clamps; jacobian = `where` diagonals × precomputed `(nq,nv)` `dq_dv` (built once on CPU `:45-54`) | `lower/upper_pos_limit` tensors; `joint_models` at init only | `.to()` on limits + `dq_dv` per call | analytic |
| 28 | `JointVelocityLimit` (`__call__` only), `JointAccelLimit`, `NullspaceResidual`, `JerkResidual`, `YoshikawaResidual` | `limits.py:89-128`, `regularization.py:155-173`, `smoothness.py:164-184`, `manipulability.py` | batched / — | limits | — | stubs (velocity-limit jacobian raises; Yoshikawa returns None → FD fallback #10) |
| 29 | `RestResidual` | `regularization.py:20-64` | `model.difference` (#22) + scale; jacobian = `weight·I` | `q_neutral` (via caller) | `torch.eye` per jacobian call `:61` | analytic (small-angle `Jr≈I`) |
| 30 | `ReferenceTrajectoryResidual` | `regularization.py:67-152` | `model.difference` on `(T,nq)`; jacobian = dense `(T·nv,T·nv)` built by `loop/T` `:135-138`; `apply_jac_transpose` = batched per-frame scale `:141-152` | — | dense J alloc | analytic + matrix-free JT |
| 31 | `VelocityResidual` / `AccelerationResidual` | `residuals/smoothness.py:33-161` | residual batched over T (`model.difference` on shifted views); jacobian = dense banded built by `loop/T` (`:77-81`, `:137-141`); `apply_jac_transpose` = 3 aligned slice-adds (batched) | — | dense `(nv(T−2), T·nv)` alloc; `torch.eye` per call | analytic banded + matrix-free JT |
| 32 | `ContactConsistencyResidual` | `residuals/contact.py:72-175` | residual batched (`(T,K,3)` diffs); jacobian = `loop/T×K` with **`float(w_pair[t,k])` per block** `:131` (sync per (t,k)); `apply_jac_transpose` batched per frame `:136-175`, K× `get_frame_jacobian` over the T-batch | `frames` via #8 | `float()` in double loop | analytic + matrix-free JT |
| 33 | `TimeIndexedResidual` | `residuals/temporal.py:20-127` | slices one knot from `Data` views, calls inner; jacobian scatters inner block into `(dim, T·nv)` zeros `:93-95`; sparse JT `:97-115` | — | dense scatter alloc | wraps inner |
| 34 | Self/world collision residuals | `residuals/collision.py` | — | `RobotCollision` (also stub) | — | **stub** |
| 35 | `CostStack.residual/jacobian/gradient` | `costs/stack.py:104-171` | Python loop over items + `torch.cat`; `gradient` prefers per-item `apply_jac_transpose` (matrix-free) | — | none | auto/analytic per item |
| 36 | `LeastSquaresProblem.residual/jacobian/gradient/jacobian_blocks/step` | `optim/problem.py:42-105` | each call re-runs `state_factory` → **1 full FK + frame pass per call** (`tasks/ik.py:212-214`) | — | — | n/a (solver treats values) |
| 37 | LM `minimize` | `optim/optimizers/levenberg_marquardt.py:61-135` | `loop/iter`: J build (#36) → IRLS reweight `:27-44` → `JtJ = JᵀJ`, `Jtr` `:94-95` → `H = JtJ + λI` (`torch.eye` per iter `:96`) → solve → step + clamp → accept/reject branch | box limits (`.to()` per iter `:105-107`) | **`float()` ×4+ per iter** (`:86,:110,:116,:126`) — full sync each; Python branch on cost; history list append | n/a |
| 38 | GN `minimize` | `optim/optimizers/gauss_newton.py:27-76` | same shell, fixed ε-damping | as #37 | `float()` `:65,:69`; `torch.eye` `:53` | n/a |
| 39 | Adam `minimize` | `optim/optimizers/adam.py:45-107` | `loop/iter`: full J built per iter just for `g = Jᵀr` `:79-80` (not the matrix-free `problem.gradient`) | — | `float()` `:96,:100` | n/a |
| 40 | LBFGS `minimize` | `optim/optimizers/lbfgs.py:50-171` | `loop/iter`: two-loop recursion over history (Python), Armijo backtracking (≤20 residual evals = ≤20 FK per iter) `:125-139` | — | `float()` ×6 per iter (`:83,:87,:114,:118,:134,:152`) | n/a |
| 41 | `MultiStage` / `LMThenLBFGS` | `optim/optimizers/multi_stage.py`, `lm_then_lbfgs.py` | sequential stage driver (cold) | — | — | n/a |
| 42 | Linear solvers | `optim/solvers/cholesky.py` (`try cholesky → except lstsq` — exception detection forces device sync / graph break), `lstsq.py`; `cg.py`/`sparse_cholesky.py` **stubs** | batched linalg | — | try/except | n/a |
| 43 | Damping strategies | `optim/strategies/{adaptive,constant}.py` (pure Python floats); `trust_region.py` **stub** | scalar | — | operates on Python `float` λ | n/a |
| 44 | Robust kernels | `optim/kernels/{l2,huber,cauchy,tukey}.py` | batched `where`/clamps | — | none | auto |
| 45 | `SolverState.from_problem` | `optim/state.py:99-113` | 1 residual eval (FK) at x0 | — | `.clone().detach()` | n/a |
| 46 | Capsule distance kernels + dispatch + `RobotCollision.*` | `collision/closest_pts.py:14-36`, `pairs.py:35-46`, `robot_collision.py:36-73` | — | frame-attached capsule tensors (declared: `local_a/local_b/radii/self_pairs/allowed_pairs_mask`) | — | **all stubs** |
| 47 | `solve_ik` | `tasks/ik.py:150-266` | build stack + problem, delegate to #37-41; `state_factory` = FK + frames per eval | name→id dicts | `.float()` casts at setup | n/a |
| 48 | `solve_trajopt` | `tasks/trajopt.py:84-230` | flatten `(T,nq)`; per-knot retraction via `Model.integrate` (#22); B-spline chain rule = dense `J_q @ kron(B, I_nq)` `:199-201,:56-65` | — | `torch.kron` dense `(T·nq, C·nq)` | n/a |
| 49 | B-spline basis (Cox–de Boor) | `tasks/parameterization.py:89-127` | double Python loop over (degree, basis fn); `if d1 > 0` on 0-dim tensors → `bool()` sync each `:121-122`; built once per (T, device) then cached; `expand` = single `B @ z` matmul `:144-153` | — | cold-path syncs | auto through `expand` |
| 50 | `Trajectory.resample/slice` (`_linear_interp_along_axis`, `_slerp_along_axis`) | `tasks/trajectory.py:151-293` | batched `searchsorted` + `index_select` + slerp | — | `float()`/`int()` in `slice` `:164-166` (cold) | auto |
| 51 | `spatial` value-type algebra: `Inertia._to_6x6` / `se3_action` / `apply`, `Motion.cross_*`, `ops.ad/ad_star` | `spatial/inertia.py:155-237`, `spatial/motion.py`, `spatial/ops.py` | batched | packed `(...,10)` inertia | `torch.eye` per `_to_6x6` call `:179` | auto |
| 52 | Backend dispatch layer | `backends/__init__.py:45-124`, `protocol.py`, `torch_native/{lie,kinematics,dynamics}_ops.py` | registry dict hit + 2 lazy-import re-export hops per whole-pass call; per-op hops for every Lie call from the facades (`lie/se3.py:28-29,42-49`) | — | ~19 µs/op dispatch overhead measured (`audit_core_architecture.md §2.1`) | n/a |
| 53 | Warp stub | `backends/warp/__init__.py` (raises), `bridge.py` (`WarpBridge.to_warp/to_torch` raise), `kernels/__init__.py` (empty; names planned: `fk.py, jacobian.py, rnea.py, aba.py, crba.py`) | — | — | — | **stub** |

Excluded as non-compute: `io/` (build-time parsing/assembly; the only runtime-relevant fact
is recorded in §4), `viewer/` (render-side; per-frame update loops exist in
`render_modes/skeleton.py` / `urdf_mesh.py` but are viser-I/O-bound, not kernel candidates),
`utils/` (`flatten_batch` is a stub; `batch_shape` trivial), `lie/types.py` (thin frozen
wrappers delegating to #1), `costs/factory.py` (cold builders), `kinematics/chain.py` (stub),
`data_model/indexing.py` (stub), `tasks/retarget.py` (stub).

---

## 2. Per-pass notes by structural family

### 2.1 Per-op Lie layer (`lie/`) — batched, loop-free, but called from inside Python loops

- Every op is pure batched elementwise/small-matmul math over `(..., 7)/(..., 4)/(..., 6)`
  tensors; no Python loop inside any op. The cost is **call volume × dispatch**:
  one `se3.compose` costs 4 Python frames + registry hit through the facade
  (`lie/se3.py:42-49` → `backends/__init__.py:78-80` → `torch_native/lie_ops.py:29-30` →
  `_torch_native_backend.py:238-247`).
- Call volume per whole pass (N = njoints, F = nframes, T = targets):
  - FK: ~2N−1 `se3_compose` + N_actuated `joint_transform` (each = axis-angle→quat + cat).
  - Frame update: F `se3_compose`.
  - Joint Jacobians: N `so3_to_matrix` + N `hat_so3`.
  - RNEA: **2N** `se3_adjoint_inv` (backward pass recomputes it, `rnea.py:195`), 2N cross ops.
  - ABA: N `se3_adjoint_inv` (cached in a Python list, `aba.py:73-75`), N `linalg.inv`.
  - CRBA/CCRBA: N `se3_adjoint_inv` + N–2N 6×6 congruence matmuls.
  - Pose residual eval: 1 inverse + 1 compose + 1 log per target; its Jacobian adds
    `right_jacobian_inv_se3` = 9 sequential 6×6 matmuls + one `linalg.inv`
    (`lie/tangents.py:192-207`).
- Numerics: Taylor-stitched `torch.where` for exp/log; θ=0 gradient is NaN through the
  unclamped `sqrt` (`_torch_native_backend.py:148,270,296` — verified in
  `audit_core_architecture.md §2.2`). Any Warp rewrite inherits the obligation to fix this
  in the adjoint.
- **No custom `torch.autograd.Function` anywhere in the library** (grep). Every backward is
  eager autograd through the op graph.

### 2.2 Tree-scan passes (the natural Warp whole-pass kernels)

All five share the same skeleton: serial Python `for j in model.topo_order`, per-joint
Python-object dispatch (`model.joint_models[j].method(...)`), per-joint reads of the
Python tuples `parents/nqs/nvs/idx_qs/idx_vs`, and per-joint slicing of `q/v/a`.
None branches on tensor *values*; all branching is on static Python ints (`nq_j > 0`,
`parent < 0`), so they unroll cleanly under `torch.compile` (5× CPU measured on raw FK,
per handoff) — but eager per-joint launch overhead is serial on GPU.

1. **FK** (#4): root→leaf; recursion state = parent world pose. Output built by list +
   `torch.stack` (autograd-clean, no in-place). Two composes/joint. The per-joint
   `joint_transform` is where joint-kind dispatch lives — no `if kind` in the loop itself,
   but the *object call* is Python dispatch and the axis tensors it uses live on CPU (#21).
2. **Joint Jacobians** (#7): root→leaf; recursion = `J[j] ← J[parent]` copy + per-joint
   column block write. In-place slice writes into a zeros tensor (autograd handles it;
   hostile to functionalization/vmap).
3. **RNEA** (#11): root→leaf (v, a, f) then leaf→root (τ, force transport). Backward pass
   recomputes `Ad_inv` instead of caching; τ assembled per scalar DOF into a Python list
   then stacked (`rnea.py:180-200`).
4. **ABA** (#12): three passes ↓↑↓; recursion state per joint = `(IA 6×6, pA 6)`;
   per-joint `U/D/u` factorisation with `torch.linalg.inv` of an `(nv_i×nv_i)` matrix —
   for revolute chains that is a batched 1×1 inverse per joint.
5. **CRBA** (#13): leaf→root composite-inertia accumulation, then per-joint ancestor
   `while`-walk writing `M` blocks — O(depth²) small matmuls, in-place writes.
6. **CCRBA** (#14): CRBA backward + per-joint column transport to the COM frame; plus the
   `_world_com` per-body map (mass-weighted `se3.act`, fully parallelizable — currently a
   Python loop with a running-sum).

Shared per-joint costs across #11–#14: `Inertia(model.body_inertias[i]...)._to_6x6()` is
re-derived (hat, parallel-axis, `torch.eye`, block-cat) **per joint per call** — a natural
precompute (a `(njoints, 6, 6)` buffer) that none of the passes cache.

### 2.3 Per-frame / per-body maps

`update_frame_placements` (#6), `_world_com` (#14), and the support-chain fallback in
`get_frame_jacobian` (#8) are embarrassingly parallel over frames/bodies but written as
Python loops. #6 and #8 also pay a **real per-frame H2D copy on GPU** because
`Frame.joint_placement` tensors are never moved by `Model.to()` (`data_model/model.py:92-119`
moves 15 named tensors — `frames` is not among them; consumed at `forward.py:215`,
`jacobian.py:200`, `pose.py:28-30`). Stacking frame placements into a `(nframes, 7)` model
tensor + a `(nframes,)` parent-joint index tensor turns #6 into one gather + one batched
compose.

### 2.4 Manifold ops (`integrate` / `difference`)

`Model.integrate/difference` (#22) is a Python loop over all joints with per-kind dispatch
and a final `torch.cat` — for the common all-revolute case it is literally `q + v` computed
as N single-column adds. It sits on several hot paths: LM/GN/Adam/LBFGS retraction
(`problem.step` via `tasks/ik.py:225`), every Rest/ReferenceTrajectory/Velocity/Acceleration
residual (once or twice per eval), the FD fallback (#10, 2·nv+1 times per Jacobian), and
`solve_trajopt`'s per-knot retraction. A flat-metadata rewrite (segment-wise: Euclidean add
for `nq==nv` blocks, SE3/SO3 retraction for the ≤2 manifold blocks) removes the loop entirely.

### 2.5 Per-residual maps

Residual *evaluations* are all batched tensor math (good). The Python-loop / sync hotspots
are in **dense Jacobian assembly** for trajectory residuals:
`ReferenceTrajectory.jacobian` loop/T (`regularization.py:135-138`), `Velocity`/`Acceleration`
banded builders loop/T (`smoothness.py:77-81,137-141`), `ContactConsistency.jacobian`
loop T×K with a `float()` sync per block (`contact.py:127-133`), and `TimeIndexedResidual`'s
dense scatter (`temporal.py:93-95`). Each of these already has a **matrix-free
`apply_jac_transpose`** that is loop-free (or loops only over K frames), used by
`CostStack.gradient` — so the LM/GN dense path is the only consumer of the dense builders.
`PoseResidual`'s per-eval `r.new_tensor(...)` (`pose.py:65,102`) is an H2D copy inside the
innermost solver loop.

### 2.6 Linalg / optimizer layer

The optimizers are Python iteration loops around whole-problem tensor ops: J assembly
(#35/#36 → one FK per call), `JᵀJ`/`Jᵀr` matmuls, λI damping with a fresh `torch.eye` per
iteration, Cholesky (with `try/except` → lstsq fallback), retraction, clamp. Every
iteration converts cost/gradient norms to Python `float` for the accept/reject and
convergence branches — **4–6 device syncs per LM iteration**, so the loop is fully
host-serialized regardless of how fast the passes get. Everything is unbatched over
problems: `r @ r`, `delta_v @ Jtr`, and `float(cost)` all assume a single flat problem
vector (a batch dimension would break `LevenbergMarquardt.minimize` at `:110-117`).
Adam (#39) builds the full dense J per iteration only to form `Jᵀr`, even though the
matrix-free `problem.gradient` (`problem.py:72-84`) exists.

### 2.7 Autograd-wrapped derivative passes

`compute_{rnea,aba,crba}_derivatives` (#15), `StateMultibody.jacobian_*` (#16), and
action-model `calc_diff` (#20) all use `torch.autograd.functional.jacobian(...,
vectorize=False)` — one full backward replay per output element. These are the passes where
either analytic recursions (Carpentier–Mansard, roadmap stubs) or a Warp `wp.Tape`-style
adjoint of the whole pass changes complexity class, not constant factor.

### 2.8 Stubs (no compute exists yet — free to design Warp-first)

Collision (all of #34/#46: point-segment, segment-segment, pair dispatch, world capsules,
self/world distances), Coriolis matrix, `compute_minverse`, dynamic integrators, centroidal
derivatives, `TrustRegion`, `CG`, `SparseCholesky`, `retarget`, jerk/nullspace/velocity-limit
Jacobians, `utils.flatten_batch`, `kinematics/chain.get_chain`, `data_model/indexing`.

---

## 3. Data: what is cached where

`Data` (`data_model/data.py:95-148`) is a mutable dataclass holding only optional tensors:

| Group | Fields (shape) |
|-------|----------------|
| inputs | `q (B...,nq)`, `v`, `a`, `tau (B...,nv)` |
| placements | `joint_pose_local`, `joint_pose_world (B...,njoints,7)`, `frame_pose_world (B...,nframes,7)` |
| vel/acc | `joint_velocity_{world,local}`, `joint_acceleration_{world,local}`, `joint_forces (B...,njoints,6)` |
| jacobians | `joint_jacobians (B...,njoints,6,nv)`, `joint_jacobians_dot` |
| dynamics | `mass_matrix (B...,nv,nv)`, `coriolis_matrix`, `gravity_torque`, `bias_forces`, `ddq` |
| centroidal | `centroidal_momentum_matrix (B...,6,nv)`, `centroidal_momentum (B...,6)`, `com_{position,velocity,acceleration} (B...,3)` |

Invalidation: `__setattr__` hook on `q/v/a` (`data.py:152-165`) demotes `_kinematics_level`
and `None`s higher caches; `require(level)` raises `StaleCacheError`. All bookkeeping is
Python-side (an `IntEnum` field), zero tensor cost. Note the **cache is bypassed in
practice**: the IK `state_factory` builds a *fresh* `Data` per residual/jacobian/step
evaluation (`tasks/ik.py:212-214` via `model.create_data`, `model.py:121-133`), so nothing
survives between solver iterations; and `rnea/aba/crba/ccrba` each re-run FK internally
rather than consuming a populated `Data` (`rnea.py:108`, `aba.py:68`, `crba.py:44`,
`centroidal.py:136`).

---

## 4. Model: current tensor layout (flat tensors vs Python objects)

Already flat device tensors, moved by `Model.to()` (`model.py:62-84,92-119`):
`joint_placements (njoints,7)`, `body_inertias (nbodies,10)`, `lower/upper_pos_limit (nq,)`,
`velocity_limit/effort_limit/rotor_inertia/armature/friction/damping (nv,)`, `gravity (6,)`,
`mimic_multiplier/mimic_offset (njoints,)`, `q_neutral (nq,)`, `reference_configurations`.

Still Python objects / CPU-resident (i.e. everything a kernel would need to index per joint):

- `parents`, `children`, `subtrees`, `supports`, `topo_order` — tuples of ints/tuples
  (`model.py:49-53`).
- `nqs`, `nvs`, `idx_qs`, `idx_vs` — tuples (`:57-60`). Contiguity is asserted at build
  time (`io/build_model.py:82-92`), and `idx_q/idx_v` are assigned by simple accumulation in
  joint order (`io/build_model.py:326-345`) — i.e. **there is no runtime q permutation**;
  any IR→model reordering happens once at build.
- `joint_models` — tuple of frozen dataclass instances; the per-kind discriminator is a
  *string* `kind` field (18 kinds, `joint_models/base.py:17-36`), and axis vectors live as
  CPU tensors inside the instances / module globals (#21). No integer kind-code tensor
  exists.
- `frames` — tuple of `Frame` dataclasses, each holding a `(7,)` CPU `joint_placement`
  tensor and a Python `parent_joint` int (`data_model/frame.py:22-28`); **not** stacked and
  **not** moved by `Model.to()`.
- `mimic_source` — tuple of ints (`model.py:78`); with #24 unconsumed, the whole mimic
  triple is dormant metadata.
- name↔id dicts, `meta`.

So the flat-metadata prep step the Warp-first plan needs is: kind-code / axis / pitch /
`idx_q/idx_v/nq/nv` / parent / topo-rank as `(njoints,)`-shaped device tensors, plus stacked
frame placements `(nframes, 7)` + `frame_parent_joint (nframes,)`, plus a precomputed
`(njoints, 6, 6)` spatial-inertia buffer (or keep `(njoints,10)` packed and expand in-kernel).

---

## 5. Consolidated sync-point / graph-break inventory (hot paths only)

| Site | Kind | Frequency |
|------|------|-----------|
| `forward.py:64` `bool((...).any())` | device→host sync | per FK call (free-flyer models) — **twice** via `forward.py:97+181` |
| `pose.py:65,102` `r.new_tensor([...])` | H2D copy | per pose-residual eval / jacobian |
| joint axis / placement `.to(device)` (`revolute.py:21,29-31`, `prismatic.py`, `helical.py`) | H2D copy (CPU module constants) | per joint per FK/Jac/dyn call on GPU |
| `frame.joint_placement.to(device)` (`forward.py:215`, `jacobian.py:200`, `pose.py:28-30`) | H2D copy (never pre-moved) | per frame per call on GPU |
| LM/GN/Adam/LBFGS `float(...)` (`levenberg_marquardt.py:86,110,116,126` etc.) | sync | 2–6× per optimizer iteration |
| `contact.py:131` `float(w_pair[t,k])` | sync | per (t,k) block in contact jacobian |
| `torch.eye` fresh alloc (`levenberg_marquardt.py:96`, `gauss_newton.py:53`, `se3_exp/log :283/:308`, `tangents.py:116,143,194`, `inertia.py:179`, `regularization.py:61,136`, `smoothness.py:75,135`, `free_flyer.py:41`) | alloc per call | hot paths throughout |
| `solvers/cholesky.py` `try/except` around `linalg.cholesky` | sync + graph break | per LM/GN iteration |
| `aba.py:144` `torch.linalg.inv(D)` (nv_i×nv_i, mostly 1×1) | small-kernel launch | per joint per ABA call |
| `parameterization.py:121-122` `if d1 > 0` on tensor | `bool()` sync | cold (basis built once per T) |
| In-place slice writes (`jacobian.py:50,76-77,196-197`, `crba.py:101,115-116`, `centroidal.py:187`, `forward.py:216`) | autograd-legal but functionalization/vmap-hostile | per pass |

---

## 6. Grouping summary (for kernel-boundary decisions)

- **Tree-scan candidates (whole-pass kernels)**: FK, joint Jacobians, RNEA, ABA, CRBA,
  CCRBA — all Python-loop-bound today, all consume the same not-yet-flat per-joint
  metadata, all autograd-only backward.
- **Per-frame/body maps (one batched gather+op each, or fold into the tree kernels)**:
  frame placements, world-COM, frame Jacobian extraction.
- **Manifold segment ops**: `Model.integrate/difference` (+ `StateMultibody`, `integrate_q`)
  — loop-bound, on every solver step and several residuals.
- **Per-residual maps**: already batched evals; dense trajectory-Jacobian builders are
  loop/T Python; matrix-free `apply_jac_transpose` paths are already loop-free and are the
  better long-horizon seam.
- **Linalg/optimizer**: stays PyTorch (`linalg.cholesky/lstsq/inv`, matmuls), but the
  iteration loops are host-serialized by design (`float()` accept/reject) and unbatched
  over problems.
- **Autograd-replay derivative passes**: `compute_*_derivatives`, `calc_diff` — the
  passes where an adjoint-of-the-pass design (analytic or `wp.Tape`) changes complexity.
- **Green-field for Warp**: the entire collision layer, Coriolis, Minverse, dynamic
  integrators — stubs with signatures already fixed.
