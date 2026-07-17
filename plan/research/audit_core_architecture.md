# Audit: Core Architecture — backends, lie, spatial, data_model, kinematics, dynamics, io

Auditor: read-only architecture subagent (Claude). Date: 2026-07-16.
Scope: `src/better_robot/{backends,lie,spatial,data_model,kinematics,dynamics,io}` (8,509 LOC total per `wc -l`).

---

## 1. Scope & method

- Read every file in the seven target packages in full (except long tails of `prismatic.py`/`translation.py`/`planar.py`, skimmed after their family pattern was established from `revolute.py`).
- Cross-checked claims in `CLAUDE.md`, `lie/CLAUDE.md`, `data_model/CLAUDE.md`, `kinematics/CLAUDE.md`, `io/CLAUDE.md`, and `docs/reference/roadmap.md` against the code.
- Ran empirical checks with `uv run python` inside the repo (CPU only — **CUDA is unavailable on this box**: driver 12060 too old for the installed torch 2.11.0+cu130 build, so all GPU-sync/H2D claims below are code-reading evidence, not measurements):
  - dispatch-overhead micro-benchmark (facade vs direct backend call);
  - `torch.compile(fullgraph=True)` on `forward_kinematics_raw` for fixed-base and free-flyer Panda;
  - NaN-gradient probes of `so3_exp/so3_log/se3_exp/se3_log` at the θ=0 singularity, plus an end-to-end rest-residual backward on the SMPL-like body;
  - batched/differentiable `Model.joint_placements` through FK and RNEA;
  - `ModelBuilder.add_helical`, `add_joint(kind=JointRX())`, and mimic-joint FK behavior through `build_model`;
  - FK / `Model.integrate` timing at several batch sizes.
- Comparison material: `references/design/{BEST_PRACTICES,pyroki,mujoco_warp,newton,pinocchio}.md`.

Everything below cites `file:line` in `src/better_robot/` unless noted. Severities: **Critical** (silent wrong results / NaN in a mainline use case), **High** (blocks a stated goal or hurts every GPU user), **Medium** (real cost, workaround exists), **Low** (hygiene/taste).

---

## 2. Confirmed problems (with evidence)

### 2.1 THE BACKENDS QUESTION — the Protocol layer is not earning its keep

**The actual call chain** for one `se3.compose(a, b)`:

1. `lie/se3.py:42-49` — facade function; calls `_lie(backend)`.
2. `lie/se3.py:28-29` — `_lie(None)` → `default_backend()`.
3. `backends/__init__.py:78-80` → `_load("torch_native")` → registry dict hit (`backends/__init__.py:45-53`).
4. `backends/torch_native/lie_ops.py:29-30` — `TorchNativeLieOps.se3_compose` → `_backend()` (a lazy `from ...lie import _torch_native_backend`, `lie_ops.py:19-21`).
5. `lie/_torch_native_backend.py:238-247` — the actual math.

Four Python frames + a registry lookup + a `sys.modules` hit per Lie op. Measured on CPU (single unbatched compose, 20k iterations):

```
facade se3.compose : 86.8 us/call
direct tn.se3_compose: 68.1 us/call
dispatch overhead: 18.8 us/call  (~27% of one small op)
```

FK does 2 composes + 1 `joint_transform` per joint (`kinematics/forward.py:110-129`), so a 25-joint model pays ~75 dispatch round-trips per FK call, all in serial Python — on GPU this is pure launch-gap time.

**The layer is also structurally dishonest.** Evidence:

- **The dependency arrow is inverted.** The documented DAG is `backends → lie → … → kinematics → dynamics` (CLAUDE.md), but the concrete backend imports *upward*: `backends/torch_native/kinematics_ops.py:27-34` lazily imports `kinematics.forward` / `kinematics.jacobian`, and `backends/torch_native/dynamics_ops.py:28-57` lazily imports `dynamics.rnea/aba/crba/centroidal`. The "bottom" layer is implemented by the layers above it; the lazy imports exist to dodge the layering contract test. This means the "backend" is not a backend — it's a re-export table.
- **`DynamicsOps` has zero consumers.** `grep` for `backend.dynamics.` finds no call sites; `rnea/aba/crba` are called directly by everything (e.g. `dynamics/derivatives.py:44`, `dynamics/action/differential.py:104`). The Protocol (`backends/protocol.py:72-109`) and its impl (60 lines) are dead weight.
- **`KinematicsOps` has exactly two consumers** — `kinematics/forward.py:183` and `kinematics/jacobian.py:96` — and both just bounce back into the same package.
- **The Lie facade half-bypasses its own dispatch.** `lie/se3.py:22` imports `_torch_native_backend as _be` directly, and `identity`/`from_axis_angle`/`from_translation`/`apply_base` (`se3.py:39,102,110,122`; `so3.py:33,94`) call `_be` unconditionally, ignoring the `backend=` kwarg. A hypothetical second backend would silently get torch-native for a third of the surface.
- **The `backend=` kwarg is never used internally.** No call site in `tasks/`, `residuals/`, or `optim/` passes it (grep). Even `forward_kinematics_raw` — what the backend dispatch returns — internally calls the default-backend `se3.compose` (`forward.py:122,129`), so passing a custom `Backend` to `forward_kinematics(...)` would *not* route its Lie ops through that backend. The dispatch is inconsistent at every level.
- **`graph_capture` is a no-op passed off as a seam** (`backends/__init__.py:110-124`) with zero users (grep), and `backends/torch_native/ops.py` is an empty 10-line module that exists "so the Warp backend has a mirror location."
- Total cost of the layer: ~530 LOC (`protocol.py` 122 + `backends/__init__.py` 139 + `torch_native/*` ~215 + `warp/*` ~50) for two real call sites.

**Evaluation of the three futures:**

- **(a) Delete now, reintroduce when a second backend is real** — recommended, with one nuance (below). The lie facades already contain the direct-call pattern (`_be.…`); making it uniform deletes ~500 lines and 19 µs/op. `forward_kinematics` / `compute_joint_jacobians` call their `_raw` siblings directly. Nothing user-visible changes except `set_backend`, which today can only select the default anyway (`backends/__init__.py:55-62` — "warp" raises unconditionally).
- **(b) Keep as-is** — pays real per-op overhead and maintenance for a hypothetical, and it is *the wrong granularity* for the stated Warp future (see next point). Rejected.
- **(c) Leaner dispatch** — the key insight from the reference designs: when Warp arrives it will replace **whole algorithm passes** (FK, Jacobian, RNEA, ABA, CRBA kernels — exactly what `docs/reference/roadmap.md` §Backends lists), not per-op quaternion multiplies. Calling into Warp per `se3_compose` would be slower than eager torch. So `LieOps` — the most elaborate of the three Protocols — is dispatch at a granularity no future backend will ever want. The useful seam is the *whole-pass* level, and that seam **already exists** as the tensor-only `forward_kinematics_raw(model, q) -> (oMi, liMi)` / `_compute_joint_jacobians_raw(model, data)` functions. mjlab's pattern (per `references/design/mujoco_warp.md:422-426` and `newton.md:450-462`) is the model: Warp kernels wrapped in `torch.autograd.Function`, torch tensors at every public boundary, no Protocol registry.

**What shape the code should take TODAY so a later Warp move is cheap:**

1. Keep/strengthen the pure tensor-in/tensor-out `*_raw` functions with `(model, q, …) → tensors` signatures — they are the natural `torch.autograd.Function.forward` bodies. (Already good.)
2. **Pack the per-joint metadata a kernel needs into flat device tensors on `Model`**: joint-kind codes, axes, `idx_q/idx_v/nq/nv`, parents. Today kinds live as Python objects (`model.joint_models`, `model.py:56`), axes live *inside* `JointModel` instances as module-level CPU tensors (`joint_models/revolute.py:63-65`), and parents/topo are Python tuples. No Warp (or `torch.compile`-friendly vectorized) kernel can consume that. This is the single highest-leverage prep step.
3. Do not add a dispatch framework until the second implementation exists; when it does, an `if backend == "warp"` at ≤6 whole-pass call sites (or a 20-line function-pointer table) is the entire requirement.

Severity: **High** (user's #1 concern; the layer costs real performance and lies about the dependency structure) — but the *fix* is cheap because the right seam already exists underneath it.

### 2.2 NaN gradients at the Lie singularity — despite explicit claims of smoothness

`lie/_torch_native_backend.py:11-18` and `lie/CLAUDE.md` claim the Taylor-stitched `torch.where` keeps exp/log "smooth and differentiable across θ = 0". **This is false at exactly θ = 0** — the value is fine but the *gradient* is NaN, because `theta = theta2.clamp(min=0.0).sqrt()` (`_torch_native_backend.py:148`, `:270`, `:296`) has an infinite derivative at 0, and it feeds `torch.cos(half)` / `atan2` terms *outside* the `torch.where` protection (e.g. `qw = torch.cos(half)`, `:159`). `0 * inf = NaN` propagates through `where`'s backward.

Measured (fp64):

```
so3_exp @ 0:  grad = [nan, nan, nan]
se3_exp @ 0:  grad = [1, 1, 1, nan, nan, nan]
so3_log @ id: NaN     se3_log @ id: NaN
2nd-order grad @ 0: NaN
(gradcheck at ||ω||≈1e-5 passes — the docs' gradcheck tests near, not at, zero)
```

End-to-end on the SMPL-like body (free-flyer + 23 spherical joints, the better_human topology):

```python
d = model.difference(q_neutral, q)   # rest residual at the rest pose
(d*d).sum().backward()
# → q.grad has NaN in 96 of 99 components
```

Any gradient-based optimizer (Adam/LBFGS paths) initialized at the neutral/rest pose of a model with spherical or free-flyer joints produces NaN on step 1. This is precisely the better_human mainline. Also hit by `PoseResidual`-style `log` at exact convergence and by `JointSpherical.difference` (`joint_models/spherical.py:58-62`) whenever two quats are equal. Fix is the standard safe-`where` idiom (mask the *input* of `sqrt`/division, e.g. `theta = torch.where(use_taylor, ones_like, theta2).sqrt()` used only in the non-Taylor branch). Same pattern needed in `lie/tangents.py:86-102,128-145`.

Severity: **Critical**. Small fix (S), high value; add gradcheck *at* zero to the test suite.

### 2.3 Mimic joints are silently not enforced

- `data_model/CLAUDE.md` claims: "Mimic joints: handled via tensors (`mimic_multiplier`, `mimic_offset`, `mimic_source`) with no Python branching. Mimic joints have nq=0, nv=0."
- Reality: `build_model._kind_to_joint_model` (`io/build_model.py:106-148`) never returns `JointMimic` — a URDF mimic joint keeps its base kind (e.g. `revolute`) and gets **its own independent DOF**. The `mimic_*` tensors are populated (`build_model.py:486-504`) but **no code anywhere reads them** (grep: only `model.py` field defs and `model.to()`).
- Verified: a 2-finger gripper with `j2 mimic j1, multiplier=-1` builds with `nq=2`, and FK with `q=[0.5, 0.0]` leaves j2 at identity (expected quaternion z ≈ −0.247). The Panda (`panda_finger_joint2` mimics joint1) loads with nq=9 — both fingers independent.
- `JointMimic.joint_transform` (`joint_models/mimic.py:32-34`) claims "the mimic offset is encoded in model.joint_placements" — also false; and it returns a CPU-float32 constant, a device landmine if it were ever called.

Silent wrong kinematics for any gripper/linkage URDF the user believes is constrained. Severity: **Critical** (silent), even though pinocchio's default loader is similarly permissive — the difference is this library's own docs claim it's handled.

### 2.4 Parametric / per-batch Model quantities — the better_human blocker

Verified behavior:

- **Differentiable (unbatched) `joint_placements` works today**: `dataclasses.replace(model, joint_placements=jp.requires_grad_())` → FK → backward flows gradients into `jp`. Good news — the FK composition path (`forward.py:112`, `.to()` is autograd-transparent) is parametric-shape-ready in the single-instance case.
- **Per-batch-element placements `(B, njoints, 7)` fail**: `forward_kinematics_raw` indexes `model.joint_placements[j]` (`forward.py:112`), which eats the *batch* dim instead of the joint dim → `RuntimeError: size of tensor a (25) must match ... (4)`. Same failure through `compute_frames` and RNEA (`rnea.py:167` `model.body_inertias[i]`, `crba.py:60`, `aba.py:121`, `centroidal.py:41,152`).

So per-subject SMPL betas → per-batch bone lengths is impossible without a rebuild-per-subject loop. What per-instance parametric models require (all mechanical, none deep):

1. Index model tensors as `[..., j, :]` instead of `[j]` in FK/Jacobian/dynamics (broadcasting then handles both unbatched `(njoints,7)` and batched `(B,njoints,7)`).
2. Store frame placements as a stacked `(nframes, 7)` Model tensor rather than per-`Frame` Python objects (`data_model/frame.py:22-28`, consumed one-at-a-time at `forward.py:213-216`), so they can be batched, moved by `.to()`, and composed in one vectorized op.
3. Decide semantics for batched limits/`q_neutral` and relax `_validate_q`.
4. A `Model.expand(batch_shape)` / documented `dataclasses.replace` recipe + tests.

Severity: **High** for the stated better_human goal (this is context item #3). Effort M.

### 2.5 Free-flyer FK: graph break and forced host sync on every call

`_validate_q` (`kinematics/forward.py:60-70`) executes `bool(((norm - 1.0).abs() > _QUAT_NORM_TOL).any())` for every free-flyer model on **every FK call** — including all `2·nv+1` FD-Jacobian evaluations per optimizer iteration. That is (i) a mandatory GPU→host sync on the happy path, and (ii) a `torch.compile(fullgraph=True)` breaker. Verified:

- fixed-base Panda: `torch.compile(fullgraph=True)` of `forward_kinematics_raw` **succeeds** (0.55 ms vs 2.78 ms eager, B=256 CPU);
- free-flyer Panda: **fails** — `Could not guard on data-dependent expression … Caused by: forward.py:64`.

CLAUDE.md's "torch.compile friendliness … No Python branching on tensor values" is therefore true only for fixed-base robots — false on the exact path humans/humanoids use. Note the hot-path lint (`tests/contract/test_hot_path_lint.py`) only bans `.item()`/`.cpu()`, so this `bool(...)` (and `float(norm.min())` on line 65) sails through. Severity: **High**, fix S (make validation opt-in/debug-level, or use `torch._check`).

### 2.6 Per-call H2D copies from module-level CPU constants (siblings of the fixed `so3_inverse` bug)

Commit 93b8c03 fixed exactly this pattern in `so3_inverse`, but siblings remain:

- `joint_models/revolute.py:63-65` — `_AXIS_X/Y/Z` are module-level **CPU** tensors; `_revolute_transform` does `axis.to(q_slice.device, q_slice.dtype)` per joint per FK call (`revolute.py:21`), and `_revolute_subspace` does three scalar `.to()`s plus in-place writes (`revolute.py:28-31`). On CUDA that is one H2D copy per revolute joint per FK/Jacobian call. Same pattern in `prismatic.py:16-46` and `helical.py:30-52`.
- `Model.to()` (`data_model/model.py:92-119`) moves 15 tensor fields but **not `frames`** — so `frame.joint_placement.to(device=…)` in `update_frame_placements` (`forward.py:215`) and `get_frame_jacobian` (`jacobian.py:200`) is an H2D copy per frame per call for GPU models. Frames also aren't dtype-converted by `Model.to`.
- `JointFixed/JointUniverse/JointMimic.joint_transform` return freshly constructed CPU-float32 constants (`fixed.py:25-26,63`) — currently unreached in FK (the `nq_j == 0` branch uses `_id7`, `forward.py:117-119`) but a landmine for any caller.

(Unmeasured on GPU — no working CUDA on this box — but the pattern is exactly what 93b8c03 was fixing.) Severity: **High** for GPU workloads; fixes are S (cache axes on device, include frames in `Model.to`, or better: pack axes into a Model tensor per §2.1-prep).

### 2.7 `JacobianStrategy` enum is 60 % dead and the FD fallback is unbatched

- `JacobianStrategy` (`kinematics/jacobian_strategy.py:11-25`) documents ANALYTIC/AUTODIFF/FUNCTIONAL/FINITE_DIFF/AUTO. `residual_jacobian` (`jacobian.py:230-284`) only ever checks `ANALYTIC`/`AUTO` and then unconditionally runs finite differences. Passing `AUTODIFF` or `FUNCTIONAL` silently produces FD — no `torch.func.jacrev/jacfwd` exists anywhere. The docstring "AUTO — prefer analytic, fall back to autodiff" is false (falls back to FD, as CLAUDE.md admits elsewhere).
- The FD fallback runs `2·nv` full FK passes (`jacobian.py:279-284`) *sequentially in Python* and produces an **unbatched** `(dim, nv)` matrix (`jacobian.py:279`), while its own docstring promises `(B..., dim, nv)` (`jacobian.py:236`). With batched state the `J[:, i] = …` assignment would shape-error (or silently mix batch into `dim` via `r0.numel()`, `jacobian.py:276`). Combined with `model.integrate` costing 3.1 ms/call at B=256 (measured; Python per-joint loop at `model.py:160-174`), an FD Jacobian for the 75-nv human is ~150 × (integrate+FK) ≈ order-of-a-second per iteration.

Severity: **Medium-High** (correctness of API contract + the main IK cost center). Effort S–M (implement `jacrev` or delete enum values; batch or forbid FD with batch dims).

### 2.8 FK/dynamics per-joint Python loop cost

`forward_kinematics_raw` walks `topo_order` in Python (`forward.py:110-129`); each joint issues ~10–15 small tensor ops. Measured, SMPL-like 25-joint model, CPU eager:

```
B=1:    4.16 ms   (pure Python/dispatch overhead dominated)
B=256:  5.23 ms   (20 us/sample)
B=4096: 17.9 ms   (4.4 us/sample)
```

The B=1→256 flatness shows ~4 ms of fixed per-call Python overhead. On GPU this becomes ~350 serial kernel launches with launch-gap stalls. RNEA/ABA/CRBA have the same structure plus per-joint 6×6 adjoint builds, and **rebuild the constant body-frame 6×6 spatial inertia from the packed 10-vector on every call** (`rnea.py:167`, `aba.py:121`, `crba.py:60`, `centroidal.py:152` — `Inertia(...)._to_6x6()` is q-independent and belongs in a Model-level cache). Mitigations, in order of leverage: `torch.compile` the raw passes (works today for fixed base — measured 5×), precompute constant inertia/axis tensors, and eventually pyroki-style same-kind joint batching (`references/design/pyroki.md`). Severity: **Medium** (it is *the* hot path, but compile already helps and tests pass).

### 2.9 ModelBuilder paths that cannot build

- `ModelBuilder.add_helical(...)` emits `kind="helical"` (`io/parsers/programmatic.py:298-318`), which `build_model._kind_to_joint_model` rejects: verified `IRError: Unknown joint kind 'helical'` (`build_model.py:148` — no helical case despite `JointHelical` being imported at `build_model.py:15`).
- The documented catch-all `add_joint(kind=<JointModel instance>)` (`programmatic.py:355-402`) pushes the instance's `.kind` string (e.g. `"revolute_rx"`), which `build_model` also rejects: verified `IRError: Unknown joint kind 'revolute_rx'`. So the "pass a JointModel" contract fails for every axis-aligned revolute/prismatic class, helical, composite, and mimic.
- `JointComposite` (`joint_models/composite.py`) and `JointHelical` are constructible but unreachable from any parser/builder → dead code shipped as API.
- Internal code bypasses the builder's own guard-railed API: `kinematic_tree.py:171,189` calls the private `b._push_joint(...)` directly — a sign the named-helper-only contract is fighting its own package.

Severity: **Medium** (crash, not silent), effort S–M.

### 2.10 Docs/roadmap vs code disagreements (beyond those above)

`docs/reference/roadmap.md:8` promises "If a symbol is *not* on this page, it is implemented and tested." Counterexamples:

- `kinematics/chain.py:16` `get_chain` → `NotImplementedError`, not on roadmap.
- `data_model/indexing.py:17` `build_name_to_id` → `NotImplementedError`, not on roadmap — and dead anyway (build_model builds the dicts inline, `build_model.py:473-475`).
- `dynamics_ops.py:3` docstring: "Most paths still raise NotImplementedError until P11 lands" — false; all four forward to live implementations.
- `rnea.py:111` sets `data._kinematics_level = 1` (raw int) where every other site uses the enum — harmless today but shows the invariant is maintained by hand.

Severity: **Low-Medium** individually; collectively they mean the roadmap/docs cannot be trusted as a stub inventory (relevant to this assessment's ground rules).

### 2.11 Pre-release legacy machinery

The library is unreleased with zero external users, yet carries: 18 deprecated `Data` property shims with `DeprecationWarning`s (`data_model/data.py:42-61, 255-281`), a deprecated `nle` alias (`rnea.py:229`), an `IR_SCHEMA_VERSION` handshake for pickled IRs that nothing pickles (`io/ir.py:70-93`, `build_model.py:232-237`), and a `tests/contract/test_no_legacy_strings.py` to police the old names. This is release-management theater for a v0. Severity: **Low** (bloat), effort S to delete.

### 2.12 `spatial/` value types are a parallel implementation the hot paths avoid

`Motion.cross_motion/cross_force` (`spatial/motion.py:49-86`) build 3×3 hat matrices + matmuls; `dynamics/rnea.py:41-68` re-implements the same operators as `_cross_motion/_cross_motion_force` with `torch.linalg.cross` (faster) and dynamics uses only those (grep: no `Motion(`/`Force(`/`spatial.ops` use outside `spatial/`). Only `Inertia` is actually consumed by dynamics. Two implementations of the same math, one of them (the public one) the slow one; `spatial/ops.py` (60 lines) has zero internal users. Severity: **Low-Medium** (drift risk), effort S–M to unify.

### 2.13 Data cache-invariant machinery: high ceremony, three consumers

The `KinematicsLevel` enum + `__setattr__` interception + three cache buckets + `invalidate`/`require` (`data_model/data.py:63-211`) exist so that exactly **three** call sites can raise a nicer error (`jacobian.py:94,114,154`). Producers bypass it (`object.__setattr__` at `forward.py:186`; raw int at `rnea.py:111`), in-place mutation is undetected by design (`data.py:17-19`), and `_model_id` is stored but never validated anywhere (grep: only constructors). The machinery isn't wrong, but it's ~90 lines of magic for what "field is None → call FK first" already communicates. Severity: **Low** (taste + maintenance), noted because it's characteristic of the codebase's speculative-ceremony pattern.

### 2.14 `dynamics/action/` — Crocoddyl skeleton with no consumer

`action/` (~290 LOC across `action.py`, `differential.py`, `integrated.py`) + `state_manifold.py` implement the Crocoddyl 3-layer split, but no DDP/iLQR solver exists (`action/action.py:10-14` admits this), nothing outside its own tests imports it (grep), and `calc_diff` uses `torch.autograd.functional.jacobian(..., vectorize=False)` per knot (`differential.py:90,96`, `integrated.py:52,103`) — unusably slow for real OC. `DifferentialActionModelFreeFwd.forward_dynamics` allocates a fresh `Data` per call (`differential.py:104`). Speculative scaffolding. Severity: **Medium** (bloat + a false "implemented" signal in CLAUDE.md's feature list), effort S to park.

---

## 3. Suspected problems needing verification

- **GPU behavior unmeasured.** All sync/H2D findings (§2.5, §2.6) are code-derived; magnitude on a real GPU (especially the per-joint `_AXIS_X.to(cuda)` copies and the free-flyer `bool()` sync inside LM/FD loops) should be profiled on a CUDA box before sizing the fix effort.
- **FD Jacobian with batched states in `optim/`**: if any optimizer path passes batched `ResidualState` into the FD fallback, it either crashes or silently mis-shapes (§2.7). The optim/tasks audit should trace this.
- **Deep autograd graphs from sequential FK composition**: `world_list[j] = compose(world_list[parent], …)` builds an O(depth) chain; combined with trajectory horizons this may produce slow backwards. Plausible, unmeasured.
- **`Model.meta = {"ir": ir, ...}`** (`build_model.py:591`) retains the entire IR (including geoms/meshes references) on every Model; probably intentional for the viewer, but it makes `Model` heavy to pickle/copy and leaks parser types into the data model. Verify viewer actually needs the whole IR rather than just geoms.
- **`test_pinocchio/` parity coverage breadth**: the suite exists (FK/RNEA/ABA/CRBA/centroidal/frame-Jacobian vs pinocchio, incl. `test_rnea_advanced_joints.py`) and is the right kind of test; I did not run it or check which joint kinds/batches it covers. Spot-run before trusting dynamics on spherical/planar joints.
- **`JointPlanar.joint_transform` sign handling**: `sign(sin_t)` at `sin_t == 0`, θ = π gives `half_sin = 0` → wrong pose at exactly θ=π (`joint_models/planar.py:33-35`); also non-differentiable `torch.sign`. Suspected edge-case bug, unverified.
- **`se3_log` near θ = π**: `cot_half` blows up as θ→π (`_torch_native_backend.py:301`); value is clamped but gradient quality near π is untested.

---

## 4. What is actually good and should be kept

- **The Model/Data split and the "free-flyer is just joint 1" design.** Single code path for fixed/floating base genuinely holds throughout FK/Jacobian/dynamics (`forward.py`, `rnea.py` never branch on base type). This is the right architecture, matching pinocchio/pyroki/brax consensus (`references/design/BEST_PRACTICES.md:15`).
- **`forward_kinematics_raw` / `_compute_joint_jacobians_raw` as tensor-only primitives.** Pure `(model, q) → tensors` signatures are exactly the seam a Warp/`torch.compile` future needs (§2.1). Verified: fixed-base FK compiles `fullgraph=True` unmodified, 5× faster.
- **The `JointModel` per-kind dispatch design.** Stateless per-kind objects with `joint_transform/joint_motion_subspace/integrate/difference` keep all kind-specific logic out of the hot loops; adding a joint kind is genuinely local. The implementation needs the axis-packing fix, but the shape is right.
- **The Lie math core (values, not gradients-at-zero).** Scalar-last convention enforced everywhere, Shepperd 4-branch matrix→quat (`_torch_native_backend.py:72-122`), Taylor-stitched *values* correct, fp64 gradcheck near singularities passes, dropping PyPose was correct.
- **The pinocchio-parity test strategy** (`tests/test_pinocchio/`) — real reference implementation, no mocking. This is the most valuable QA asset in the repo.
- **IR → `build_model` → frozen Model pipeline.** Drake-style suffix dispatch (`io/__init__.py:36-110`), flat order-free IR, one factory with explicit topology invariant checks (`build_model.py:46-93`) — right-sized, not over-abstracted (the answer to key question 6 is: io is mostly fine; the defects are the specific kind-mapping/mimic bugs, not the architecture).
- **`build_kinematic_tree_model` / `make_smpl_like_body`** — the better_human entry point exists, takes per-body mass/com/inertia, and works (verified build + FK). With the §2.4 indexing fix it becomes genuinely useful.
- **Contract tests as an idea** (layering via AST, hot-path lint, frozen public API) — keep, but extend the lint to catch `bool(tensor)`/`float(tensor)` (§2.5 escaped it).
- **Featherstone implementations read correctly** (body-frame RNEA two-pass matches RBDA; ABA three-pass with `U/D/u` factorization matches pinocchio's local convention; CRBA ancestor walk correct; LWA frame handling in `get_frame_jacobian` and its documented adjoint pitfall are right, `jacobian.py:210-227`).

---

## 5. Recommendations

| # | Change | Effort | Risk |
|---|--------|--------|------|
| R1 | **Delete the Backend/LieOps/KinematicsOps/DynamicsOps Protocol layer + registry + warp stubs (~530 LOC).** `lie/se3.py`/`so3.py` call `_torch_native_backend` directly (pattern already present); `forward_kinematics`/`compute_joint_jacobians` call the `_raw` functions directly. Keep the `_raw` tensor-only functions as the designated future backend seam; reintroduce whole-pass dispatch only when a Warp kernel exists. | M | Low — two internal call sites; `set_backend` today can only pick the default anyway. |
| R2 | **Fix NaN-at-singularity gradients** in `so3_exp/so3_log/se3_exp/se3_log` and `tangents.py` coefficients via safe-`where` (mask sqrt/div inputs). Add gradchecks *at* θ=0 and at identity, incl. 2nd order. | S | Low |
| R3 | **Make Model quantities batch/parametric-capable**: index `joint_placements`/`body_inertias` as `[..., j, :]` everywhere; stack frame placements into a `(nframes, 7)` Model tensor (also fixes the `Model.to()` frames gap); document + test `dataclasses.replace`-based parametric models. This is the better_human enabler. | M | Medium — touches every recursion; pinocchio-parity suite is the safety net. |
| R4 | **Mimic joints: enforce or reject.** Either implement the q-expansion gather before FK (PyRoki trick the tensors were built for) or raise `NotImplementedError` in `build_model` when `mimic_source` is set. Never silent. Fix the false claims in `data_model/CLAUDE.md`. | S (reject) / M (implement) | Low |
| R5 | **Remove the data-dependent `bool(...)` validation from the FK hot path** (opt-in debug flag or `torch._check`), restoring fullgraph-compilability and sync-free FK for free-flyer models. Extend hot-path lint to ban `bool(`/`float(` on tensors. | S | Low |
| R6 | **Kill remaining per-call H2D/constant construction**: pack joint axes + kind codes into Model device tensors (doubles as Warp prep, §2.1-3); precompute constant 6×6 body spatial inertias on Model (or first-use cache) for RNEA/ABA/CRBA/centroidal. | S–M | Low |
| R7 | **Rationalize `JacobianStrategy`**: implement AUTODIFF/FUNCTIONAL via `torch.func` or delete them; make FD batched or raise on batch dims; fix the return-shape docstring. | S–M | Low |
| R8 | **Park `dynamics/action/` + `state_manifold`** until a DDP solver is planned (move to an `experimental/` namespace or delete; keep the git history). | S | Low |
| R9 | **Delete pre-release legacy machinery**: 18 `Data` alias shims, `nle`, IR schema-version handshake, no-legacy-strings test. | S | Low |
| R10 | **Fix ModelBuilder**: map `JointModel`-instance kinds (or carry the instance through IR), wire or remove helical/composite; stop internal `_push_joint` bypass. | S–M | Low |
| R11 | **Unify spatial cross-operators**: make `Motion.cross_*` delegate to the fast `linalg.cross` implementations (single source of truth), or demote `spatial/ops.py` value-type layer to a thin documented convenience. | S | Low |
| R12 | **Simplify Data cache machinery** to "None ⇒ not computed" + explicit errors, dropping `__setattr__` interception unless the invalidation is extended to producers consistently. (Do alongside R3 since both touch `Data`.) | M | Medium (behavioral, but only 3 consumers) |
| R13 | For FK throughput: apply `torch.compile` to the raw passes as the near-term win (verified working, 5× on CPU fixed-base — blocked for free-flyer only by R5), and evaluate same-kind joint batching (pyroki-style) as the long-term GPU answer. | M–L | Medium |

Suggested order: R2 → R5 → R4 (correctness, all S) → R1 + R6 (structure/perf, unlocks Warp-prep) → R3 (better_human) → the rest.

---

## 6. Open questions

1. **Warp commitment level.** If the plan is "move FK/dynamics fully to Warp kernels with torch autograd at the boundary," the right prep is R1+R6 (flat device metadata + whole-pass seams + `torch.autograd.Function` wrappers later). If instead the plan is "stay torch-native, maybe compile," R13 changes priority. The current Protocol layer serves *neither* future well.
2. **Should mimic joints be first-class** (gather-based, nq=0 as documented) or explicitly unsupported? Affects R4 sizing and the Panda gripper story.
3. **What are the batched-Model semantics for limits/neutral** when shape varies per batch element (R3)? Per-batch limits complicate the LM clamp step.
4. **Is `Model.meta["ir"]` retention** (full IR incl. geom params on every Model) intentional API for the viewer, or should Model carry a slimmed geom table?
5. **How much of `test_pinocchio/` runs in CI and over which joint kinds/batch shapes?** Parity coverage for spherical/planar/free-flyer batched dynamics determines how safe R3's indexing sweep is.
6. **GPU profiling pass** (blocked here by the broken CUDA driver): quantify §2.5/§2.6 before/after to confirm the sync/H2D fixes matter as much as the code reading suggests.
7. **`get_chain`/`build_name_to_id` stubs**: implement or delete? They contradict the roadmap's "not on this page ⇒ implemented" guarantee either way.
