# M3 — Parametric Model Breadth: Agent Execution Instructions

> Read `plan/for_agents/README.md` first. It carries the standing rules
> (deletion ordering, honesty rules, kernel requirements, test commands).
> All paths below are relative to `/data3/rikhat.akizhanov/better/BetterRobot`
> unless prefixed with a consumer-repo name.

## Mission

Make `ModelValues` (joint placements, body inertias, frame placements)
**batched and differentiable across every compute pass**, so one topology
can carry a batch of shapes whose parameters flow gradients — the thing
`better_human` needs to fit body shape (`betas`) through BR's FK/dynamics.
Concretely: (1) sweep every dim-0 model-tensor index to `[..., j, :]` under
a *defined* q-batch × value-batch broadcast contract; (2) make
`Model.with_values(...)` a public, validated, batched rebind primitive and
move frame placements into a batched table; (3) implement the mimic-joint
reduced coordinate map across FK/Jacobians/dynamics/limits (M0 only made
mimic *loading* fail fast — this makes it *work*); (4) add an
order-preserving build path plus a vectorized `q_permutation`; (5) add
human-joint residuals (swing/twist limits, rotation prior) and a
differentiable batched `Inertia.from_mesh`; (6) vectorize
`integrate`/`difference` by joint kind. The safety net — an extended
pinocchio-parity suite — lands **before** any indexing change touches a
pass. Done when a `better_human` prototype builds a betas-parameterized SMPL
skeleton that runs batched FK + IK with gradients flowing to `betas`, and
BVR's `smplx_robot/` and BHF's remap shims become deletable.

## Prerequisites

M3 extends the M1 seam in *breadth only*; it consumes M2's evaluation
protocol. Confirm all of the following before starting.

- **M0 landed** (`m0_truth_and_correctness.md`):
  - θ=0 gradient fix. Verify:
    ```bash
    uv run python -c "
    import torch; from better_robot.lie import so3
    g = torch.autograd.grad(so3.exp(torch.zeros(3, requires_grad=True)).sum(),
                            torch.zeros(3, requires_grad=True))[0]" 2>/dev/null; \
    uv run python -c "
    import torch; from better_robot.lie import so3
    x=torch.zeros(3,requires_grad=True)
    g=torch.autograd.grad(so3.exp(x).sum(),x)[0]
    assert torch.isfinite(g).all(), f'theta=0 NaN — M0 not landed: {g}'
    print('M0 theta=0 OK')"
    ```
    As of 2026-07-17 this prints `[nan, nan, nan]` — **M0 has NOT landed**
    (`lie/_torch_native_backend.py:148` still computes
    `theta = theta2.clamp(min=0).sqrt()`). T3.6 and T3.7 manifold gradchecks
    require it.
  - Mimic **reject at build**. `grep -n mimic src/better_robot/io/build_model.py`
    should show a `raise` guiding the user (M0 item 3). As of 2026-07-17
    there is **no reject** — `build_model.py:493-504` silently resolves the
    arrays. T3.4 replaces that reject with real enforcement, so M0's reject
    must exist first (else T3.4 has nothing to swap out and the library is
    still lying — see the false claim still live in
    `data_model/CLAUDE.md:33`).
- **M1 landed** (`m1_two_lane_seam_and_hygiene.md`, not yet written as of
  2026-07-17 — confirm its actual symbol names when it exists):
  - The structure/value split exists:
    `grep -rn "class ModelStructure\|class ModelValues" src/better_robot/`
    must hit. As of 2026-07-17 **neither exists** — `Model` is still one
    frozen dataclass holding the value tensors directly
    (`data_model/model.py:62-85`). **Every "`values.joint_placements`" below
    is the post-M1 path; on today's tree the same tensor is
    `model.joint_placements`. Confirm M1's landed attribute path first and
    translate.**
  - The **execution-batch ABI is frozen** (03 §2.4): flat `E`, per-input
    batch-index maps, in-kernel shared-value gradient reduction, and its
    tests (batched-q × unbatched-values, converse, multi-axis, mismatch
    errors) pass. **M3 extends this ABI's breadth across passes; it never
    changes the ABI.** `m1_two_lane_seam_and_hygiene.md` owns these tests;
    T3.2 adds *value-batched* cases under the same contract.
  - The first FK Warp kernel exists (opt-in, warp-CPU parity). T3.2 adds
    its value-batched coverage tests.
  - Per-call constants hoisted (M1 item 6): the per-joint `.to()` inside
    FK/RNEA loops should already be gone. If M1 did **not** remove them,
    T3.2 removes the ones on `joint_placements`/`body_inertias` as part of
    the indexing sweep (they are at `forward.py:112`, `rnea.py:167`,
    `crba.py:60`, `aba.py:121`, `centroidal.py:41,152` today).
- **M2 landed** (`m2a`/`m2b`/`m2c`): `with_values` and the provider hooks
  must fit `m2a_variable_blocks_and_slice.md`'s `Problem`/provider design —
  `RobotStateProvider(model, var="q")` (T2a.4) runs FK on a `model`, and
  `RobotConfig(model)` (T2a.1) wraps `model.integrate`/`difference`. T3.3's
  `with_values` is what a `better_human` provider calls to swap in
  betas-derived values before FK; T3.7's vectorization sits *under*
  `RobotConfig` (which "wraps what exists" per m2a T2a.1's pitfall). The
  done-when's *batched IK* is M2b/M2c machinery — M3 only makes the *model
  values* it consumes batched. Verify M2a's public `Problem` exists and its
  contract-test freeze passed.
- **Consumer repos** readable at
  `/data3/rikhat.akizhanov/better/BetterVideoReconstruction` (BVR) and
  `/data3/rikhat.akizhanov/better/BetterHumanForce` (BHF), plus
  `/data3/rikhat.akizhanov/better/better_human`. **Read-only evidence** —
  never modify them here (deletion of their shims is M4).

## Sizing & parallelism

Roadmap label: **M–L**. Seven tasks (roadmap M3's six items, with the
parity safety net split out as T3.1 so it can land first).

Dependency order (explicit):

```
T3.1  (safety net) ─── MUST land first; blocks any pass-touching task
   │
   ├─► T3.2 (indexing sweep + broadcast) ─► T3.3 (with_values + frame table)
   │
   ├─► T3.4 (mimic reduced map)     [start after T3.2 so passes already index [...,j,:]]
   │
   ├─► T3.5 (order-preserving build)   ─┐
   ├─► T3.6 (human residuals + from_mesh)├─ independent; parallel after T3.1
   └─► T3.7 (vectorize integrate/diff) ─┘
```

Ordered chain: **T3.1 → T3.2 → T3.3**. **T3.4** touches every pass, so it
lands after T3.2's `[..., j, :]` sweep to avoid two agents editing the same
lines. **T3.5, T3.6, T3.7** are independent of the value spine and of each
other — hand them to parallel agents once T3.1 is green. T3.6's
`Inertia.from_mesh` feeds T3.3's `body_inertias` but does not block it (a
static inertia tensor works until `from_mesh` exists).

Effort per task: T3.1 **M**, T3.2 **M–L**, T3.3 **M**, T3.4 **L** (spans
FK+Jacobians+RNEA+ABA+CRBA+centroidal+limits+indexing), T3.5 **S–M**, T3.6
**M**, T3.7 **S–M**.

**Coordination with `better_human` (scope fence, `plan/02_vision.md`).** BR
provides the *hooks*; body-model specifics stay outside BR. The betas →
joint-placements/inertias mapping (LBS, `shapedirs`, `J_regressor`,
landmark tables, density tables, `from_classic` adapters) lives in
`better_human` and **must not** enter BR core. Validate M3's hooks with
BR-side **synthetic** tests (a spherical-joint tree from
`build_kinematic_tree_model` with batched placements from a fake "betas"
leaf) — do **not** import `better_human` into BR tests. The done-when's
"betas-parameterized SMPL skeleton" is a `better_human` deliverable that
exercises these hooks; M3's acceptance is the hooks + their synthetic
tests.

## Tasks

### T3.1 — Parity-suite safety net: audit, then extend  [M] — FIRST

**Goal / done-when:** the pinocchio-parity suite (`tests/test_pinocchio/`)
is extended over **joint kinds × batch shapes × base types** *before* any
pass indexing changes, so it can catch a regression in the T3.2/T3.4
refactors. Assessment §4 flags current breadth as **suspect** —
**the executor audits the actual coverage and reports the matrix before
writing any new case.**

**Current state (audited 2026-07-17 — reproduce and confirm, don't
trust):**
- Robots exercised: Panda (fixed base; 7 revolute + 2 prismatic fingers),
  G1 (free-flyer + revolute, `test_rnea_advanced_joints.py:25-33`), a
  2-body programmatic spherical + `revolute_rz` chain
  (`test_rnea_advanced_joints.py:76-116`).
- Joint kinds exercised: revolute (generic + `rz`), prismatic (Panda
  fingers), free-flyer, spherical. **Not exercised:** planar, translation,
  helical, `revolute_unbounded`, `revolute_unaligned`, mimic.
- Batch shapes: **FK parity is UNBATCHED** (`q = qs[i]`, single config,
  `test_fk_matches_pinocchio.py:19`). RNEA/CRBA/ABA have **single-axis**
  batched-q tests only (`test_rnea_matches_pinocchio.py:165` batch `(3,)`;
  `test_crba_matches_pinocchio.py:79` and `test_aba_matches_pinocchio.py:72`
  batch `(4,)`). **No multi-axis batch. No batched-VALUES parity anywhere.**
- Base types: fixed (Panda, spherical chain) + free-flyer (G1). Both.
- dtype: **fp64 only** (~66 `float64` refs vs 2 `float32`). No fp32 parity.

So today's suite is a safety net for *q-batched dynamics on one shape*, and
nothing else — exactly the gap Assessment §4 warns about. It does **not**
cover the batched *model values* that T3.2/T3.3 introduce.

**Implementation plan:**
1. **Audit and report first.** Run a coverage sweep
   (`grep`/read across `tests/test_pinocchio/`) and produce the matrix
   {joint kind} × {batch shape: unbatched / 1-axis-q / multi-axis-q /
   batched-values} × {base type: fixed / free-flyer} × {dtype: fp32 /
   fp64}. Post it in the PR description. This is the gate that stops the
   sweep from proceeding on a false sense of safety.
2. **Batched-q FK parity.** Add `test_fk_batched_matches_pinocchio` and a
   frame-Jacobian batched twin: BR takes a `(B, nq)` (and a `(B1, B2, nq)`
   multi-axis) `q`; pinocchio is looped per slice; compare. This closes the
   biggest gap — FK is the pass T3.2 changes most.
3. **fp32 parity band.** Add fp32 variants of FK + one dynamics pass at the
   fp32-appropriate tolerance (the existing fp32-URDF FK band is `atol=2e-6`,
   `test_fk_matches_pinocchio.py:37`; do **not** apply fp32 tolerances to the
   fp64 tests — keep their `1e-10`/`1e-12`).
4. **Missing joint kinds** where a pinocchio equivalent exists: planar
   (`pin.JointModelPlanar`), translation (`pin.JointModelTranslation`),
   `revolute_unbounded` (`pin.JointModelRUB*`), helical
   (`pin.JointModelHelicalUnaligned`). Build the 2-body programmatic chain
   pattern (`_build_spherical_chain`, `test_rnea_advanced_joints.py:76`) for
   each. Where no clean pinocchio equivalent exists, add a **property test**
   (roundtrip / group-axiom, jaxlie-style) instead — record which kinds got
   which oracle.
5. **Value-batched parity uses a BR-vs-BR-loop oracle, not pinocchio**
   (pinocchio cannot batch `joint_placements`). Scaffold it here as a helper
   `assert_value_batched_matches_loop(...)` but the *cases* land with T3.2
   (they need `with_values`/`[...,j,:]`). Note the cross-reference so T3.2's
   executor uses this helper.

**What to test:** all new cases live under `tests/test_pinocchio/`
(pinocchio-oracle) and `tests/kinematics/` / `tests/dynamics/` (property /
value-batched). Every existing parity test stays green and untouched (they
are the reference the sweep must not break). fp32 tests obey the
`write-tests` skill's float32 rule.

**Pitfalls / do-not-forget:**
- This suite is the *only* thing that makes T3.2/T3.4 safe — the audit
  report is not optional ceremony; it decides what the sweep is actually
  protected against.
- Do not weaken any existing tolerance to make a new case pass; add a new
  test with its own band.
- pinocchio is a dev-only dependency (`pytest.importorskip`, `conftest.py:12`)
  — new cases must skip cleanly when it is absent.

### T3.2 — The `[..., j, :]` indexing sweep + broadcast contract  [M–L]

**Goal / done-when:** every model-tensor index across
FK/Jacobians/RNEA/ABA/CRBA/centroidal reads `[..., j, :]` (trailing dim),
so values may carry leading batch dims and `requires_grad`. The **execution
batch is DEFINED as the broadcast of the q batch with every value batch**;
`Data` allocation derives from that broadcast (today from `q.shape[:-1]`
alone). Tests cover batched-values × unbatched-q, the converse, multi-axis,
and mismatch (which raises a clear error).

**Current state:**
- Only FK indexes placements on dim 0:
  `model.joint_placements[j].to(device, dtype)` (`forward.py:112`) — dim-0
  index **plus** a per-call `.to()`.
- Dynamics index **`body_inertias[i]`** on dim 0 (the audit's crash sites —
  verified they index inertias, *not* placements): `rnea.py:167`,
  `crba.py:60`, `aba.py:121`, `centroidal.py:41` and `:152`, each with a
  per-call `.to()`.
- `Data.batch_shape` returns `q.shape[:-1]` (`data.py:220`); FK allocates
  `Data` with `batch_shape=tuple(q.shape[:-1])` (`forward.py:174`). So a
  batched model with unbatched q has **no defined behavior**.
- EXP2/EXP4 (`audit_better_human_integration.md §1.2`): `(4, njoints, 7)`
  placements and `(4, nbodies, 10)` inertias both crash with a dim-0
  size-mismatch. Autograd through *unbatched* placements/inertias already
  works (EXP1/EXP3) — this is an indexing + allocation change, not an
  algorithm rewrite.
- `se3.compose` already broadcasts `(...,7)×(...,7)`
  (`lie/_torch_native_backend.py:238-247`), and Jacobians consume only
  `data.joint_pose_world` + motion subspaces (`jacobian.py:30-79`), so once
  FK accepts batched placements the analytic Jacobians and every
  pose/position residual work **with zero changes** — except the frame
  placement read (`jacobian.py:200`), which T3.3 moves to the table.

**Implementation plan:**
1. **Indexing.** Replace `model.joint_placements[j]` →
   `values.joint_placements[..., j, :]` (`forward.py:112`) and
   `model.body_inertias[i]` → `values.body_inertias[..., i, :]`
   (`rnea.py:167`, `crba.py:60`, `aba.py:121`, `centroidal.py:41,152`).
   On the post-M1 tree these read from `ModelValues`; on today's tree they
   read from `model` — confirm M1's landed path first.
2. **Drop the per-call `.to()`** on those reads (M1 item 6 should already
   have; if not, do it here). The value tensors live on the model/values
   device; a one-time device/dtype check at pass entry replaces the
   per-joint `.to()`.
3. **The broadcast contract — the load-bearing part (codex B4, A10).**
   Compute the execution batch once at each pass entry:
   ```python
   # sketch — direction, not final:
   E = torch.broadcast_shapes(
       q.shape[:-1],
       values.joint_placements.shape[:-2],
       values.body_inertias.shape[:-2],
       values.frame_placements.shape[:-2],   # T3.3
   )
   ```
   `Data` allocation uses `E`, not `q.shape[:-1]`. A shared (unbatched)
   value is passed once and *indexed/broadcast*, never physically repeated.
   This is the **frozen M1 ABI's** torch-lane view — do not invent a new
   batching scheme; mirror the flat-`E` + batch-index-map contract M1
   already tests, extended to these passes.
4. **Mismatch → honest error**, not a raw torch broadcast failure deep in
   `se3.compose`. Draft wording:
   > `ShapeError: cannot broadcast q batch (4,) with joint_placements batch
   > (3,) — model-value batch dims must be right-aligned-broadcastable
   > against the q batch. For (B, T) trajectories with per-person shapes,
   > pass placements as (B, 1, njoints, 7). See docs on the value-batch
   > broadcast contract.`
   Note the heterogeneous-rank trap (audit S-1): `(B, njoints, 7)`
   placements + `(B, T, nq)` q mispair `B` with `T` under naive
   right-alignment. Decide the rule and test it: **implement torch-native
   right-aligned broadcast** and require the caller to unsqueeze
   (`(B, 1, njoints, 7)`); if `with_values` should auto-unsqueeze instead
   (audit open Q1), that is an **owner decision** — produce the two-option
   writeup and STOP for review rather than picking silently.
5. **Warp lane.** The M1 FK kernel gains **value-batched coverage tests
   here** (roadmap M3 item 1): parity + gradcheck (warp-CPU) with batched
   `joint_placements` against the torch lane, under the same frozen ABI. No
   kernel-facing ABI change — breadth only. If M1's kernel does not yet
   accept batched placements as differentiable inputs, that is an M1 gap to
   report, not to redesign here.

**What to test** (`tests/kinematics/test_batched_values.py`,
`tests/dynamics/test_batched_values.py`, using T3.1's
`assert_value_batched_matches_loop`; float32 per `write-tests`, plus fp64
tightening):
- **Oracle = per-slice loop.** For a batch of value slices, build
  `model.with_values(slice_k)` (T3.3), run the unbatched pass, stack, and
  compare to one batched call. This is the correct oracle (pinocchio can't
  batch values).
- batched-values × unbatched-q; batched-q × unbatched-values (existing
  behavior preserved); multi-axis `(B1, B2)` values; q `(B, T, nq)` with
  `(B, 1, njoints, 7)` placements.
- Mismatch cases raise the drafted `ShapeError` (exact message asserted).
- **Autograd:** a `(B, njoints, 7)` placement leaf → batched FK →
  `.backward()` fills `.grad` (batched EXP1); same for `(B, nbodies, 10)`
  inertias → RNEA (batched EXP3).
- Warp-lane value-batched parity/gradcheck (warp-CPU), skipped when warp
  absent.

**Pitfalls / do-not-forget:**
- **Do not change the frozen ABI.** If a case seems to need a new batching
  rule, it is a T3.1/M1 test gap, not an ABI change. Cross-referenced from
  `m1_two_lane_seam_and_hygiene.md` (freezes the ABI) — M3 is breadth only.
- `Data.batch_shape` (`data.py:220`) documents itself as `q.shape[:-1]`;
  fix its docstring when the source of truth becomes the execution batch.
- Keep `se3.compose`'s broadcasting — it is the reason FK math needs only
  the index change. Don't "optimize" it into a fixed-shape op.
- Frame placements (`jacobian.py:200`, `forward.py:215`) are still per-frame
  `(7,)` structs until T3.3 — do not batch them here; leave the frame path
  for T3.3 so the two changes don't collide.

### T3.3 — Public `Model.with_values(...)` + frame table + exhaustive `.to()`  [M]

**Goal / done-when:** `Model.with_values(...)` is public, validates
shapes/devices, is batched and tested; frame placements move into a
batched table `frame_placements: (*value_batch, nframes, 7)` in
`ModelValues` with parent-joint indices in `ModelStructure`;
`update_frame_placements` is vectorized; `Model.to()` moves frames; `.to()`
is exhaustive and casts float values only (never index/kind tensors);
inertia 6×6 caching applies to static values only, live parametric
inertias derive the 6×6 form once per evaluation context.

**Current state:**
- **No** `with_values`/`with_params` exists (grep: NONE). Autograd through
  placements/inertias already works via `dataclasses.replace` (EXP1/EXP3,
  `audit_better_human_integration.md §4`) — this task makes it public,
  validated, batched.
- `Frame.joint_placement` is a frozen `(7,)` struct (`frame.py:27`).
  `update_frame_placements` is a **per-frame Python loop** with a per-frame
  `.to()` (`forward.py:213-216`) — "one third of FK+frames time"
  (`01_assessment.md §2.4`). `get_frame_jacobian` reads
  `frame.joint_placement` (`jacobian.py:200`).
- `Model.to()` (`model.py:92-119`) moves `joint_placements`,
  `body_inertias`, all limits, mimic tensors, `q_neutral`,
  `reference_configurations` — but the `frames` tuple (`model.py:81`) is
  passed by reference, so **frame placements are never moved** (the
  cautionary example in 03 §5; confirmed).
- `Data._model_id = id(self)` is set (`model.py:133`) but **never checked**
  (audit S-4 — grep-confirmed no consumer). `with_values` returns a new
  `Model` with a new `id()`.
- `Inertia._to_6x6()` (`spatial/inertia.py:155`) rebuilds the 6×6 from the
  packed 10-vector every dynamics call.

**Implementation plan:**
1. `Model.with_values(*, joint_placements=None, body_inertias=None,
   frame_placements=None) -> Model` — sketch (direction, not final;
   roadmap/03 §5 name it `with_values`, audit R1 named it `with_params` —
   use `with_values`):
   - Validate trailing event shapes `(njoints, 7)` / `(nbodies, 10)` /
     `(nframes, 7)`; validate device matches the structure device;
     renormalize the quaternion slice of any placement input. Leading dims
     are the value batch (any rank).
   - Return a new `Model` pairing the same `ModelStructure` with the new
     `ModelValues` (`dataclasses.replace` under the hood — autograd flows,
     verified). O(1), **no rebuild**, no topology recompute.
   - Validation errors are explicit (name the offending field and both
     shapes).
2. **Frame table.** Add `frame_placements: (*value_batch, nframes, 7)` to
   `ModelValues`; add `frame_parent_joints: int32 (nframes,)` to
   `ModelStructure`. `Frame` objects keep `name`/`parent_joint`/
   `frame_type` as **metadata views** — the placement's source of truth is
   the table. Vectorize `update_frame_placements`:
   ```python
   T_parent = joint_pose_world[..., frame_parent_joints, :]   # (E..., nframes, 7)
   frame_pose_world = se3.compose(T_parent, values.frame_placements)  # one batched compose
   ```
   Replace the `forward.py:213-216` loop and update the frame-Jacobian read
   (`jacobian.py:200`) to gather the table row. This gives **shape-dependent
   markers** for free: a `better_human` landmark set is just extra rows in
   the table, and every existing `PositionResidual`/`PoseResidual` + analytic
   Jacobian serves them unmodified (audit CP-5/R3).
3. **Exhaustive `.to()`.** Move `frame_placements` (and every float value)
   with the other floats; **never** cast index/kind tensors
   (`frame_parent_joints`, `parents`, `nqs`, joint-kind codes stay their
   integer dtype). Test that `.to(dtype=float64)` leaves int tensors int and
   that frames now move device.
4. **Inertia caching vs differentiability (03 §5, codex B4).** Precomputed
   6×6 spatial inertias (M1 item 6 / perf §7 step 2) apply to **static**
   `body_inertias` only. When `body_inertias` is a live parametric tensor
   (batched via `with_values` and/or `requires_grad`), derive the 6×6 form
   **once per evaluation context** (per pass call) so gradients stay
   attached and caches cannot go stale. Branch on identity: if
   `values.body_inertias is` the build-time static tensor → use the cached
   6×6; else derive fresh via `Inertia(...)._to_6x6()`.
5. **Resolve `Data._model_id`** (audit S-4): it is dead weight that *appears*
   to forbid the replace/`with_values` pattern while checking nothing. Either
   drop it, or validate against structure identity (not `id()`). Recommend
   drop; if kept, `with_values` must set it consistently. Record the choice.

**What to test** (`tests/data_model/test_with_values.py`,
`test_frame_table.py`, `test_model_to.py`):
- `with_values` shape/device validation errors (wrong `njoints`, wrong
  trailing dim, device mismatch) — exact messages; quaternion
  renormalization applied.
- Autograd end-to-end: a `(B, nb)` fake-`betas` leaf → a mapping →
  `with_values(joint_placements=..., body_inertias=...)` → batched FK + RNEA
  → `.backward()` fills the leaf's `.grad` (this is the synthetic stand-in
  for the done-when; **no `better_human` import**).
- Frame table: add a marker frame; `PositionResidual` on it works and its
  analytic Jacobian matches FD; vectorized `update_frame_placements` matches
  the old per-frame-loop result (parity, tight tol).
- `Model.to(device/dtype)`: frames move (frame FK computes on the new device
  with no H2D copy); int tensors keep their dtype; round-trip identity.
- Inertia caching: a static model reuses the cached 6×6 across calls (assert
  tensor identity); a parametric-inertia model derives fresh and grad flows
  to the inertia leaf.

**Pitfalls / do-not-forget:**
- `Frame.joint_placement` is read at `jacobian.py:200` and `forward.py:215`
  — **both** must switch to the table, or the frame Jacobian silently reads
  a stale struct.
- BVR/BHF build models via `build_kinematic_tree_model` (`io.builders`) —
  `with_values` and the frame table must keep those models working; run the
  consumer import-smoke check (standing rule 1).
- The layer DAG: `with_values` lives on `Model` (`data_model`); it must not
  import from `kinematics`/`optim`. Frame gather uses only `data_model` +
  `lie`.
- Keep `Frame` frozen — the table is the mutable-per-query surface (in
  `ModelValues`/`Data`), `Frame` stays immutable metadata.

### T3.4 — Mimic-joint reduced coordinate map  [L]

**Goal / done-when:** the mimic relationship is *enforced* across
FK/Jacobians/dynamics/limits/torque accumulation and all nq/nv indexing.
Acceptance is an **explicit reduced-map assertion** — a mimic-constrained
gripper has the correct reduced `nq` and *coupled* motion — **not**
pinocchio-default parity (pinocchio's default loader is equally permissive;
codex A3).

**Current state (half-wired):**
- Parser reads mimic tags (`urdf.py:247-272`); `build_model` resolves
  `mimic_multiplier`/`offset`/`source` arrays (`build_model.py:486-504`);
  `Model` carries them (`model.py:76-78`); a zero-DOF `JointMimic`
  placeholder exists (`joint_models/mimic.py`, identity transform). But **no
  kinematics/dynamics file reads the arrays** — grep confirms every hit is
  `io/`, the `JointMimic` docstring, or `data_model/CLAUDE.md:33` (which
  still falsely claims "Handled via tensors … nq=0, nv=0").
- A Panda loads `nq = nv = 9` with **two independently movable fingers** —
  the mimic is silently ignored (`01_assessment.md §1.3`).
- M0's job was to **reject** mimic URDFs at build with guidance; T3.4
  replaces that reject with real enforcement. (As of 2026-07-17 the reject
  is not present — see Prerequisites.)

**Implementation plan — a reduced↔full coordinate map, not an FK gather
(03 §5, codex A3/B4):**
1. In `ModelStructure`, build the reduced map at build time: a mimic joint
   contributes **0** DOFs to the *reduced* configuration; its full-space
   coordinate is `q_full[j] = mult[j] * q_reduced[src] + off[j]`. Store the
   gather-with-multiplier operator `G: (nq_full, nq_reduced)` (and the `nv`
   analogue) plus the offset vector `g0` so:
   - **expand:** `q_full = G @ q_reduced + g0`, `v_full = G_v @ v_reduced`.
   - **reduce Jacobian:** `J_reduced = J_full @ G_v` (chain rule).
   - **reduce torque/limits:** `tau_reduced = G_vᵀ @ tau_full` (a mimic's
     generalized force accumulates onto its source);
     `lower/upper_pos_limit` reduce through the coordinate map.
2. `model.nq`/`model.nv` become the **reduced** counts; keep separate
   `nq_full`/`nv_full` for the internal expanded space. Every public pass
   accepts reduced `q`/`v`, expands internally before the joint loop, runs
   the existing FK/RNEA/ABA/CRBA/centroidal unchanged on full space, then
   reduces Jacobians/torques on the way out. Apply `expand`/`reduce`
   **uniformly** — a pass that forgets one makes FK and dynamics disagree
   (codex).
3. Once enforcement lands, remove M0's build-time reject (mimic URDFs now
   load with the reduced map) and delete the false claim in
   `data_model/CLAUDE.md:33`.

**What to test** (`tests/data_model/test_mimic_reduced_map.py`,
`tests/dynamics/test_mimic.py`):
- **Reduced nq/nv:** a mimic-constrained gripper (the Panda finger mimic, or
  a synthetic 2-finger gripper where `finger2 = 1.0*finger1 + 0`) has
  reduced `nq` one less than unconstrained (Panda: reduced `nq = 8`, not 9);
  `nq_full`/`nv_full` are the unreduced counts.
- **Coupled motion:** setting the source coordinate moves *both* fingers by
  `mult*source + off`; FK of the two finger frames is coupled; the frame
  Jacobian column for the source drives both fingers.
- **Dynamics coupling:** RNEA generalized force on the source joint
  accumulates both mimic branches (`G_vᵀ` reduction); CRBA/ABA operate in
  reduced space consistently with FK.
- **Limits:** `JointPositionLimit` on the reduced gripper penalizes the
  source coordinate's box, not a phantom finger DOF.
- **Explicitly NOT** a "matches pinocchio default" assertion — if a
  pinocchio reference is used, it must be an *explicitly constrained*
  `pin.Model` (mimic built as a real constraint), never `buildModelFromUrdf`
  defaults.

**Pitfalls / do-not-forget:**
- This is the milestone's riskiest task: **every** pass and the limit
  residual and all nq/nv indexing must go through the same reduced map, or
  FK/dynamics silently disagree. Put the map in `ModelStructure` so **both
  lanes** (torch + warp) see it.
- `mimic_source` uses the self-index-otherwise convention
  (`model.py:78`) — a non-mimic joint names itself; only genuine mimic
  joints have `src != self`.
- The M1 execution-batch ABI and the reduced map compose: reduce/expand act
  on the trailing q/v dim, batched leading dims pass through untouched.
- The `JointMimic` placeholder's identity `joint_transform` stays — the
  coupling now lives in the coordinate map + placements, not in a per-joint
  branch (matching the hot-path-no-branching rule).

### T3.5 — Order-preserving build + public vectorized `q_permutation`  [S–M]

**Goal / done-when:** `build_model(..., preserve_joint_order=True)` (or the
default when the input is already topological) plus a public vectorized
`q_permutation`, so the ~180-line name-keyed remap shims both consumers
wrote become **deletable** (deletion itself is M4, per standing rule 1).

**Current state:**
- `_ir_topo_sort` is an **unconditional DFS** (`build_model.py:160-196`).
  EXP6: the SMPL kintree order (which is *already* topological — parents
  before children) comes out permuted.
- BVR shims: `remap_q_bh_to_model` (`smplx_robot/model.py:87`) /
  `remap_q_model_to_bh` (`model.py:112`), each a **per-joint Python loop**
  (`for j in range(1, num_joints)` at lines 103, 122). The whole
  `smplx_robot/` dir (`model.py`, `inertia.py`, `dynamics.py`,
  `vertex_weights.py`) is the consumer bridge M3 makes redundant.
- BHF shims: `remap_smpl_q_to_model` (`robot_motion/motion.py:71`) /
  `remap_model_q_to_smpl` (`motion.py:109`), plus
  `remap_smpl_contacts_to_model`/`remap_model_contacts_to_smpl`
  (`motion.py:135,162`).

**Implementation plan:**
1. Add `preserve_joint_order: bool = False` to `build_model` (thread through
   `load()`). When `True`, use a **stable Kahn** topo-sort that keeps input
   order whenever it is already valid, instead of forced DFS. Keep DFS as
   the default and the non-topological fallback.
2. **Keep DFS the default (audit R2).** The pinocchio-parity q/v indexing
   aligns 1-to-1 *because* BR switched to DFS
   (`test_rnea_advanced_joints.py:8-10,153-162`). Changing the default would
   re-baseline the entire parity suite. "Default-when-topological" (03 §5)
   is therefore an **owner decision** — produce the evidence (which fixtures
   change order, what the parity suite does) and STOP for review; do not flip
   the default silently.
3. Add `Model.q_permutation(other_joint_order: Sequence[str]) ->
   (perm_q, perm_v)` returning index tensors, so `q_br = q_ext[..., perm_q]`
   is one gather (no per-joint Python loop). Vectorized; batched-safe on the
   trailing dim.

**What to test** (`tests/io/test_preserve_order.py`,
`tests/data_model/test_q_permutation.py`):
- `preserve_joint_order=True` on the SMPL kintree → `model.joint_names`
  matches the input order (the EXP6 inverse); FK/RNEA still correct.
- DFS default unchanged: existing parity + ordering tests
  (`test_rnea_advanced_joints.py:150-171`) stay green.
- `q_permutation` round-trip: `q_ext[..., perm_q]` then the inverse gather
  recovers `q_ext`; the result matches a reference name-keyed remap on a
  fixture (reproduce the consumer's mapping *structure* synthetically — do
  not import the consumer).
- Batched `q_permutation`: a `(B, T, nq)` gather works on the trailing dim.

**Pitfalls / do-not-forget:**
- The shim/`smplx_robot/` **deletion happens in the consumer repos in M4**,
  not here. M3 delivers the hooks; the done-when says the shims *become
  deletable*. Don't touch BVR/BHF.
- Don't break `idx_qs`/`idx_vs` accumulation: order-preserving must still
  assign contiguous q/v slices in the chosen order.
- `q_permutation` maps *joint* order to q/v *slice* order — a spherical
  joint is 4 q-slots / 3 v-slots, free-flyer 7/6; the permutation is over
  slices, not scalar indices.

### T3.6 — Human-joint support: swing/twist limit + rotation prior + `Inertia.from_mesh`  [M]

**Goal / done-when:** a swing/twist limit residual and a per-joint rotation
prior for spherical joints (the current spherical "limits" are inert), plus
a differentiable batched `Inertia.from_mesh` that replaces the three
hand-rolled trimesh loops across the workspace.

**Current state:**
- Spherical position "limits" are packed as `[-1, 1]` per quaternion
  component (`build_model.py:390-392`) — **inert**, since unit-quaternion
  components never leave `[-1, 1]`. `JointPositionLimit` zeroes Jacobian
  rows for `nq != nv` joints (`residuals/limits.py:48`:
  `if jm.nq == 0 or jm.nq != jm.nv: continue`), so for an all-spherical
  human the whole limit residual is a **no-op**.
- `RestResidual` *does* work for spherical joints (tangent-space
  `model.difference`) — a mean-pose prior is already expressible; per-joint
  weighting and swing/twist are not (audit CP-6).
- `Inertia` has `from_sphere`/`box`/`capsule`/`ellipsoid`/`mass_com_*`
  (`spatial/inertia.py:42-153`) but **no `from_mesh`**.
- Three hand-rolled trimesh loops (audit R4):
  - BVR `tools/smplx_robot/inertia.py`: `compute_smplx_body_inertia`
    (`inertia.py:53`), convex-hull, `for j in range(J)` (`inertia.py:88`).
  - BHF `tools/robot_motion/smpl_inertia.py`: `compute_smpl_body_inertia`
    (`smpl_inertia.py:76`), face-segmentation + density table,
    `for j in range(_N_SMPL_JOINTS)` (`smpl_inertia.py:136`).
  - `better_human` `smpl/mass.py`: `compute_Mass_Inertia` (`mass.py:11`),
    a **double** Python loop `for b … for j …` (`mass.py:27-28`) with
    `.detach().cpu().numpy()` (non-differentiable, CPU-bound).

**Implementation plan:**
1. `SwingTwistLimitResidual(model, joint_ids, twist_axis, swing_max,
   twist_range)` in `residuals/` — decompose each spherical joint's
   quaternion into swing (about `twist_axis`) and twist; one-sided clamp
   penalty (like `JointPositionLimit`) on `swing > swing_max` and twist
   outside `twist_range`. Analytic tangent-space Jacobian over the joint's
   `nv=3` columns. Operates on q slices.
2. `JointRotationPrior(model, q_mean, per_joint_weight)` in `residuals/` —
   tangent-space deviation like `RestResidual` but with a `(njoints,)` or
   `(nv,)` weight vector (humans need strong spine / weak shoulder priors);
   generalizes `RestResidual` per block.
3. `Inertia.from_mesh(vertices: (..., V, 3), faces: (F, 3),
   density: float | Tensor) -> Inertia` in `spatial/inertia.py` —
   closed-form signed-tetrahedron volume integrals over faces, batched in
   torch (no trimesh, no `.cpu().numpy()`); autograd flows vertices → betas.
   Plus `inertia_from_vertex_parts(vertices, faces_per_body | vertex_to_body,
   densities, joints) -> (..., nbodies, 10)` with COMs joint-relative,
   feeding T3.3's `body_inertias`.

**What to test** (`tests/residuals/test_swing_twist.py`,
`test_rotation_prior.py`; `tests/spatial/test_inertia_from_mesh.py`;
float32 per `write-tests`):
- SwingTwist: a spherical config inside the cone → zero residual; outside →
  positive; fp32 gradcheck; the Jacobian hits the right `nv=3` columns of
  the target joint and zeros elsewhere.
- RotationPrior: matches the weighted tangent deviation against a manual
  computation; per-joint weights applied correctly; works at the θ=0
  singularity (needs M0).
- `Inertia.from_mesh`: **numerical parity against trimesh** on the same
  meshes — reproduce the two consumer implementations' meshes as fixtures
  and assert mass/COM/inertia match to tolerance; autograd flows to
  `vertices`; batched `(..., V, 3)`; both face-winding orientations tested
  (signed volume is winding-sensitive).

**Pitfalls / do-not-forget:**
- The inert `[-1, 1]` spherical limit packing (`build_model.py:390-392`)
  stays (harmless) but document that swing/twist is the real limit; don't
  try to "fix" the quaternion box.
- Layer DAG: residuals sit above `data_model`; `Inertia.from_mesh` is in
  `spatial/` (below `data_model`) — fine. `from_mesh` must not import
  trimesh in BR core (it is the thing being replaced); trimesh stays a
  consumer/test-only dependency.
- These are library *additions*; nothing in `residuals/`/`spatial/` is
  deleted here.

### T3.7 — Vectorize `integrate`/`difference` by joint-kind grouping  [S–M]

**Goal / done-when:** `Model.integrate`/`difference` group joints by kind
and run one vectorized op per kind (pyroki-style), matching the per-joint
loop bit-for-parity and measurably faster. This is also the prerequisite
for M6's decision on whether an `integrate`/`difference` Warp kernel is ever
needed (M6 boundary table: a kernel must beat *this* rewrite, not today's
loop).

**Current state:**
- `Model.integrate` (`model.py:160-174`) and `difference`
  (`model.py:176-190`) are **per-joint Python loops** dispatching to
  `jm.integrate`/`jm.difference`. Measured 2.4× (audit, 24 joints) / 3.66×
  (codex, 24-joint SMPL-like, T=200) slower than a vectorized twin;
  EXP5 confirms the loop is *correct* (roundtrip 1e-8).
- BVR already wrote the vectorized twin to steal back: `tangent_difference`
  (`tools/human_optim/motion.py:39`) — root SE3 via
  `se3.log(q0⁻¹∘q1)`, each parent-local spherical joint via
  `so3.log(q0⁻¹∘q1)`, **vectorized over joints** (stride-4 in `q[7:]`,
  `motion.py:50-53`). Its docstring explicitly calls itself "the vectorized
  twin of BetterRobot's `Model.difference`".

**Implementation plan:**
1. Precompute per-kind index groups in `ModelStructure` at build: for each
   joint kind (1-DOF revolute/prismatic/helical → Euclidean add; spherical
   → so3; free-flyer → se3), the gathered q-slice and v-slice indices.
2. `integrate`: for each kind group, one vectorized op over *all* joints of
   that kind at once (one `so3.compose∘exp` for every spherical joint
   together, one `se3` op for the free-flyer, one add for all 1-DOF joints),
   then scatter back into the full q. Same structure for `difference`
   (steal BVR's `tangent_difference` gather). Keep the **right/local
   perturbation** semantics identical to `spherical.py:52-62` /
   `free_flyer.py` — a left-perturbation rewrite would pass roundtrip tests
   and silently break analytic-Jacobian parity.
3. **Do not change the signature.** `RobotConfig(model)` (m2a T2a.1) wraps
   `model.integrate`/`difference` as-is and gets the speedup for free;
   m2a's T2a.1 pitfall deferred this vectorization here explicitly.

**What to test** (`tests/data_model/test_integrate_vectorized.py`):
- **Parity oracle = the current per-joint loop** (it is correct): the
  vectorized `integrate`/`difference` matches the loop bit-for-parity
  (fp32 and fp64) on Panda, G1 free-flyer, and an all-spherical SMPL-like
  tree.
- Roundtrip: `difference(q, integrate(q, v)) ≈ v` (EXP5 tolerance), at the
  θ=0 singularity (needs M0).
- Batched: `(B, T, nq)` inputs work on the trailing dim.
- **Committed benchmark** (standing rule 4 — hardware, dtype, shapes,
  warmup, statistics checked in): speedup vs the loop on the 24-joint
  SMPL-like T=200 case. **Report the number; do not gate on a bare ratio.**
  This benchmark is what M6 uses to decide whether an integrate kernel is
  ever built (M6 item 2). If the vectorized version does not beat the loop,
  that is a finding to report, not to hide.

**Pitfalls / do-not-forget:**
- Manifold ops must stay autograd-clean at θ=0 — depends on the M0 θ=0 fix
  for spherical `so3.log`/`exp` at identity.
- `random_configuration` (`model.py:192-205`) has the same per-joint loop
  shape but is not hot — leave it unless trivial; the roadmap item is
  `integrate`/`difference` only.
- The per-kind groups belong in `ModelStructure` (both lanes see them),
  co-located with T3.4's reduced map and the M1 kind codes.

## Milestone acceptance checklist

- [ ] Prerequisites confirmed: M0 θ=0 one-liner prints OK; M0 mimic reject
      present; M1 `ModelStructure`/`ModelValues` + frozen execution-batch
      ABI + FK warp kernel landed; M2a `Problem`/providers public.
- [ ] **T3.1 first:** coverage audit matrix posted; pinocchio-parity
      extended over joint kinds × batch shapes × base types × dtype; the
      value-batched loop-oracle helper exists; all pre-existing parity tests
      untouched and green.
- [ ] T3.2: FK/Jacobians/RNEA/ABA/CRBA/centroidal index `[..., j, :]`;
      execution batch = broadcast(q, all value batches); `Data` allocation
      derives from it; batched-values × unbatched-q, converse, multi-axis,
      and mismatch-error tests pass; batched autograd to placements/inertias
      verified; FK warp kernel has value-batched coverage; **no ABI change**.
- [ ] T3.3: `Model.with_values(...)` public, validated, batched, tested;
      frame table `(*value_batch, nframes, 7)` + `frame_parent_joints` in
      structure; `update_frame_placements` vectorized; `Model.to()` moves
      frames and casts float-only; inertia 6×6 caching static-only, live
      parametric derived per eval; `_model_id` resolved.
- [ ] T3.4: mimic reduced coordinate map across FK/Jacobians/dynamics/limits
      /torque + all nq/nv indexing; acceptance is the explicit reduced-map
      assertion (reduced `nq`, coupled motion), **not** pinocchio-default
      parity; M0 reject removed; `data_model/CLAUDE.md:33` false claim
      deleted.
- [ ] T3.5: `preserve_joint_order` build path (DFS default kept; flip is an
      owner decision with evidence); public vectorized `q_permutation`;
      remap shims **deletable** (deletion is M4).
- [ ] T3.6: `SwingTwistLimitResidual`, `JointRotationPrior`,
      `Inertia.from_mesh` + `inertia_from_vertex_parts` — all differentiable,
      batched, trimesh-parity-tested.
- [ ] T3.7: vectorized `integrate`/`difference` (per-kind groups); loop
      parity + roundtrip + committed benchmark; signature unchanged
      (`RobotConfig` still wraps it).
- [ ] A **synthetic** BR-side test proves the done-when's shape: a fake
      `betas` leaf → `with_values` → batched FK + IK → gradients reach the
      leaf (no `better_human` import).
- [ ] Full suite green (`uv run pytest tests/ -v`; 897 baseline + M0/M1/M2
      deltas + M3 additions); pinocchio-parity suite green throughout; both
      consumer repos still import cleanly.

## Out of scope

- **Any ABI change to the execution batch / kernel seam** — frozen in M1
  (`m1_two_lane_seam_and_hygiene.md`); M3 is breadth only.
- **Batched second-order / bounded LM, per-element status** →
  `m2b_batched_second_order_solvers.md`. M3 makes model *values* batched;
  the batched *solver* the done-when's IK uses is M2b/M2c.
- **Deleting the consumer shims / `smplx_robot/` / migrating BVR/BHF** →
  `m4_consumer_packs_and_migration.md`. M3 makes them *deletable*.
- **Vision residuals (projection/chamfer/SDF), Geman-McClure, viewer
  `SkinnedMeshMode`, retarget** → M4 (`m4_...`). The done-when needs only
  FK + IK + gradient flow, not the vision pack.
- **Sparse/banded trajectory assembly** → `m5_sparse_trajectory_structure.md`.
  Batched *values* here are dense per pass.
- **Warp `integrate`/`difference`, RNEA/CRBA/ABA kernels, CUDA-graph
  capture** → `m6_warp_fast_path_and_cuda_graphs.md`. M3 only adds
  value-batched *coverage tests* to M1's existing FK kernel; T3.7 produces
  the benchmark M6 gates the integrate kernel on.
- **Per-batch joint limits** — limits stay per-model (unbatched) in v1
  (03 §5); out of scope until a use case exists.
- **SMPL/body-model specifics** (LBS, blendshapes, `J_regressor`, landmark
  tables, density tables, `from_classic`) — stay in `better_human` per the
  scope fence (`plan/02_vision.md`); never enter BR core.
- Do not modify the consumer repos; do not delete anything with a live
  consumer import.

## References

- `plan/04_roadmap.md` — the M3 section (six items → seven tasks here) and
  the standing rules; the done-when this file expands.
- `plan/03_architecture.md §5` — parametric batched Model values, the
  broadcast contract, frame table, inertia caching, mimic reduced map,
  order-preserving build, human-joint support (the primary design source);
  `§2.4` — the frozen execution-batch ABI M3 extends; `§8` — the
  parity-suite-before-the-sweep rule (T3.1); `§7 step 4` — the
  vectorize-integrate/difference performance step (T3.7).
- `plan/01_assessment.md §2.3` — the verified better_human blockers
  (dim-0 indexing, frozen frames, DFS reorder, inert spherical limits);
  `§4` — parity-coverage breadth is **suspect**, audit it first (T3.1).
- `plan/research/audit_better_human_integration.md` — the primary evidence:
  CP-1…CP-9 (confirmed problems), R1–R9 (recommendations), EXP1–EXP6 (raw
  experiments), §6 (open questions incl. the S-1 broadcast-shape question).
- `plan/research/codex_plan_review.md` — §A3/§A10/§B4 (calibrations:
  `[...,j,:]` necessary but not sufficient; mimic enforcement spans
  Jacobians/limits/dynamics/indexing; pinocchio-default parity proves
  nothing for mimic; `Data` allocation must derive from the broadcast, not
  `q` alone; frame table needs the model batch axes).
- `plan/for_agents/m1_two_lane_seam_and_hygiene.md` — freezes the
  execution-batch ABI and builds `ModelStructure`/`ModelValues` (T3.2/T3.3/
  T3.7 build on them); its FK warp kernel gains value-batched tests in T3.2.
- `plan/for_agents/m2a_variable_blocks_and_slice.md` — `RobotConfig` wraps
  `integrate`/`difference` (T3.7 sits under it); `RobotStateProvider`/
  `Problem` are what `with_values` must fit (T3.3); T2a.1's pitfall defers
  the integrate/difference vectorization to M3 item 6 (T3.7).
- Consumer evidence (read-only): BVR `tools/smplx_robot/`
  (`model.py:87,112` remap shims; `inertia.py:53,88` trimesh loop),
  BVR `tools/human_optim/motion.py:39` (vectorized `tangent_difference` to
  steal), BHF `tools/robot_motion/motion.py:71,109` (remap shims) +
  `smpl_inertia.py:76,136` (trimesh loop),
  `better_human/src/better_human/smpl/mass.py:11,27` (double-loop trimesh
  inertia).
- Source anchors verified 2026-07-17: `kinematics/forward.py:112`
  (`joint_placements[j]` + per-call `.to()`), `:194,213-216`
  (per-frame loop); `dynamics/{rnea:167,crba:60,aba:121,centroidal:41,152}`
  (`body_inertias[i]` + per-call `.to()`); `data_model/model.py:76-78`
  (mimic tensors), `:80-81` (frames tuple), `:92-119` (`.to()` misses
  frames), `:160-190` (integrate/difference loops), `:133` (dead
  `_model_id`); `data_model/joint_models/mimic.py` (zero-DOF placeholder);
  `data_model/frame.py:27` (`(7,)` struct); `residuals/limits.py:48`
  (nq!=nv zeroing); `io/build_model.py:160-196` (DFS), `:390-392` (inert
  spherical limits), `:486-504` (mimic array resolution, no enforcement);
  `spatial/inertia.py` (no `from_mesh`); `io/parsers/urdf.py:247-272`
  (mimic tags).
