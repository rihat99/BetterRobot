# Order 04 — Tangent groups, per-block freeze, higher-order smoothness

Read `plan/README.md` and `DESIGN_RULES.md` first. Independent of orders
02–03 in content; lands after them to keep merge noise down. Order 01's
weight semantics are assumed. This is the widest-wired order of the round —
its inventory below is part of the order, not advice.

## Motivation (verified)

- Weighting root vs joints requires reaching into
  `ModelStructure.manifold_free_flyer_v_indices` /
  `manifold_spherical_v_indices` and tiling by hand (BVR's
  `split_weight`, `residuals.py:50-71` — note it also splits root
  *translation* from root *rotation*, not just root from joints).
- `trainable` is a whole-variable bool; no tangent subset can be frozen,
  so the classic "place the root first, then bend the joints" warm-up is
  inexpressible. The old `mask`/`scale` subsystem was deleted at `1d124b7`
  for having zero callers — this is the real caller arriving. Prior art:
  `git show 1d124b7~1:src/better_robot/optim/variables.py` (free-index
  gather machinery); redesign smaller, do not resurrect wholesale.
- Smoothness ships only velocity and acceleration; jerk/snap require a
  fork (`smoothness.py`; roadmap "jerk smoothness" deferred entry).

## Target semantics

1. **Named tangent groups.** `RobotVariable.tangent_groups()` returns an
   ordered `Mapping[str, LongTensor]` of per-knot v-index groups: one per
   joint name, plus the unions `"root"` (the free-flyer/base block, when
   present), `"root_lin"` and `"root_ang"` (its translation and rotation
   halves — BVR's orientation prior weights them differently), and
   `"joints"` (the complement of `"root"`). Derived from model topology;
   no new state.
2. **`tangent_weight({...}, default=1.0)`** on `RobotVariable` builds a
   `(nv,)` per-coordinate vector from `{group_or_joint_name: value}`.
   Values are **square-root-information row multipliers** (they multiply
   rows; the objective sees their square) — state this in the docstring
   and the guide; the helper does not take square roots for the caller.
3. **Construction-time freeze.**
   `RobotVariable(..., frozen_groups=("joints", ...))` — group names
   only, no raw mask input (no second caller for it). Frozen coordinates
   are excluded from `free_dim`. The wiring is deliberately narrow:
   - `variables.py`: immutable free-index table at construction;
     `retract(delta)` scatter-embeds the `(…, free_dim)` delta with zeros
     at frozen coordinates; add `gather_tangent`/`expand_tangent` helpers
     (the prior-art names). **Public `difference()` is unchanged** — it
     keeps returning the full tangent; the `Difference` residual
     (`base.py:247-254`) and every other geometry consumer must not
     change. Gathering to free coordinates happens only in optimizer-
     facing plumbing (`Problem.difference`, `Problem.gradient`,
     `TorchOptimizer` buffers), which already goes through
     `variable`-level hooks.
   - `optim/problem.py`: analytic Jacobian blocks are validated against
     `free_dim` columns (`problem.py:504-519`) while residuals keep
     returning full-`nv` columns — reduce columns **centrally** at that
     validation point (one gather, every residual unaffected). AD block
     paths differentiate through `retract`, so they produce free columns
     already.
   - `optim/lm.py`: the bounds layout maps state coordinates to tangent
     columns (`lm.py:142-207`) — it must map to *free* columns; a bound
     on a frozen coordinate is simply never active (assert, don't
     special-case).
   - `optim/temporal.py` + `variables.py`: `temporal_tangent_width` and
     banded assembly width become the per-knot *free* width; the mask is
     per-knot and time-uniform by construction (groups are joint-level),
     so eligibility logic stays shape-static.
   - `optim/implicit.py`: solution coordinates are the free ones; add a
     guard test.
   The mask is immutable after construction — changing the frozen set
   means constructing a new variable and problem for the next phase,
   handing values over via the vertical-slice segment pattern.
4. **`SmoothnessResidual(q, *, order, dt, coordinate_weight=None, ...)`**
   for `order ∈ {2, 3, 4}`: the `order`-th forward difference of the
   knot sequence in tangent space (differences of the first-difference
   sequence `δ_i = model.difference(q_i, q_{i+1})`), scaled `1/dt^order`;
   `dim = (T − order)·nv`. `order=2` reproduces `AccelerationResidual`'s
   rows exactly (same 3-point stencil) — delete `AccelerationResidual`
   with a MIGRATION row saying so. **`VelocityResidual` stays unchanged**
   (central difference; the wishlist asks for orders 2–4 and no caller
   wants a forward order-1 — do not invent one). `coordinate_weight` is a
   `(nv,)` tensor applied per knot-row (a `row_weight` under order 01's
   algebra; same √-info semantics as `tangent_weight`). `T ≤ order`
   raises at construction (the existing convention); no auto-skip —
   callers guard, the how-to shows the `if T > order` pattern.

## The analytic-block question — verify before building

Current smoothness residuals declare **constant identity** temporal blocks
(`smoothness.py:58-68`). For scalar joints `model.difference` is a plain
subtraction and the blocks are exact; for spherical/free-flyer joints the
difference is a manifold log whose derivative is state-dependent, and the
existing FD test only covers Panda (`tests/residuals/test_smoothness.py`).
First **verify** the suspected pre-existing defect: compare declared blocks
against AD on a spherical-joint trajectory model. Then, for both
`VelocityResidual` and the new `SmoothnessResidual`: declare analytic /
direct-banded blocks **only** for models whose difference is affine (all
scalar joints — compute this from model topology, not from tensor values);
other models take the AD/dense path with an actionable
`AutodiffFallbackWarning` naming the joint kind. Do not ship
state-dependent log-Jacobian blocks in this order — that is its own
contract; record it in the roadmap's deferred list. Report the
verification result either way; if the existing Panda-only blocks were
wrong for spherical models in released behavior, say so in the results
file and the changelog.

## Migration surface (exhaustive, verified by grep)

`residuals/__init__.py:15,52-53` exports; `examples/05_panda_trajopt.py:
30,77`; `tests/residuals/test_smoothness.py`;
`tests/residuals/test_temporal_structure.py`;
`tests/bench/bench_trajopt_sparse.py:207,301-302`;
`docs/concepts/residuals_costs_and_solvers.md:154` and the generated
smoothness API page. `tasks/trajopt.py` and `tasks/smoothing.py` do **not**
construct smoothness residuals (factories come from callers) — do not
"recalibrate" them.

## Acceptance

- `tangent_weight` reproduces a hand-built root_ang/root_lin/joints
  diagonal on a free-flyer + spherical test model; works for fixed-base
  models (no `"root"` group; asking for it raises with the model name).
- Frozen-root warm-up end to end: phase 1 optimizes only the root block of
  an SMPL-like model (joints provably immobile to zero tolerance), phase 2
  a fresh unfrozen problem continues from its result and converges.
- LM (with bounds) and TorchOptimizer both solve a frozen-subset problem;
  banded trajopt routes with a frozen group and matches dense; analytic
  blocks validated at free width; `Difference` residual on a frozen
  variable keeps its full-tangent `dim` and passes.
- `SmoothnessResidual` orders 2–4: rows match a hand-computed stencil;
  `order=2` matches the deleted `AccelerationResidual` bit-for-bit on
  Panda; block-vs-AD agreement on scalar-joint models; the spherical
  model warns and routes dense; `T ≤ order` raises.
- MIGRATION rows; roadmap jerk entry resolved; full gate green; net src
  growth ≤ +350.
