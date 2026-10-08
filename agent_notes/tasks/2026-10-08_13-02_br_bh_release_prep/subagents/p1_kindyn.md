# P1 kindyn — kinematics and dynamics review (lie, spatial, data_model, kinematics, dynamics)

Worktree head `23070f4`. All paths are relative to `/home/rikhat.akizhanov/better/dev/BetterRobot`. Probe script:
`/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/kindyn_probe.py`.
Budget note: the code was read through signatures, guides and hot spots, not line by line in full.

## Current state (facts, with paths)

- Size: lie 1.1k lines, spatial 1.0k, data_model 2.6k, kinematics 1.1k, dynamics 0.56k (`wc -l`).
- Public dynamics (`src/better_robot/dynamics/state.py`): `inverse_dynamics(model, state, *, v, a, fext=)`,
  `mass_matrix`, `forward_dynamics`, `center_of_mass`, `centroidal_map`, `centroidal_momentum`. All read a
  `KinematicState`; world geometry is memoised in `state._memo["world_geometry"]` (`state.py:43-46`).
- One implementation, frax-style: everything in world frame; ancestor/subtree sums are products with constant
  support matrices (`DynamicsPlan`, `dynamics/_world.py:35-68`); op count is independent of joint count
  (Panda RNEA 379 ops, tree200 382; `benchmarks/results/p2_world_dynamics_cpu.json`).
- ABA is not Featherstone ABA: `forward_dynamics` = RNEA bias + dense `cholesky(M)` + `cholesky_solve`
  (`state.py:83-91`). Dense is the correct form for mimic joints (`dynamics/CLAUDE.md`).
- FK (`kinematics/_fk_lean.py:1-15`): joint transforms are affine in features of `q` (sin/cos, quaternion
  products), one batched product per model plus one matmul per tree level; rare kinds (planar, translation,
  helical, composite) go through `data_model/joint_dispatch.py` one by one.
- Jacobians: `frame_jacobian`, `joint_jacobian` and their time variations (`kinematics/state.py:185-260`),
  built from a shared world column table `C (N, 6, nv_full)` (`kinematics/_jacobian_columns.py:103`).
- Joint kinds (`data_model/joint_models/`): revolute (axis-aligned, unaligned, unbounded), prismatic, spherical,
  free flyer, planar, translation, helical, composite, fixed, mimic (`model_structure.py:46`, `mimic_source`).
- Reduction: only mimic expand/reduce helpers (`data_model/reduced_coordinates.py`). No closed loops and no
  constraint model anywhere in `data_model` (grep for `closed|loop|constraint` is empty).
- Not present: analytic derivatives, Coriolis matrix, `compute_minverse`, kinetic/potential energy, CoM
  acceleration, contact/constrained/impulse dynamics. Roadmap lists derivatives and accessors as "not in
  progress" (`docs/reference/roadmap.md:72-75`; CoM acceleration "not provided", line 24).
- Benchmarks: `benchmarks/run.py` measures wall time and torch op counts for FK, Jacobian, RNEA, CRBA, ABA, with
  Pinocchio batch-1 rows; results are per-commit JSON in `benchmarks/results/`. CUDA results cover FK only.
- Compile: `tests/kinematics/test_fk_compile.py` compiles the free-flyer FK kernel fullgraph, not the public path.

Measured (CPU, float32, this machine, `p2_world_dynamics_cpu.json` and the probe):

| case | BR | Pinocchio | ratio |
|---|---|---|---|
| Panda RNEA, B=1 | 1.07 ms | 0.0016 ms | ~650x |
| Panda ABA, B=1 | 1.30 ms | 0.0054 ms | ~240x |
| G1 ABA, B=1 | 1.40 ms | 0.0188 ms | ~75x |
| Panda RNEA, B=4096 | 52 ms (12.7 us/sample) | 1.6 us/sample (1 thread) | ~8x |
| tree200 ABA, B=4096 | 4.6 s fwd, 17.3 s fwd+bwd | — | dense M is 655 MB |
| Panda FK+RNEA, B=1, probe | 0.54 ms eager, 0.28 ms `torch.compile` (graph breaks) | | 2x from compile |
| Panda d tau/dq by `jacrev`, B=1 | 4.39 ms, output shape `(1, 8, 1, 8)` | | 8x one RNEA |

## How references evolved (with sources)

Digests are under `/home/rikhat.akizhanov/better/BetterRobot/references/design/`. Items marked (mem) come from my
knowledge of the projects' history and were not re-fetched in this session; verify before quoting publicly.

- Pinocchio: every algorithm ships a matching `*-derivatives.hpp`, plus centroidal, contact and constrained
  dynamics (`pinocchio.md:27-28`), and the same templates run on `CppAD`/`CasADi` scalars with code generation
  (`pinocchio.md:31-34,56`). History (mem): analytic RNEA/ABA derivatives came in 2.0 (Carpentier and Mansard,
  RSS 2018) because finite differences and AD were too slow for MPC; contact-dynamics derivatives followed; 3.0
  generalised contacts to `RigidConstraintModel` and added closed loops and proximal constrained dynamics
  (Carpentier, Budhiraja, Mansard, RSS 2021). Lesson: derivatives and constraints became first-class later,
  and needed new data structures (constraint models and datas) that the 2.x API had no slot for.
- frax: same world-frame, mask-matrix design as BR (`frax.md:5,175,216`); it has no derivative code at all and
  relies on JAX AD (`frax.md:253`); only two joint types by design (`frax.md:39`). BR already went further
  (joint kinds, mimic); frax shows the ceiling of the "AD everything" strategy.
- mujoco_warp: separate files per stage: `smooth.py` (kinematics, `crb`, `factor_m`, `solve_m`, `rne`),
  `constraint.py` (equality/limit/friction/contact rows), `derivative.py`, `inverse.py` (`mujoco_warp.md:13-17`).
  Kernel infrastructure cost: `cache_kernel`, `event_scope` timing, and a custom AST linter for kernel signatures
  (`mujoco_warp.md:21-24`). History (mem): MJX (JAX) was slow on contact-rich scenes, which motivated the Warp
  rewrite with hand-written kernels; MuJoCo factors `M` with its sparse tree structure, not dense Cholesky.
- Newton: one `Model/State/Control` vocabulary, many solvers behind a `SolverBase` (Featherstone, MuJoCo-warp,
  XPBD, VBD, semi-implicit, Kamino) (`newton.md:3,30-45`). Lesson: the data model is the stable brick; solvers
  are swappable bricks on top.
- Brax: started with several pipelines (`generalized/` Featherstone, positional, spring) with penalty contact and
  a joint-limit/contact constraint module (`brax.md:20-35`); (mem) later steered users to MJX. Lesson: owning a
  contact solver is a large, separate commitment from owning rigid-body algorithms.

## Findings (ranked, F1..Fn: severity, evidence, consequence)

- **F1 high — `torch.compile(fullgraph=True)` of the public path fails.** Evidence: probe on
  `forward_kinematics_state` + `inverse_dynamics` raises `Unsupported ... function marked as skipped`
  (`torch._C._functorch...`); the checks are `torch._C._functorch.is_functorch_wrapped_tensor` in
  `kinematics/_fk_lean.py:213` and `dynamics/_world.py:176`. Non-fullgraph compile works with 5 graph breaks
  in `_fk_lean` and gives only 2x (0.54 -> 0.28 ms). The existing test compiles only the FK kernel. Consequence:
  compile and CUDA graphs, the main lever for batch-1 latency, are blocked on the public API.
- **F2 high — batch-1 latency is dispatch bound by construction.** ~380 torch ops per RNEA at ~2.8 us each;
  650x slower than Pinocchio for Panda RNEA. Batch size does not change op count, so this is a fixed floor.
  Consequence: single-robot MPC/DDP (many small sequential calls) and 1 kHz control loops are not viable
  without compile/graph capture or a kernel. Large batches are fine (8x of single-thread Pinocchio per sample).
- **F3 high — no derivatives API, and the autograd route is costly and shape-surprising.** `jacrev` of a
  batched RNEA costs 8x one RNEA and returns `(B, nv, B, nv)` (cross-batch blocks), so users must know to wrap
  in `vmap`. For free-flyer models autograd w.r.t. `q` gives ambient (`nq`) gradients, not tangent (`nv`)
  partials; the user must compose with `Model.integrate`. Consequence: the optim theme's DDP/iLQR, MPC and
  contact-implicit work all need `f_x, f_u` and will each re-invent this.
- **F4 high (for BH) — forward dynamics is dense O(nv^3) with O(nv^2) memory.** tree200 at B=4096: 4.6 s forward,
  17.3 s with backward, 655 MB for `M` (`dynamics/CLAUDE.md`, results JSON). An SMPL-X body (free flyer + 54
  ball joints, nv ~ 168) sits in this regime. No O(n) articulated-body pass, no tree-sparse `LTDL` factor
  (as in MuJoCo/Featherstone), no `compute_minverse`. Dense remains right for mimic models, so a second path
  must be selected by structure, not exposed as a second API.
- **F5 medium — no constraint brick: contacts, closed loops and impulses have no home.** `fext` per joint in
  joint frame is the only interaction input (`dynamics/CLAUDE.md` contracts). There is no stacked contact
  Jacobian, no KKT/constrained forward dynamics, no impulse dynamics, no loop-closure model. Pinocchio needed a
  new structure (`RigidConstraintModel`) for exactly this. Two in-tree callers already exist: BH contact frames
  and humanoid/legged terms; OpenSim-style models (BH) need loop closures and coupled coordinates.
- **F6 medium — missing cheap accessors.** Kinetic and potential energy, Coriolis matrix, CoM velocity and
  acceleration, `dh/dt` (centroidal dynamics). All are a few lines on the existing world quantities, but their
  absence pushes users to autograd hacks.
- **F7 medium — joint set lacks biomechanics joints.** No Euler-angle spherical (ZYX/XYZ), universal, or
  OpenSim `CustomJoint` (spline-coupled coordinates, e.g. knee translation as a function of flexion). Composite
  and mimic cover part of it; mimic is linear only. BH/OpenSim users hit this wall first.
- **F8 medium — no kernel seam defined.** Raw passes take dataclasses (`WorldKinematics`, `DynamicsPlan`,
  `LeanPlan`) rather than flat tensors, and nothing documents which function a future Warp/Triton kernel would
  replace. Without a seam a kernel becomes a parallel API (the "duality" the owner disliked). mujoco_warp shows
  the infrastructure tax of kernels (F8 is about deciding the seam, not writing kernels).
- **F9 medium — hidden mutable memo in a frozen state.** `state._memo` is a dict mutated on first dynamics call
  (`dynamics/state.py:43-46`). Works today, but it is untyped, unkeyed by dtype/device, and is a mutation inside
  what compile and `vmap` treat as a pure function. Adding derivatives or contacts will add more memo keys.
- **F10 low — benchmarks have no regression gate and no compile, memory or derivative columns.** CUDA results
  exist for FK only (`p2_world_dynamics_cuda.json`). Pinocchio comparison is batch 1 only.

## Roadmap (table: brick | why | prerequisite | when)

| brick | why (which user) | prerequisite in the architecture | when |
|---|---|---|---|
| Fullgraph-compile clean public FK/Jacobian/RNEA/CRBA + contract test | everyone at batch 1; enables CUDA graphs | replace functorch probe (F1) | now |
| Benchmark modes: `--compile`, peak memory, `jacrev` rows, CUDA dynamics rows | measure F2-F4 before choosing work | none | now |
| Derivatives contract (doc only): `*_derivatives(model, state, ...)` returns tangent-space partials `(B..., nv, nv)` | DDP/MPC/contact-implicit (optim theme) | names reserved, tangent convention fixed | now |
| `inverse_dynamics_derivatives`, `forward_dynamics_derivatives` via `vmap(jacrev)` + `Model.integrate` | trajopt users now, correct shapes | contract above | next |
| Analytic RNEA/ABA derivatives (Carpentier-Mansard) behind the same signature | speed for MPC | autograd version as oracle; Pinocchio parity tests | later |
| Energy, Coriolis matrix, CoM vel/acc, `dh/dt`, `compute_minverse` | controllers, analysis, BH | none | next |
| Constraint brick: `ContactSet`/`ConstraintModel` (frames, type 3D/6D, Baumgarte), stacked Jacobian, KKT forward dynamics, impulse dynamics | legged/humanoid, BH ground contact | decision on where constraint specs live (data_model vs dynamics) | next |
| Loop closures via the same constraint brick | OpenSim/BH models, parallel mechanisms | constraint brick | later |
| Structure-selected forward dynamics: tree-sparse `LTDL` or O(n) ABA when no mimic | humans (nv ~ 170), large batches | kernel seam (F8); benchmark gate | later |
| Kernel backend (Warp or Triton) as `torch.library.custom_op` for raw passes, with fake impl and autograd formula | GPU batch throughput | seam = raw-pass signatures; derivative formulas exist | maybe |
| Biomechanics joints: Euler spherical, universal, spline-coupled custom joint | BH/OpenSim users | joint dispatch for "rare kinds" already exists | next (Euler, universal), later (custom) |
| Model composition/reduction (append, lock joints to q0) | toolbox theme; BH reduction | ModelStructure rebuild helpers | next (shared with toolbox) |
| Constrained/contact dynamics derivatives | contact-implicit trajopt | both bricks above | maybe |

## Prep changes recommended now (each ≤ 1 day, non-breaking)

1. **Unblock fullgraph compile (F1).** In `_fk_lean.py:213` and `_world.py:176`, skip the
   `is_functorch_wrapped_tensor` probe when `torch.compiler.is_compiling()` (or take the conservative branch
   under compile). Add `tests/kinematics/test_public_compile.py`: fullgraph compile of
   `forward_kinematics_state` + `frame_jacobian` + `inverse_dynamics` + `mass_matrix` on a free-flyer model,
   checking equality with eager. Verify with `TORCH_LOGS=graph_breaks` (zero breaks).
2. **Benchmark columns (F2, F4, F10).** Add `--compile` mode, peak-memory field, and a `rnea_dq` (vmap+jacrev)
   algo to `benchmarks/run.py`; commit one CPU baseline JSON. This decides whether analytic derivatives or
   sparse ABA come first.
3. **Write the derivatives contract (F3), doc only.** One ADR under `docs/contracts/decisions/`: names, return
   shapes `(B..., nv, nv)`, tangent convention for `q` (partials w.r.t. `dq` in `Model.integrate(q, dq)` at
   `dq = 0`), and the rule "one public function; autograd or analytic is an internal choice validated against
   each other". Same ADR states the kernel seam rule (F8): kernels replace raw passes behind
   `torch.library.custom_op`, never a second public API.
4. **Type the memo (F9).** Document allowed `_memo` keys in `kinematics/CLAUDE.md` (or a small key enum), so the
   next bricks (derivatives, contacts) add keys deliberately.
5. **Reserve the constraint vocabulary (F5), doc only.** A one-page design note: contact = (frame id, type 3D/6D,
   reference frame), how it reuses `frame_jacobian`, and how `fext` maps to it. No code until the two callers
   (BH contacts, legged terms) are scheduled.

## Open questions for the owner

1. Target regime: is batch-1 latency (MPC, control) a goal, or is BR batch-first (fitting, RL, BH)? This decides
   compile/CUDA-graph work vs sparse ABA vs kernels.
2. Derivatives: accept autograd-based derivatives with a fixed contract first (recommended), or wait for analytic
   ones?
3. Contacts: should BR own a constrained-dynamics brick (KKT, impulses), or only provide contact Jacobians and
   leave solving to `optim`/an external simulator (MuJoCo, Newton)? Brax shows owning a contact solver is a large
   commitment.
4. GPU kernels: is Warp acceptable as an optional lazy dependency behind `custom_op`, or is Triton (ships with
   torch) the only allowed kernel language?
5. Biomechanics joints (Euler spherical, spline-coupled custom joint): BR core or a BH extension?
