# Plan — simplification, model redesign, test prune, Warp FK+RNEA

State at planning time (2026-07-20, branch `dev`, 64f074c): the optimization
API v2 round is delivered and verified first-hand — full CPU gate 1,585
passed in 2:12, CUDA suite 15/15 passed on this host, Pinocchio parity and
contracts green, every claimed deletion grep-confirmed. The numerics are
healthy (Panda IK 4 iters/160 ms; trajopt T=30 converges in 9 iters with the
banded route; implicit gradients flow). The debt this round removes is
**speculative surface, indirection layers, duplicated tests, and the silent
AD fallback** — found by a four-way audit (optim, core packages, GPU/Warp,
tests) with every load-bearing claim re-verified against the code.

Audit facts driving this round:

- `optim/` carries features with zero production callers: `block_step_limits`,
  the `LU` solver, and the `mask`/`scale` tangent-reduction subsystem.
  `manifolds.py` is 34 lines holding one dataclass. `implicit.py` is correct
  (a real IFT/KKT backward, more capable than Theseus's because it handles
  bounds) but unreadable: 14 flat helpers and a duplicated 9-field state
  spec.
- `auto` Jacobian strategy falls back to `torch.func` AD **silently** when a
  residual has no `jacobian()` (`problem.py:506-516`). Built-in kinematic
  residuals are analytic (verified empirically: analytic 88 ms vs forced
  jacrev 1,096 ms on Panda IK), but `chamfer`, `human`, and `scene_sdf` ship
  no analytic blocks and would fall back without a word.
- LM computes a projected-gradient candidate step every iteration with no
  bounds guard (`lm.py:678-698`) — two extra JVPs per iteration on every
  unconstrained solve.
- `Model` stores ~44 fields flat **and** copies of them inside
  `structure`/`values`, kept in sync by hand-maintained field lists in
  `to()`/`with_values()`/`_shallow_rebind`. Triplicated data, silent-desync
  fragility.
- `lie/` has a facade layer (`so3.py`/`se3.py` → `_impl.py`) whose reason to
  exist (a removed PyPose backend) is gone: ~23 one-line forwards, every
  `_impl` function single-caller. `types.py` re-imports `so3` lazily inside
  ~18 methods with no cycle to justify it.
- `dynamics/derivatives.py` is thin `torch.autograd.functional.jacobian`
  sugar (no analytic content, one test caller); `integrators.py` is a
  one-line wrapper with only its own test as caller;
  `tasks/parameterization.py` is 156 lines whose accepted class is never
  invoked and whose implemented class is rejected by its only consumer.
- The test suite is mostly justified (parity, LM numerics, gradchecks stay
  untouched) but ~740 LOC is verified duplication from the v2 migration,
  concentrated in `*_v2.py` files sitting next to migrated originals.
- GPU truth: the eager pipeline is kernel-launch-bound (~46k launches, 200 ms
  of CPU dispatch per B=1024 Panda IK solve; GPU compute itself 66 ms). Small
  problems run faster on CPU; trajopt already wins 1.75× on GPU; dense
  algebra is not a bottleneck. Warp FK is 24× faster than eager torch on the
  SMPL-like model — but Panda is excluded from the Warp lane entirely because
  the kernel does not support mimic joints, and the fallback is silent.

## The four orders

| Order | What | Acceptance in one line |
|---|---|---|
| `for_agents/01_simplify.md` | Delete speculative surface, flatten `lie/`, restructure `implicit.py`, add fallback warnings, prune dead dynamics/tasks modules | src ≤ 20,700 lines; warning fires in a test; full gate green |
| `for_agents/02_test_prune.md` | Delete verified-duplicate tests, drop the `_v2` suffixes | ~700+ test LOC gone with zero coverage loss; gate count accounted |
| `for_agents/03_model_redesign.md` | `Model` becomes a thin `ModelStructure` + `ModelValues` combiner; no field triplication | `model.py` ≤ ~320 lines; no `field_updates` dicts; parity green |
| `for_agents/04_warp_fk_rnea.md` | Mimic-joint support in Warp FK, Warp RNEA kernel, explicit fallback warning, kernel polish | Panda runs the Warp lane; RNEA parity+gradcheck+timing recorded; CUDA suite green |

Run them in order; each ends with the full gate green. 01 and 02 are
independent in content but 02 assumes 01's deletions have landed. 03 is the
widest churn; 04 is additive.

## Decisions taken in this plan (owner may veto before launch)

1. **`mask`/`scale` on `Variable` are deleted**, not documented-as-future.
   Zero production callers; the machinery runs only in identity form. Git
   history recovers them when a real masked-solve caller appears.
2. **`LU` is deleted.** The LM normal matrix is SPD by construction; a
   non-SPD path does not exist in-tree.
3. **`SO3Variable`/`SE3Variable` stay.** Small, public, plausibly wanted for
   object-pose fitting — but they are currently test-only surface; noted
   here so the choice is conscious.
4. **`Node` gets a trim, not a redesign.** Dead `_epoch` field, bool depth,
   simplified merge key. The larger idea (pass a shared `RobotState` instead
   of auto-wrap-and-merge) is deferred; the ergonomics of auto-wrap win at
   alpha.
5. **`joint_dispatch.py` stays separate from `joint.py`.** Audited: `joint.py`
   is a zero-import leaf enum; the dispatcher imports `joint_models` and
   `model_structure`. Merging would pollute a leaf every package depends on.
6. **No blanket per-package `utils.py`.** Single-caller helpers stay next to
   their caller (moving them scatters algorithms). The convention adopted
   instead: when a package has cross-module private helpers, they live in one
   file named `utils.py` — so `optim/_solver_common.py` and
   `residuals/_variables.py` are renamed to `utils.py`;
   `residuals/_temporal_jacobian.py` keeps its name (a cohesive feature
   module, not a helper pile); `tasks/` gains `utils.py` for its two
   verified duplicate helpers.
7. **`contact_forces.py` and the vertical-slice tests stay.** Bulky but
   inherent; real integration value.

## Ground rules

1. **Read `DESIGN_RULES.md` (repo root, untracked) before writing code.** It
   is the owner's taste encoded; orders assume it.
2. **No backward compatibility.** Removed or renamed public symbols get a row
   in root `MIGRATION.md`; never an alias, shim, or deprecation path.
3. **The safety net stays green.** Pinocchio parity, contracts, and the full
   non-bench/non-CUDA gate pass at the end of every order. Never weaken
   parity or architecture tests to land a change; each order lists the
   contract files it may touch.
4. **CUDA is agent-runnable here.** 8 idle-ish RTX 6000 Ada GPUs; pin one
   with `CUDA_VISIBLE_DEVICES` (2, 3, 7 usually free) and run
   `uv run pytest tests/ -q -m cuda` yourself. "CUDA was unavailable" is not
   an acceptable results line on this host. If `torch.cuda.is_available()`
   is False in your sandbox, say so explicitly and stop — do not ship
   device-conditional code unvalidated.
5. **Deletion needs evidence** — cite the zero-caller grep in the results
   file. Budgets are hard; an order that cannot meet its budget stops and
   reports rather than redefining success.
6. **Docs stay true per order** — every page, docstring, and `CLAUDE.md` an
   order falsifies is updated in the same order.
7. **Honest results files.** Each order writes `for_agents/NN_results.md`:
   delivered, deviations (findings, not failures), exact test counts, line
   accounting by the `wc -l` method.
8. **Subagents run on Opus.** Codex is available for adversarial review
   (it now shares the same skills and reads `CLAUDE.md` via its project-doc
   fallback); treat its taste for abstraction with suspicion, its factual
   findings with respect.
9. **Everything on `dev`.** No feature branches, no worktrees unless asked.
