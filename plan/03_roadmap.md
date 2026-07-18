# 03 — Roadmap

Six phases, ordered by dependency. Phases 1–2 are the heart (one optimizer
stack); 3–4 open and polish the core; 5 rewrites the docs against the settled
API; 6 sweeps the floor. Each phase has a work order in `for_agents/` and ends
with the full gate green: `uv run pytest tests/ -m "not bench and not cuda"`,
pinocchio parity, contracts, scoped ruff, and a successful Sphinx build.

## Phase 1 — Delete the legacy stack

Remove everything with zero production callers: the legacy
`LeastSquaresProblem`/`CostStack`/optimizer/strategy stack, the `costs/`
package, the graph-capture harness, dead solver options. The only real code
change is `solve_trajopt`'s input — it takes residual items directly instead
of a `CostStack` it was only using as a list. Examples updated; every removed
public symbol appended to `migration_ledger.md`.

*Acceptance:* `grep -r "CostStack\|LeastSquaresProblem" src/` returns nothing;
`optim/optimizers`, `optim/strategies`, `optim/problem.py`, `optim/state.py`,
`optim/cost_stack.py`, `costs/`, `optim/_graph_executor.py` no longer exist;
net `src/` delta ≤ −1,700 lines; full gate green.

## Phase 2 — Rebuild the optimizer core

Flatten `optim/blocks/` into `optim/`, introduce the builder surface
(`add_variable` / `add_residual`), collapse the shadow API under the new
validation policy, replace custom Adam with the `torch.optim` adapter, inline
the phase engine, drop scalar objectives and the provider DAG, slim implicit
differentiation, merge kernels and solvers into single files, collapse the
residual protocol to one shape. The work order carries the API sketch from
`02_architecture.md` §1.

*Acceptance:* `optim/` ≤ 3,800 lines; the line-fit example from the
architecture doc runs as written; direct IK takes one screen; `solve_ik` /
`solve_trajopt` / `solve_contact_forces` keep their solution-quality
regression tests (same solutions within tolerance) — solver-state semantics
for Adam and the removed options change deliberately, with their tests; full
gate green.

## Phase 3 — Open the core

Export the raw passes uniformly with one naming convention and `*Result`
returns; make `ModelStructure`/`ModelValues` top-level; make `Data` optional
for dynamics wrappers; wire `differentiable=True` through `solve_ik` to the
implicit path; stop detaching `solve_contact_forces` outputs; delete stub
exports and the `nle` alias; make `br.spatial` reachable; reconcile
`ReferenceFrame` with the `Literal` frame strings.

*Acceptance:* a documented gradcheck test passes through every public entry
point w.r.t. `q` and (where meaningful) model values; `solve_ik(...,
differentiable=True)` back-propagates to targets; none of the enumerated
stub exports (six dynamics + the listed residual stubs) remains; full gate
green.

## Phase 4 — Polish the core

The validation diet and style pass: boundary-once validation (frozen values
validated at construction, not per call — counter contract updated
deliberately), shared task-boundary validators replacing the per-task walls,
no value-content policing, belt-and-suspenders branches removed, the
joint-dispatch transform duplication resolved to one source of truth, uniform
return/naming/error conventions across sibling functions.

*Acceptance:* the named validation walls (`tasks/contact_forces.py`,
`tasks/ik.py`, `data_model/model.py` per the work order) are each ≤ ⅓ their
current length with equal or better error messages; duplicate helper
definitions are gone; full gate green including updated contracts.

## Phase 5 — Rewrite the documentation

Restructure to the four-quadrant layout (~24 hand-written pages), rewrite the
machine prose for humans, add the beginner explainers (what a Jacobian is —
analytic vs autodiff vs finite difference; SE(3)/quaternion primer; FK/IK/
least-squares/LM in plain words), surface every design decision with its
rejected alternative, purge all milestone/process jargon, keep every snippet
executable. The work order carries the full audit: per-file verdicts, the
jargon census, and the target tree.

*Acceptance:* zero matches for milestone/process jargon per the work-order
grep list; docs build clean; snippet tests green; the named beginner questions
each have one obvious page answering them.

## Phase 6 — Loose ends

The small fixes and honest bookkeeping accumulated by the reviews: the
inactive-residual objective inconsistency, the contact-point limitation
documented, the stray `utils/` directory, `CLAUDE.md` files re-synced to the
polished reality, the trajectory-benchmark baseline either measured or
removed (no `PENDING_MEASUREMENT` shipping), changelog rewritten without
milestone framing.

*Acceptance:* the work-order checklist is fully resolved — each item fixed,
or moved to `for_future.md` with a reason; no placeholder artifacts in the
tree; full gate green.

## What is deliberately not here

GPU/Warp expansion, CUDA-graph capture productization, collision, external
competitor benchmarks, new residuals or solvers, native neural-network
integration — all in `for_future.md`. This phase makes the library smaller
and better, not bigger.
