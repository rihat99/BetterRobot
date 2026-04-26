# 00 · Principles behind these proposals

These proposals were drafted with the brief: *imagine you are a senior
engineer at a big tech company shipping a robotics library that the
community will use for years*. That brief is the lens for every
recommendation below.

## What "library used for years" actually means

The decisions that hurt later — once a library has users — are not the
ones reviewers usually flag. They are the ones that look fine on day
one and only become painful at scale:

1. **Public surface that's too easy to add to, too hard to remove
   from.** Every public name is a contract. Adding the wrong name
   costs nothing today and a deprecation cycle later.
2. **Implicit coupling between layers.** A function that "just for
   convenience" pulls a higher-level type one cycle eventually pulls
   the whole stack with it.
3. **String literals where enums belong.** `reference="local"` is
   harmless until a user passes `"world_local"` (typo, no error) or a
   downstream library wants to introduce a new mode.
4. **One backend baked into the data path.** A `lie.compose` that
   talks straight to PyPose locks you into PyPose; a `lie.compose`
   that goes through `backends.current()` does not.
5. **A frozen `Model` whose fields are tensors but whose schema isn't
   versioned.** A schema change in v1.3 breaks every external user
   who pickled a v1.2 model.
6. **Types that wrap tensors but quietly subclass them.** PyPose's
   `LieTensor` is the cautionary tale. Every `__torch_function__`
   override is a shipping liability.

The proposals here are weighted toward fixing those problems
*now*, before the code has external users, because each is at
least 10× more expensive once the deprecation clock starts.

## What we are *not* trying to do

- We are not trying to reach feature parity with Pinocchio, Drake, or
  Crocoddyl. The roadmap (`docs/status/18_ROADMAP.md`) tracks that.
- We are not relitigating naming
  (`docs/conventions/13_NAMING.md`), the SE(3) layout (`[tx, ty, tz,
  qx, qy, qz, qw]`), or the layered DAG. Those are stable.
- We are not adding speculative abstractions. A `Backend` ABC before
  there is a second backend would be wrong. A `Pose` type that
  someone might one day want is wrong. Every proposal here cites a
  concrete code location it improves.

## Decision filters

Each proposal had to clear all five:

1. **Concrete pain point.** What's broken or fragile today? Cite a
   file and line range.
2. **Cheap-now, expensive-later.** Either the change is reversible
   for a short window, or postponing it forces a deprecation later.
3. **Doesn't break the layered DAG.** Or, if it does, the rewrite
   path is part of the proposal.
4. **Composes with the existing extension seams.** A new abstraction
   that doesn't fit into [15_EXTENSION.md](../conventions/15_EXTENSION.md)
   has to either generalise the seams or shrink itself.
5. **Has an acceptance criterion that fits in one paragraph.** If
   "done" is fuzzy, the proposal isn't ready.

Proposals that failed any of these were dropped — not deferred,
dropped — to keep this folder focused.

## Recurring tensions

Several proposals trade against each other. Naming the tensions
explicitly so reviewers can see the trade-offs:

### Tension 1 — Types vs. tensors

Typed value classes (`SE3`, `Inertia`, `Motion`) are ergonomic at the
call site but obscure the underlying tensor for autograd, batching,
and `torch.compile` analysis. The current code splits the difference:
`lie/` is functional; `spatial/` is value-typed. **Proposal 01 keeps
this split** but extends `spatial/` with `SE3`/`SO3` typed wrappers
for user-facing call sites — internal hot paths still call the
functional `lie/` API on raw tensors. **Proposal 05** keeps `Inertia`
as a single packed `(..., 10)` tensor (avoid two ways to do the same
thing).

### Tension 2 — Public surface size vs. completeness

The 25-symbol cap (
[01_ARCHITECTURE.md §Public API contract](../design/01_ARCHITECTURE.md))
is a strong forcing function but slightly arbitrary. **Proposal 06**
proposes a small, motivated expansion: `SE3`/`SO3`/`Pose` types,
`IKResult`, `IKCostConfig`, `OptimizerConfig`, `ReferenceFrame` enum,
`SolverState`. The principle becomes: *the public surface is exactly
what users need to write idiomatic IK / FK / trajopt code*, not a
fixed integer.

### Tension 3 — Strict typing vs. ergonomics

mypy `--strict` plus full jaxtyping shape annotations are a tax on
internal contributors. **Proposal 04** keeps strictness for the
public surface only; internal modules may opt out per-file. The
public-surface contract is what users program against; the rest is
internals where readability matters more than the last mypy nitpick.

### Tension 4 — Future-proof vs. simple now

The Warp backend is the largest deferred decision. **Proposal 02**
makes the backend dispatch real *now*, even when there is only one
backend, because the cost of routing the half-dozen primitive ops
through `backends.current()` is small and the cost of retrofitting
later is large. **Proposal 03** plans the PyPose replacement so the
known autograd bug stops constraining residual design.

### Tension 5 — Documentation completeness vs. tractability

`docs/` today is a complete *internal* spec set. It is not yet a
*user* manual: there are no tutorials, no how-to guides, and no
auto-generated reference. **Proposal 10** lays out a Diátaxis-shaped
Sphinx site without doubling the documentation footprint —
auto-generated reference + a small handful of hand-written tutorials.

## A short style note

Several proposals call for renaming, deprecation shims, or moves
across the directory tree. The library is at the right point — phase
1 closed, no external users — for these to be cheap. Once the
project has 10 users, every rename is a discussion. The proposals
here cluster the cheap renames into a single "rename sprint" so
contributors live through one disruption, not five.

## Reading order

The proposals are numbered roughly by altitude (foundational →
hygiene). Read the foundational triplet (01, 02, 03) together,
because each conditions on the other two. The mid-altitude
proposals (04–09) are each independent. The hygiene cluster (10–15)
matters but does not block any of the others.
