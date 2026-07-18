# 01 — Vision

## What BetterRobot is

A **PyTorch-native toolkit for differentiable robot kinematics, dynamics, and
optimization**. Load a robot (or build one in code), compute where its parts
are and how they move, define what you want as residuals, and solve — batched,
on CPU or GPU, with gradients flowing through everything.

The product, in one phrase: *batched differentiable optimization over
articulated structures*. IK, trajectory optimization, model fitting, and
contact-force estimation are all the same thing to this library — a set of
named variables on manifolds, a set of residuals, and a solver.

## The LEGO principle

This phase exists because the library drifted toward the opposite of what it
should be. The redesign built correct machinery, then wrapped it in facades,
validation walls, and "trust us" surfaces until the machinery disappeared.
The corrective principle:

**BetterRobot is a box of blocks, not a vending machine.**

- Every block works standalone. `forward_kinematics` in your training loop,
  `rnea` inside your `nn.Module`, a Jacobian in your own solver, `se3.exp` on
  a bare tensor — all first-class, all documented, no ceremony.
- Every block is differentiable. Gradients w.r.t. configurations *and* model
  parameters are a guarantee, not a lucky property. Where a solver returns a
  detached answer by design, an opt-in differentiable path exists and is easy
  to find.
- Blocks snap together. The high-level calls (`solve_ik`, `solve_trajopt`) are
  short compositions of public blocks — convenience, never the only door.
- Hiding is reserved for true internals: kernel launch machinery, capture
  plumbing, scratch layouts. If a robotics user could want it, it is public.
- We are a **torch extension**, not a framework. First-order optimization
  belongs to `torch.optim`. Training loops belong to the user. We add what
  torch does not have — manifolds, whole-pass robot algorithms, batched
  manifold-aware least squares — and we stay out of everything else.

## Design principles

1. **Simplicity, elegance, and efficiency are one requirement.** Complex things
   built compactly. If a function needs a wall of `if`-statements, the design
   above it is wrong. Every class and function is polished to the same shape
   as its siblings — same naming, same return style, same error style.
2. **No abstraction without a second concrete caller.** Single-caller
   protocols, speculative flexibility, and config options nobody sets are
   deleted on sight.
3. **Validate once, at the boundary.** Public entry points check inputs once
   with helpful errors. Everything beneath trusts its inputs. No re-checking
   an immutable object on every call, no re-policing typed dataclasses at
   runtime, no shadow copies of the API split by trust level.
4. **Batched by default.** Leading batch axes `(B..., feature)` everywhere;
   the Python loop runs over topology, never over the batch.
5. **Own the second-order numerics; borrow the first-order ones.** LM/GN with
   damping, robust kernels, manifolds, and bounds are our craft. Adam is
   torch's.
6. **No backward compatibility before 1.0.** Unshipped code owes nothing to
   its past. The migration ledger records removals; that is the whole debt.
7. **Honest docs, written for humans.** A beginner can learn robotics from
   them; an expert can find the design rationale and every "why X not Y".
   No internal jargon, no milestone names, no machine prose.

## Conventions that never change

SE(3) pose `[tx, ty, tz, qx, qy, qz, qw]` (scalar-last quaternion); tangent
`[linear(3), angular(3)]`; `get_frame_jacobian` defaults to
LOCAL_WORLD_ALIGNED; batched `(B..., feature)`. The pinocchio-parity suite is
the numerical anchor of the whole library.

## Scope fence

Inside: Lie/SE(3) math, spatial algebra, kinematic trees, Featherstone
dynamics, the residual library, manifold least squares, model construction
(URDF/MJCF/builder), visualization as an optional extra.

Outside, permanently unless deliberately re-decided: physics simulation,
parametric body models (blend shapes, LBS — the hooks are ours, the models are
not), task pipelines and datasets, camera models beyond a pinhole residual.

## Where this goes

After this phase: a library around **22k source lines** with one optimizer
stack instead of two, a public surface where every exported name works and
every working block is exported, code a careful human would be proud of, and
documentation that reads like a good book. That is the release candidate.
