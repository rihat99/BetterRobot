# Design decisions

This page records the five choices that shape BetterRobot and why each was made. Read it when a design looks odd
and you want to know whether it is deliberate. It is the one page that names the old APIs; they are history and no
longer exist in the library.

## Torch is the only implementation

Every computation is a direct PyTorch formula: Lie groups, spatial algebra, kinematics, dynamics, collision,
optimization and the offline renderer. There is no second backend and no custom kernel.

Earlier versions carried NVIDIA Warp lanes next to the torch code for forward kinematics. Before removing them, the
project recorded both on the same CPU (no GPU was free for the measurement):

| Case (batch 4096) | Warp | torch |
|---|---|---|
| Panda FK, forward only | 2.3 ms | 6.6 ms |
| G1 FK, forward only | 5.9 ms | 27.8 ms |
| Panda FK, with gradients | 106 ms | 23 ms |
| G1 FK, with gradients | 317 ms | 75 ms |

At batch 1 torch won both cases. BetterRobot exists to differentiate through kinematics and dynamics, so the case
that matters is the one with gradients, where Warp was 4× slower. A second backend also doubled every test and
every contract. The Warp lanes were removed. A GPU record was never taken; if one ever shows a large win with
gradients, the question can be reopened.

What torch-only buys: autograd everywhere, one dtype and device story,
and the only hard dependency is torch.

## A forward-kinematics query returns an immutable state

`forward_kinematics_state(model, q)` returns a frozen `KinematicState`: joint transforms, frame poses and the
quantities derived from them for exactly that `q`. Jacobians and the functions in `better_robot.dynamics.state`
take the state as input. See [Model and state](model-and-state.md).

The old design was Pinocchio's: `Model.create_data()` made a mutable `Data` object, and
`forward_kinematics(model, data, q)` and `rnea`, `crba`, `aba` wrote their results into it. That model fits C++,
where the cache saves allocations. In PyTorch it caused three problems:

- **Stale reads.** Nothing tied a `Data` field to the `q` that produced it. Reading a `Data` field after changing `q`
  but before re-running kinematics returned old values silently.
- **Autograd hazards.** In-place writes into cached tensors break the autograd graph or raise version-counter
  errors when the same `Data` is reused across steps.
- **Hidden coupling.** Which fields were valid depended on which functions had run before, so every caller had to
  know the call order.

A frozen state has none of these: it is valid by construction, it never changes under you, and two queries never
share storage. Allocation in PyTorch is cheap next to the kernels.

## Terms are plain functions

An optimization term is a function of named variables and data, such as `lambda q, q_start: q - q_start`, registered
on a `Problem` with a name, weight, kernel and optional `window`. Robot terms in `better_robot.terms` are factories
that return such functions. See [Optimization](optimization.md) and [Write a term](../guides/write-a-term.md).

The old `better_robot.residuals` package defined one class per residual, and `better_robot.tasks` wrapped them into
task objects for the old `better_robot.optim`; the new optimizer was developed as `optim2` next to it. Classes
added a layer that did not pay for itself: each new residual needed a subclass, state lived in instances, and the
solver had to discover what a residual read. With functions, the argument names *are* the reads: the `Problem`
inspects them, hands each term exactly the variables, data and cached values it names, and can tell which terms
are time-local for the banded solver. `optim` itself imports only torch and knows nothing about robots. When the
function-based optimizer covered every use, the class-based packages were removed and `optim2` was renamed `optim`.

## One viewer, built on viser

There is one viewer: `better_robot.viewer.Viewer`, built on viser, with layers fed whole `(T, ...)` clips and an
offline torch rasterizer for recordings. See [Viewer](viewer.md).

viser gives a
browser scene, GUI widgets and a static recording format from one Python dependency, with no OpenGL context on the
machine running the code, which suits remote GPU servers. The offline renderer is torch, so recording works where
viser's browser client cannot run, such as CI. The viewer sits at the top of the import graph and nothing imports
it, so the computational core never pays for it; viser is imported only when a `Viewer` is created.

## Configurations and velocities have different sizes

A configuration `q` has `nq` entries and a velocity `v` has `nv`. They differ whenever a joint lives on a curved
space: a free-flyer root stores a position and a unit quaternion (7 numbers) but moves with a 6-D twist, so a
floating-base robot has `nq = nv + 1`. See [Joints, bodies and frames](joints-bodies-frames.md).

Adding a velocity to a configuration with `+` would push the quaternion off the unit sphere and mix units. So the
model provides the two maps every solver needs:

- `Model.integrate(q, v)`: move from `q` along the tangent vector `v`; the result is a valid configuration.
- `Model.difference(q0, q1)`: the tangent vector that takes `q0` to `q1`.

Solvers step in the tangent space (`nv` coordinates) and map back with `integrate`; residuals between
configurations use `difference`. This is why `terms.RobotManifold(model)` exists and why a phase's `free` mask has
`nv` entries per knot, not `nq`. Fixed-base robots with only revolute and prismatic joints have `nq = nv`, but code
should not assume it.

!!! note "Rule"
    Never add or subtract configurations directly. Use `Model.integrate` and `Model.difference`.
