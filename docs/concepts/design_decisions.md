# Design decisions

A library is an accumulation of decisions, and most libraries hide them. This page states
BetterRobot's load-bearing decisions openly: what we chose, what we rejected, and why. Each section
names the alternative we said no to, because a decision without its alternative is just an
assertion.

If you disagree with one of these, this page should at least tell you whether we considered your
option and what it would have cost.

(decision-pytorch-native)=
## PyTorch-native, not bindings and not JAX

The classical way to build a fast robotics library is a C++ core with Python bindings — Pinocchio
is the outstanding example. We chose pure Python on PyTorch tensors instead, for one dominant
reason: **autograd is the substrate of everything we do.** Fitting articulated models to
observations, differentiating kinematics inside a training loop, computing solver derivatives —
all of it needs gradients through the library, not just around it. With a C++ core, every
derivative you want is a derivative someone had to hand-write and bind. With PyTorch, every
operation composes with every other differentiable operation the ecosystem has — including your
neural network.

JAX would have offered the same, and libraries like MJX and Pyroki prove it works. We chose
PyTorch because the surrounding ecosystem we interoperate with (vision models, human body models,
learned priors) lives overwhelmingly in PyTorch, and crossing the JAX–PyTorch boundary in a hot
loop costs more than either framework's advantages are worth.

The honest cost: eager PyTorch is slow at short serial loops, and a kinematic-tree sweep is
exactly that. We pay that cost deliberately and buy it back in two ways — `torch.compile` on the
reference implementations, and the current opt-in fused FK lane where it helps (see below).

(decision-functional-lie)=
## Functional Lie operations, not tensor subclasses

Rotations, poses, twists, and wrenches use ordinary tensors plus explicit functions. A tensor
subclass could carry a semantic tag, but it would complicate interoperability, compilation, and
serialization at nearly every boundary. The functional API keeps conventions visible while
letting Lie operations compose directly with the rest of PyTorch.

(decision-model-data)=
## The Model / Data split

Borrowed from Pinocchio, deliberately. `Model` is shallowly frozen structure — topology, joint
types, index maps, and tensors that callers treat as read-only. `Data` is a per-query workspace
of results. The alternative — one mutable `Robot` object
that holds both — collapses the moment you want the same robot evaluated at a thousand
configurations on a GPU, or two threads asking different questions of one model, or gradients
flowing through results while the structure stays constant. Splitting identity from state is what
makes batching, sharing, and differentiation compatible with each other.
{doc}`model_and_data` develops this in depth.

(decision-one-ir)=
## One model-building IR, not parser-specific models

URDF, MJCF, and the programmatic `ModelBuilder` all produce the same `IRModel`, and one
`build_model()` path validates and lowers it into a `Model`. Keeping parser details out of the
compute model gives every input format the same joint semantics, indexing rules, and validation.
Adding a format therefore means translating into the IR, not adding another model family.

(decision-batched)=
## Batched by default

The documented tensor APIs accept leading batch axes `(B..., feature)`. This looks like a style
choice and is actually the performance model: the Python loop in a pass runs over the robot's
*topology* (a fixed tuple, unrolled by `torch.compile`), never over the batch. One robot or a
thousand robots is the same Python work. A library that is batch-aware in some functions and
scalar-only in others ends up writing every algorithm twice; we refused that fork from the start.

(decision-unified-base)=
## One code path for fixed and floating base

A floating-base robot is a robot whose first joint is a free-flyer (`nq=7`, `nv=6`). That is the
entire difference. There is no floating-base flag, no parallel `_floating` solver family, no
special base-pose argument. Libraries that split the two cases pay for the split in every
operation forever; unifying them means humanoid IK and manipulator IK are literally the same call.

(decision-whole-pass)=
## Whole-pass functions, not per-operation dispatch

BetterRobot once had a `backends/` protocol layer that dispatched each small Lie operation through
an interface, in the hope of swapping implementations later. We deleted it. Per-operation
dispatch has the wrong granularity for acceleration: no GPU kernel worth writing replaces a single
quaternion multiply. The unit of architecture is the **whole pass** — the FK sweep, RNEA, a
residual with its Jacobian — as a plain function: tensors in, tensors out. A pass has a reference
PyTorch implementation (the correctness oracle, the CPU path, the second-order-gradient path) and
may have a fused kernel implementation behind the same signature, selected by an explicit
argument at the call site. No registries, no protocols, no hidden global state.

(decision-warp-kernels)=
## Warp for kernels, not hand-written CUDA

Where a pass earns a GPU kernel, we write it in
[NVIDIA Warp](https://nvidia.github.io/warp/stable/index.html) — a Python-syntax kernel language —
and not in raw CUDA. Raw CUDA can win specialized benchmarks, but it would create a second source
language and a much larger maintenance surface. We accept that measured performance trade and buy
back distance with kernel fusion and graph capture. The other boundary we hold: kernels are whole
passes, never per-op. PyTorch is the truth; Warp is the speed.

(decision-own-lm)=
## Our own Levenberg–Marquardt, not our own Adam

Nonlinear least squares on manifolds is the library's core competence. When this solver stack was
chosen, we found no maintained PyTorch-native implementation that combined what robotics
estimation needs: batched independent problems with per-problem damping and convergence,
manifold-aware steps, robust kernels, and honest treatment of joint limits. So LM and
Gauss–Newton are ours, and we own their numerics — adaptive damping, iteratively reweighted
robust losses, bound handling — as a responsibility, not an implementation detail.

First-order methods are the opposite case: `torch.optim` already ships excellent Adam and friends,
battle-tested by the entire deep-learning world. Reimplementing them inside a robotics library adds
code without adding value. Our job there is only the thin manifold layer that first-order steps
need — compute gradients in tangent space, retract steps back onto the manifold — and to stay out
of the way.

(decision-residual-problems)=
## Residual least squares, not arbitrary scalar objectives

`Problem` is deliberately residual-valued: providers return error vectors, and least-squares
solvers use their Jacobians and block structure. That surface makes robust losses, diagnostics,
and Gauss–Newton structure explicit. Arbitrary scalar objectives remain possible in ordinary
PyTorch, but they belong with `torch.optim` rather than being disguised as least squares.

(decision-variable-blocks)=
## Optimization over named variable blocks, not one flat vector

Real estimation problems optimize heterogeneous things jointly: joint configurations, object
poses, shape parameters, camera intrinsics, contact forces. An optimizer hardwired to a single
flat `q` vector forces every consumer to invent its own packing/unpacking layer — which is
exactly how hand-rolled optimization code proliferates. So the problem surface is a set of *named
variable blocks*, each with its own geometry and optional bounds, and residuals declare which blocks
they read. IK is then just a problem with one block; a body-fitting pipeline is a problem with
five. Same solver, no packing code.

(decision-pose-layout)=
## Quaternions scalar-last, tangents linear-first

Every SE(3) pose is `[tx, ty, tz, qx, qy, qz, qw]` — quaternion scalar **last** — and every
tangent/velocity is `[linear(3), angular(3)]`. Both are ecosystem alignments: scalar-last matches
SciPy, ROS, Eigen's coefficient/storage order, and Pinocchio (MuJoCo is the notable scalar-first
holdout), and linear-first matches Pinocchio's motion-vector convention. Within one process you can pass poses between
BetterRobot, SciPy, and ROS tooling without a reorder. There is no deep mathematical argument for
either order — which is precisely why we align with the largest neighborhood and then never,
ever deviate. Conventions only help when they are absolute.

(decision-jacobian-frame)=
## Jacobians default to LOCAL_WORLD_ALIGNED

`get_frame_jacobian` answers, by default, the question users actually ask: "how does the world
position and world orientation of this frame change with the joints?" That is the
local-world-aligned convention — linear rows are the world-frame velocity of the frame origin.
The classical alternatives (`WORLD`: velocity measured at the world origin; `LOCAL`: everything
in the body frame) are available by argument. The default was chosen for the common case, and the
frame details are documented in {doc}`kinematics_and_jacobians`: converting
`LOCAL_WORLD_ALIGNED` to `LOCAL` uses a rotation-only block transform, while converting `WORLD`
to `LOCAL` also shifts the reference point and therefore uses the full adjoint.

(decision-analysis-only)=
## Analysis and optimization, not simulation

BetterRobot computes kinematics, dynamics quantities, and solutions to optimization problems. It
does not ship a physics simulator: no contact solver, no integrator loop marching a world forward
in time. That fence is deliberate — simulation is a different product with different correctness
criteria, and excellent ones exist (MuJoCo, Drake, Isaac). BetterRobot is the layer you use to
*analyze* and *fit*; if you need to simulate, pair it with a simulator.

## What these decisions cost

Honesty requires the bill, not just the benefits: pure PyTorch means eager CPU tree sweeps are
slower than Pinocchio's C++ and always will be; batching-by-default makes some single-robot code
carry an extra dimension in its head; refusing hand-written CUDA leaves measured throughput on
the table against specialized engines; owning LM numerics means we maintain solver code that a
wrapper library would not. We consider every one of these a good trade. When one stops being
good, the decision — and this page — changes.
