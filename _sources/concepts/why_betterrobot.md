# Why BetterRobot

BetterRobot is for robot calculations that need to live inside a PyTorch
program. It computes kinematics, rigid-body dynamics, residuals, and
optimization steps with ordinary tensors. Those calculations can be batched,
moved between CPU and GPU, and differentiated with PyTorch.

That combination is the point. A robotics library can be fast and complete
without fitting naturally into a learning or estimation pipeline. A tensor
library can differentiate arbitrary code without knowing that a quaternion is
not four independent numbers. BetterRobot joins the two: robotics supplies the
structure, while PyTorch supplies tensor execution and automatic
differentiation.

## The questions it answers

Given a robot model and a joint configuration, BetterRobot can answer
questions such as:

- Where are the robot's joints, bodies, and tool frames?
- How does a small joint motion move an end effector?
- Which joint forces produce a requested acceleration?
- Which configuration best matches several pose or observation targets?
- How can the same calculation run for a large batch of configurations?

The public API keeps these questions close to their mathematical form. Forward
kinematics takes a model and `q`. Inverse kinematics takes pose targets and
returns a candidate `q`. Direct optimization uses variables and residuals
without asking callers to flatten every kind of state into one anonymous
vector.

## Why PyTorch is the foundation

Robot estimation rarely ends at a robot-only boundary. A target may come from
another differentiable model, and the fitted configuration may feed a larger
loss. Keeping the robot calculation in PyTorch lets gradients cross those
boundaries without a second array type or a hand-written derivative bridge.

This does not mean every convenience function returns an attached autograd
graph. Task solvers normally return detached results, because an iterative
solve needs an explicit differentiation policy. The tensor operations beneath
them are differentiable, and eligible optimization problems have a guarded
implicit-differentiation path. The distinction matters: “written in PyTorch”
is a foundation, not a promise that every call should be differentiated in the
same way.

The alternative and its cost are recorded in
{ref}`decision-pytorch-native`.

## One robot, one set of algorithms

A floating-base robot does not enter a separate part of the library. Its root
joint is a free-flyer with seven stored configuration numbers and six tangent
degrees of freedom. Forward kinematics, Jacobians, dynamics, and optimization
then use the same functions as a fixed-base arm.

This choice removes an entire axis of special cases. It also makes manifold
rules unavoidable in the right place: joint models define how configurations
are integrated and compared, while algorithms consume that common interface.
See {ref}`decision-unified-base` for the rejected split design.

## Batching is part of the model

BetterRobot uses trailing event dimensions and leading batch dimensions. A
single configuration has shape `(nq,)`; a batch may have shape
`(B..., nq)`. The topology loop still runs over the robot's joints, while
PyTorch carries every batch entry through the tensor operations.

Batching is therefore not a separate “GPU API.” The same function handles one
configuration, a time-and-sample grid, or thousands of independent queries.
The detailed argument is in {ref}`decision-batched`; the practical introduction
is {doc}`/getting_started/05_batched_gpu`.

## A deliberately bounded library

BetterRobot analyzes articulated systems and fits their state. It is not a
physics simulator. It does not advance a world through time, solve contact
impulses, or manage sensors and rendering as one simulated environment.
MuJoCo, Drake, and other simulators already solve that different problem.

The boundary is useful. Kinematics and dynamics quantities can be compared
against independent references, and optimization behavior can be tested
without also owning a contact engine. See {ref}`decision-analysis-only`.

## What the choices cost

The design has real costs. Short serial tree walks are not where eager
PyTorch is strongest. Leading batch axes require careful shape conventions.
Manifold-aware least squares is more code to maintain than a wrapper around a
general optimizer. An optional fused kernel can improve a measured pass, but
it also needs a second implementation and gradient parity evidence.

Those costs are accepted openly rather than hidden behind slogans. The full
record, including alternatives, is {doc}`design_decisions`.

## Where to continue

- {doc}`architecture` explains how the packages enforce these boundaries.
- {doc}`model_and_data` explains the robot identity/workspace split.
- {doc}`lie_and_spatial` introduces the geometry used by poses and motion.
- {doc}`residuals_costs_and_solvers` explains how fitting becomes least
  squares on manifolds.
