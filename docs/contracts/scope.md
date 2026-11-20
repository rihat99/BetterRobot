# Scope

BetterRobot computes kinematics, dynamics and least-squares solutions for robots, batched and differentiable in
PyTorch, and shows the results in a browser viewer. A feature outside this page needs a decision record that
changes the page first.

## In scope

- Robot models as values: a `Model` built from URDF, MJCF or a `ModelBuilder`, with its shapes in a separate
  `Geometry`.
- Lie and spatial algebra on tensors, forward kinematics, Jacobians, rigid-body dynamics (RNEA, CRBA, ABA,
  centroidal quantities) and sphere and capsule collision distances.
- Nonlinear least squares and scalar minimisation over named tensors (`optim`), robot terms and inverse
  kinematics (`terms`), differentiable through `solve_layer`.
- A viser-based viewer for recorded and live scenes.

## Non-goals

- Simulation: the library evaluates dynamics; it does not step time, resolve contacts or integrate a scene.
- Motion planning, task scheduling and control loops.
- A second numerical backend: torch is the only implementation.
- A native model file: a model is rebuilt from its source description; `torch.save` of the tensors is the
  caller's business.
- MJCF beyond the passive rigid tree: actuators, tendons, sensors, contact settings, joint springs, keyframes,
  cameras, lights and simulation options other than gravity are ignored by rule; what lies inside the rigid tree
  but cannot be expressed (a body with two joints, a limited ball joint, an equality other than a linear coupling of
  two joints) raises `ModelError` ([Robot descriptions](../concepts/io.md)).
- Downloading robot descriptions or keeping a registry of them; `robot_descriptions` serves the examples.
- Kinematic Hessians as functions: autograd gives them.

## Inclusion bar

A feature lands when it:

1. has an oracle test (Pinocchio, MuJoCo, an analytic formula or finite differences) that names its oracle;
2. works for every joint kind, every float dtype and any leading batch shape;
3. keeps the hot paths free of host syncs and of Python loops over the batch (contracts C-TENS-4 and C-PERF-2);
4. removes more from its callers than it adds to the library.

## Commonly rejected

Reopening one of these needs new evidence, not preference.

| Idea | Reason |
|---|---|
| A second backend (Warp, JAX, custom CUDA kernels) | doubles every test and every contract (C-PERF-5) |
| A mutable `Data` filled by the algorithms | stale reads, autograd hazards, call-order coupling |
| Residual classes or cost objects | data held in instances, heavier to compose |
| Another viewer (meshcat, rerun, native OpenGL) | weaker recording, duplicate state, a display on the GPU host |
| Rotation vectors so that `nq == nv`, or `q + v` on configurations | singular at pi; use `integrate` |
| Value classes for poses, motions, forces and inertias | a second API beside the tensor functions |
| A joint, parser or solver registry | a second way in beside the class or function itself |
| A native model file | a format to version forever for a value that rebuilds from its source |
| An MJCF parser of our own | a second reading of the format; MuJoCo's compiler is the reference |
| Scalar-first `wxyz` quaternions | the layout is `xyzw` everywhere (C-NAME-3) |