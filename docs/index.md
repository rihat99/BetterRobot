# Welcome to BetterRobot!

**BetterRobot** is a PyTorch-native, GPU-ready library for robot
kinematics, dynamics, and trajectory optimisation. It follows
[Pinocchio](https://github.com/stack-of-tasks/pinocchio)'s `Model` /
`Data` architecture, runs on plain PyTorch tensors with autograd, and
uses **one code path** for fixed-base and floating-base robots.

The core objectives of the framework are:

- **PyTorch-first.** Pure PyTorch on the hot path. No PyPose, no JAX, no
  C extensions for the default backend.
- **Batched by default.** Every tensor carries a leading batch
  dimension; there is no scalar fast path that diverges from the batched
  one.
- **Differentiable end-to-end.** FK, dynamics, residuals, costs, and
  solver iterates all participate in autograd.
- **Single residual / cost / solver stack.** IK, trajectory
  optimisation, retargeting, and (eventually) filtering and optimal
  control share the same substrate.
- **Future-proof backend.** A `Backend` Protocol with `LieOps`,
  `KinematicsOps`, and `DynamicsOps` sub-Protocols lets us drop in Warp
  without touching call sites.

A minimal example — load a Panda URDF, solve IK to a target pose, read
back the joint solution:

```python
import better_robot as br

model  = br.load("panda.urdf")
result = br.solve_ik(model, {"panda_hand": target_pose})

result.q                          # (nq,) joint solution
result.frame_pose("panda_hand")   # (7,) SE(3) pose at the solution
```

Implemented today: forward kinematics, analytic + autograd Jacobians,
the full residual library (pose, position, orientation, joint limits,
rest, smoothness, contact consistency, reference trajectories),
`CostStack`, LM / GN / Adam / LBFGS / multi-stage optimisers, IK on
fixed- and floating-base robots, trajectory optimisation
(`solve_trajopt`) with knot and B-spline parameterisations,
Featherstone dynamics (RNEA / ABA / CRBA / CCRBA), centroidal momentum,
autograd-derived `compute_*_derivatives`, three-layer Crocoddyl-style
action models, and a viewer (Skeleton, URDF mesh, grid, frame axes,
target gizmos, force vectors, ViserBackend, joint panel, trajectory
player).

What is specified but still stubbed: dynamic integrators
(`semi_implicit_euler` / `symplectic_euler` / `rk4`), `compute_minverse`,
`compute_coriolis_matrix`, analytic Carpentier–Mansard derivatives,
`solve_retarget`, jerk / Yoshikawa / collision / nullspace residuals,
viewer COM / PathTrace / ResidualPlot overlays, `VideoRecorder`, and
the Warp backend kernels. The full list lives in
{doc}`reference/roadmap`.

## Status

The package is pre-1.0. The 26-symbol public API is **frozen** under
`tests/contract/test_public_api.py` — additions require a SemVer minor
bump, removals are forbidden in v0.x. See the
{doc}`reference/changelog` for release notes.

## License

BetterRobot is open-sourced under the BSD-3-Clause license.

## Table of Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started
:titlesonly:

getting_started/index
concepts/index
```

```{toctree}
:maxdepth: 1
:caption: Design Specs
:titlesonly:

design/index
conventions/index
```

```{toctree}
:maxdepth: 1
:caption: Reference
:titlesonly:

reference/index
```

```{toctree}
:hidden:
:caption: Project Links

GitHub <https://github.com/rihat99/BetterRobot>
PyPI <https://pypi.org/project/better-robot/>
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
