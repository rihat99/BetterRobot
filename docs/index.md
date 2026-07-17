# Welcome to BetterRobot

**BetterRobot** is a PyTorch-native, GPU-ready library for robot
kinematics, dynamics, and trajectory optimisation. It follows
[Pinocchio](https://github.com/stack-of-tasks/pinocchio)'s `Model` /
`Data` architecture, runs on plain PyTorch tensors with autograd,
and uses **one code path** for fixed-base and floating-base robots.

The five commitments that shape every other decision:

- **PyTorch on the hot path.** Forward kinematics, Jacobians,
  residuals, costs, and solver iterates all participate in autograd.
  No `AutoDiffXd` scalar to switch into, no JAX mode flag, no C
  extension that breaks the gradient graph.
- **Batched tensor math.** FK, residuals, and analytic Jacobians accept
  `(B..., feature)`. The current optimizer stack is single-problem; batched
  solving is scheduled for M2b.
- **One code path for fixed and floating base.** A floating-base
  robot is one whose root joint is `JointFreeFlyer`. The IK solver
  does not know the difference.
- **One residual / cost / solver stack.** IK, trajectory
  optimisation, retargeting (and future filtering, optimal-control)
  share a `Residual` Protocol, a `CostStack`, a
  `LeastSquaresProblem`, and an `Optimizer`.
- **A backend that does not leak.** The math layer routes through a
  `Backend` Protocol with a torch-native default. Users see
  `torch.Tensor` in and out at every public surface.

A minimal example — load a Panda URDF, solve IK to a target pose,
read back the joint solution:

<!-- front-page-example:start -->
```python
import better_robot as br
from better_robot.tasks.ik import IKCostConfig, OptimizerConfig
from robot_descriptions import panda_description

model = br.load(panda_description.URDF_PATH)
q0 = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit)
q_goal = q0.clone()
q_goal[0] = 0.25
target_pose = br.forward_kinematics(
    model, q_goal, compute_frames=True
).frame_pose_world[model.frame_id("body_panda_hand")].clone()

result = br.solve_ik(
    model,
    {"body_panda_hand": target_pose},
    initial_q=q0,
    cost_cfg=IKCostConfig(limit_weight=0.0, rest_weight=0.0),
    optimizer_cfg=OptimizerConfig(max_iter=100),
)

result.q  # (nq,) joint solution
result.frame_pose("body_panda_hand")  # (7,) SE(3) pose at the solution
```
<!-- front-page-example:end -->

## What ships today

Forward kinematics; analytic Jacobians with an unbatched central-FD fallback; the residual
library (pose / position / orientation, joint position limits, rest,
contact consistency, reference trajectories, velocity and
acceleration smoothness, time-indexed residuals); `CostStack`;
LM, GN, Adam, L-BFGS, and multi-stage optimisers; pluggable linear
solvers (Cholesky, LSTSQ); pluggable robust
kernels (L2, Huber, Cauchy, Tukey) and damping strategies (Constant,
Adaptive); single-problem IK on fixed and floating-base robots;
trajectory optimisation with knot and B-spline parameterisations;
Featherstone dynamics (RNEA / ABA / CRBA / CCRBA), centroidal
momentum, and autograd-derived `compute_*_derivatives`; a three-layer
Crocoddyl-style action model; URDF and MJCF parsers; a programmatic
`ModelBuilder`; a viewer with skeleton / URDF-mesh render modes,
draggable IK target gizmos, and trajectory playback.

A small set of named symbols are deliberately stubbed and listed in
{doc}`reference/roadmap`. They have the correct signatures and raise
`NotImplementedError`.

## Status

The 26-symbol public API is **frozen** under
`tests/contract/test_public_api.py`. Additions require a SemVer
minor bump; removals are forbidden in v0.x. See
{doc}`reference/changelog` for release notes.

## License

BetterRobot is open-sourced under the BSD-3-Clause license.

## Table of Contents

```{toctree}
:maxdepth: 2
:caption: Get Started

getting_started/index
```

```{toctree}
:maxdepth: 2
:caption: Concepts

concepts/index
```

```{toctree}
:maxdepth: 1
:caption: Conventions

conventions/index
```

```{toctree}
:maxdepth: 1
:caption: Reference

reference/index
```

```{toctree}
:hidden:
:caption: Project Links

GitHub <https://github.com/rihat99/BetterRobot>
PyPI <https://pypi.org/project/better-robot/>
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
