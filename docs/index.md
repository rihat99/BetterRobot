# Welcome to BetterRobot

**BetterRobot** is a PyTorch-native, GPU-ready library for robot
kinematics, dynamics, and trajectory optimisation. It follows
[Pinocchio](https://github.com/stack-of-tasks/pinocchio)'s `Model` /
`Data` architecture, runs on plain PyTorch tensors with autograd,
and uses **one code path** for fixed-base and floating-base robots.

The five commitments that shape every other decision:

- **A PyTorch tensor surface on the hot path.** Eager Lie maps, kinematics,
  dynamics, residuals, costs, and named-block problem evaluation have
  path-specific autograd coverage. Task facades return detached results; small
  generic LM/GN problems can explicitly request the documented dense
  first-order implicit backward. The optional Warp FK lane still takes and
  returns Torch tensors.
- **Batched tensor math.** FK, residuals, and analytic Jacobians accept
  `(B..., feature)`. The named-block `Problem` also evaluates independent
  batches; named-block Adam/LM/GN, `solve_ik`, and `solve_trajopt` preserve
  those axes with per-element solver state.
- **One code path for fixed and floating base.** A floating-base
  robot is one whose root joint is `JointFreeFlyer`. The IK solver
  does not know the difference.
- **An explicit optimization migration.** New multi-block code uses named
  `VarSpec`s, a `Problem`, structural residuals, and evaluation-local
  providers. IK and knot trajectory optimization use that named-block stack;
  declared temporal problems can route through block-banded or explicit
  normal-operator solves, with dense fallback for undeclared structure.
- **A whole-pass compute seam that does not leak.** Torch raw passes consume
  `ModelStructure` plus `ModelValues` by default. An eligible opt-in kernel
  may replace an entire pass without changing the public `torch.Tensor`
  surface; individual Lie operations do not switch implementations at
  runtime.

A minimal example — load a Panda URDF, solve IK to a target pose,
read back the joint solution:

```bash
git clone --branch dev https://github.com/rihat99/BetterRobot.git
cd BetterRobot
python -m pip install '.[demos]'
```

The `demos` extra supplies `robot_descriptions`; it is not part of the core
installation.

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

Forward kinematics; analytic Jacobians with an unbatched central-FD fallback;
named optimization-variable blocks with Euclidean, SO(3), SE(3), and robot
configuration manifolds; batched `Problem` evaluation with mask-eliminated
tangent coordinates, structural residuals, scalar objective terms, and lazy
provider DAGs; the residual library (pose / position / orientation,
joint position limits, rest, contact consistency, reference trajectories,
velocity and acceleration smoothness, time-indexed residuals); named-block
LM, GN, Adam, and functional phases; dense, block-banded,
and normal-operator linear solvers (Cholesky, LSTSQ, BandedCholesky,
NormalCG); pluggable robust
kernels (L2, Huber, Cauchy, Tukey, Geman–McClure); batched IK on fixed and floating-base robots;
trajectory optimisation with knot parameterisation and automatic banded/dense
routing (the Euclidean B-spline basis is numerical-only pending a separate
robot-manifold design); floating-base contact-force fitting; opt-in dense
first-order implicit differentiation for eligible generic named-block LM/GN
problems;
Featherstone dynamics (RNEA / ABA / CRBA / CCRBA), centroidal
momentum, and the autograd-derived `compute_rnea_derivatives`,
`compute_aba_derivatives`, and `compute_crba_derivatives` helpers; URDF and MJCF parsers; a programmatic
`ModelBuilder`; a viewer with skeleton / URDF-mesh render modes,
draggable IK target gizmos, and trajectory playback; and a CUDA-validated,
explicitly selected fused Warp FK lane whose VJP recomputes the Torch oracle.
Public solver drivers remain eager; no captured solver mode ships.

A small set of named symbols are deliberately stubbed and listed in
{doc}`reference/roadmap`. They have the correct signatures and raise
`NotImplementedError`.

## Status

The top-level API is deliberately compact. Named-block construction lives at
`better_robot.optim`; in particular, `SE3Manifold` is distinct from the
top-level `SE3` pose wrapper. Before 1.0 the contract test pins a required core
and validates `__all__`, but does not freeze an exact symbol count. See
{doc}`reference/changelog` for release notes.

## License

BetterRobot is licensed under Apache-2.0. See
{doc}`conventions/source_and_license` for the source-ledger and third-party
provenance rules that still apply to adapted work.

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
:caption: Guides

guides/index
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
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
