# BetterRobot

BetterRobot is a PyTorch library for batched, differentiable robot kinematics, dynamics and optimisation.

![Sixteen Panda arms, each at its own inverse-kinematics answer, all solved in one batched call](docs/images/readme_ik.gif)

> **Status:** this is an early version (0.3.0) in active development, and breaking changes may happen. Development happens on the `dev` branch.

## Install

```bash
git clone https://github.com/rihat99/BetterRobot && cd BetterRobot
uv sync --all-extras   # Python 3.12+, PyTorch, and the viewer, MJCF, mesh and demo extras
```

## Quickstart

```python
import torch
from robot_descriptions import panda_description
import better_robot as br

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
generator = torch.Generator().manual_seed(0)
q0 = model.q_neutral.expand(64, -1)                             # (64, nq): a batch of 64 configurations
q = br.terms.random_start(model, q0, generator=generator)       # 64 random configurations within the limits

state = br.forward_kinematics(model, q)                         # every joint and frame pose, for the whole batch
hand = state.frame_pose_of("panda_hand")                        # (64, 7): [tx, ty, tz, qx, qy, qz, qw]
jacobian = br.frame_jacobian(model, state, frame="panda_hand")  # (64, 6, nv), rows [linear, angular]
print(hand.shape, jacobian.shape)

result = br.solve_ik(model, {"panda_hand": hand}, q0=q0, seeds=8, generator=generator)  # 64 IK problems, one call
reached = br.forward_kinematics(model, result.x["q"]).frame_pose_of("panda_hand")
print(f"median hand error after IK: {(reached[:, :3] - hand[:, :3]).norm(dim=-1).median():.1e} m")
```
```output
torch.Size([64, 7]) torch.Size([64, 6, 8])
median hand error after IK: 5.0e-13 m
```

Every function keeps the leading batch axes of its inputs, and gradients flow through every call. Show the 64
answers in the browser viewer as a clip, one answer per frame:

```python
from better_robot.viewer import RobotVisual

viewer = br.Viewer()  # a local viser server: open viewer.url in a browser
geometry = br.io.load_geometry(panda_description.URDF_PATH, model)
viewer.add(RobotVisual(model, result.x["q"], geometry=geometry, root=panda_description.URDF_PATH))
viewer.serve()  # plays the clip until Ctrl-C
```

## What is inside

| Package | What it does |
|---|---|
| `io` | Reads URDF and MJCF files into a `Model`; `ModelBuilder` builds one in code. |
| `model` | The robot as one frozen record of tensors: joints, bodies, frames and geometry. |
| `kinematics` | Forward kinematics, frame and joint Jacobians, `integrate` and `difference`, and `compiled`, the opt-in `torch.compile` path. |
| `dynamics` | Inverse and forward dynamics, the mass matrix, the center of mass and centroidal momentum. |
| `lie`, `spatial` | SE(3) and SO(3) operations; twists, wrenches and inertias. Plain torch functions. |
| `optim` | Least squares over named tensors: Gauss-Newton, Levenberg-Marquardt, sequential QP, L-BFGS. It knows nothing about robots. |
| `terms` | Robot costs for `optim` (pose targets, joint limits, clearance, manipulability) and `solve_ik`. |
| `collision` | Signed distances between spheres, capsules, boxes, planes and voxel grids; a robot as spheres or capsules. |
| `viewer` | A browser viewer on [viser](https://viser.studio): robots, clips, points and meshes on one timeline, and video recording. |

`exceptions` holds the error classes, and `testing` the checks that BetterRobot and the projects built on it share.

## How it compares

Five other libraries cover parts of the same ground. The table was checked against each project's documentation
and code in November 2026; a dash means we did not find the answer there. Corrections are welcome.

| | BetterRobot | [Pinocchio](https://github.com/stack-of-tasks/pinocchio) | [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) | [frax](https://github.com/StanfordASL/frax) | [PyRoki](https://github.com/chungmin99/pyroki) | [cuRobo](https://github.com/NVlabs/curobo) |
|---|---|---|---|---|---|---|
| Focus | batched kinematics, dynamics, least squares | rigid-body algorithms, their derivatives, contact solvers | physics simulation of many scenes in parallel | kinematics and dynamics for controllers | kinematic optimisation | motion generation |
| Built on | PyTorch | C++, with Python bindings | JAX or Warp | JAX | JAX | PyTorch, CUDA, Warp |
| Runs on | CPU, CUDA GPU | CPU | CPU, GPU, TPU | CPU, GPU, TPU | CPU, GPU, TPU | NVIDIA GPU |
| Derivatives | autograd | analytical; CppAD, CasADi | JAX (MJX-JAX only) | JAX | JAX or analytical | differentiable kinematics |
| Robot files | URDF, MJCF | URDF, SDF, MJCF, SRDF | MJCF, URDF | URDF | URDF | URDF |
| Floating base | yes | yes | yes | yes | base pose as a variable | virtual joints |
| Inverse and forward dynamics, mass matrix | yes | yes, with analytical derivatives | yes | yes | no | - |
| Inverse kinematics | yes, batched with multiple starts | through Pink | - | yes | yes | yes |
| Trajectory optimisation | yes | - | - | - | yes | yes, with collision-free planning |
| Collision shapes | robot spheres or capsules; boxes, planes, voxel grids | through Coal | planes, height fields, spheres, capsules, boxes, meshes | spheres | spheres, capsules, planes, height maps | robot spheres; cuboids, capsules, cylinders, meshes, voxel grids |
| Viewer | browser (viser) | Meshcat, Viser, Gepetto Viewer, Panda3D, RViz | MuJoCo's desktop and web viewers | in the examples only | viser | viser |
| License | Apache-2.0 | BSD-2-Clause | Apache-2.0 | MIT | MIT | Apache-2.0 |

Speed, in microseconds per configuration, float64, on one core of an AMD EPYC 9654 with PyTorch 2.13 (measured
2026-11-20). The Panda here has its fingers fixed (7 joints); the Unitree G1 has a floating base (35 velocities).

| | Pinocchio, one call | BetterRobot, batch 1, eager | BetterRobot, batch 4096, compiled |
|---|--:|--:|--:|
| **Panda** | | | |
| Forward kinematics | 0.39 | 110 | 0.25 |
| Frame Jacobian | 0.76 | 200 | 0.84 |
| Inverse dynamics | 1.4 | 440 | 1.8 |
| Mass matrix | 0.92 | 380 | 5.4 |
| Forward dynamics | 4.2 | 690 | 7.1 |
| **Unitree G1** | | | |
| Forward kinematics | 1.3 | 190 | 2.4 |
| Frame Jacobian | 1.2 | 280 | 4.2 |
| Inverse dynamics | 5.0 | 530 | 10 |
| Mass matrix | 4.5 | 480 | 26 |
| Forward dynamics | 18 | 820 | 40 |

For one configuration at a time, Pinocchio is the faster choice: 45 to 420 times faster here, because every PyTorch
call carries a fixed overhead. BetterRobot is built for batches. With 4096 configurations in one compiled call
(`kinematics.compiled`), its cost per configuration on one core runs from 0.6 times Pinocchio's (Panda forward
kinematics) to 6 times (mass matrix). Compiling takes 2 to 16 seconds per function and batch size. The table has no
GPU column; the same script measures one with `--device cuda`. The command, run on one thread pinned with `taskset`
to an idle core:

```bash
uv sync --all-extras --all-groups
uv run python benchmarks/speed.py run --device cpu --dtype float64 --robots panda7,g1 --batches 1,4096 \
  --functions forward_kinematics,frame_jacobian,inverse_dynamics,mass_matrix,forward_dynamics --modes eager,compile
```

## How it is tested

Kinematics, Jacobians and dynamics are compared with Pinocchio and MuJoCo on the same robots, each quantity under
its own tolerance (`tests/parity`). The README's Python examples, and the docs' runnable ones, are tests: each must
print what the page shows. Three tiers run the suite:

```bash
uv sync --all-extras --all-groups                            # adds Pinocchio and the development tools
uv run pytest tests/contract -q -m "not slow and not cuda"   # fast tier, seconds
uv run pytest tests -q -m "not cuda"                         # full tier, minutes on a CPU
bash tools/release_tier.sh --gpu 0                           # release tier: every check, plus the GPU tests
```

## Documentation and related projects

- The documentation starts at [docs/index.md](docs/index.md); `uv run mkdocs serve` serves it as a site.
- [examples/](examples/) holds short programs, from batched IK to dynamics and the viewer.
- [BetterHuman](https://github.com/rihat99/BetterHuman) builds SMPL-family human body models on BetterRobot.

## License

Apache License 2.0 ([LICENSE](LICENSE)). Copyright 2026 BetterRobot contributors.
