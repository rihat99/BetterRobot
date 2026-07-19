# Concepts

These chapters explain why BetterRobot is shaped the way it is. They start
with robot models and motion, then build toward dynamics, optimization, and
the optional GPU seam. Tutorials show what to type; this section explains
what the operations mean and why the alternatives were rejected.

```{toctree}
:maxdepth: 1

why_betterrobot
design_decisions
architecture
model_and_data
joints_bodies_frames
lie_and_spatial
kinematics_and_jacobians
dynamics
residuals_costs_and_solvers
the_compute_seam
parsers_and_ir
viewer
```

## A useful reading path

| Chapter | Question it answers |
|---|---|
| {doc}`why_betterrobot` | What kind of robotics work is this library for? |
| {doc}`design_decisions` | Which alternatives were considered, and what did each choice cost? |
| {doc}`architecture` | Which package owns each responsibility? |
| {doc}`model_and_data` | Why are robot identity and per-call results separate? |
| {doc}`joints_bodies_frames` | How do joints, rigid bodies, and named frames fit together? |
| {doc}`lie_and_spatial` | How are rotations, poses, twists, forces, and inertias represented? |
| {doc}`kinematics_and_jacobians` | How do joint coordinates determine poses and local motion? |
| {doc}`dynamics` | How do motion, force, inertia, and acceleration relate? |
| {doc}`residuals_costs_and_solvers` | How does a desired outcome become a least-squares problem? |
| {doc}`the_compute_seam` | How do batching, compilation, and the opt-in Warp pass coexist? |
| {doc}`parsers_and_ir` | How do files and builders become the same model? |
| {doc}`viewer` | How does visualization stay separate from robot computation? |

Jump to any chapter if you already know its prerequisites. Terms used by the
tutorials are also defined in the {doc}`/reference/glossary`.
