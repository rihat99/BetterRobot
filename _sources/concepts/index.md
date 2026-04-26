# Concepts

The chapters in this section read end-to-end. Together they explain
*why* BetterRobot is shaped the way it is — what each layer of the
architecture exists to do, what alternatives we considered, and what
goes wrong without that piece. If you want to call the API, the
{doc}`/getting_started/index` tutorials are the right entry point.
If you want the rules contributors must follow, the
{doc}`/conventions/index` chapters are the place. This is the third
quadrant: explanation.

```{toctree}
:maxdepth: 1

vision
architecture
model_and_data
joints_bodies_frames
lie_and_spatial
parsers_and_ir
kinematics
dynamics
residuals_and_costs
solver_stack
tasks
collision_and_geometry
batching_and_backends
viewer
```

## Reading order

The chapters work in order: each one builds on the previous and
points forward to the next. Read top to bottom for a complete tour.
You can also jump in at any layer — every chapter ends with pointers
to the chapters it depends on and the ones that depend on it.

| Chapter | What it covers |
|---------|----------------|
| {doc}`vision` | Why the library exists; the commitments that decided every other choice. |
| {doc}`architecture` | The layered DAG, the 26-symbol public API, the contract tests that enforce both. |
| {doc}`model_and_data` | Frozen `Model`, mutable `Data`, and the cache invariant that prevents stale Jacobians. |
| {doc}`joints_bodies_frames` | The universal joint taxonomy and the free-flyer convention that unifies fixed and floating base. |
| {doc}`lie_and_spatial` | SE(3) / SO(3) ops, the spatial-algebra value types, why we do not subclass `torch.Tensor`. |
| {doc}`parsers_and_ir` | URDF / MJCF / programmatic builder all converging on a single intermediate representation. |
| {doc}`kinematics` | Forward kinematics and the unified Jacobian dispatch. |
| {doc}`dynamics` | RNEA, ABA, CRBA, the centroidal map, and the action-model framework. |
| {doc}`residuals_and_costs` | The residual library plus `CostStack`. |
| {doc}`solver_stack` | `LeastSquaresProblem` and the four pluggable axes (Optimizer, LinearSolver, RobustKernel, DampingStrategy). |
| {doc}`tasks` | `solve_ik`, `solve_trajopt`, `Trajectory`. |
| {doc}`collision_and_geometry` | Geometry primitives, pair dispatch, `RobotCollision`. |
| {doc}`batching_and_backends` | Tensor and device conventions; the `Backend` Protocol. |
| {doc}`viewer` | The viser-backed visualisation layer. |
