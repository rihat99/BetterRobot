# M3.5 removed surfaces

M3.5 removes unused pre-1.0 surfaces after verifying that they have no
BetterRobot-internal callers. This ledger records the migration path for
downstream repositories.

| Removed surface | Migration |
|---|---|
| `better_robot.kinematics._warp_bridge.require_warp` | No replacement is needed. Importing the private Warp bridge initializes Warp before its active forward entry point is used. |
| `better_robot.optim.solve` | For a legacy `LeastSquaresProblem`, instantiate an optimizer from `better_robot.optim.optimizers` and call `minimize`. For a named-block `Problem`, call `better_robot.optim.LevenbergMarquardt().run(values, problem)`, `GaussNewton().run(...)`, or the relevant phase runner. |
| `better_robot.optim.ResidualSpec`, `better_robot.optim.jacobian_spec`, and the built-in residual `.spec` producers | No runtime replacement is needed: no shipped solver consumed this metadata. The temporal and sparse semantics are retained in `plan/design_notes/residual_sparsity.md` for the M5 structured declaration. |
| `better_robot.optim.blocks.LMState.iter_num` | Read `LMState.iterations`. |
| `better_robot.optim.blocks.Problem.normal_matrix` | Form `J.mT @ J` from `Problem.dense_jacobian(values)` when a dense normal matrix is needed. |
| `better_robot.tasks.IKResult.q_only` | Read `IKResult.q` directly. |
