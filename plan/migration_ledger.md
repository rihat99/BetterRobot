# Migration ledger — removed public surfaces

For the owner's use when migrating downstream projects (BetterHumanForce,
BetterVideoReconstruction) onto the current branch. This file is not shipped
documentation — it moved here from `docs/reference/m*_removed_symbols.md`
because a released library's docs should not carry internal migration tables.

Downstream repositories are out of scope for agents working in this
repository. Agents only **append** rows here when a work order removes a
public symbol; the owner resolves the consumer side.

## Removed during the redesign (already gone on this branch)

| Removed surface | Replacement |
|---|---|
| `Data.oMi`, `oMf`, `liMi` | `joint_pose_world`, `frame_pose_world`, `joint_pose_local`. |
| `Data.ov`, `oa`, `v_joint`, `a_joint` | The `joint_velocity_*` / `joint_acceleration_*` fields. |
| `Data.M`, `C`, `g`, `nle` | `mass_matrix`, `coriolis_matrix`, `gravity_torque`, `bias_forces`. |
| `Data.J`, `dJ`, `Ag`, `hg`, `com`, `vcom`, `acom` | `joint_jacobians`, `joint_jacobians_dot`, `centroidal_momentum_matrix`, `centroidal_momentum`, `com_position`, `com_velocity`, `com_acceleration`. |
| `register_residual`, `get_residual`, `registered_residuals`, `better_robot.residuals.registry` | Construct a residual explicitly and add it to a problem. |
| `retarget`, `RetargetCostConfig`, `better_robot.tasks.retarget` | No replacement ships; build the task from trajectory residuals and `solve_trajopt`. |
| `IR_SCHEMA_VERSION`, `IRModel.schema_version`, `IRSchemaVersionError` | IR is an in-process boundary; re-parse source assets after upgrades. |
| `better_robot.dynamics.action.*` | No replacement ships; optimal-control contracts are future work. |
| `better_robot.costs.factory.factory` | Implement a small residual object explicitly. |
| `better_robot.kinematics.chain`, `better_robot.data_model.indexing`, `better_robot.utils.*` | Unimplemented/unused helpers; no replacement. |
| `better_robot.kinematics._warp_bridge.require_warp` | Not needed; the active Warp entry point initializes Warp itself. |
| `better_robot.optim.solve` | Use a solver object directly (`LevenbergMarquardt().run(values, problem)`). |
| `better_robot.optim.ResidualSpec`, `better_robot.optim.jacobian_spec`, residual `.spec` producers | No runtime replacement; no shipped solver consumed this metadata. |
| `better_robot.optim.blocks.LMState.iter_num` | `LMState.iterations`. |
| `better_robot.optim.blocks.Problem.normal_matrix` | Form `J.mT @ J` from `Problem.dense_jacobian(values)`. |
| `better_robot.tasks.IKResult.q_only` | `IKResult.q` directly. |
| `better_robot.CostStack`, `better_robot.optim.CostStack`, `better_robot.optim.cost_stack.CostStack`, `better_robot.costs.CostStack`, `better_robot.costs.stack.CostStack` | Build `Problem` from `ResidualItem` objects; pass residual items directly to `solve_trajopt`. Known downstream caller: BHF `scripts/motion/optimize_motion.py:210`. |
| `better_robot.optim.CostItem`, `better_robot.optim.cost_stack.CostItem`, `better_robot.costs.CostItem`, `better_robot.costs.stack.CostItem` | `better_robot.optim.ResidualItem`. |
| `better_robot.optim.CostKind`, `better_robot.optim.cost_stack.CostKind`, `better_robot.costs.stack.CostKind` | No replacement; pass only the residual items that should participate in a solve. |
| `better_robot.LeastSquaresProblem`, `better_robot.optim.LeastSquaresProblem`, `better_robot.optim.problem.LeastSquaresProblem` | `better_robot.optim.Problem` with named variables, residual items, and providers. Recorded downstream use was protocol-level. |
| `better_robot.optim.SolverState`, `better_robot.optim.state.SolverState` | Use the concrete state returned by the selected solver, such as `LMState` or `AdamState`. |
| `better_robot.optim.state.SolverStatus` | Use the selected solver's status enum, such as `LMStatus` or `AdamStatus`. |
| `better_robot.optim.Optimizer`, `better_robot.optim.optimizers.Optimizer`, `better_robot.optim.optimizers.base.Optimizer` | No shared optimizer protocol replaces it; call a concrete named-block solver's `run(values, problem)` method. |
| `better_robot.optim.OptimizationResult`, `better_robot.optim.optimizers.OptimizationResult`, `better_robot.optim.optimizers.base.OptimizationResult` | Use the concrete solver state returned by `run`, such as `LMState` or `AdamState`. |
| `better_robot.optim.optimizers.LevenbergMarquardt`, `better_robot.optim.optimizers.levenberg_marquardt.LevenbergMarquardt` | `better_robot.optim.LevenbergMarquardt` (`init_state` / `update` / `run`). |
| `better_robot.optim.optimizers.GaussNewton`, `better_robot.optim.optimizers.gauss_newton.GaussNewton` | `better_robot.optim.GaussNewton` (`init_state` / `update` / `run`). Known downstream caller: BHF `tools/geometry/icp.py:58,330-332`. |
| `better_robot.optim.optimizers.Adam`, `better_robot.optim.optimizers.adam.Adam` | `better_robot.optim.Adam` (`init_state` / `update` / `run`). |
| `better_robot.optim.optimizers.LBFGS`, `better_robot.optim.optimizers.lbfgs.LBFGS` | `torch.optim.LBFGS` with an explicit objective closure. |
| `better_robot.optim.optimizers.LMThenLBFGS`, `better_robot.optim.optimizers.lm_then_lbfgs.LMThenLBFGS` | Use `run_phases` for staged named-block solves; invoke `torch.optim.LBFGS` explicitly when a separate first-order stage is required. |
| `better_robot.optim.optimizers.MultiStageOptimizer`, `better_robot.optim.optimizers.multi_stage.MultiStageOptimizer` | `better_robot.optim.run_phases`. |
| `better_robot.optim.optimizers.OptimizerStage`, `better_robot.optim.optimizers.multi_stage.OptimizerStage` | `better_robot.optim.Phase`. |
| `better_robot.optim.strategies.base.DampingStrategy` | No pluggable damping protocol replaces it; configure `LevenbergMarquardt` directly. |
| `better_robot.optim.strategies.Constant`, `better_robot.optim.strategies.constant.Constant` | `LevenbergMarquardt(fixed_damping=True, damping_parameter=...)`. |
| `better_robot.optim.strategies.Adaptive`, `better_robot.optim.strategies.adaptive.Adaptive` | `LevenbergMarquardt(damping_parameter=...)` with its default adaptive damping. |

## To be removed by the polish phases (agents append exact rows as they land)

The polish roadmap (`plan/03_roadmap.md`) deletes the legacy optimization
stack outright. Known consumer call sites recorded before the repository
boundary was closed (paths are in the consumer repos, unverified since
2026-07-17):

| Legacy surface being deleted | Known consumer site | Replacement |
|---|---|---|
| `optim.kernels.cauchy.Cauchy`, `.huber.Huber` deep paths | BHF `tools/object_align/sdf_fit.py:40-41` | Same kernels at their public import path. |
| `ResidualState` legacy evaluation protocol | BHF `scripts/motion/optimize_motion.py:211-216` | Residuals evaluated through a `Problem` context. |

`Trajectory` stays public and is not scheduled for deletion.
