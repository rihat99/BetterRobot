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
| `better_robot.optim.SolverState`, `better_robot.optim.state.SolverState` | Use the concrete result returned by the selected algorithm, such as `LMState` or `FirstOrderResult`. |
| `better_robot.optim.state.SolverStatus` | Use `LMStatus` for LM/GN; first-order results expose a boolean `converged` tensor. |
| `better_robot.optim.Optimizer`, `better_robot.optim.optimizers.Optimizer`, `better_robot.optim.optimizers.base.Optimizer` | No shared optimizer protocol replaces it; call a concrete named-block solver's `run(values, problem)` method. |
| `better_robot.optim.OptimizationResult`, `better_robot.optim.optimizers.OptimizationResult`, `better_robot.optim.optimizers.base.OptimizationResult` | Use the concrete result returned by the selected algorithm, such as `LMState` or `FirstOrderResult`. |
| `better_robot.optim.optimizers.LevenbergMarquardt`, `better_robot.optim.optimizers.levenberg_marquardt.LevenbergMarquardt` | `better_robot.optim.LevenbergMarquardt` (`init_state` / `update` / `run`). |
| `better_robot.optim.optimizers.GaussNewton`, `better_robot.optim.optimizers.gauss_newton.GaussNewton` | `better_robot.optim.GaussNewton` (`init_state` / `update` / `run`). Known downstream caller: BHF `tools/geometry/icp.py:58,330-332`. |
| `better_robot.optim.optimizers.Adam`, `better_robot.optim.optimizers.adam.Adam` | `run_first_order(..., optimizer_factory=lambda params: torch.optim.Adam(params, ...))`. |
| `better_robot.optim.optimizers.LBFGS`, `better_robot.optim.optimizers.lbfgs.LBFGS` | `torch.optim.LBFGS` with an explicit objective closure. |
| `better_robot.optim.optimizers.LMThenLBFGS`, `better_robot.optim.optimizers.lm_then_lbfgs.LMThenLBFGS` | Call LM and a `torch.optim` stage sequentially, rebuilding the problem when stage weights differ. |
| `better_robot.optim.optimizers.MultiStageOptimizer`, `better_robot.optim.optimizers.multi_stage.MultiStageOptimizer` | Call the selected solver functions sequentially in task code. |
| `better_robot.optim.optimizers.OptimizerStage`, `better_robot.optim.optimizers.multi_stage.OptimizerStage` | No stage record replaces it; ordinary sequential calls make staging explicit. |
| `better_robot.optim.strategies.base.DampingStrategy` | No pluggable damping protocol replaces it; configure `LevenbergMarquardt` directly. |
| `better_robot.optim.strategies.Constant`, `better_robot.optim.strategies.constant.Constant` | `LevenbergMarquardt(fixed_damping=True, damping_parameter=...)`. |
| `better_robot.optim.strategies.Adaptive`, `better_robot.optim.strategies.adaptive.Adaptive` | `LevenbergMarquardt(damping_parameter=...)` with its default adaptive damping. |
| `better_robot.optim.blocks` and `better_robot.optim.blocks.*` import paths | Import the same surviving records and solvers directly from `better_robot.optim`; implementation modules are flat under `better_robot.optim.*`. |
| `better_robot.optim.kernels.{base,cauchy,geman_mcclure,huber,l2,tukey}` import paths | Import `RobustKernel`, `Cauchy`, `GemanMcClure`, `Huber`, `L2`, or `Tukey` directly from `better_robot.optim`. |
| `better_robot.optim.solvers.{base,banded_cholesky,cholesky}` import paths | Import the surviving solver protocols, result types, `BandedCholesky`, or `Cholesky` directly from `better_robot.optim`. |
| `better_robot.optim.LSTSQ`, `better_robot.optim.solvers.LSTSQ`, `better_robot.optim.solvers.lstsq.LSTSQ`, and `OptimizerConfig(linear_solver="lstsq")` | Use the default SPD `Cholesky` route; call `torch.linalg.lstsq` explicitly when a separate least-squares fallback is genuinely required. |
| Rank-deficient fallback in `Cholesky.solve` | Inputs must be SPD; use `Cholesky.solve_with_info` for per-element failure status or call `torch.linalg.lstsq` explicitly. |
| `ObjectiveItem`, `ObjectiveTerm`, and `Problem.require_least_squares` | Express every term as a residual and add it with `Problem.add_residual`. |
| `ResidualState`, `residual_jacobian`, and residual `.jacobian()` / transpose-apply hooks | Residuals read a named evaluation context; provide `jacobian_blocks` only when an analytic block is worthwhile. |
| `better_robot.JacobianStrategy` and `better_robot.kinematics.JacobianStrategy` | Pass a `better_robot.optim.JacobianStrategy` string: `"auto"`, `"analytic"`, `"jacrev"`, `"jacfwd"`, or `"finite_difference"`. |
| Provider `inputs` declarations | Rename the declaration to `reads`; outputs and lazy evaluation remain unchanged. |
| `better_robot.optim.Adam`, `AdamState`, and `AdamStatus` | Use `run_first_order` with any `torch.optim.Optimizer`; inspect `FirstOrderResult`. |
| `better_robot.optim.Phase`, `PhaseResult`, and `run_phases` | Call algorithms sequentially and rebuild a problem explicitly when stage weights change. |
| `NormalCG`, `NormalOperator`, `linearization="matrix_free"`, and `LinearSystemKind` | Use automatic dense/block-banded routing, or force `linearization="dense"` / `"structured"`. |
| `better_robot.optim.structure` | Import `BlockBandedMatrix`, `TemporalAnalysis`, and `LinearizationReason` from `better_robot.optim`; route records live with LM. |
| `LMState.previous_linear_step`, `projected_gradient`, and `linear_solve_*` diagnostics | Inspect convergence, cost, gradients, `factorization_ok`, and status; iterative-solver warm-start diagnostics disappeared with `NormalCG`. |
| `better_robot.dynamics.compute_minverse`, `better_robot.dynamics.crba.compute_minverse` | Call `torch.linalg.inv(crba(...))` when an explicit inverse is acceptable; no direct ABA-factorization inverse ships. |
| `better_robot.dynamics.compute_coriolis_matrix`, `better_robot.dynamics.rnea.compute_coriolis_matrix` | Use `bias_forces` when the needed quantity is `C(q, v) v + g(q)`; no standalone Coriolis-matrix pass ships. |
| `better_robot.dynamics.compute_centroidal_dynamics_derivatives`, `better_robot.dynamics.derivatives.compute_centroidal_dynamics_derivatives` | Differentiate through `compute_centroidal_map` or `compute_centroidal_momentum`; no named analytic helper ships. |
| `better_robot.dynamics.semi_implicit_euler`, `symplectic_euler`, `rk4` and their `better_robot.dynamics.integrators` paths | `integrate_q` retracts configurations; full physics integration belongs to the caller's simulation layer. |
| `better_robot.dynamics.nle`, `better_robot.dynamics.rnea.nle` | `better_robot.dynamics.bias_forces`. |
| `better_robot.residuals.YoshikawaResidual`, `better_robot.residuals.manipulability.YoshikawaResidual` | No replacement ships; implement an explicit residual when its conditioning contract is defined. |
| `better_robot.residuals.SelfCollisionResidual`, `WorldCollisionResidual` and their `better_robot.residuals.collision` paths | No replacement ships; collision residuals wait on the owner-gated collision package decision. |
| `better_robot.residuals.JointAccelLimit`, `better_robot.residuals.limits.JointAccelLimit` | No replacement ships; the model currently has no acceleration-limit values. |

## To be removed by the polish phases (agents append exact rows as they land)

The polish roadmap (`plan/03_roadmap.md`) deletes the legacy optimization
stack outright. Known consumer call sites recorded before the repository
boundary was closed (paths are in the consumer repos, unverified since
2026-07-17):

| Legacy surface being deleted | Known consumer site | Replacement |
|---|---|---|
| `ResidualState` legacy evaluation protocol | BHF `scripts/motion/optimize_motion.py:211-216` | Residuals evaluated through a `Problem` context. |

`Trajectory` stays public and is not scheduled for deletion.
