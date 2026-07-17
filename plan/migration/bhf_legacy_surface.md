# BHF Legacy Surface Migration Ledger

Status: M2c seed for M4, 2026-07-17.

This inventory is inherited from the pre-verified tables in
`plan/for_agents/m2c_first_order_phases_tasks.md` and
`plan/for_agents/m4_consumer_packs_and_migration.md`. It was **not re-verified**
against BetterHumanForce in this change because consumer repositories are
explicitly out of scope. M4 must re-grep the consumer at migration time and
expand or correct this ledger before deleting any corresponding BetterRobot
surface.

## Symbol map

| Inherited consumer symbol and site | BetterRobot replacement | M2c state | Later action |
|---|---|---|---|
| `better_robot.costs.stack.CostStack` — `scripts/motion/optimize_motion.py:210` | Named-block `optim.Problem` + `ResidualItem`s + the phase engine | `CostStack`/`CostItem` now live in `better_robot.optim.cost_stack`; `better_robot.costs.stack` is an identity-preserving forwarder still needed by BetterRobot's legacy problem/trajopt path | Re-verify the site in M4, migrate the objective to named blocks, then remove the forwarding package once no BetterRobot-local caller remains |
| `better_robot.optim.optimizers.gauss_newton.GaussNewton.minimize` — `tools/geometry/icp.py:58,330-332` | Named-block `better_robot.optim.GaussNewton` using `init_state` / `update` / `run` and solver configuration | Legacy class remains; the named-block solver is available | Re-verify and migrate the ICP loop before retiring the legacy optimizer stack |
| `better_robot.optim.kernels.cauchy.Cauchy` and `.huber.Huber` — `tools/object_align/sdf_fit.py:40-41` | Per-`ResidualItem` robust kernels using the same `rho` / `weight` contract | Kernels remain importable | Re-verify usage in M4; keep the kernel behavior, and decide whether deep legacy module paths still have an in-tree caller |
| `AccelerationResidual`, `ContactConsistencyResidual`, `ReferenceTrajectoryResidual`, and `ResidualState` — `scripts/motion/optimize_motion.py:211-216` | Block residuals declaring `reads`, evaluated through the M2a context/provider DAG | Legacy residual entry points remain | Re-verify and port the motion objective before deleting the legacy evaluation state |
| `better_robot.tasks.trajectory.Trajectory` — inherited sites in `motion.py`, `playback.py`, `smoothing.py`, and `view_motion.py` | `Trajectory` (kept public API) | Kept | Re-verify signatures only; this symbol is not scheduled for deletion |
| Deprecated `Data.oMi` — inherited sites in `playback.py` and `motion.py` | `Data.joint_pose_world` | Alias was already removed in M1; M2c does not resurrect it | Re-verify and migrate any remaining consumer reads before switching the consumer to the redesign branch |
| `better_robot.optim.problem.LeastSquaresProblem` and legacy optimizer consumers | Named-block `optim.Problem` plus `LevenbergMarquardt`, `GaussNewton`, or `Adam` lifecycle APIs | Remains for BetterRobot's legacy task/trajopt code | Delete only after all BetterRobot-local flat-problem callers are migrated; M4 must also account for inherited consumer use of the legacy optimizer protocol |

## Already-retired supporting surface

- `better_robot.costs.factory.factory` was removed in M1. The replacement is a
  small residual object or plain named-block residual callable. The M2c plan's
  instruction to delete it was already satisfied and this change does not
  recreate it.
- The deprecated `Data` field aliases, including `oMi`, were removed in M1.
  Their historical consumer sites above are migration obligations, not a
  reason to restore the aliases on the redesign branch.

## Ordering for M4

1. Re-grep the then-current consumer source and attach exact paths/lines to
   every row; this inherited seed is not deletion evidence.
2. Migrate and verify the consumer on its intended BetterRobot branch without
   changing the meaning of kernels, trajectory containers, or pose fields.
3. Remove a BetterRobot compatibility path only after both its BetterRobot-
   local callers and the corresponding migration-table row are resolved.
4. Run the full BetterRobot contracts and the committed consumer parity gates,
   then record every deletion in the M4 results report.

The 2026-07-17 branch decision means consumer compatibility does not by itself
block redesign-branch cleanup. In this M2c slice, however,
`better_robot.costs.stack` remains necessary for BetterRobot's own legacy
imports, so retaining that small forwarding module is not a consumer shim
promise.
