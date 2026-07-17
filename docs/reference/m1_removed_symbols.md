# M1 removed surfaces

M1 removes pre-1.0 surfaces that had no BetterRobot-internal callers. This is
the migration ledger for downstream repositories.

| Removed surface | Migration |
|---|---|
| `Data.oMi`, `oMf`, `liMi` | Use `joint_pose_world`, `frame_pose_world`, `joint_pose_local`. |
| `Data.ov`, `oa`, `v_joint`, `a_joint` | Use the corresponding `joint_velocity_*` / `joint_acceleration_*` fields. |
| `Data.M`, `C`, `g`, `nle` | Use `mass_matrix`, `coriolis_matrix`, `gravity_torque`, `bias_forces`. |
| `Data.J`, `dJ`, `Ag`, `hg`, `com`, `vcom`, `acom` | Use `joint_jacobians`, `joint_jacobians_dot`, `centroidal_momentum_matrix`, `centroidal_momentum`, `com_position`, `com_velocity`, `com_acceleration`. |
| `register_residual`, `get_residual`, `registered_residuals` and `better_robot.residuals.registry` | Construct a `Residual` explicitly, give it a stable `name`, and add it to `CostStack`. |
| `retarget`, `RetargetCostConfig`, `better_robot.tasks.retarget` | No replacement ships yet; build the task explicitly from trajectory residuals and `solve_trajopt`. |
| `IR_SCHEMA_VERSION`, `IRModel.schema_version`, `IRSchemaVersionError` | Treat IR as an in-process boundary and re-parse source assets after upgrades. |
| `better_robot.dynamics.action.*` | No replacement ships yet; executable optimal-control contracts belong to a later milestone. |
| `better_robot.costs.factory.factory` | Implement a small `Residual` object explicitly. |
| `better_robot.kinematics.chain`, `better_robot.data_model.indexing`, `better_robot.utils.*` | These unimplemented or unused helpers have no direct replacement. |

Workspace audit found four live `Data.oMi` accesses in BetterHumanForce:
`tools/robot_motion/motion.py:201` and
`tools/robot_motion/playback.py:115,180,190`. BetterHumanForce must migrate
those call sites before it follows the redesign branch. BetterVideoReconstruction
had no matching BetterRobot API references. `CostStack` remains because
BetterRobot itself uses it; BetterHumanForce also imports
`better_robot.costs.stack.CostStack` in
`scripts/motion/optimize_motion.py:210`. Since M2c, `CostStack` is canonically
defined in `better_robot.optim.cost_stack`; the old module forwards to the same
class identity while BetterRobot's legacy flat-problem callers remain.
`ResidualSpec` was subsequently removed in M3.5; see
`m3_removed_symbols.md` for its migration note.
