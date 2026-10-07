# A2 inventory — mutable `Data` API -> state API

`state = forward_kinematics_state(model, q)` (frozen `KinematicState`: `q`, `joint_transform (B..., nj, 3, 4)`,
`frame_transform (B..., nf, 3, 4)`; properties `joint_pose`/`frame_pose (…, 7)`, `joint_/frame_translation`,
`joint_/frame_rotation`, `frame_pose_of(id)`). NEW = added in this stage (verified against Pinocchio, see a2_state.md).

| Removed (Data API) | Replacement |
|---|---|
| `Model.create_data()`, `data_model/data.py::Data`, `_kinematics_level.py` (`KinematicsLevel`, `data.require`) | `forward_kinematics_state(model, q)` -> `KinematicState` |
| `kinematics.forward_kinematics(model, q, compute_frames=)` | `forward_kinematics_state(model, q)` (frames always computed) |
| `update_frame_placements(model, data)` | none needed (`state.frame_transform`) |
| `data.joint_pose_world` / `data.frame_pose_world` | `state.joint_pose` / `state.frame_pose` |
| `compute_joint_jacobians` + `get_joint_jacobian(model, data, j, reference="world")` | NEW `joint_jacobian(model, state, j, *, reference="local_world_aligned")` — default differs: pass `reference="world"` where old code relied on the default |
| `get_frame_jacobian(model, data, f, reference=)` | `frame_jacobian(model, state, f, *, reference=)` |
| `compute_joint_jacobians_time_variation` + `get_joint_jacobian_time_variation` (needed `data.v`) | NEW `joint_jacobian_time_variation(model, state, v, j, *, reference=)` |
| `get_frame_jacobian_time_variation` (needed `data.v`) | NEW `frame_jacobian_time_variation(model, state, v, f, *, reference=)` |
| `dynamics.rnea(model, data, q, v, a, fext=)` | `dynamics.state.inverse_dynamics(model, state, v, a, fext=)` |
| `dynamics.bias_forces(...)` | `inverse_dynamics(model, state, v, zeros_like(v))` |
| gravity torque | `inverse_dynamics(model, state, zeros, zeros)` |
| `dynamics.crba(model, data, q)` | `mass_matrix(model, state)` |
| `dynamics.aba(model, data, q, v, tau, fext=)` | `forward_dynamics(model, state, v, tau, fext=)` |
| `ccrba`, `compute_centroidal_map` | `centroidal_map(model, state)` |
| `compute_centroidal_momentum(model, q, v)` | NEW `centroidal_momentum(model, state, v)` = `A_g v` |
| `dynamics.center_of_mass(model, q, ..., data=)` (+ `data.com_velocity`) | `dynamics.state.center_of_mass(model, state)`; COM velocity = linear rows of `centroidal_momentum` / total mass (check row order of `centroidal_map` first) |
| `exceptions.StaleCacheError` | raised only by the Data getters in `kinematics/jacobian.py:421` -> remove with them |
| Data-cache fields (`data.joint_forces`, `joint_velocity_local`, `joint_acceleration_local`, `tau`, `com_velocity`) | none; tests asserting only their population are Data-cache tests |
| `JointMimic` placeholder | `io/build_model.py:395` does `isinstance(joint_model, JointMimic)` — check whether anything instantiates it before trashing |

Kernels the state API calls (keep): `kinematics/_fk_lean.py`, `_jacobian_columns.py`, `forward.py::_validate_q`
(move into `state.py` when trimming `forward.py`), `jacobian.py::{joint_jacobians_time_variation_raw,
frame_jacobian_time_variation_raw, _time_variation_in_reference}` (+ their deps), `dynamics/_world.py`
(`inverse_dynamics_full`, `mass_matrix_full`, `centroidal_map_full`, `get_dynamics_plan`),
`data_model/reduced_coordinates.py`. Trash candidates once no test/BH uses them: `forward_kinematics_raw`,
`frame_placements_raw`, `joint_jacobians_raw`, `frame_jacobian_raw` (users today: kinematics/__init__,
_fk_matrix.py, _jacobian_columns.py, jacobian.py, forward.py), `ccrba_raw`, `dynamics/{rnea,crba,aba}.py` Data entry
points, `dynamics/_execution.py::prepare_dynamics_inputs` (check users).

Root `__init__` today (additive step done): Data exports still present; state exports added
(`forward_kinematics_state, KinematicState, frame_jacobian, frame_jacobian_time_variation, joint_jacobian,
joint_jacobian_time_variation, inverse_dynamics, mass_matrix, forward_dynamics, centroidal_map, centroidal_momentum`).
Root `center_of_mass` still points at the Data-era `dynamics.center_of_mass`; swap to `dynamics.state.center_of_mass`
in the removal step.
