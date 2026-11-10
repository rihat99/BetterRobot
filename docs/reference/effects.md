# Effects table

One row for each public function of `kinematics`, `dynamics`, `lie` and `spatial` that the contract tests iterate
(`tests/contract/_public_ops.py`; the few without a tensor to test, such as `subtree` or `se3.identity`, are listed
there with their reason), one column for each contract test that runs over them. `yes` means the test passed for that entry point in the run that
rendered this page. A blank cell means no test covers the pair: `another dtype raises` covers the robot functions
that take a plain tensor besides the model and the state, and `another model's state raises` covers the functions
that take a `KinematicState`. `tests/docs/test_effects_table.py` reruns the tests and fails while this page differs
from a fresh render; `uv run python -m tests.docs.test_effects_table` rewrites the table.

| Column | What the test checks | Test |
|---|---|---|
| inputs untouched | the version counter of every tensor reachable from the arguments is unchanged after the call (C-STATE-2, C-STATE-3) | `tests/contract/test_no_inplace.py` |
| dtype and device kept | float32 and float64 inputs on CPU give outputs in the same dtype on the same device (C-TENS-1) | `tests/contract/test_dtype_device.py` |
| batch axes per element | inputs with leading axes `(2,)` and `(3, 1)` give results with those axes whose every element equals the call on that element alone (C-TENS-2) | `tests/contract/test_dtype_device.py` |
| another dtype raises | a tensor argument in another dtype than the model's raises `DtypeMismatchError`, never a silent promotion | `tests/contract/test_dtype_device.py` |
| inputs validated once | the public call checks each floating tensor input once, under its name, at its boundary (C-ERR-1) | `tests/contract/test_boundary_validation_count.py` |
| another model's state raises | a `KinematicState` of another model raises `ModelError` or the mismatch error (C-STATE-6) | `tests/contract/test_state_attachment.py` |
| same operators at every batch size | a batch of two and a batch of seven dispatch the same number of torch operators, so no Python loop runs per batch element (C-PERF-2) | `tests/contract/test_performance_rules.py` |

No test over these entry points checks autograd, `torch.func.vmap`, `torch.compile(fullgraph=True)`, host
synchronisation or CUDA yet, so the table has no such columns.

<!-- effects table: rendered by tests/docs/test_effects_table.py -->
| Entry point | inputs untouched | dtype and device kept | batch axes per element | another dtype raises | inputs validated once | another model's state raises | same operators at every batch size |
|---|---|---|---|---|---|---|---|
| `forward_kinematics` | yes | yes | yes | yes | yes |  | yes |
| `frame_jacobian` | yes | yes | yes |  | yes | yes | yes |
| `joint_jacobian` | yes | yes | yes |  | yes | yes | yes |
| `point_jacobian` | yes | yes | yes | yes | yes | yes | yes |
| `frame_jacobian_time_variation` | yes | yes | yes | yes | yes | yes | yes |
| `joint_jacobian_time_variation` | yes | yes | yes | yes | yes | yes | yes |
| `inverse_dynamics` | yes | yes | yes | yes | yes | yes | yes |
| `mass_matrix` | yes | yes | yes |  | yes | yes | yes |
| `forward_dynamics` | yes | yes | yes | yes | yes | yes | yes |
| `centroidal_map` | yes | yes | yes |  | yes | yes | yes |
| `centroidal_momentum` | yes | yes | yes | yes | yes | yes | yes |
| `center_of_mass` | yes | yes | yes |  | yes | yes | yes |
| `integrate` | yes | yes | yes | yes | yes |  | yes |
| `difference` | yes | yes | yes | yes | yes |  | yes |
| `full_configuration` | yes | yes | yes | yes | yes |  | yes |
| `lie.umeyama` | yes | yes | yes |  | yes |  | yes |
| `se3.act` | yes | yes | yes |  | yes |  | yes |
| `se3.adjoint` | yes | yes | yes |  | yes |  | yes |
| `se3.adjoint_inverse` | yes | yes | yes |  | yes |  | yes |
| `se3.compose` | yes | yes | yes |  | yes |  | yes |
| `se3.exp` | yes | yes | yes |  | yes |  | yes |
| `se3.from_axis_angle` | yes | yes | yes |  | yes |  | yes |
| `se3.from_matrix` | yes | yes | yes |  | yes |  | yes |
| `se3.from_translation` | yes | yes | yes |  | yes |  | yes |
| `se3.inverse` | yes | yes | yes |  | yes |  | yes |
| `se3.log` | yes | yes | yes |  | yes |  | yes |
| `se3.normalize` | yes | yes | yes |  | yes |  | yes |
| `se3.sclerp` | yes | yes | yes |  | yes |  | yes |
| `se3.to_matrix` | yes | yes | yes |  | yes |  | yes |
| `so3.act` | yes | yes | yes |  | yes |  | yes |
| `so3.compose` | yes | yes | yes |  | yes |  | yes |
| `so3.exp` | yes | yes | yes |  | yes |  | yes |
| `so3.from_axis_angle` | yes | yes | yes |  | yes |  | yes |
| `so3.from_euler` | yes | yes | yes |  | yes |  | yes |
| `so3.from_matrix` | yes | yes | yes |  | yes |  | yes |
| `so3.inverse` | yes | yes | yes |  | yes |  | yes |
| `so3.log` | yes | yes | yes |  | yes |  | yes |
| `so3.normalize` | yes | yes | yes |  | yes |  | yes |
| `so3.slerp` | yes | yes | yes |  | yes |  | yes |
| `so3.to_euler` | yes | yes | yes |  | yes |  | yes |
| `so3.to_matrix` | yes | yes | yes |  | yes |  | yes |
| `tangents.hat_se3` | yes | yes | yes |  | yes |  | yes |
| `tangents.hat_so3` | yes | yes | yes |  | yes |  | yes |
| `tangents.left_jacobian_inverse_se3` | yes | yes | yes |  | yes |  | yes |
| `tangents.left_jacobian_inverse_so3` | yes | yes | yes |  | yes |  | yes |
| `tangents.left_jacobian_se3` | yes | yes | yes |  | yes |  | yes |
| `tangents.left_jacobian_so3` | yes | yes | yes |  | yes |  | yes |
| `tangents.right_jacobian_inverse_se3` | yes | yes | yes |  | yes |  | yes |
| `tangents.right_jacobian_inverse_so3` | yes | yes | yes |  | yes |  | yes |
| `tangents.right_jacobian_se3` | yes | yes | yes |  | yes |  | yes |
| `tangents.right_jacobian_so3` | yes | yes | yes |  | yes |  | yes |
| `tangents.vee_se3` | yes | yes | yes |  | yes |  | yes |
| `tangents.vee_so3` | yes | yes | yes |  | yes |  | yes |
| `spatial.cross_motion` | yes | yes | yes |  | yes |  | yes |
| `spatial.cross_force` | yes | yes | yes |  | yes |  | yes |
| `spatial.transform_motion` | yes | yes | yes |  | yes |  | yes |
| `spatial.transform_force` | yes | yes | yes |  | yes |  | yes |
| `spatial.transform_inertia` | yes | yes | yes |  | yes |  | yes |
| `spatial.inertia_matrix` | yes | yes | yes |  | yes |  | yes |
| `spatial.rotational_inertia` | yes | yes | yes |  | yes |  | yes |
| `spatial.inertia_from_mass_com` | yes | yes | yes |  | yes |  | yes |
| `spatial.inertia_of_sphere` | yes | yes | yes |  | yes |  | yes |
| `spatial.inertia_of_box` | yes | yes | yes |  | yes |  | yes |
| `spatial.inertia_of_capsule` | yes | yes | yes |  | yes |  | yes |
| `spatial.inertia_of_ellipsoid` | yes | yes | yes |  | yes |  | yes |
| `spatial.inertia_of_mesh` | yes | yes | yes |  | yes |  | yes |
<!-- end of effects table -->
