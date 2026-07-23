# Order 01 — `local_world_aligned` for joint Jacobians + parity coverage

Read `plan/README.md` first. Smallest order of the round: one additive
branch and the joint-Jacobian parity tests that should have existed
already.

## Change

`src/better_robot/kinematics/jacobian.py:174-200` — add an
`elif reference == "local_world_aligned":` branch to `get_joint_jacobian`
mirroring the frame path (`frame_jacobian_raw` lines 140-142), using the
joint's own world position:

```python
p_j = data.joint_pose_world[..., joint_id, :3]
linear = J_j[..., :3, :] - torch.matmul(hat_so3(p_j), J_j[..., 3:, :])
return torch.cat((linear, J_j[..., 3:, :]), dim=-2)
```

Semantics (verified vs pinocchio 3.9.0, err ≤ 1.1e-16): world-frame J with
linear rows translated to the joint origin; angular rows unchanged. Update
the docstring to document all three frames the way `get_frame_jacobian`'s
docstring does (`jacobian.py:210-227`), including that `"local"` uses the
full adjoint while `"local_world_aligned"` is translation-only. The
`_ReferenceFrame` Literal already includes the value; no signature change.

## Tests

1. **Net-new parity file** `tests/test_pinocchio/test_joint_jacobian_matches_pinocchio.py`
   — mirror the structure, tolerances (`atol=2e-6, rtol=1e-5`), and fp64
   model convention of `test_frame_jacobian_matches_pinocchio.py`:
   - all three references vs `pin.getJointJacobian(pin_model, pin_data,
     joint_id, pin.ReferenceFrame.X)` after `pin.computeJointJacobians`,
     over `sample_panda_q` configurations, several joints including the
     tip;
   - one batched case (stacked q, per-sample comparison);
   - one free-flyer + spherical case following the pattern of
     `test_rnea_advanced_joints.py:28-121` (nq≠nv coverage — this is where
     a wrong frame convention would hide on Panda). Those fixtures are
     module-local, not in conftest: either duplicate the small builders or
     promote them to `tests/test_pinocchio/conftest.py` (promotion
     preferred if order 03 will need them too — it will).
2. **`tests/kinematics/test_jacobians.py`** — the existing
   reference-parametrized test (`:105-121`) covers **frames** only, and
   the joint test (`:91-102`) is world-only; `get_joint_jacobian` is not
   exercised with any `reference=` today. Add a new joint-reference
   parametrization mirroring the frame test at `:105-121` (fp32, builder
   models, all three references).

## Acceptance

- New parity file green; `tests/kinematics` green; full CPU gate green.
- No changes outside `jacobian.py` docstring/branch and the two test
  files. `get_joint_jacobian` has zero internal callers — nothing else
  moves.

## Out of scope

Docs pages (order 04). Time variation (order 03). Any change to the world
"local" branch or to `compute_joint_jacobians`.
