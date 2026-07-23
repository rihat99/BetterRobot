# Order 03 — Results: Jacobian time variation (J̇)

WORLD `J̇` cache (`data.joint_jacobians_dot`) plus LOCAL/LWA getters for joints
and frames, built on the order-02 batched column table. One raw pass
(`joint_jacobians_time_variation_raw`) computes the WORLD `J̇` transport
`dJ[:, c] = ad(ov_owner(c)) @ columns[:, c]` on full-width owner columns
(`ov = J v` from the reduced per-joint Jacobian), masks per joint, reduces last
— exactly like `joint_jacobians_raw`. Getters derive LWA/LOCAL from the WORLD
cache via the verified closed forms; `frame_*` reuses the parent joint's WORLD
`J`/`J̇`. Transport-only: every shipped joint's local motion subspace is
constant (no `dS/dt` term).

## Parity vs pinocchio 3.9.0 (fp64, max abs error over configs/joints)

Oracle: `pin.computeJointJacobiansTimeVariation` →
`getJointJacobianTimeVariation` / `getFrameJacobianTimeVariation`.

| Model / target       | world    | local_world_aligned | local    |
|----------------------|----------|---------------------|----------|
| Panda joints (1/4/7) | 8.40e-07 | 8.40e-07            | 1.12e-06 |
| Panda frame (hand)   | 8.40e-07 | 8.40e-07            | 1.13e-06 |
| G1 free-flyer joints | 7.30e-08 | 7.30e-08            | 6.57e-08 |
| Spherical chain      | 8.88e-16 | 8.88e-16            | 6.66e-16 |

Kept tolerance `atol=2e-6, rtol=1e-5` (the static frame-Jacobian file's) — **no
widening**. Panda/G1 residuals are dominated by the URDF fp32 placement
round-trip (`io/parsers/urdf.py`), identical to the static frame-Jacobian
parity; the programmatic fp64 spherical chain hits machine precision, proving
the formulas exact.

## Finite-difference cross-check (fp32, builder chain, dt=1e-3, max abs error)

Central difference of the same-frame Jacobian along `v` via `model.integrate`
retraction vs the getter. Fixture is non-degenerate: `v` nonzero in every
coordinate, tip joint/tool frame at a nonzero world origin with 3 angular DOF
above it; `WORLD J̇ − LWA J̇` is O(1) here (`> 0.1` asserted), so the
`−v_pt × J_ang` transport term is genuinely exercised.

| target | world    | local_world_aligned | local    |
|--------|----------|---------------------|----------|
| joint  | 4.36e-05 | 4.36e-05            | 8.78e-05 |
| frame  | 4.36e-05 | 4.36e-05            | 9.69e-05 |

fp32 roundoff floor (~1e-4 abs, dt-independent) sets the FD test tolerance
`atol=3e-4, rtol=1e-2` — still ~10⁴× tighter than the O(1) trap term it guards.

## Launch count — combined J + J̇ pass (MHR-203-like, batch 256, fp32)

Model: free-flyer root + 66 spherical (njoints 68, nv 204). Protocol = order 02
(`cudaLaunchKernel` events from `torch.profiler`, joint poses precomputed under
`no_grad`, GPU 2 pinned via `CUDA_VISIBLE_DEVICES`).

| Pass                                     | launches | /joint |
|------------------------------------------|----------|--------|
| J alone (`joint_jacobians_raw`)          | 52       | 0.76   |
| combined J+J̇ (`..._time_variation_raw`) | 60       | 0.88   |
| incremental J̇                           | 8        | —      |

J alone reproduces order-02's mhr203 figure (52). J̇ adds a **constant 8**
launches (einsum for `ov`, owner gather, two cross pairs, cat, transpose, second
mask-multiply, second mimic reduce) — independent of njoints, matching the
performance bar.

## Gate

- Focused `tests/kinematics tests/test_pinocchio tests/contract`: 591 passed.
- New tests: `test_jacobian_time_variation_matches_pinocchio.py` 22, fp32
  `test_jacobian_time_variation.py` 15, `test_mimic.py` +1 covector-reduction.
- Full CPU gate `-m "not bench and not cuda"`: **1737 passed, 61 deselected**.
- `ruff check` on the order-03 diff: clean (the 7 reported errors are
  pre-existing `PLC0415` local imports in `conftest.py`, untouched lines).
- No fp64 outside `tests/test_pinocchio`; contract tests updated additively.
