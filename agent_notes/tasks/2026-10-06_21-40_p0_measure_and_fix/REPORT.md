# REPORT — P0 measure and fix (BetterRobot)

Task `2026-10-06_21-40_p0_measure_and_fix`, branch `dev`, commits `0953f28` and `d1a96de`
(plus the commit that adds this report). First phase of the redesign in
`/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_20-13_architecture_plan/DESIGN.md`.
Nothing was pushed or merged. BetterHuman was not touched.

## Result in short

1. The dev environment now has pinocchio, Warp and the other declared extras, so the pinocchio
   comparison tests run instead of skipping.
2. Two real bugs fixed with tests; the third reported bug is not a bug; one new defect found and
   left for P2 (URDF values are rounded to float32).
3. A benchmark runner exists (`benchmarks/run.py`) and the CPU baseline is stored.
4. A leaner FK prototype needs 33 to 57 operator calls instead of 246 to 336 and is 2.6 to 7
   times faster on CPU with identical results. The pose format `(B, joints, 7)` alone costs more
   than the whole lean FK.
5. The immutable FK result exists as a thin shim: `kinematics.forward_kinematics_state(model, q) -> KinematicState`,
   `kinematics.frame_jacobian(model, state, frame)`.
6. Reference outputs of today's IK and trajectory optimization are stored for comparing the new optim.
7. Not done: GPU baselines and the Warp record (no free GPU all evening), a BVR clip reference.

## 1. Environment and test gate

`uv sync --all-extras` (only packages already declared in `pyproject.toml`): pinocchio 3.9.0,
warp 1.15.0, viser 1.0.26, robot_descriptions 1.23.0, torch 2.13.0+cu129.

Full suite on CPU (`CUDA_VISIBLE_DEVICES="" uv run pytest tests -n 8`): 1765 pass, 2 fail, 62 skip.
Both failures are present at the starting commit `190cd34` with this environment and were not
touched:

| Test | Error | Note |
|---|---|---|
| `tests/optim/test_temporal_structure.py::test_banded_and_dense_lm_match_with_outer_reduction_and_activity` | 1.8e-4, allowed 4e-5 | float32 comparison of two LM routes; the optim is replaced in P1 |
| `tests/kinematics/test_fk_regression.py::test_fk_matches_oracle[g1]` | 2e-8, allowed 1e-10 | was silently skipped before (robot_descriptions missing); stored reference likely from another G1 file revision, not checked |

## 2. Bugs

| Bug | Status | Evidence |
|---|---|---|
| SE3 right Jacobian was a 9-term series; error 2.6e-4 at 2 rad, 1e-2 at 3 rad, 0.34 at 4.5 rad | fixed: closed form (Barfoot eq. 7.86) with a Taylor branch below 0.2 rad; inverse in closed form, no `linalg.inv` | `tests/lie/test_tangents.py` (60-term series reference, float64 1e-10, float32 2e-5), pinocchio `Jexp6` at large angles |
| `right_jacobian_inv_so3` returned about 1e30 beyond pi (clamped a negative denominator) | fixed: half-angle form | same test file |
| `Model.random_configuration`: float32 on a float64 model; NaN for unlimited revolute joints | fixed: model dtype and device; unlimited revolute sampled in [-pi, pi]; other unlimited joints raise `ValueError` naming the joint | `tests/data_model/test_model_data.py` |
| "FK crashes on mismatched dtype" | not a bug: `forward_kinematics` raises `DtypeMismatchError`; the crash needs a direct call of an internal function | reproduction script output in this report's history |
| NEW: the URDF parser casts every value to float32 (`io/parsers/urdf.py:29-30` and others) | open, for P2 | a float64 Panda has placement quaternions off unit norm by 6.7e-8, so the analytic Jacobian and autograd through FK agree only to 1.4e-7 |

Does the Jacobian bug explain the poor batched IK convergence? No.
`subagents/ik_jacobian_fix_check.json` (Panda, batch 1024, float32, default LM): converged
fraction 0.840 with both the old and the exact Jacobian at spread 0.3, 0.725 vs 0.735 at spread 1.0.

What does explain most of it (`golden/summary.json`, float64, batch 256, reachable targets):

| IK cost | spread | median position error | within 0.1 mm | "converged" flag |
|---|---|---|---|---|
| default (rest 0.01, limits 0.1) | 0.3 rad | 0.41 mm | 7% | 95% |
| default | 1.0 rad | 1.24 mm | 1% | 86% |
| no regularizers | 0.3 rad | 0.64 um | 93% | 93% |
| no regularizers | 1.0 rad | 2.7 um | 67% | 73% |

The default IK solves a regularized problem, so it stops about half a millimetre from a
reachable target, and the `converged` flag does not mean "target reached". The remaining
failures at large spread are local minima (worst case 0.4 to 0.7 m). Both are inputs for P1.

## 3. Benchmark runner and CPU baseline

Command: `CUDA_VISIBLE_DEVICES="" uv run python benchmarks/run.py --device cpu`.
Output: `/home/rikhat.akizhanov/better/dev/BetterRobot/benchmarks/results/0953f28_cpu.json`.
Grid: FK, frame Jacobian, RNEA, CRBA, ABA x Panda, G1, 25- and 200-joint random trees x batch
1, 64, 256, 4096 x forward / forward+backward, plus pinocchio rows at batch 1.
"ops" = operator calls dispatched per call (torch profiler, top-level `aten::` events); it does
not depend on timing noise. CPU: AMD EPYC 9654, 1 thread, float32. Another benchmark ran in
parallel on other cores, so times are good to roughly 10%.

Median ms / ops:

| algo | model | fwd B=1 | fwd+bwd B=1 | fwd B=4096 | fwd+bwd B=4096 | pinocchio fwd B=1 (ms) |
|---|---|---|---|---|---|---|
| fk | panda | 0.655 / 246 | 2.03 / 721 | 32.9 / 246 | 70.5 / 721 | 0.00047 |
| fk | g1 | 0.901 / 336 | 3.25 / 1193 | 78.4 / 336 | 234 / 1193 | 0.0013 |
| fk | tree25 | 0.716 / 265 | 2.42 / 873 | 51.9 / 265 | 132 / 873 | - |
| fk | tree200 | 1.72 / 615 | 8.41 / 3148 | 666 / 615 | 7.43e+03 / 3148 | - |
| jacobian | panda | 0.981 / 384 | 3.05 / 1082 | 38.8 / 384 | 97.5 / 1082 | 0.00066 |
| jacobian | g1 | 1.24 / 473 | 4.29 / 1549 | 271 / 473 | 669 / 1549 | 0.0019 |
| jacobian | tree25 | 1.03 / 402 | 3.41 / 1229 | 109 / 402 | 297 / 1229 | - |
| jacobian | tree200 | 2.13 / 752 | 9.61 / 3504 | 3.2e+03 / 752 | 1.04e+04 / 3504 | - |
| rnea | panda | 2.11 / 1053 | 7.01 / 2621 | 53.4 / 1053 | 165 / 2621 | 0.0016 |
| rnea | g1 | 4.81 / 2648 | 17.9 / 6841 | 174 / 2648 | 670 / 6841 | 0.0049 |
| rnea | tree25 | 3.43 / 1866 | 12.1 / 4632 | 111 / 1866 | 350 / 4632 | - |
| rnea | tree200 | 20.6 / 12366 | 86.6 / 31569 | 1.32e+03 / 12366 | 3.75e+04 / 31569 | - |
| crba | panda | 1.86 / 909 | 6.69 / 2894 | 91.5 / 910 | 233 / 2895 | 0.0012 |
| crba | g1 | 4.5 / 2550 | 19.1 / 9016 | 300 / 2551 | 1.17e+03 / 9017 | 0.0046 |
| crba | tree25 | 2.74 / 1430 | 10.1 / 4553 | 185 / 1431 | 478 / 4554 | - |
| crba | tree200 | 23.1 / 14183 | 116 / 49704 | 2.6e+03 / 14184 | 8.23e+05 / 49705 | - |
| aba | panda | 6.09 / 3017 | 16.6 / 6573 | 200 / 3018 | 468 / 6574 | 0.0056 |
| aba | g1 | 6.91 / 3416 | 29.2 / 10483 | 359 / 3417 | 956 / 10484 | 0.019 |
| aba | tree25 | 5.2 / 2491 | 19.2 / 6920 | 216 / 2492 | 545 / 6921 | - |
| aba | tree200 | 34.4 / 17366 | 152 / 49898 | 2.15e+03 / 17367 | 2.6e+04 / 49899 | - |

Reading the table:
- At batch 1 the library is 1,000 to 1,500 times slower than pinocchio (FK 0.67 ms vs 0.45 us on
  Panda). At batch 4096 on one CPU thread it is still 17 times slower per sample (7.7 us vs 0.45 us).
- Operator counts do not depend on batch size, and for dynamics they grow with joint count
  (RNEA: 1,053 on Panda, 12,366 on the 200-joint tree).
- Backward is where large trees collapse: CRBA forward+backward on the 200-joint tree at batch
  4096 takes 800 s per call.
- A full run takes about 50 minutes, almost all of it in 200-joint dynamics with backward. The
  runner has `--max-call-s` to skip larger batches after a slow row.

Reproducibility: two complete runs gave identical operator counts on all 160 library rows (`artifacts/run_a_cpu.json` against the stored file). Counts differ by exactly 1 between batch 1 and larger batches for CRBA and ABA, which is a property of the code, not noise.

## 4. Lean FK prototype (E3) and compile probe (E2)

Prototype: `subagents/proto_lean_fk.py`, note `subagents/proto_lean_fk.md`. One affine map turns
`sin q`, `cos q` (and quaternion products for free joints) into all local transforms; then one
matrix product per tree level. Output is `[R | p]` of shape `(B, joints, 3, 4)`.

| Model | ops forward: library → lean | ops forward+backward | CPU ms at batch 4096, forward | forward+backward |
|---|---|---|---|---|
| Panda | 246 → 33 | 720 → 179 | 35.2 → 5.2 | 73.3 → 15.8 |
| G1 | 336 → 57 | 1192 → 248 | 138 → 19.4 | 235 → 57.6 |
| 25-joint tree with spherical joints | 298 → 45 | 978 → 190 | 83.6 → 25.5 | 149 → 97.2 |

Parity in float64: positions 1e-15, gradients 1e-14. Converting to the `(B, joints, 7)` pose
format adds about 94 operator calls forward and 300 forward+backward, more than the lean FK itself.

Independent re-measurement and compile probe (`subagents/e2_compile_probe.json`, CPU, 1 thread):

| Lane | Panda B=1 fwd | Panda B=4096 fwd | Panda B=4096 fwd+bwd | G1 B=4096 fwd | G1 B=4096 fwd+bwd | first call |
|---|---|---|---|---|---|---|
| library eager | 0.66 | 35.1 | 69.9 | 139 | 283 | 0 s |
| library `use_compile=True` | 0.21 | 19.3 | 47.4 | 86.7 | 205 | 5 to 22 s |
| lean eager | 0.082 | 5.1 | 15.3 | 36.8 | 96.9 | 0 s |
| lean + `torch.compile` | 0.091 | 5.4 | 15.5 | 42.7 | 87.4 | 5 to 29 s |

- The Panda numbers reproduce. G1 at batch 4096 came out at 36.8 ms here against 19.4 ms in the
  prototype's own run; the gain on G1 is therefore between 3.8 and 7 times, not pinned down.
- `torch.compile` works for forward and backward on both lanes, but lean eager already beats the
  compiled library, and compiling the lean version gains nothing on CPU. Compile stays an option, not a foundation.
- Limits of the prototype: no planar, translation, helical or composite joints, no per-batch
  placements, and its dense map grows with the square of the joint count (a per-family version
  costs about 2 more calls per family and stays linear).

Consequence for P2: store rotation matrices and translations in `KinematicState`; quaternion
poses on request only.

## 5. `KinematicState` shim

`src/better_robot/kinematics/state.py`: frozen dataclass with `q`, `joint_pose`, `frame_pose`
and the accessors `joint_rotation`, `joint_translation`, `frame_rotation`, `frame_translation`;
`forward_kinematics_state(model, q)`; `frame_jacobian(model, state, frame, reference=...)`. It wraps today's
passes. New code should read rotations and translations through the accessors, which stay valid
when P2 changes the storage. Velocities (`v`) are not part of the shim yet.
Test: `tests/kinematics/test_state.py`.

## 6. Reference outputs for P1

`/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-06_21-40_p0_measure_and_fix/golden/`:
`ik_panda.npz` (256 problems with targets and today's solutions for two cost settings),
`trajopt_panda.npz` (the example problem: 9 iterations, goal error 2.8e-10 m, 0.3 s),
`summary.json`, and `make_golden.py` to regenerate. Compare achieved errors and costs, not exact `q`.

## Not verified / not done

- GPU baselines and the Warp-versus-torch record (E1): all 8 GPUs were busy at every check
  (21:40, 22:30). Run `uv run python benchmarks/run.py --device cuda` on a free GPU.
- BVR clip reference output: not started.
- The cause of the G1 reference mismatch (2e-8) was not investigated.
- The lean FK operator-count breakdown per stage comes from reading the code, not from profiling.
- BetterHuman: nothing run.

## Assumptions made off-hours (after 22:00)

- Name `KinematicState` and its field and function names (`forward_kinematics_state`, `frame_jacobian`,
  `joint_pose`, `frame_pose`); small and reversible, the owner has not confirmed them.
- The URDF float32 rounding is left unfixed because fixing it changes stored float64 references.
- The two failing tests were left failing rather than loosened.
- The function is named `forward_kinematics_state`, not `kinematics.forward` as in the design:
  `better_robot.kinematics.forward` is an existing module and exporting a function of that name
  broke two Warp tests. In P2 `forward_kinematics` itself should return the state and this name goes away.
- GPU baselines and the Warp record move to the next task's to-do list.
