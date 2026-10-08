# REPORT — "now" items implemented, roadmap documented (BR + BH)

Task `2026-10-08_15-45_br_bh_now_items`, 2026-10-08 16:37 → ~19:30 Dubai. Worktrees BR
`/home/rikhat.akizhanov/better/dev/BetterRobot`, BH `/home/rikhat.akizhanov/better/dev/BetterHuman`, branch `dev`.
Nothing pushed or merged. Owner decisions are recorded verbatim in `PLAN.md`; they override the review site.

## What landed (BetterRobot)
| Theme | Change | Where (contract / ADR / test) |
|---|---|---|
| toolbox | joint ids accept `int \| str` everywhere; unknown names raise `UnknownNameError` (a `KeyError`); ids were already keyword-only (contract test existed) | contract 06; `tests/kinematics/test_joint_names.py`, `tests/contract/test_keyword_only.py` |
| toolbox | `io.parse(source, format=) -> IRModel`; `load = build_model(parse(...))`; URDF `<dynamics>` and MJCF `damping`/`frictionloss`/`armature` reach `ModelValues` (were silently zero; MJCF path verified with the main checkout's mujoco) | `docs/concepts/io.md`; `tests/io/test_joint_dynamics.py` |
| toolbox | `Trajectory` frozen dataclass (`q (T,nq)`, `times (T,)` strictly increasing, `v` optional, `at(step)`) | glossary; `tests/data_model/test_trajectory.py` |
| toolbox | SE(3) vocabulary C-NAME-5: placement (parent frame, 7-vector, constant or q-dependent), pose (world 7-vector), transform (world 3x4); `JointModel.joint_transform` → `joint_placement` (owner); contract test walks functions, methods, properties incl. viewer | contract 01, glossary; `tests/contract/test_se3_vocabulary.py` |
| toolbox | `BackendNotAvailableError` documented as the reserved kernel hook; MJCF marked experimental with its known gaps | contract 07; io page |
| optim | per-step diagnostics `grad_norm`, `step_norm`, `accepted` (LM: `damping`) in `Result.diagnostics[phase][key]` `(steps, B)`; `Result.state` + `solve/run(state=)` warm start (LM damping, L-BFGS memory); dense-fallback warning for `time=` problems; L-BFGS `done` from the new gradient; `solve_constrained(outer<1)` → `ValueError`; warm-start shape check | contract 08 (sink seam), `optimization.md` (reserved `equality`/`inequality`); `tests/optim/test_solver_state.py` |
| kin/dyn | placement-cache path guarded by `torch.compiler.is_compiling()` → FK, RNEA, CRBA run under `torch.compile(fullgraph=True)` and match eager; ADR 0007 derivatives contract (autograd first, tangent-space partials, batch-first, Warp reserved); `_memo` keys documented | contract 03/04; `tests/kinematics/test_fk_compile.py` |
| kin/dyn | `so3.to_euler` exact at pitch ±π/2 with finite gradients (float32/64), band chosen by angle; bit-identical elsewhere (200k quats) | `tests/lie/test_euler_matrix_interop.py` |
| kin/dyn | benchmark smoke test; `pytest -m bench` collects `bench_*.py`; README fixed | `tests/bench/test_run_smoke.py`, `pyproject.toml` |
| collision | keyword-only `frame`/`target`/`pairs` in `terms`; `self_collision_pairs(ignore=)`; `RobotSpheres.to()`; batched offsets documented+tested; ADR 0008 geometry brick (separate `RobotGeometry`, spheres+capsules, lazy trimesh, `*_closest` functions, contact in BR, `meta["ir"]` frozen) | contract 08 (`distance_fn`); `tests/collision/test_collision.py` |
| viewer | `Viewer.add(layer, times=)` per-layer seconds, `Viewer._index` nearest-earlier sample, integer `step` default; `on_pointer`/`clear_pointer` own viser's single slot; `tools.nearest_on_ray`; `layers/camera.py`; "Lazy inputs" + "Limits" docs; viser-only and `server.gui` sanctioned (ADR 0004 note); "frame" → "step" in the timeline | contract 01 C-NAME-4; `tests/viewer/test_time_pointer.py` |
| docs | `docs/reference/roadmap.md` rewritten: how to read now/next/later/maybe, "Landed in this version", current limits, plan by theme (5 tables); changelog "Unreleased" entries incl. breaking changes; contract 08 intro lists five seams; ADRs 0007/0008 in the nav | — |

## What landed (BetterHuman)
`Rig.corrective_features(..., *, identity=None)` (breaking for third-party Rigs; memo no longer shared across
`replace`); `META_KEYS` documented in `FamilySpec` with a scan test; fps/`times` semantics in the addendum;
`HumanState.vertices`/`frame_poses` are properties; studio uses `Viewer.on_pointer` and BR's `nearest_on_ray`,
resets `frames.active` on model switch, `.detach().cpu().numpy()` in editors; MHR tree cache hit for bare `cuda`;
reduction test compares at one ulp (rtol 1e-6 f32, 1e-15 f64 — owner accepted); scope section: families in-tree
only, retargeting and fitting are not BH's job; roadmap section (STAR next, mocap import next, marker sets later,
muscles maybe). Live studio check: `vis/studio/` (4 PNGs).

## Review
Read-only adversarial pass on the BR diff (`subagents/w2_aux_findings.md`): 10 findings — NaN gradient at the exact
gimbal lock (fixed), float32 near-lock snapping (fixed), weak vocabulary test (fixed), MJCF path unverified (verified
with mujoco 3.8), viewer `times` persistence/NaN (fixed), warm-start shape error (fixed), `q_permutation` error type
(fixed), `Trajectory` monotonic check (added), `ignore=` docstring (fixed), `joint_transform` conflict (renamed per
owner). Everything the pass probed holds (see the file's "Confirmed" list).

## Gates (final, clean runs)
| Repo | Tests | Ruff | Docs |
|---|---|---|---|
| BR | 1650 passed, 13 skipped, 0 failed | 80 pre-existing findings (was 82) | strict clean |
| BH | 339 passed, 1 skipped, 0 failed (the former known failure now passes at one-ulp tolerance) | clean | strict clean |

## Commits
| Repo | Commit |
|---|---|
| BR | `f9245f0` (code + docs), notes commit after it |
| BH | `d0d06aa` |

## Left for the owner
- `Trajectory` has no consumer yet (`Viewer.add` takes numpy `times`); wiring it in is a roadmap `next`.
- The MJCF parser stays experimental (owner decision); its joint-dynamics test only runs where mujoco is installed.
- 80 pre-existing ruff findings in untouched files (was 82).
- Previous task's site (`…/2026-10-08_13-02_br_bh_release_prep/site/`) still describes the pre-implementation state;
  the roadmap pages in the docs are now the source of truth.
