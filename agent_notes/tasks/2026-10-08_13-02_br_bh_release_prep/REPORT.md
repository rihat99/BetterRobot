# REPORT — BR + BH release preparation

Task `2026-10-08_13-02_br_bh_release_prep`, 2026-10-08 13:02 → ~15:30 Dubai. Worktrees BR
`/home/rikhat.akizhanov/better/dev/BetterRobot` and BH `/home/rikhat.akizhanov/better/dev/BetterHuman`, branch `dev`.
Nothing pushed or merged.

## What was asked
1. Adversarial architecture review of both libraries against how reference projects evolved; gap list + roadmap
   (not implemented), delivered as an HTML site. 2. Docs videos showed as white space. 3. HumanStudio: frames files
   tracked, gizmo usable inside the body, `Reset pose` / `Deselect joint` / `Deselect frame` above the tabs.
4. Final code sweep: bugs, correctness, bottlenecks. Owner's choices at intake: gizmo on top + slightly bigger;
   three buttons + a status line; fix small bugs, report the rest; six reviewers.

## Deliverables
- **Site**: `/home/rikhat.akizhanov/better/dev/BetterRobot/agent_notes/tasks/2026-10-08_13-02_br_bh_release_prep/site/index.html`
  (open in a browser; pages: Overview, Optim, Kin/Dyn, Collision, Viewer, Human, Toolbox, Decisions). Rebuild with
  `cd site && uv run --no-project --python 3.12 python build.py`.
- Reviews: `subagents/p1_{optim,kindyn,collision,viewer,human,toolbox}.md`. Sweep reports:
  `subagents/p2_{br_core,br_optim,br_optim2,br_io_viewer,bh,bh2,perf,studio}.md`. Studio screenshots `vis/studio/`
  (gitignored), site renders `vis/site/`, benchmark raw outputs `artifacts/perf/`.

## 1. Architecture review — headline gaps (details and roadmap on the site)
| Theme | Biggest gaps |
|---|---|
| optim | time-coupled problems without `window` silently go dense (T=256, n=7 → 1792² Cholesky); autograd Jacobian = one reverse pass per residual row; constraints/multipliers live outside `Problem`/`Result`, so SQP/QP-IK/DDP cannot plug in |
| kin/dyn | a functorch probe blocks `torch.compile(fullgraph=True)`; batch-1 is launch-bound (RNEA ~650× Pinocchio, 8× per sample at B=4096); no derivatives API; dense O(nv³) forward dynamics (SMPL-X nv≈168) |
| collision | parsed collision geometry never reaches `Model` (only `meta["ir"]`, read by the viewer); no sphere fitting from meshes; self-collision filter is topology-only (160 spheres → 8448 pairs) |
| viewer | one integer time base (no mixed-rate mocap/video/forces); viser's single pointer slot means tools cannot coexist; no GUI-panel brick (HumanStudio uses the escape hatch); BVR overlays missing (video pane, reprojection, wrenches, trails, labels) |
| human | `Rig.corrective_features` and `Body.bake` closed to STAR/SUPR/Anny; ~13 undeclared `FamilySpec.meta` keys; contact frames SMPL-only; no fitting recipe / Motion type / retargeting namespace |
| toolbox | three names for one SE(3) quantity (pose/placement/transform); joint ids not `int \| str`; URDF `<dynamics>` damping/friction silently dropped; 83 bare `ValueError`s bypass the error hierarchy; no model editing/composition; no trajectory type |

Each theme page lists "prep changes now" (each ≤ 1 day, non-breaking); none were implemented — they are decisions
for the owner (Decisions page).

## 2. Docs videos — fixed
Raw `<video>`/`poster` paths were `../clips/…`; mkdocs rewrites Markdown links but not raw HTML, and with directory
URLs a page sits one level deeper in the built site, so the browser asked for `/getting-started/clips/…` (404 →
white box). Seven pages now use `../../clips/…`; `tests/docs/test_snippets.py::test_referenced_images_exist` resolves
raw-HTML paths the way the built site does (fails on the old path, passes now; same change in BH's copy).

## 3. HumanStudio — done
- Frames files `studio_data/frames/smplx_neutral/{contacts,better_contacts}.json` were already tracked in both main
  (`b3341b2`) and dev, byte-identical; nothing to copy.
- Gizmo: `depth_test=False` (drawn through the mesh) and default scale 0.15 → 0.20 (`viz/studio/gizmo.py`).
- Above the tabs: status line `smplx neutral · joint: left_elbow · frame: frame_0` and three full-width buttons
  (viser's button group never wraps and clipped at the sidebar edge, so plain buttons). Verified on a live server
  (`subagents/drive_studio_p2.py`): screenshots `vis/studio/studio_{before_select,joint_selected,after_deselect}.png`.
- Left: after a model switch `frames.active` keeps the old name (label hides it); finger-joint spheres still large.

## 4. Sweep — fixed (each with a pinning test)
| Repo | Bug | Where |
|---|---|---|
| BR | zero-size batch crashed FK, Jacobians, dynamics (`reshape(0, -1)`) | `kinematics/state.py`, `kinematics/_fk_lean.py`, `dynamics/state.py` |
| BR | every solver crashed on B=0 (`x[0].numel()`); `RobotManifold` too | `optim/manifold.py`, `optim/solvers.py`, `terms/robot.py`, `optim/problem.py` |
| BR | LM/GN crashed when a `scalar` term did not depend on the variables | `optim/problem.py` |
| BR | URDF parser ignored `<inertial><origin rpy>` rotation (now R I Rᵀ; Pinocchio parity test) | `io/parsers/urdf.py` |
| BR | `Viewer.url`/`record` used the requested port, not the one viser bound | `viewer/viewer.py` |
| BR | all three benchmark scripts broken since keyword-only args (`31a37ed`) | `benchmarks/*.py`, BH `benchmarks/run.py` |
| BH | repeated `drop_joints` corrupted `meta["native_joint_ids"]` (classic pose axes read wrong) | `core/spec.py` |
| BH | `CLAUDE.md` benchmark command lacked the required `--out` | `CLAUDE.md` |

Pinocchio parity (G1 floating base, float64): FK 7.8e-16, Jacobians ≤ 1.3e-15, RNEA 2.6e-13, CRBA 2.1e-14,
ABA 9.4e-10, integrate/difference ≤ 8.9e-16 (`subagents/p2_br_core_probes/pin_spot.py`).

## 4b. Sweep — reported, not changed (owner decides)
1. **MJCF parser (critical)**: drops body pos/quat of every body with a joint; joint ranges not converted from
   degrees; `limited="auto"` read as true → limits [0,0]; ignores iquat/fullinertia/geom mass. The dev venv has no
   mujoco so these tests skip. Suggested: rebuild the parser on the compiled `MjModel` (`p2_br_io_viewer.md`).
2. `so3.to_euler` wrong at pitch = ±π/2 (gimbal lock; round-trip error 1.76) — needs a branch (`p2_br_core.md`).
3. `joint_jacobian*` take only an int; names/negative ids unvalidated (`p2_br_core.md`, matches toolbox F2).
4. L-BFGS flags convergence one step late; `TorchOptim` never sets `done`; `solve_constrained(outer=0)` bare error
   (`p2_br_optim.md`).
5. dJ/dt recomputes every joint per call: 3.78 ms vs 0.95 ms FK (G1, B=64) (`p2_br_core.md`).
6. BH: studio editors call `.numpy()` without `.cpu()`; `mhr_rig.build_model` compares `device("cuda")` with
   `cuda:0` so the MHR tree is rebuilt every `Body` (`p2_bh2.md`).
7. BH known failure `test_corrective_slice_is_exact_when_dropped_joint_is_at_rest`: the reduction is exact; the test
   compares a 27-term and an 18-term float32 sum at tolerance 0 and they differ by 1 ulp (7.6e-6 at 112.9).
   Loosening the tolerance is the owner's call (`p2_bh.md`).
8. Benchmarks: no test runs the scripts (how the break went unnoticed); `pytest -m bench` collects 1 test because
   bench files are `bench_*.py`; README names a missing `bench_trajopt_sparse.py`; 8 threads on contended cores makes
   FK 50–80× slower (use `--threads 1`) (`p2_perf.md`).
9. Not read by any sweep agent (budget): BR `io/build_model.py`, `io/programmatic.py`, viewer layers/tools bodies,
   `optim/linear.py` bodies, parts of `optim/problem.py`, `collision/robot_spheres.py` bodies; BH `io/schema.py`,
   `io/convert_chumpy.py`, `families/mhr.py`, `mhr_archive.py`, studio layers/persistence/posing bodies.

## 5. Performance (CPU only; no GPU was free; machine load 600–900 during the task)
No regressions vs `benchmarks/results/0953f28_cpu.json` (same EPYC 9654, 1 thread): every row faster, new/old
0.12–0.94 (Jacobian g1 B=4096 271 → 50 ms; ABA panda B=1 6.1 → 0.8 ms). Docs timings still hold. Profiles:
`frame_jacobian` is one bmm (97 %); `solve_ik` LM is launch-bound (~60 bmm + 58 cat per iteration, 52 ms / 10 it at
B=64). Raw: `artifacts/perf/`.

## Gates
| Repo | Tests | Ruff | Docs |
|---|---|---|---|
| BR | 1589 passed, 11 skipped, 0 failed (`-m "not bench and not cuda"`) | 82 pre-existing findings, unchanged | `mkdocs build --strict` clean |
| BH | 334 passed, 1 skipped, 1 failed (known `test_reduction`, diagnosed in 4b.7) | clean | — |

Site checked by a read-only adversarial pass (`subagents/p3c_aux_findings.md`): 25 claims verified against the code,
4 corrected (a false optim gap already covered by contract 02-tensors; the compile fix understated; "three names for
one SE(3) quantity" overstated — placement vs pose is Pinocchio's split, the real gap is the 3x4 `*_transform` name;
"frame counter" wording breaks contract 01-naming), biomechanics items added (C3D/BVH/AMASS import, marker sets,
muscles, force-plate input).

## Commits
| Repo | Commit | Content |
|---|---|---|
| BR | `05c46d2` | sweep fixes, docs clip paths, benchmark scripts |
| BR | notes commit (below) | this task folder incl. `site/` |
| BH | `2b8da78` | studio row + gizmo, `drop_joints` fix, benchmark call site, CLAUDE.md, test doubles |

Trash to empty after review: none added by this task.
