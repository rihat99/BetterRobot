# Wave 1 — BetterRobot "now" items (one slice per agent; disjoint files)

Read `context.md` in this folder, then `../PLAN.md` (owner decisions — binding). Your source of detail is the review
for your theme: `../../2026-10-08_13-02_br_bh_release_prep/subagents/p1_<theme>.md` (and `p3c_aux_findings.md` there,
which corrected some claims) and the sweep reports `p2_*.md`. The API is not fixed: breaking changes are allowed, but
every public change needs its docstring, its docs page and its tests updated in the same slice (`tests/docs` runs
every snippet in `docs/`; `mkdocs build --strict` must stay clean). Match the repo's style and contracts
(`docs/contracts/*.md`, nearest `CLAUDE.md`). Other agents edit other slices of the same worktree at the same time:
touch only the files listed for your slice (plus their tests and docs pages); if you must touch a file outside your
slice, say so in the report instead of editing it. Run your slice's tests while iterating and the full suite once at
the end (`CUDA_VISIBLE_DEVICES="" UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests -q -m "not bench and
not cuda" -p no:cacheprovider`), `uv run ruff check src tests` (82 pre-existing findings; add none),
`uv run mkdocs build --strict`. Do not commit.

Deliverable: `subagents/w1_<slice>.md` (≤ 150 lines): what changed (file: change), tests added, docs touched, gate
output tails, anything left or any call site outside your slice that now breaks (BH is handled in wave 2: list BH
call sites you know break, do not edit BH). Final message ≤ 15 lines + path. Budget ≈ 50k tokens: do the items in
the order given; if you cannot finish, say which items are undone.

## A `br_api` — `src/better_robot/{kinematics,dynamics,data_model,io,__init__.py,exceptions.py}`, `docs/contracts/01-naming.md`, `docs/reference/glossary.md`, `docs/concepts/io.md` (or the io page), `examples/`
1. Identifiers keyword-only after `(model, state)` in kinematics/dynamics public functions (`frame`, `joint`, …);
   joint ids accept `int | str` everywhere a frame id does (`joint_jacobian`, `joint_jacobian_time_variation`,
   `get_subtree`, …) via `model.joint_id`; a bad name raises the repo's typed error. Update every in-repo call site
   (src, tests, examples, docs snippets). Add a contract test that public kinematics/dynamics functions take ids
   keyword-only (extend `tests/contract/` in the existing style).
2. `IRJoint` gets `damping`, `friction`, `armature`; URDF `<dynamics damping friction>` and MJCF `damping`,
   `frictionloss`, `armature` are parsed and wired into the existing `ModelValues` fields (`model_values.py:68-70`).
   Test with a small URDF (the sweep's repro: `/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/p1tb/a.urdf`).
3. Public `io.parse(path, format=…) -> IRModel` (the IR exists; expose it; `build_model` consumes it), documented as
   the composition point ("composition lives on the IR").
4. `Trajectory`: minimal frozen dataclass in `src/better_robot/data_model/trajectory.py` (or the module the repo's
   layout suggests): `q (T, nq)`, `v (T, nv) | None`, `times (T,)`, `__len__`, `at(step)`, shape validation at
   construction; exported from `better_robot`; one test; one glossary line. Nothing else uses it yet.
5. Naming contract: add the SE(3) vocabulary (placement = model constant in the parent frame, 7-vector; pose =
   computed world 7-vector; transform = the same world quantity as a 3x4 matrix) to `docs/contracts/01-naming.md`
   and the glossary; add a contract test that new public names follow it (regex on `__all__` / signatures is enough).
6. `BackendNotAvailableError`: docstring + contract 07 line saying it is reserved for a future kernel brick.
7. MJCF: add an "experimental" note (known gaps: body pos/quat for jointed bodies, degree ranges, `limited="auto"`,
   iquat/fullinertia/geom mass; see `p2_br_io_viewer.md`) to the MJCF docstring and the io docs page. No parser fix.

## B `br_optim` — `src/better_robot/optim/`, `docs/contracts/08-extension-points.md`, `docs/concepts/optimization.md`, `src/better_robot/optim/CLAUDE.md`
1. Solvers write fixed diagnostic keys into `State.extra` (e.g. `grad_norm`, `step_norm`, `accepted`, `damping` for
   LM; document the key set in one place) and `history` exposes them; test per solver.
2. `Result` carries the final solver `state`; `solve`/`run` accept `state=` to warm-start (LM damping, L-BFGS memory);
   test: a warm-started second solve takes fewer iterations than a cold one on a small problem.
3. Warn once (`warnings.warn`, repo style) when a `time=` problem falls back to the dense sink because a term lacks
   `window`; test with `pytest.warns`.
4. Contract 08: name the sink seam (`DenseSink`/`BandSink`/`ArrowSink`, `add`/`finish`) as the extension point for a
   future block-sparse sink; `docs/concepts/optimization.md`: reserve `equality`/`inequality` term kinds (vocabulary
   only, not implemented) and cross-reference the batch-axis rule in `02-tensors.md:81-84`.
5. Fix the two low items from `p2_br_optim.md` if local: L-BFGS `done` from `g_new`; `solve_constrained(outer=0)` →
   `ValueError`.

## C `br_kindyn` — `src/better_robot/{lie,spatial}/`, `src/better_robot/kinematics/_fk_lean.py` and the placement-cache code it names, `src/better_robot/dynamics/_world.py`, `benchmarks/`, `tests/bench/`, `docs/contracts/decisions/`, `docs/contracts/04-differentiability.md`, `docs/concepts/kinematics-and-jacobians.md`
(Agent A edits the public signatures in `kinematics/*.py` and `dynamics/*.py`; you edit only the internal cache path in
`_fk_lean.py:190-225` and `dynamics/_world.py:170-180`. If a line you need is in A's files, report it.)
1. Make the placement-cache path compile-safe: guard the functorch probe and the `_version`/`object.__setattr__`
   cache under `torch.compiler.is_compiling()` (see `p1_kindyn.md` F1 and `p3c_aux_findings.md` item 2); add tests
   that `torch.compile(fullgraph=True)` of FK, RNEA and CRBA on the Panda runs and matches eager (float64, B=2).
2. ADR 0007 "derivatives contract": autograd first with a fixed contract (tangent-space partials, one public entry per
   quantity, analytic implementations as a later drop-in behind the same signature; batch-first target, compile-clean
   as the path to batch-1; Warp stays a reserved option). Link it from contract 04 and `decisions/index.md`.
3. Document the allowed `KinematicState._memo` keys (contract 03 or the state docstring).
4. `so3.to_euler` gimbal-lock branch at pitch = ±π/2 (yaw = 0, combined roll) + test that the round trip is exact
   there; all other inputs bit-identical (assert on a random batch before/after).
5. Benchmarks: a smoke test that imports and runs `benchmarks/run.py` on one tiny config (so the break found in
   `p2_perf.md` cannot recur); make `pytest -m bench` collect the `bench_*.py` files (`python_files` in the pytest
   config, or rename — pick the smaller change) and fix `tests/bench/README.md` (missing `bench_trajopt_sparse.py`,
   note to use `--threads 1` on shared cores).

## D `br_collision` — `src/better_robot/collision/`, `src/better_robot/terms/`, `docs/concepts/collision.md`, `docs/contracts/decisions/0008-*.md`, `docs/contracts/08-extension-points.md` (collision subsection only — agent B edits the optim part; append a separate section)
1. Keyword-only ids/pairs after `(model, state)` in `terms` and `collision` (`pairs` in `self_clearance`,
   `self_distances`, `frame` in terms, …) consistent with agent A's rule; update call sites in src/tests/examples/docs.
2. `self_collision_pairs(..., ignore=None)`: frame-name pairs to drop; test.
3. `RobotSpheres.to(device=, dtype=)`; test batched offsets `(B, N, 3)` explicitly (the review says they work by accident).
4. ADR 0008 "geometry is a separate brick": `collision.RobotGeometry` built from the IR (Pinocchio GeometryModel style),
   spheres and capsules as robot primitives, lazy `trimesh` allowed for mesh → primitive fitting, closest point /
   normal as separate `*_closest` functions (scalar distances stay), contact for dynamics lives in BR (generic);
   `meta["ir"]` geometry schema frozen until it lands. Link from `decisions/index.md` and the collision page.
5. Docs: SDF outside-grid behaviour (zero gradient), batched offsets, and the `distance_fn` protocol as an extension
   point (contract 08, collision section).

## E `br_viewer` — `src/better_robot/viewer/`, `docs/concepts/viewer.md`, `docs/contracts/01-naming.md` (time-axis lines 62-63 only — agent A edits the SE(3) part; coordinate by editing only those lines), `docs/contracts/decisions/0004-viser-viewer.md`
1. Time model prep: optional per-layer `times: (T,) seconds` (default `None` = integer steps at the timeline fps);
   one private `Viewer._index(layer, step) -> int` maps the timeline step to the layer's sample (nearest earlier
   sample when `times` is set); layers keep indexing by the int they receive. Update contract 01 lines 62-63 (time
   axis is `step`; `times` is optional seconds). Tests: a 10 Hz layer and a 30 Hz layer on one timeline.
2. `Viewer.on_pointer(kind, callback)` / `Viewer.clear_pointer()` own viser's single scene-pointer slot (tools go
   through them; `tools/` updated). Test with the fake server.
3. Move BH's `_nearest_on_ray` logic into BR `viewer/tools/picking.py` (read it in
   `/home/rikhat.akizhanov/better/dev/BetterHuman/src/better_human/viz/studio/` — do not edit BH; wave 2 switches BH)
   and `CameraLayer` into its own `layers/camera.py` (re-export where it was).
4. Docs: lazy `step -> array` callable input documented as the convention for every layer (implement for the layers
   where it is a 3-line change; otherwise document which layers accept it); a "Limits" section on
   `docs/concepts/viewer.md` (one integer timeline, viser only, no GUI brick: `viewer.server.gui` is the sanctioned
   studio API, record needs Chromium); ADR 0004 gets a dated note: viser stays the only backend, `server.gui`
   sanctioned, npz export planned.
5. `timeline.py` docstring and any label: "frame" → "step" (naming contract).
