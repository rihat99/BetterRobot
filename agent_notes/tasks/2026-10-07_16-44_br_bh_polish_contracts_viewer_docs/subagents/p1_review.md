# P1 adversarial review of docs/contracts (mars_aux, 2026-10-07 18:20) — fix list for P1c

Verdict: right structure, not ready. ~¼ of rules describe code that doesn't exist; DAG rule fails on today's
imports; most enforcers name tests/ruff codes that don't exist or aren't enabled. Line numbers = each page's own.
Orchestrator decisions appended in **[O]**.

## S1 wrong / contradictory
1. 08:9 C-EXT-1 says a term is `(variables) -> Tensor`; ADR 0003:16 and optim/CLAUDE.md:8-9 say argument names
   are the reads and terms are factories returning `TermFn` (terms/robot.py:62 `frame_pose(...) -> TermFn`;
   optim/problem.py:28 TermFn, :46 Term). `pose_term` example doesn't exist; "no Term base class" (08:23)
   ambiguous next to `optim.Term`. Fix: restate as ADR 0003; use `frame_pose` as the example.
2. 07:8-22 DAG lacks terms→optim (terms/{robot,collision,__init__}.py import optim) → DAG test fails day one.
   07:20 "dynamics reuses kinematics helpers by explicit exception" contradicts 07:4 and C-EXT-3 (08:44); real
   edge is private (`dynamics/_world.py:30` → `kinematics._jacobian_columns`). Fix: add terms→optim; rank
   kinematics below dynamics or make the helper public. **[O] make `_jacobian_columns` entry point public-internal
   (`kinematics.jacobian_columns` or re-export) — ledger item; DAG: kinematics rank 5, dynamics rank 5.5/6.**
3. 08:28-36 C-EXT-2 describes `JOINT_TYPES.register` and a loader protocol — neither exists; only
   `io.register_parser(suffix, fn)` (io/__init__.py:42); isinstance checks remain in kinematics/state.py,
   io/build_model.py, io/parsers/programmatic.py, joint_models/composite.py. Fix: mark joint registry [planned,
   needs ADR]; use `register_parser` as the example.
4. 05:51-57 C-API-4 says ids resolve via `model.frame_id(x)` but it takes str only (model.py:195,
   model_structure.py:172); `frame_pose_of(int)` ints only (state.py:146); `state.pose` doesn't exist. Fix: say
   the helper WILL be widened to `int | str` (ledger item); real names.
5. Non-existent paths cited "from this codebase": kinematics/forward.py (02:15, 06:14, 09:20), terms/pose.py
   (07:25, 08:14), optim/solve.py (04:15, 07:27), viewer/server.py (07:41), `_validation.check_configuration`
   (06:16; only `check_tensor` exists). Fix: cite real files (FK = kinematics/state.py:151) or label illustrative.
6. 02:74-75 and 09:9-10 say hot-path lint scans lie, spatial, kinematics, dynamics, optim, terms;
   tests/contract/test_hot_path_lint.py:28-32 scans kinematics, dynamics, lie, optim/solvers.py only.
   `_fk_lean.py:51` `.tolist()` at plan-build time has no exemption. Fix: align lists; exempt once-per-model code.
7. 03:9-10 "every optim result type frozen" — `Phase`, `Result` (optim/phases.py:15,38) are not;
   `KinematicState._memo` mutable (state.py:110) conflicts with C-STATE-4 (03:62) and ADR 0002:17. Fix: freeze
   or exempt by name. **[O] freeze Phase/Result in polish; exempt `_memo` explicitly as a documented memo cache.**
8. 04:10 bans `.detach()`/`no_grad` in library math; dynamics/state_manifold.py:64,72 detach; problem.py:278-303
   detaches (intended solve behaviour). Fix: list documented exceptions; state_manifold needs a section or a fix.

## S2 enforcement missing
9. Tests named but absent: test_dtype_device, test_frozen, test_no_inplace, test_keyword_only, test_import_dag,
   test_optional_deps, tests/autograd/test_vmap.py (04:39). **[O] P1 continuation is writing these.**
10. Ruff codes named but not selected: N (01:23,39), D (10:28), SLF001 (08:47), ANN001 (10:57). Would flag D102
    ×164, ANN001 ×132, N806 ×96, D107 ×35, SLF001 ×31, N803 ×17. "Per-file ignores for math packages" (01:23)
    and "PLC0415 ignored per-file" (07:54) don't exist. TID251 lacks `torch.manual_seed` and thread-count calls
    banned by C-DEP-3 (07:59). Fix: tag `[planned:<ledger id>]`; add the bans.
11. Matrix 00:26-46 omits C-STATE-4, C-DIFF-4, C-API-5, C-ERR-4, C-DEP-4, C-EXT-4, C-PERF-4, C-TEST-1, C-TEST-2
    (breaks C-DOC-3, 10:40). Disagreements: C-DIFF-2 (matrix test_no_inplace vs page test_vmap); C-ERR-2 (matrix
    test_boundary_validation_count vs page test_public_api; real enforcer is tests/test_exceptions_wired.py);
    C-NAME-3 (matrix ruff N + test_naming vs page [review]). test_naming.py only checks old names, not C-NAME-2.
    Fix: a test comparing matrix vs rule IDs on pages; tag missing enforcers `[planned:…]`.

## S3 over-reach and gaps
12. Count naming has three styles: 01:61 `num_steps`, 02:51 `num_frames`, Model `nframes`/`njoints`, viewer
    `n_joints`. **[O] decided: `n<thing>` everywhere (`nsteps`, `njoints`); P3a-2 brief updated.**
13. C-DOC-1 (10:6-29) numpy Parameters/Returns on every public symbol = mass churn (1 file has `Returns` today;
    D → 213 errors). **[O] decided: MUST for new/rewritten code; existing code needs only the shape line.**
14. Gaps: (a) string-wired cache names (`state="state"`) — rule: a missing read raises at Problem build;
    (b) Problem mutability vs `with_` view; (c) batching semantics of optim variables; (d) `Model.to` + cache
    invalidation (`dynamics/_world`); (e) serialization/format versioning of Model and IR; (f) how to add a solver
    (init/step), a viewer layer, a package; (g) Sphinx roles (`:class:` 47×) unaddressed; (h) version scheme
    behind "one minor release" (pre-1.0 policy). **[O] add rules for a, b, c, d, f, g, h; e = ADR "planned".**
15. 06:5 "never about values" vs `check_quaternion_norm` (state.py:151). Fix: opt-in value checks behind a
    keyword flag defaulting off.
16. Readability: "Array API broadcasting" (02:5) unexplained; monogram legend missing "o" (origin) and Jacobian
    prefix (01:10-11); C-DIFF-4 and C-PERF-1 restate other rules under new IDs.
17. Workspace `/home/rikhat.akizhanov/better/CLAUDE.md` points to `docs/conventions/` (doesn't exist). **[O] fix
    in BR task? It's the workspace file, outside both repos — orchestrator edits it (one line).**

Counts: 42 rules + 5 ADRs checked; 11 wrong claims; 7 contradictions; 15 rules with missing/disabled enforcer;
8 gaps.
