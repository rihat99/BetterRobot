# P5d-2 review — README, guides, contributing, glossary/roadmap, contracts, agent guides (mars_aux, read-only)

14 findings (3 wrong API, 6 wrong concept, 5 stale/style). `~` lines in 00-index.md counted from the table.

WRONG API
1. `src/better_robot/dynamics/CLAUDE.md:9-10` — `inverse_dynamics(model, state, v=v, a=0)` / `v=0, a=0`: ints fail
   in `broadcast_to_execution_batch` (`execution_batch.py:118` reads `.shape`). Fix: zero tensors `(B..., nv)`.
2. `src/better_robot/dynamics/CLAUDE.md:7-8` — memo key is `state._memo["world_geometry"]`, not
   `"world_kinematics"`; inertias are not memoised (`dynamics/state.py:38-46`).
3. `src/better_robot/kinematics/CLAUDE.md:10-11` — `frame_jacobian_time_variation(model, state, v=v, frame=…, *,
   reference=…)`: v, frame/joint, reference are all keyword-only (`kinematics/jacobian.py`).

WRONG CONCEPT
4. `docs/contracts/00-index.md` C-DOC-2 `[review]` and `10-docs-tests.md:37` "`tests/docs/` (planned)" —
   `tests/docs/test_snippets.py` exists and runs every page.
5. `00-index.md` C-DOC-3 `[review]` and `10-docs-tests.md:44` — `tests/contract/test_contract_matrix.py` is the
   C-DOC-3 enforcer and says so.
6. `00-index.md` C-NAME-1, C-NAME-2, C-EXT-3 (SLF001), C-DOC-1 (D), C-TEST-2 (ANN001) show ruff codes without the
   `[planned: L-7]` tag their pages carry; pyproject does not select N/D/ANN001/SLF001. Cause:
   `test_contract_matrix.py:24,32` drops the planned flag for ruff codes.
7. `docs/figures/contracts_map.svg` legend "ruff: lint rule in CI" but C-NAME's ruff codes are all planned; C-EXT
   and C-DOC ruff planned only.
8. `contracts_map.svg` C-PERF "review only" and `00-index.md` C-PERF-3 `[review]` vs `09-performance.md:33`
   "enforced by `tests/bench/`" vs `contributing/testing.md:19` + pyproject "bench: advisory" — pick one story.
9. `docs/contributing/extending.md:47-48` — "suffix has no leading dot and is matched case-insensitively":
   `register_parser` stores the suffix verbatim (`io/__init__.py:44`), only the file suffix is lowercased (:52).

STALE / STYLE
10. `src/better_robot/viewer/CLAUDE.md:7-8` — "`as_frames` promotes…", "Per frame write arrays" → per step
    (code still has `as_frames` in `viewer/_math.py:10`, "Frame counter" in `timeline.py:12`).
11. `docs/reference/glossary.md:129` — "`Viewer.record(...)` renders a clip offline" → screenshots the live scene
    in headless Chromium; also takes `gpu`, `camera`.
12. `docs/reference/glossary.md:74-75` — "Gauss--Newton", "Levenberg--Marquardt" literal double dashes.
13. `docs/contracts/10-docs-tests.md:29-30` — C-DOC-1 lists `tests/contract/test_docstrings.py` twice.
14. `docs/contracts/09-performance.md:6` — no C-PERF-1 (heading redirects to C-TENS-4); index/figure start at
    C-PERF-2 — note the gap.

Verified OK: every dotted `better_robot.*` name in scope resolves; optim signatures, Solver protocol, float64 linear
solves; Result fields + `__getitem__`; kinematics/dynamics signatures apart from 1–3; exception names incl. IRError;
BetterHuman tables in use-with-betterhuman.md; bench `median_ms`/`min_ms`; test_contract_matrix passes; no
ConvergenceError/SingularityWarning/Layer.geometry/`:func:` in scope.

Not reached (grep only): README.md; guides prose (load-a-robot, fit-a-clip, hard-constraints, differentiate,
write-a-term, own-the-loop, benchmark-and-profile); roadmap.md; io/lie/data_model CLAUDE.md prose; top-level
CLAUDE.md DAG vs test_layer_dependencies.
