# PLAN — BetterRobot docs rewrite + legacy removal (2026-10-07_10-52_br_docs_rewrite)

## Owner decisions (intake, 2026-10-07 10:45–10:55 Dubai)
- Docs describe the new design only; **remove all legacy from `dev`**: `optim` (old), `residuals`, `tasks`,
  examples 01–07, the mutable `Data` API (`create_data`, `forward_kinematics(model, data)`, `rnea/crba/aba/ccrba`
  on `Data`), Warp leftovers, `JointMimic` placeholder if unused, legacy aliases. The state API
  (`forward_kinematics_state`, `dynamics.state.*`) is the only API.
- Rename `optim2` -> `optim` (BetterRobot and BetterHuman, both on `dev`).
- Toolchain: MkDocs Material + mkdocstrings (new dependencies, owner approved); Sphinx stack trashed.
- Visuals: SVG schematics (figure skill) + rendered screenshots (`viewer.record`, offline rasterizer) + one
  archify interactive map of the package.
- Public API docstrings rewritten (shapes, contracts, examples); no behaviour change.
- Everything on `dev`; never push or merge; trash, never rm; parity tests ported, never weakened.

## Success criteria
1. `uv run pytest tests -q -m "not bench and not cuda"` green in BR; BH suite green on the renamed import.
2. No module `better_robot.{optim2,residuals,tasks}`; no `Data` class; `grep -rn "optim2\|create_data"` empty in
   src/tests/examples/docs of both repos (except the changelog).
3. `uv run mkdocs build --strict` passes (no broken links/anchors); every code snippet in the docs runs
   (`tests/docs/test_tutorial_snippets.py` ported to the new pages).
4. Pages: Getting started (install, first model, FK, IK, floating base, batched GPU, visualize), Concepts
   (architecture, model/structure/values/state, joints-bodies-frames, Lie & spatial, kinematics & Jacobians,
   dynamics, optimisation: problems/terms/solvers/time, viewer, io/IR), How-to guides, Reference (API via
   mkdocstrings, glossary, changelog, roadmap), Contributing (conventions). Each concept page has at least one
   figure; getting-started pages have rendered screenshots.
5. REPORT.md with the page list, figure list, what was removed (line counts), review findings.

## Stages
A. Code (sequential, main agent + one mars_top):
   A1 trash legacy optim/residuals/tasks + their tests + examples 01–07; promote `examples/optim2/*` to
      `examples/`; fix `__init__`, contract tests, CLAUDE.md guides; rename `optim2` -> `optim` in BR + BH.
   A2 remove the `Data` API: port `tests/test_pinocchio` and the other 40 test files to the state API with the
      same tolerances; delete `data_model/data.py`, `_kinematics_level.py`, `dynamics/{rnea,crba,aba,centroidal}`
      Data entry points (keep the `_raw` kernels the state API uses); drop Warp leftovers, unused `JointMimic`.
   Commit after A1 and after A2 (BR), BH rename commit.
B. Parallel (≤ 6 agents): docstrings (4 agents by package group), MkDocs scaffold + API reference + snippet
   test, figures + screenshots + archify map.
C. Parallel writers: getting started, concepts (2 agents), guides + contributing + reference pages.
D. `mkdocs build --strict`, snippet test, mars_aux review of the docs against the code, REPORT.md, cleanup.
