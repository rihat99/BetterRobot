# Writers brief (shared by the page-writing agents)

Read `context.md` in this folder first (rules). Then `nav.md` (the page tree and what each page is for) and
`figures.md` (figures and screenshots with captions and their target page). The MkDocs scaffold exists:
`mkdocs.yml`, `docs/index.md` (landing, a worked example you can copy the style of), `docs/reference/api/*.md`
(mkdocstrings), `docs/reference/{glossary,roadmap,changelog}.md`. Your pages are placeholders (a title and a
`TODO(writer)` line) — replace the whole file.

## Hard rules
- Truth comes from the code, not from memory or old docs. Read the module you write about and its tests
  (`tests/<pkg>/`), and the examples (`examples/01_ik.py` … `08_dynamics.py`, `examples/viewer/*`). The old
  Sphinx pages are in `trash/concepts`, `trash/guides`, `trash/getting_started`, `trash/conventions` — use
  them only to recover intent and good explanations; every API name and claim must be re-checked against
  the current code. The `Data` API, `residuals`, `tasks`, `optim2`, Warp do not exist: never mention them
  except in `design-decisions.md` as history.
- Kinematics/dynamics API to use: `forward_kinematics_state(model, q) -> KinematicState`, its fields
  (`joint_transform`, `joint_translation`, `frame_pose_of(id)`, ... read `kinematics/state.py`),
  `kinematics.frame_jacobian(model, state, frame_id, reference=...)`, `better_robot.dynamics.state.*`.
  Another agent is removing the `Data` API concurrently and may add functions to the state API; if something
  you need is missing, say so in the page as a `TODO(api)` HTML comment and in your final message.
- Every ```python block runs: `tests/docs/test_snippets.py` executes the blocks of a page in order in one
  namespace (CPU, fake viser server so `Viewer()` is safe, 120 s per page). Tag a block that must not run
  with ```python no-run (shell/pseudo-code goes in ```bash / ```text). Run `uv run pytest tests/docs -q -k
  <your page stem>` before you finish; the whole file must pass. Robot assets: `better_robot.load` accepts
  paths or the `robot_descriptions` names used in the examples (Panda via `robot_descriptions`; check
  `examples/01_ik.py`); a `ModelBuilder` chain is the dependency-free option.
- Figures: `![caption](../figures/<name>.svg)` / `../images/<name>.png` with the caption from `figures.md`;
  put a figure where the reader first needs it, not at the top as decoration. Use MkDocs Material features
  sparingly: `!!! note` for contracts, `??? note "Details"` for deep dives, `=== "Tab"` for alternatives,
  `$...$` math (arithmatex) where a formula says it better than prose.
- Voice: short sentences, present tense, second person for tutorials. Every page starts with a 2–3 line
  summary of what the reader will be able to do or will understand. Headings are claims or tasks. No
  marketing, no filler, no "in this section we". Define a term the first time it appears or link to the
  glossary (`../reference/glossary.md#term`). Shapes in backticks: `q (B, nq)`.
- Length: tutorials 80–180 lines, concept pages 120–250 lines, guides 60–150 lines. Tables for numbers,
  code for commands and paths.
- Cross-link pages with relative Markdown links; `mkdocs build --strict` fails on a broken link — run
  `uv run mkdocs build --strict` once at the end (it builds the whole site; fix only your pages' warnings,
  report the others).

## Final message (≤ 30 lines)
Pages written with line counts, snippets run (pass/fail), figures used, `TODO(api)` items, claims you could
not verify and left out, warnings from the strict build that are not yours.
