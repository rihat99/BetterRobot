# P5d brief — docs-vs-code adversarial review (read-only)

Repo `/home/rikhat.akizhanov/better/dev/BetterRobot` (branch `dev`, HEAD 171509d). The docs under `docs/` were just
rewritten as a guidebook by three writer agents. Your job: find every place where the docs lie about the code.
You cannot edit files; return findings only.

Rules: every ```python block without `{ .no-run }` runs in CI (`tests/docs/test_snippets.py`), so running code is
mostly right — focus on PROSE, TABLES, FIGURES (`docs/figures/*.svg`, read the SVG text labels), CAPTIONS, and
`{ .no-run }` blocks. Check each claim against the source: function names and signatures (keyword-only args after
`*`), module paths, default values, enum names, shapes, return types, which exceptions exist
(`src/better_robot/exceptions.py`), what the viewer API is now (`src/better_robot/viewer/`: `Layer.set_step`,
`nsteps`, `Viewer.record`, `skeleton_spec`, `RobotView` slots), the optim API (`src/better_robot/optim/`: `Problem`,
`CacheRef`, `Result.__getitem__`, `solve/run/solve_layer/solve_constrained`), the contract IDs in
`docs/contracts/*.md`. Also flag: a page that contradicts another page; a figure whose boxes name packages or
classes that do not exist or sit in the wrong package; a caption that does not match the clip (clips in
`docs/clips/`, you can read the `.png` posters); stale terms (`frame`/`nframes` as time axis, `ConvergenceError`,
`SingularityWarning`, `render/` rasterizer, `Layer.geometry`, Sphinx `:func:` roles).

Scope for THIS agent: __SCOPE__

Final message ≤ 40 lines, one finding per line:
`<file>:<line> — <what the doc says> — <what the code says (file:line)> — fix: <one phrase>`. Order by severity
(wrong API > wrong concept > stale name > style). End with a count and the pages you did not get to.
