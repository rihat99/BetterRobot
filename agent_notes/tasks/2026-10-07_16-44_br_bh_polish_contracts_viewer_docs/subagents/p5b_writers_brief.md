# P5b-3/4/5 brief — rewrite the BetterRobot docs pages as a guidebook

Read `context.md`, then this whole file, then `p0_docs_refs.md` Part 1 (c, d: features, admonition palette,
typography) and `p5a` conventions below, then YOUR section's pages and the code they describe. Repo
`/home/rikhat.akizhanov/better/dev/BetterRobot`. Each writer owns ONE section (see split); never edit another
section's pages, `mkdocs.yml`, figures or clips. Final message ≤ 25 lines: pages rewritten (lines before/after),
figures/clips embedded, claims you could not verify, gate tails.

## Owner's words
"Formatting of text should be better. Easy read for users. Logical flow and coherence. More visuals, better
visuals. GIFs, not just images. Like a guidebook: see the code, see some text and understand what's happening
visually. Mathematical formulas where needed. Examples. After reading our new docs I want to fully understand
the overall architecture, high-level ideas etc. Now it's confusing."

## Conventions (P5a already set the theme; follow them exactly)
- Every ```python block runs in CI in page order (`tests/docs/test_snippets.py`, fake viser server). A block
  that must not run: ```` ```python { .no-run } ````. A block that prints gets a ```` ```output ```` fence
  DIRECTLY after it with the real output (run the page: `BR_DOCS_CAPTURE=1 BR_DOCS_CAPTURE_DIR=<scratch> uv run
  pytest tests/docs -q -k <page>` writes `<page>.outputs.json`; trim tensors to ≤ 6 lines + `...`).
  `tests/docs::test_getting_started_outputs_fresh` checks the first line of each pane on getting-started pages.
- Clips: `<figure markdown="span"><video autoplay loop muted playsinline preload="metadata"
  poster="../clips/x.png" width="960"><source src="../clips/x.webm" type="video/webm"></video>
  <figcaption>…</figcaption></figure>` — paths relative to the page as Markdown links would be (test_snippets
  checks they exist). Available clips (`docs/clips/`): panda_trajectory, batched_ik, g1_clip_fit, viewer_layers,
  obstacle_trajectory (+ `.png` posters). Images (`docs/images/`): panda_rest, panda_ik, g1_floating,
  trajectory_strip, banded_sparsity. Figures (`docs/figures/`): architecture, model_values_state,
  joints_bodies_frames, lie_spatial, jacobians, optim_problem, banded_normal, viewer_layers, pipeline_overview,
  ik_loop, contracts_map, time_structure (`![alt](../figures/x.svg){ width=… }` + a one-sentence caption line in
  italics under it). If an asset is missing when you start (agents are producing them), embed it anyway with the
  listed name and say so in the final message.
- Math: `$…$` / `$$…$$` (arithmatex). Use it where it carries meaning (Jacobian, LM step, exp/log, RNEA
  recursion), with symbols defined in one line. No decorative math.
- Page shape: 1-paragraph lead that says what the reader will be able to do after the page; sections whose
  headings are claims ("A `Model` is immutable; queries return a new state"); code → output → one short
  paragraph of what happened; one visual per major idea; a closing "Where next" list of 2–3 links. Admonitions:
  `note`, `tip`, `warning`, `example` only. Tables for numbers and option lists. ≤ 160 lines per page.
- Names from the code as of HEAD: `forward_kinematics_state` / `model.fk(q)`, `state.frame_pose_of("name")`,
  `frame_jacobian(model, state, *, frame=…)`, `inverse_dynamics(model, state, *, v, a)`, `result["q"]`,
  `Problem.cached(...) -> CacheRef`, viewer `set_step`/`nsteps`, `Viewer.record(path, *, steps, size, fps)`,
  `skeleton_spec`, `Layer.listed/visible/world_points`. Grep before you write; the docs are checked against the
  code by a reviewer afterwards.
- Verify by rendering: `uv run mkdocs build --strict` must stay clean; `uv run pytest tests/docs -q -k <your pages>`
  green; after your last edit, screenshot your pages
  (`BETTER_ROBOT_CHROME=<see scripts/docs_screenshots.mjs header> node scripts/docs_screenshots.mjs <out_dir>`
  → look at 2–3 of yours in light mode with the Read tool) and fix layout problems.

## Split
- Writer A (P5b-3): `docs/index.md`, `docs/getting-started/*.md` (8). Index = pipeline_overview figure + a 20-line
  "what BetterRobot is" + the batched-IK snippet + cards to the three sections. Tutorials are a single story:
  load Panda → fk → IK (ik_loop figure, panda_ik image) → floating base (g1_floating) → trajectories
  (panda_trajectory clip, time_structure figure) → GPU/batching (batched_ik clip) → visualize (viewer_layers clip).
- Writer B (P5b-4): `docs/concepts/*.md` (11). Architecture page first: the architecture figure + a walk through
  one call (`model.fk`) across the layers + the contracts_map figure with a link to contracts. Then each concept
  page with its figure (model_values_state, joints_bodies_frames, lie_spatial, jacobians, optim_problem,
  banded_normal + banded_sparsity.png, time_structure, viewer_layers + viewer_layers clip, collision with
  obstacle_trajectory clip, io).
- Writer C (P5b-5): `docs/guides/*.md` (8), `docs/contributing/*.md` (3), `docs/reference/{glossary,roadmap}.md`,
  `README.md` (keep it short: pitch, install, the one IK snippet with output, 3 links; the GIF for GitHub is
  produced by another agent as `docs/images/readme_ik.gif` — embed it). Guides get the g1_clip_fit clip
  (fit-a-clip), obstacle_trajectory (hard-constraints or own-the-loop), and the `scripts/docs_screenshots.mjs`
  + `Viewer.record` mention in benchmark-and-profile only if relevant.

Do not spawn agents. Do not commit. If budget runs low, finish the page you are on and list the rest.
