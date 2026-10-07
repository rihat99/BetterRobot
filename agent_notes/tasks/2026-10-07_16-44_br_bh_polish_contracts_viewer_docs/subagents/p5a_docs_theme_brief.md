# P5a brief — docs theme and tooling (BetterRobot)

Read `context.md`, then `p0_docs_refs.md` Part 1 (tested settings) and the assets in `subagents/p0_assets/`
(`guidebook.css`, `demo.md`, `shoot.mjs`, `chrome.sh`), then `mkdocs.yml`, `docs/stylesheets/extra.css`,
`tests/docs/test_snippets.py`. Repo `/home/rikhat.akizhanov/better/dev/BetterRobot` at `31a37ed`. Scope:
`mkdocs.yml`, `docs/stylesheets/**`, `docs/**/*.md` (fence changes only — no prose rewrites, that is P5b),
`tests/docs/test_snippets.py`, `scripts/docs_screenshots.mjs` (new), `pyproject.toml` (mkdocs pin only).

Owner's words: "not great design, color. Use navy blue as main color. Code snippets hard to read — Python
coloured like in editors, output in a separate window below the code. Better text formatting."

1. Theme: apply the tested settings — `primary: custom`/`accent: custom` both schemes; `theme.font` Inter /
   JetBrains Mono; features add `content.code.annotate`, `content.tooltips`, `navigation.footer`,
   `navigation.path`, `toc.follow`; `pymdownx.highlight` `anchor_linenums`, `line_spans: __span`,
   `pygments_lang_class: true`; `pymdownx.snippets`, `pymdownx.keys`; custom superfence `output` (class
   `br-output`); `extra_css` adds `stylesheets/guidebook.css` (copy from p0_assets, adjust code font to 0.85em,
   hide the copy button on output panes). Pin `mkdocs>=1.6,<2` in pyproject (owner approved).
2. Output panes: for every ```python snippet in `docs/**/*.md` and `README.md` that prints something, add a
   ```output fence directly after it containing the REAL output. Get it by running the snippets:
   `tests/docs/test_snippets.py` executes each page's blocks in order with a fake viser server — extend it with
   an opt-in mode (env `BR_DOCS_CAPTURE=1`) that captures stdout per block and writes it next to the page as
   `<page>.outputs.json` in the scratchpad (not in docs/); then paste the outputs by hand into the fences
   (trim long tensors to the first lines + `...`). Add a test: every ```output fence directly follows a
   ```python fence (same column), and for pages in `docs/getting-started/` the captured stdout of the preceding
   block starts with the fence's first line (a cheap staleness guard; skip blocks tagged `no-run`).
3. Clips: extend the `FIGURE` regex in test_snippets.py to `svg|png|jpg|gif|webm|mp4|webp` and make it also
   check `<source src=...>`/`poster=` paths inside raw HTML `<video>` tags, so a missing clip fails CI.
4. Self-review tooling: `scripts/docs_screenshots.mjs` (from `shoot.mjs`; Node 22 is on this machine; Chromium
   via `$BETTER_ROBOT_CHROME`, same env var as `viewer/capture.py`; document the libasound `LD_LIBRARY_PATH`
   workaround in a comment) → `mkdocs build` + full-page PNGs of every page in light and dark at 1280 px into
   a given out dir. Run it on the themed site into the task folder `vis/p5a/` and LOOK at: index, one
   getting-started page, one concepts page, one contracts page, in both schemes (Read tool). Fix what is wrong
   (contrast, code colours, output pane docking, nav tab labels in dark mode — p0 noted they looked invisible).
5. `uv run mkdocs build --strict` clean; `uv run pytest tests/docs -q` green.

Final message ≤ 25 lines: what changed, number of output fences added, screenshot paths you looked at and what
you fixed, gate tails, anything left. Do not spawn agents. Do not commit. No servers beyond the screenshot
script's own local static server (127.0.0.1).
