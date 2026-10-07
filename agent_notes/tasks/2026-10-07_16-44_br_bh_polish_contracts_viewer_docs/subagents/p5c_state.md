# P5c — BetterHuman docs site (mars_top_opus report, saved by the orchestrator)

Committed as BH dev `bad0de3`. Pages: `docs/index.md` (69 lines), `docs/getting-started.md` (82),
`docs/concepts/architecture.md` (160, moved from `docs/architecture.md`, adds shared-vs-human table and the
H-LAYER-1 DAG from the test's ALLOWED table), `docs/concepts/segmentation.md` (130, moved, guide folded in; old
`guide_render_segmentation.md` in `BH/trash/`), `docs/reference/roadmap.md` (moved from `docs/vision.md`),
`docs/studio.md` and `docs/contracts/*` kept as they were. Assets copied from BR: `mkdocs.yml`,
`docs/stylesheets/guidebook.css`, `docs/javascripts/mathjax.js`, `scripts/docs_screenshots.mjs`,
`tests/docs/test_snippets.py`. `site/` gitignored. `pyproject.toml` dev group: mkdocs trio (same pins as BR);
`uv sync --group dev` alone drops viser → sync with `--all-extras`.

Gates: `tests/docs` 22 passed with `BETTERHUMAN_MODELS_DIR` (17 passed + 5 env-guard skips without);
`mkdocs build --strict` clean; ruff clean; full BH suite 332 passed + the known pre-existing reduction failure.
Facts: the human API is `identity.forward(q)` (no `model.fk` on the subject); `model.integrate` used because
nq=99, nv=75. Orchestrator fixed afterwards: README/CLAUDE.md/render_segmentation.py paths to the moved pages,
roadmap link to `../../BetterRobot` (plain text now). Screenshots:
`/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/bh_shots/`.
