# P5c brief — BetterHuman docs site in the BetterRobot guidebook style

Repo `/home/rikhat.akizhanov/better/dev/BetterHuman` (branch `dev`, HEAD 81ca789). Reference site:
`/home/rikhat.akizhanov/better/dev/BetterRobot` (`mkdocs.yml`, `docs/stylesheets/guidebook.css`,
`tests/docs/test_snippets.py`, `docs/index.md`, `docs/concepts/architecture.md` as page-shape examples). Read
`context.md` in this folder first, then the BR `mkdocs.yml` + CSS + snippet test, then every file under BH `docs/`
and `README.md`, then the BH source (`src/better_human/`: `__init__.py`, `model.py`/loader, `terms.py`,
`viz/human_layers.py`, `viz/studio/app.py`) so every claim you write is true.

Deliverables (do all; commit nothing; do not spawn agents):
1. `pyproject.toml` dev group: add exactly `"mkdocs>=1.6,<2"`, `"mkdocs-material>=9.5"`, `"mkdocstrings[python]>=0.26"`
   (same pins as BetterRobot); `uv sync --group dev` (UV_CACHE_DIR=/tmp/betterrobot-uv-cache).
2. `mkdocs.yml`: copy BetterRobot's (site_name "BetterHuman", same theme/palette/fonts/extensions/superfence
   `output` block/arithmatex/mkdocstrings for `better_human`), `docs/stylesheets/guidebook.css` copied verbatim.
3. Pages (≤ 160 lines each; lead paragraph → claim headings → code → ```output pane → short explanation → "Where
   next"; `note/tip/warning/example` admonitions only; math only where it carries meaning):
   - `docs/index.md` — what BetterHuman is (SMPL/SMPL-X/MHR bodies as a BetterRobot `Model`), 20-line pitch,
     one load+fk snippet with output, three cards (Getting started / Concepts / Studio).
   - `docs/getting-started.md` — install (`uv sync`, `BETTERHUMAN_MODELS_DIR`), load a model, pose it, run
     `model.fk`, view it in the viewer (the clip/still if present, else the studio screenshots in `docs/images/`).
   - `docs/concepts/architecture.md` — from the existing `docs/architecture.md`: layers on top of BetterRobot,
     what is shared vs. human-specific, the package DAG as enforced by `tests/contract/test_layer_dependencies.py`.
   - `docs/concepts/segmentation.md` — from `docs/segmentation.md` + `guide_render_segmentation.md` (the PNGs in
     `docs/segmentation/` stay and are embedded with captions).
   - `docs/studio.md` — keep the existing page, restyle to the conventions (it is already ≈ right).
   - `docs/contracts/index.md`, `docs/contracts/bh-addendum.md` — keep, make sure they render (relative links).
   - `docs/vision.md` → `docs/reference/roadmap.md` (move with `git mv`).
   Old flat pages replaced by the new ones: `git mv` them into place or trash with
   `python3 ~/.claude/skills/autoresearch/scripts/trash.py <path>` — never `rm`. Keep `docs/render_segmentation.py`
   where the segmentation page references it.
4. `tests/docs/test_snippets.py`: port BetterRobot's (runs every ```python fence in page order, honours
   `{ .no-run }`, checks ```output fences follow python fences, checks referenced images exist). BH models come
   from `BETTERHUMAN_MODELS_DIR=$PWD/models`; snippets that need model files must skip cleanly with a clear reason
   when that env var is unset (that is a real env guard, not a weakened test).
5. Gates, all green before you report: `CUDA_VISIBLE_DEVICES="" BETTERHUMAN_MODELS_DIR=$PWD/models
   UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests/docs -q -p no:cacheprovider`;
   `uv run mkdocs build --strict`; `uv run ruff check src tests`. Then serve on 127.0.0.1:8766 only, screenshot
   index + architecture with BetterRobot's `scripts/docs_screenshots.mjs` (copy it to BH `scripts/`; env
   `BETTER_ROBOT_CHROME=~/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell`,
   `LD_LIBRARY_PATH=/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/libasound/x/usr/lib/x86_64-linux-gnu`),
   LOOK at the PNGs with the Read tool, fix layout problems, stop the server.

Final message ≤ 25 lines: pages written (lines), gate tails, screenshot paths, claims you could not verify, anything
left undone. Budget is tight: finish pages in the order listed; if you run low, say exactly which pages are missing.
