# P0 — Docs tooling and contract references (mars_aux, 2026-10-07 17:25)

Tested in a scratch copy of the site (light+dark screenshots checked). Artifacts copied to `subagents/p0_assets/`:
`guidebook.css`, `demo.md`, `shoot.mjs` (Node 22 drives Chromium over the devtools protocol, no Playwright),
`chrome.sh` (wrapper supplying libasound). Env: pygments 2.20.0, pymdown-extensions 12.1, mkdocs-material 9.7.7,
mkdocs 1.6.1. `use_pygments` on, `linenums` unset (keep), `anchor_linenums` true, arithmatex on (MathJax).

## Part 1 — guidebook docs
**Output pane: custom superfence `output`.** `tests/docs/test_snippets.py:19` matches only column-0 python
fences; fences inside `=== "Code"` tabs or admonitions are silently skipped. Tabs hide output behind a click
(reject); `!!! output` admonition looks like a callout (second choice); custom fence is column-0, never matches
the python regex, styles as a terminal (recommended). Caveat: a code-annotation list must follow its block
directly, so annotated blocks cannot dock an output pane. Optional test guard: every ```output follows a ```python.

mkdocs.yml (tested): palette `primary: custom`, `accent: custom` in both schemes; features add
`content.code.annotate`, `content.tooltips`, `navigation.footer`, `navigation.path`, `toc.follow` (skip
`toc.integrate`); superfences `custom_fences: [{name: output, class: br-output, format:
!!python/name:pymdownx.superfences.fence_code_format}]`; `pymdownx.highlight: {anchor_linenums: true,
line_spans: __span, pygments_lang_class: true}`; `pymdownx.snippets`, `pymdownx.keys`, `pymdownx.emoji`;
`extra_css: [stylesheets/extra.css, stylesheets/guidebook.css]`. Material ignores `pygments_style`; colours come
from `--md-code-hl-*` variables set per scheme in guidebook.css: navy primary `#1f3a68` (light `#2f5597`, dark
`#14284a`), slate `#1a2f55`, bg `#0e1726`; light code IntelliJ-like (kw `#0033b3`, str `#067d17`, num `#1750eb`,
fn `#00627a`, comment `#8c8c8c`); dark Darcula-like (`#cf8e6d`, `#6aab73`, `#2aacb8`, `#56a8f5`, `#7a7e85`, bg
`#1e1f22`). Code 0.82em/1.55 (suggest 0.85em). Output pane: `<pre class="br-output">`, dark bg, navy top border,
"OUTPUT" label via `::before`, `margin-top:-0.6em`, `pre-wrap`; hide its copy button with
`.md-typeset pre.br-output .md-clipboard{display:none}`.

Clips: raw HTML `<figure markdown="span"><video autoplay loop muted playsinline preload="metadata"
poster="../../clips/x.png" width="960"><source src="../../clips/x.webm" type="video/webm"></video>
<figcaption>…</figcaption></figure>` (raw-HTML paths need one more `../` than Markdown links under
`use_directory_urls`; not verified). No `mkdocs-video`. Extend `FIGURE` regex at `test_snippets.py:21` to
`svg|png|jpg|gif|webm|mp4|webp`. Playwright ffmpeg (`~/.cache/ms-playwright/ffmpeg-1011/ffmpeg-linux`) encodes only
VP8 WebM + PNG (no H.264, no GIF/palettegen): `ffmpeg-linux -framerate 24 -i f_%04d.png -vf scale=960:-2 -c:v
libvpx -b:v 0 -crf 30 -deadline good -cpu-used 2 -auto-alt-ref 0 -an clip.webm`. Budget 4–10 s, 960 px, 24 fps,
≤1.5 MB/clip, ≤20 MB site media. GIF only for README via Pillow (in venv through imageio): 720 px, 12–15 fps,
`quantize(128, MEDIANCUT)`, ≤3 MB. Always set `poster=`.

Typography (in guidebook.css): `.md-typeset{font-size:.8rem;line-height:1.7}`, h2 top border + 2.2em margin,
h3 1.6em, h1 700/-.01em; keep extra.css 76ch prose column; fonts Inter / JetBrains Mono via `theme.font`
(Google Fonts — owner's call). Admonitions: note, tip, warning, example only; `abbr` + snippets auto-append
`includes/abbreviations.md` for glossary tooltips; grids/cards for landing + next-steps.

Self-review: `CHROME=subagents/p0_assets/chrome.sh node subagents/p0_assets/shoot.mjs <site_dir> <out_dir>` →
`out/{light,dark}/<page>.png` full-page at 1280 px, 1.5 s wait for MathJax, serves on 127.0.0.1, clears storage
between schemes (else Material reuses cached palette). ~2.5 min for 94 shots. Observed: dark-mode nav tab labels
nearly invisible in screenshots — check by eye.

Other: build prints Material "MkDocs 2.0" notice; `pyproject.toml:36` doesn't pin mkdocs → suggest
`mkdocs>=1.6,<2` (owner's call). Workspace `CLAUDE.md` mentions `docs/conventions/` which doesn't exist on dev;
`docs/contributing/conventions.md` + `docs/concepts/design-decisions.md` overlap the planned contracts → fold in.

## Part 2 — contracts
(i) `docs/contracts/` TOC; rule IDs like `C-TENS-3`, MUST/SHOULD (RFC 2119), tagged [ruff]/[test]/[review]:
00-index (how to read + enforcement matrix); 01-naming (Drake monogram `X_AB`, `p_AoBo_E`, `V_WB`; units,
quaternion/twist order in names; full words outside math); 02-tensors (batch leads `(..., n)`, Array API
broadcasting; preserve dtype/device; shapes in docstrings; no `.item()`/`.cpu()` in library paths);
03-state-mutability (Model frozen; queries return new state; never mutate inputs; caches explicit);
04-differentiability (every public float op differentiable unless documented; torch.func/vmap-safe: no in-place on
inputs, no Python branching on tensor values; singularities have documented safe gradients); 05-api-shape
(functions for math, frozen dataclasses for data, classes only for resources/lifecycles; args after the second
keyword-only; no boolean positionals); 06-validation-errors (validate only at public boundaries; specific types,
messages say expected vs received; no silent fallbacks, no bare except); 07-dependencies-layering (acyclic DAG;
optional deps lazy with install hint; no process-wide torch settings); 08-extension-points (terms plain
functions; joint types/loaders via registries or protocols; public = no `_`; deprecate one minor release);
09-performance (no host sync in loops; batch not loop; benchmarks guard hot paths); 10-docs-tests (numpy
docstrings with shapes; every snippet executes; every rule has an enforcer or is [review]); decisions/ (ADRs).

(ii) Sources (✓ fetched): Drake monogram notation ✓ https://drake.mit.edu/doxygen_cxx/group__multibody__notation.html;
Google pyguide ✓ https://google.github.io/styleguide/pyguide.html; Drake pyguide ✓ https://drake.mit.edu/styleguide/pyguide.html;
Array API broadcasting ✓ https://data-apis.org/array-api/latest/API_specification/broadcasting.html; Kornia
CONTRIBUTING ✓ https://github.com/kornia/kornia/blob/main/CONTRIBUTING.md; pytorch3d batching (nv);
Pinocchio CONTRIBUTING ✓; MuJoCo mjModel/mjData (nv); JAX sharp bits (nv); torch.func limitations ✓
https://docs.pytorch.org/docs/stable/func.ux_limitations.html; Theseus ✓ https://github.com/facebookresearch/theseus;
PyTorch design philosophy ✓ https://docs.pytorch.org/docs/stable/community/design.html; Drake code review checklist ✓
https://drake.mit.edu/code_review_checklist.html; Drake code style ✓ https://drake.mit.edu/code_style_guide.html;
SPEC 1 lazy loading (nv); ruff banned-api ✓ https://docs.astral.sh/ruff/rules/banned-api/; JAX compat policy (nv);
NEP 23 (nv); MuJoCo CONTRIBUTING ✓; mujoco_warp ✓; MADR ✓ https://adr.github.io/madr/; Nygard (nv).

(iii) Enforcement. ruff today `E,F,PLC,PLE,PLR,PLW` (`pyproject.toml:70`). Add: `N` (per-file ignores for math
names), `D` numpy convention with `D1` on public only, `ANN001/ANN201`, `B006/B008`, `FBT001/FBT002`, `BLE001`,
`T201`, `TID251` banned-api (`torch.set_default_dtype/device`, `set_grad_enabled`,
`use_deterministic_algorithms`), `SLF001`, `ERA001`; keep `PLC0415` with per-file ignores for lazy imports.
`tests/contract/`: `test_import_dag.py` (ast-based, replaces import-linter), `test_optional_deps.py`
(`sys.modules["viser"]=None` subprocess), `test_frozen.py`, `test_no_inplace.py` (`tensor._version`),
`test_dtype_device.py`, `test_gradcheck.py`, `test_vmap.py`, `test_public_docstrings.py` (shape line).
Review-only: name meaning, class-vs-function, message quality, ADR need.

(iv) ADRs at `docs/contracts/decisions/NNNN-kebab-title.md` + `decisions/index.md`; seed from
`docs/concepts/design-decisions.md` (torch-only, immutable FK state, terms as plain functions, viser viewer,
nq≠nv). Template: `# NNNN Title` / Status (Proposed|Accepted|Superseded by NNNN) + Date / Context (3–6 lines) /
Decision ("We will …") / Consequences / Alternatives considered / Enforcement (rule IDs, ruff codes,
tests/contract/*). Accepted ADRs never edited except Status; supersede with a new one.
