# P0 synthesis (orchestrator, 2026-10-07 17:35)

Sources: p0_architecture_br.md, p0_bh_studio.md, p0_tests.md, p0_viewer.md, p0_docs_refs.md (+ p0_assets/).

## What the audits agree on
- Core layering and frozen data model are sound and test-enforced. Problems are at the edges: inconsistent
  naming/signatures, string-wired optim, under-used typed errors, Sphinx roles, private leaks across packages.
- The viewer's defects (unconditional Timeline, auto per-layer checkboxes ignoring `set_visible`, no
  viewer-mediated rebuild, custom lights, own rasterizer, isinstance chains) are the direct cause of 3 of the
  studio's S1 defects. Fix the viewer first, then rebuild the studio UI shell on it (keep its domain core).
- Gizmo lag = viser's 32-thread callback pool × 192 torch threads; fix with a latest-wins worker in `PoseGizmo`.
- Headless capture: bare headless Chromium + `client.get_render`; 86 ms/frame on GPU, 478 ms on CPU with viser
  default lights (1577 ms with our shadows → drop them anyway).
- Docs: custom `output` superfence, navy custom palette via CSS vars, Inter/JetBrains Mono, WebM clips
  (Pillow GIF only for README), Node-based page screenshot script for self-review.
- Tests: safe trim ≈ −22 % (128 fns); −40 % would remove the only non-Panda joint coverage.

## Contract families (P1), merged from refs + audits
00 index/enforcement matrix · 01 naming (incl. the frame-vs-step fix, `njoints` style, `is_/has_`) · 02 tensors
(batch-leading, dtype/device, shapes in docstrings, no host sync) · 03 state & mutability (frozen values, queries
return new objects, explicit caches) · 04 differentiability (torch.func-safe, documented singularities) ·
05 API shape (functions for math, frozen dataclasses for data, classes for resources; kw-only after 2nd arg; no
bool positionals; `int | str` ids everywhere) · 06 validation & errors (public boundary only, `BetterRobotError`
tree incl. `IRError`, drop never-raised types) · 07 dependencies & layering (DAG test, lazy optional deps, no
process-wide torch settings) · 08 extension points (terms plain fns, registries/protocols, `_` = private,
deprecation) · 09 performance (bench guards) · 10 docs & tests · decisions/ (ADRs seeded from design-decisions.md).
BH adopts the same set with a short BH addendum (families, assets, studio).

## Owner decisions needed (asked 17:35)
1. Trim depth −22 % vs −40 %. 2. Viewer time axis `frame`→`step`. 3. Drake monogram naming scope.
4. Studio: rebuild UI shell on BR viewer. Small ones taken by me: pin `mkdocs>=1.6,<2`; fonts via Material's
`theme.font` (Google Fonts); recording uses GPU when `gpu_free.py` says FREE else CPU; ffmpeg stays optional for
WebM; Chrome wrapper + libasound handled by a `scripts/chrome.sh`-style helper, documented as a dev requirement.
