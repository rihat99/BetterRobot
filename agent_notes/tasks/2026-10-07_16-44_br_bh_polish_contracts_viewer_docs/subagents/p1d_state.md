# p1d state (dtype/device + no-inplace in BR; BH contracts)

Done:
- BR tests/contract/_public_ops.py (shared registry, 25 entry points), test_dtype_device.py (50), test_no_inplace.py (25).
  `pytest tests/contract`: 679 passed, 10 skipped. ruff tests/contract clean.
- BH docs/contracts/index.md, docs/contracts/bh-addendum.md (H-LAYER-1, H-API-1/2, H-ASSET-1/2, H-STUDIO-1/2,
  H-PARITY-1 + folded contracts.md and design_principles.md). Originals trashed to BH trash/. Links fixed in
  CLAUDE.md, README.md (x2), docs/architecture.md.
- BH tests/contract/test_layer_dependencies.py, test_keyword_only.py (allowlist 11). tests/contract: 5 passed, ruff clean.

Remaining:
- BH full gate (`pytest tests` with BETTERHUMAN_MODELS_DIR) exceeded 600 s; was running in background
  (task ba07uzoqn). Rerun and record tail. `uv run ruff check src tests` tail not yet recorded.
- 00-index "(pending, ledger)" note for C-TENS-1/2, C-STATE-3 not dropped (contract pages owned by another agent).
- BH CLAUDE.md "Layering rule" says core -> families -> terms/viz; it omits `io` (below families). Not edited.
