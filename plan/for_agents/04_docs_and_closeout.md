# Order 04 — Docs, changelog, closeout gates

Read `plan/README.md` first. Depends on orders 01–03 being merged. This
order writes the reader-facing story and runs the round's final gates.
Follow `diataxis-docs` and `sphinx-docs`: concepts pages explain and may
carry executable examples; reference stays austere; no quadrant mixing.

## Docs to touch (verified list)

- `docs/concepts/kinematics_and_jacobians.md` — the primary page:
  - §"World, local, and local-world-aligned" (lines ~168-189) is written
    frame-centrically today; add a sentence stating that
    `get_joint_jacobian` now supports all three references too (after
    order 01), with the joint default remaining `"world"`.
  - New section on Jacobian time variation: what J̇ is, the three
    reference frames, and prominently the acceleration pairing (WORLD/
    LOCAL ↔ spatial acceleration, LWA ↔ classical acceleration) with the
    one-line reason (the LWA basis point moves). Include one runnable
    `{testcode}`/`{testoutput}` example (doctest gate runs it): compute
    `J`, `J̇`, and verify `a ≈ J v̇ + J̇ v` numerically for the LWA frame
    against a finite difference — a reader-checkable identity, not a
    print of opaque tensors.
  - Mention transport-only J̇ (constant local subspaces) in one sentence —
    concepts-level why, not implementation detail.
- `docs/concepts/design_decisions.md` — `(decision-jacobian-frame)=`
  section: extend with the J̇ default-frame choice (joints `"world"`,
  frames `"local_world_aligned"`, mirroring the static getters) and the
  single-WORLD-cache decision. Keep it a decision record (why), not a
  how-to.
- `src/better_robot/kinematics/CLAUDE.md` — update "Entry Points" (new
  functions), "Jacobian Reference Frames" (joint LWA now exists; J̇
  acceleration pairing warning), and add a short "Jacobian Hot Path"
  paragraph documenting the batched pass + memoised static artifacts
  (mirroring the "FK Hot Path" section's role).
- `docs/CHANGELOG.md` — one Unreleased bullet covering the round: joint
  LWA, J̇ API (joints + frames, three references), batched Jacobian pass
  with the measured launches/joint numbers from `02_results.md` /
  `03_results.md`.
- `docs/reference/roadmap.md` — update the Jacobian line (~line 74) to
  reflect delivered J̇/LWA.
- `docs/reference/api/` — regenerate autodoc pages for the new public
  symbols (same regen flow the FK round used for `use_compile`).
- **Not** `MIGRATION.md` — the round is purely additive.

## Closeout gates

```bash
UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests/ -q -m "not bench and not cuda"
uv run ruff check src/ tests/
uv run sphinx-build -b html docs docs/_build/html      # strict; doctests must pass
UV_CACHE_DIR=/tmp/betterrobot-uv-cache uv run pytest tests/ -q -m cuda   # on a free GPU, CUDA_VISIBLE_DEVICES pinned
```

Record final counts in `04_results.md` together with the round's headline
bench table (before/after launches per joint and wall-clock for the
Jacobian pass; combined J+J̇ numbers from order 03).

## Out of scope

Any source change beyond docstrings already landed in 01–03. Tutorials/
how-to restructuring. Pushing (owner does that).
