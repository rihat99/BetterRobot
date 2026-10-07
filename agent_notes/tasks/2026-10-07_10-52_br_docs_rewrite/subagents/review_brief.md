# Review brief — docs vs code (read-only)

Repo `/home/rikhat.akizhanov/better/dev/BetterRobot` (branch `dev`). The docs tree `docs/` (MkDocs Material; nav in
`mkdocs.yml`) was rewritten by several writer agents from the code. Your job: find claims in the docs that the code
does not support. You do not edit anything; you report. Every ```python block already runs in CI
(`tests/docs/test_snippets.py`), so focus on PROSE claims: numbers, defaults, contracts, shapes, behaviours, names.

Ignore: `README.md`, any `CLAUDE.md`, and any mention of the old `Data`/`create_data`/`rnea`/`crba`/`aba` API —
another agent is removing those right now (do not read `src/` files that are mid-edit as truth for THAT topic;
`docs/concepts/design-decisions.md` mentions them as history on purpose).

Method: for each page in `docs/` (31 pages: index, getting-started/*, concepts/*, guides/*, contributing/*,
reference/{glossary,roadmap,changelog}), read it, list each checkable prose claim, and verify against the source
module (`src/better_robot/<pkg>`), its tests (`tests/<pkg>`), or by running a 5-line script with
`CUDA_VISIBLE_DEVICES="" uv run python`. Budget: ~1 verification per claim; prioritise contracts and numbers over
wording.

Writers flagged these as unverified — check them first:
1. `concepts/optimization.md` / `time-and-structure.md`: "LM time grows linearly with the number of knots" (banded
   normal). Verify by timing `examples/02_trajectory.py`-style problems at 10/20/40 knots, or from the solver code.
2. `getting-started/install.md` (or wherever): a cu129 / Blackwell note about torch wheels.
3. `guides/load-a-robot.md`: the error table (which exception for which bad input) — check `src/better_robot/exceptions.py`
   and `io/`.
4. `contributing/testing.md`: Pinocchio parity tolerances by dtype — check `tests/test_pinocchio/*`.
5. `concepts/joints-bodies-frames.md`: `JointComposite` nq/nv, `JointRevoluteUnbounded` (nq=2, nv=1?), and "frame
   placements live only on `model.values`".
6. `concepts/design-decisions.md`: the "why" sections vs `DESIGN.md` (if it exists in the repo or `agent_notes/`);
   the Warp numbers quoted from `agent_notes/tasks/*p2_model_kin_dyn*/REPORT.md`.
7. `guides/hard-constraints.md`: "an infeasible element runs all `outer` rounds and ρ keeps growing" — check
   `src/better_robot/optim/constrained.py`.
8. `concepts/viewer.md`: `record` fits the camera over eight sampled frames; group `offset` applies offline;
   `GroundLayer` follows `Theme.up` — check `src/better_robot/viewer/{viewer.py,render/__init__.py,layers/ground.py}`.

Final message (≤ 40 lines), ranked by severity: `page:line — claim — what the code actually does — evidence
(file:line or script output)`. Then a one-line count: pages read, claims checked, wrong, unverifiable. No style
comments, no suggestions beyond the fix for each wrong claim.

## Second round (split in two reviewers; the first round covered load-a-robot, testing, joints-bodies-frames,
hard-constraints, viewer, design-decisions, time-and-structure, installation, gpu-and-batching — skip those)
- Reviewer B1: index, getting-started/{first-model, floating-base, forward-kinematics, inverse-kinematics, trajectories,
  visualize}, concepts/{architecture, model-and-state, kinematics-and-jacobians, lie-and-spatial}.
- Reviewer B2: concepts/{collision, dynamics, io, optimization}, guides/{benchmark-and-profile, differentiate, fit-a-clip,
  own-the-loop, use-with-betterhuman, write-a-term}, contributing/{conventions, extending}, reference/{glossary, roadmap}.
Budget: 11–13 pages in ~50k tokens — about 3 verifications per page, prose contracts and numbers first. Do not
re-check the numbered list above (done).
