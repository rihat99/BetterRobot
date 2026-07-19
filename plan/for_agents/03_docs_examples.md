# 03 — Documentation style and examples

> **Implementation log (2026-07-19):** Complete: `1583 passed, 2 skipped,
> 16 deselected`; docs/examples `31 passed`; doctest `30/30`; all eight
> examples run headlessly. The honesty sweep also fixed three stale Unreleased
> bullets; strict Sphinx remains blocked by 1,145 baseline cross-reference warnings.

**Goal:** finish the documentation the way the owner reads it: every snippet
*shows* its result instead of asserting it, concept pages that explain a
capability also demonstrate it, and `examples/` covers what the library is
actually for. Runs last, on the settled v2 API. This order restyles and adds;
orders 01/02 already made every page *true*.

**Contract-test authorization:** none. `tests/docs/test_tutorial_snippets.py`
and the doctest harness are ordinary tests and grow with the pages.

---

## T1 — Snippets print; output is shown

House style, applied everywhere: a snippet ends with `print(...)` of the
thing the reader came for, followed by a `{testoutput}` block with the real
output (the Sphinx doctest builder then verifies it). Where output is
device- or environment-dependent, print something deterministic instead
(shapes, comparisons) — see `docs/getting_started/05_batched_gpu.md` for the
pattern (already converted; it shipped broken precisely because an `assert`
compared `torch.device("cuda")` to `cuda:0` and no CPU sandbox ever ran it).
No `torch.testing.assert_close`, no bare `assert`, anywhere in a snippet.

Convert (audit of 2026-07-19):

- `assert_close` sites: `guides/custom_residual.md:49`,
  `guides/own_your_optimization_loop.md:39,53`,
  `concepts/residuals_costs_and_solvers.md:58`.
- `assert`-style verification: `getting_started/installation.md:44-47`,
  `01_robot_model.md:28-32`, `02_forward_kinematics.md:22-24`,
  `03_inverse_kinematics.md:36-38`, `04_floating_base.md:23-25`,
  `guides/differentiate_through_kinematics.md:36-37`,
  `guides/load_a_robot.md:18-19,60-61`, `guides/visualize.md:23`.
- `docs/index.md:56-57` ends on bare expressions that display nothing — end
  the front-page example with prints and show the output. Its extraction
  test accepts only a literal ` ```python ` fence
  (`tests/docs/test_front_page.py:9`) — keep the fence type or update the
  extractor with it.
- Silent `{testcode}` blocks (execute but show nothing):
  `concepts/lie_and_spatial.md:107,141,200`,
  `concepts/residuals_costs_and_solvers.md:89`,
  `conventions/extension.md:39,113` — same print-plus-output conversion.

Harness mechanics to respect while converting: the tutorial harness pins an
exact snippet count per page (`tests/docs/test_tutorial_snippets.py:17`) —
update counts when a page gains a Python fence; the custom-residual guide is
exact-extracted between markers (`tests/optim/slice_support.py:50`) — keep
`{testoutput}` blocks outside the extraction marker or update the extractor.

Keep the prose caution at `concepts/lie_and_spatial.md:170` (it *discusses*
`assert_close`; it is not a snippet).

## T2 — Concept pages demonstrate what they define

Add small executable snippets (print + output, same style) where a page
currently defines a capability in prose or dead `text` blocks only:

- `concepts/dynamics.md` — zero snippets today: one `rnea` gravity-torque
  print and one `crba` mass-matrix-shape print.
- `concepts/kinematics_and_jacobians.md` — promote the `text` blocks at
  `:17-26`, `:36-39` to a real FK + `get_frame_jacobian` snippet printing the
  `(6, nv)` shape.
- `concepts/residuals_costs_and_solvers.md` — the lifecycle,
  `TorchOptimizer`, and implicit-differentiation blocks (`:199-206`,
  `:210-212`, `:250-260`, `:272-276` pre-v2 numbering) become executable
  against the v2 API.
- `concepts/the_compute_seam.md` — the raw-pass examples at `:57-63`,
  `:111-113` become executable.

Every new snippet joins the snippet harness or the doctest suite (whichever
the page already uses). Wrap at 99 chars; MyST directives.

## T3 — Four new examples

`examples/` today is four viewer demos (01, 02, 04, 05 — note the gap). Add:

| File | Scope (one line) |
|---|---|
| `03_batched_ik.py` | ~1000 IK problems in one batched call, CUDA if available; prints convergence rate and wall time |
| `06_differentiable_ik.py` | `solve_ik(..., differentiable=True)`, backward from the solution to the target pose; prints the gradient |
| `07_contact_forces.py` | `solve_contact_forces` on a standing model; prints per-contact wrenches with the at-joint-origin caveat in the header |
| `08_dynamics.py` | RNEA gravity compensation, then ABA recovers the acceleration; prints torques and the round-trip error |

Rules: runnable top-to-bottom with printed, explained output; `--no-viewer`
not needed (these are console examples — no viewer dependency at all);
matching headless smoke tests beside the existing example tests; keep each
under ~80 lines. The four existing viewer examples keep their *behavior* —
their source was already migrated to the v2 API by order 01.

## T4 — Small honesty items

- `docs/CHANGELOG.md`: one line noting the v0.2.0 "public API" list
  (`:110-122`) describes a superseded surface, with a pointer to Unreleased
  and `MIGRATION.md`. History stays; the reader stops being misled.
- Verify the five `MIGRATION.md` links in the changelog resolve (moved from
  `plan/` at repo root, 2026-07-19).

## Acceptance

- `grep -rn "assert" docs/ --include="*.md"` inside fenced snippets → 0 hits
  (write the check as a small script or extend the snippet harness; prose
  mentions excluded).
- Sphinx doctest count grows (every `{testoutput}` verified); snippet harness
  green; full gate green.
- All 8 examples run headless; smoke tests green.
- Results file with per-page disposition.
