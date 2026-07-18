# 04 — Polish the core

**Implementation log (2026-07-18):** Complete on `dev`. The user revised the
source budget to 250–310 net lines removed; the final Task-owned delta is
**−282**. The full gate is green (`1443 passed, 2 skipped`), as are parity,
compile/hot-path, lint, type, and documentation checks. See `04_results.md`.

**Goal:** the validation diet and the style pass. After this phase the code
reads like one careful author wrote it: inputs checked once at the boundary
with helpful messages, nothing re-checked below, no duplicated logic, uniform
conventions. The policy is `plan/02_architecture.md` §3–4; this order lists
the evidence-backed targets.

**Size budget (user-revised 2026-07-18):** remove 250–310 net `src/` lines,
with no behavior change (all regression and parity tests untouched and green).

**Contract-test authorization:** `tests/contract/
test_boundary_validation_count.py` only (per T1). No other contract file
changes.

---

## T1 — Stop re-validating frozen values on the hot path

`ModelValues.validate(structure)` runs inside `_validate_q` on **every**
public FK call (`kinematics/forward.py:55`) and again per dynamics call via
`prepare_dynamics_inputs`. Change: validate values **where a
`ModelStructure` is available to check against** — at `Model` construction
and `with_values` (`model.py:224`) — and let public passes trust the
model-attached values, validating only the *call inputs* (`q`, `v`, `a`,
`fext`) once. Honest scope notes: `ModelValues` alone cannot self-validate
(no structure in reach, `model_values.py:63`), and `from_model` / `.to()` /
pytree unflatten construct values without validation
(`model_values.py:116,158,185`) — route the `Model`-attaching paths through
one check; standalone `ModelValues` handed to the raw `*_raw` seam is
caller-trusted by documented contract (that is what "raw" means). Tensors
are only shallowly frozen; in-place mutation after attach is out of contract,
same as `Data` documents today. Update the boundary counter contract
(`tests/contract/test_boundary_validation_count.py`) deliberately: **zero**
`ModelValues.validate` calls per public pass, one input validation per
boundary. Record the change in the results file.

## T2 — Shrink the task-boundary walls

Three walls, one shared toolkit. Build a tiny internal validator helper
(`_validate.py`-style: `check_tensor(name, t, shape=..., dtype=..., device=...)`
returning good messages) and rewrite:

- `tasks/contact_forces.py:261-329` (~68 straight lines of prevalidation,
  `# noqa: PLR0912, PLR0915`) → ≤ 25 lines using the helper.
- `tasks/ik.py:110-141` (`_validate_optimizer_config`) — **keep exactly one
  runtime check per config value**: a `Literal` annotation on a plain
  dataclass enforces nothing at runtime, and invalid-string construction is
  intentionally tested (`tests/tasks/test_ik_config_truth.py:37`). The diet
  here is compression (the helper, table-driven membership checks), not
  deletion; cross-field rules stay. `tasks/ik.py:188-217`
  (`_broadcast_initial_configuration`) gets the helper treatment.
- `data_model/model.py:166-192` (`with_values.checked` closure) → the helper.

Rules while rewriting: error messages keep or improve their specificity
(`"<argument> must <rule>, got <actual>"`); no check is silently dropped —
each is either kept (boundary), moved (to the one boundary), or deleted with
a one-line reason in the results file (typed-invariant re-check /
impossible-by-construction).

## T3 — Delete belt-and-suspenders branches

A branch is deletable only when a **single retained check** provably
excludes it. With T2 keeping the entry-point config validation, the
downstream duplicates become dead: the `else: raise ValueError("Unknown
optimizer")` at `tasks/ik.py:406-407` / `tasks/trajopt.py:322` (same value
already rejected at entry — never delete both ends, only the duplicate), and
the `isinstance(self._kinematics_level, KinematicsLevel)` guard on a field
only ever set to that enum (`data_model/data.py:175-179`). Grep for the
pattern across `tasks/` and `data_model/` and remove what you can prove
dead; when in doubt, keep.

## T4 — One source of truth for joint transforms

`data_model/joint_dispatch.py:44-62` re-implements revolute/prismatic/helical
transform math inline that also lives in the `JointModel` subclasses
(`joint_models/revolute.py:75,104,133` …), with the class method as the
authoritative fallback (line 62). Resolve the duplication without losing the
hot-path property: either the dispatch table calls small shared functions
that the `JointModel` classes also call, or the classes delegate to the
table. One implementation per formula. Verify with the pinocchio parity suite
and the compile/hot-path contract (`tests/contract/` hot-path lint) — the
dispatch must stay compile-friendly.

## T5 — Uniform conventions

- **Result types:** raw passes → frozen `*Result` (done in 03); public
  wrappers document one rule per family: kinematics fills/returns `Data`,
  dynamics returns tensors (and optionally fills `data=`), `ccrba`'s bare
  tuple becomes fields on its result or a named return consistent with its
  siblings.
- **Error voice:** one pattern everywhere —
  `"<what> must <rule>, got <actual>"`. Sweep the messages T2 didn't touch.
- **Naming:** one concept, one name. The known offenders list from the audit:
  raw-pass naming (fixed in 03), `ReferenceFrame` (fixed in 03), residual
  protocol names (fixed in 02). Sweep for stragglers, especially in
  `residuals/` after the protocol collapse.
- **`residuals/` diet:** with the legacy protocol gone (02), re-measure the
  worst validation ratios (`_point_cloud.py` 27%, `structure.py` 27%,
  `_temporal_jacobian.py` 22%, `human.py` 19%, `chamfer.py` 19%) and apply
  the T2 helper + boundary policy. Padded-NaN handling and index-detach
  semantics are load-bearing — do not touch the math.

## T6 — Test hygiene

Where a source rewrite obsoletes a test that pinned implementation details
(exact private call counts, exact error strings you improved), update it to
pin behavior instead. Exact-error-string tests are allowed only for
messages that are themselves contract (the ledger-worthy public ones); use
`match=` fragments elsewhere. Do not reduce coverage of behavior.

## Acceptance

- The three named walls are each ≤ ⅓ of their current line count with
  equal-or-better messages (`git diff --stat` evidence in the results file).
- `ModelValues.validate` has zero call sites in `kinematics/` and
  `dynamics/`; the updated counter contract enforces it stays that way.
- One implementation per joint-transform formula; parity suite green.
- Duplicate helper definitions: none (`grep -rn "def _broadcast_weight\|def _is_inactive\|def _state_coordinates" src | wc -l` ≤ 1 each — locations moved by 02).
- Full gate green; net delta and per-task decisions in the results file.
