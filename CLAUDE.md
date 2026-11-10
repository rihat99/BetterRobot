# BetterRobot

A PyTorch library for batched, differentiable robot kinematics, dynamics and least squares. This file holds what the
code cannot tell. `AGENTS.md` sends every other agent here; a nested `CLAUDE.md` holds the traps of its package.

## Commands

From the repository root. Times measured on a shared 384-thread AMD EPYC host at load average 6 to 42.

```bash
uv sync --all-extras --all-groups                            # the environment; under a second once installed
uv run pytest tests/contract -q -m "not slow and not cuda"   # fast tier, about 16 s
uv run pytest tests -q -m "not cuda"                         # full tier, about 4.5 min
bash tools/release_tier.sh --gpu <index>                     # release tier, about 10 min, on a GPU nobody else uses
uv run pytest tests/parity -q                                # Pinocchio and MuJoCo parity, about 6 s
uv run pytest tests/docs -q                                  # every docs snippet against its output, about 100 s
uv run mkdocs build --strict                                 # the docs site into site/, about 5 s
uv run ruff check . && uv run ruff format --check .          # lint and format, under a second
uv run python tools/api_snapshot.py --check                  # the public API against its snapshot, about 1.5 s
uv run tools/ratchet_counts.py                               # no ratchet grew, under a second
uv run python benchmarks/run.py --counters-only              # the counter gate, about 50 s
```

While editing a package, run its folder beside the fast tier (`uv run pytest tests/kinematics -q`, about 10 s).
`.pre-commit-config.yaml` runs ruff and the fast tier before a commit and the full tier before a push, once
installed (its header says how). `docs/guides/run-the-tests.md` explains the tiers; `docs/reference/tests.md` says
what each group of tests guards.

## What the tiers need

- Nothing skips: the plugin `better_robot.testing.no_skips` turns every skip and xfail into a failure
  (`tests/contract/test_no_skips.py`). A test that needs a GPU carries the `cuda` marker, which only the release
  tier selects; a missing oracle, fixture or browser fails.
- The full tier needs a headless Chromium for `tests/viewer/test_capture.py` (`chromium`, `google-chrome` or
  `chrome-headless-shell` on `PATH`, or `BETTER_ROBOT_CHROME` pointing at one) and the `robot_descriptions` cache,
  which the docs and example tests fill from the network on first use. Unit, contract and parity tests run offline
  on the tracked robots of `tests/fixtures/robots/` (record 0039).
- `tests/conftest.py` sets `BETTER_ROBOT_INTERNAL_CHECKS=1` unless the environment sets it; `benchmarks/run.py`
  refuses `1`, because the invariants read tensor values. Do not export it in a shell that runs both.

## Gates that must not be weakened

Never skip, xfail, loosen a tolerance or delete an assertion to get green; report the number instead. Gate data (a
tolerance, a counter threshold, the edge table) changes only in a commit of its own that says why.

- Pinocchio parity, `tests/parity/pinocchio/`: 188 tests, 999 compared values, each under its own tolerance
  (`docs/reference/tests.md`); MuJoCo parity, `tests/parity/mujoco/`: 21 tests.
- Counters: graph breaks, recompiles, operators and host syncs of the counter rows never exceed
  `benchmarks/thresholds/torch-2.13.0.json` (rule C-PERF-3, `tests/contract/test_compile_counters.py`).
- Ratchets in `tests/contract/ratchets/` only shrink, and an entry leaves in the commit that fixes it
  (`tools/ratchet_counts.py`, `tests/contract/test_ratchet_history.py`). Open today: ruff 11 sites, pyright 59 errors
  (the `CAP` of `tests/contract/test_pyright_ratchet.py`), keyword_only 6, value_checks 3, vocabulary 3; the other
  five are empty.
- The public API equals `tests/contract/public_api.json` (`tests/contract/test_api_snapshot.py`); an intended change
  is recorded with `uv run python tools/api_snapshot.py --write` and an entry in `docs/reference/changelog.md`.
- `lie` and `spatial` keep 100 % line and branch coverage (the coverage step of `tools/release_tier.sh`).

## Absences, each with its enforcer

- No package imports `viewer`, `terms` or `testing`; each package imports only its row of the edge table, `ALLOWED`
  in `tests/contract/test_layer_dependencies.py` (`docs/concepts/architecture.md` prints it).
- No name joins the root `__all__` beyond its twelve packages and fourteen names; a specialised function stays
  under its package (`tests/contract/test_public_surface.py`).
- `lie`, `spatial` and `optim` import nothing outside the package but torch, `optim` knows nothing about robots, and
  `import better_robot` loads no optional package (`tests/contract/test_optional_imports.py`).
- No host sync and no Python branch on a tensor value in the hot paths (`tests/contract/test_hot_path_lint.py`).
- No `torch.compile` call or kernel builder in `src` (`tests/contract/test_performance_rules.py`); torch is the only
  backend (record 0001).
- No cache beyond `Model._plans`, `Model._memo` and `KinematicState._memo`, no `object.__setattr__`, no read of a
  tensor's `_version` (`tests/contract/test_caches.py`).
- No public kinematics, dynamics, Lie or spatial function mutates its tensor inputs
  (`tests/contract/test_no_inplace.py`); no public dataclass is mutable outside the reasoned list
  (`tests/contract/test_frozen.py`).
- No registry and no `register_*` function: a joint kind, term, parser, layer or solver is passed where it is used
  (`tests/contract/test_extension_points.py`, record 0025).
- No builtin `ValueError`, `TypeError`, `RuntimeError`, `KeyError` or `IndexError` raised in `src` outside the
  reasoned `PLAIN_RAISES`: every error is a class of `better_robot.exceptions` (`tests/contract/test_error_tree.py`).
- No deprecated alias or shim in this major (`tests/contract/test_versioning.py`).
- No module named `utils`, `common`, `misc` or `helpers`, and no word of a public name outside
  `docs/reference/glossary.md` (`tests/contract/test_vocabulary.py`).
- No dead path citation and no line-number citation in tracked text, and no history on a docs page outside the
  changelog, the migration pages and the records (`tests/contract/test_citations.py`).
- No tracked symlink, file over 1 MiB, or directory without a tracked file (`tests/contract/test_repository_layout.py`).
- No agent file over its length and no capitals for emphasis (`tests/contract/test_agent_files.py`).

## Conventions the code does not announce

- Poses are `[tx, ty, tz, qx, qy, qz, qw]` (quaternion scalar last); twists and wrenches are `[linear, angular]`.
  Every layout another repository reads is in `docs/contracts/coupling-standard.md`, which only grows.
- Frame Jacobians default to `reference="local_world_aligned"`; the body-frame form is a rotation of both row blocks,
  never the full adjoint (`src/better_robot/kinematics/CLAUDE.md`).
- `nq` and `nv` differ for quaternion and planar joints: `kinematics.integrate(model, q, v)` and
  `kinematics.difference(model, q0, q1)`, never `q + v` or `q1 - q0`.
- Public tensors carry arbitrary leading batch axes and trailing event axes. Preserve dtype and device and never
  change process-wide torch settings (`tests/contract/test_dtype_device.py`).
- Public functions check each tensor input once at their boundary and never scan values for finiteness; a check
  that reads values sits behind `BETTER_ROBOT_INTERNAL_CHECKS` or an opt-in argument
  (`tests/contract/test_boundary_validation_count.py`, `tests/contract/test_internal_checks.py`).

## Working rules

These come from the principles repository (its README's ten laws and chapters 01-12; chapter 10 governs this
file); no test checks them, a reviewer does.

- Write the least code that solves the problem; a change's net line count per package is a design signal
  (principle 4.1).
- Add an abstraction only when two in-tree callers need it.
- No forwarding layer: a module whose public functions each forward to one other module merges into it (1.2).
- One obvious place per concept (Law 2); prefer a few deep modules to many tiny ones, and a family of modules
  becomes a folder at three (1.2, 3.8).
- Delete only after a search shows no caller; back every claim in a report with its command and output (4.1, 10.5).
- Update the docs and the nearest agent file in the change that changes a contract. A change of a contract or a
  layout gets a decision record (`docs/contracts/00-index.md` says when); an accepted record changes only in its
  Status line (`tests/contract/test_decision_records.py`).
- Before reopening a settled idea, read "Commonly rejected" in `docs/contracts/scope.md`.

## Where things live

- Rules and their enforcers: `docs/contracts/00-index.md`; scope, non-goals and rejected ideas:
  `docs/contracts/scope.md`; the public plan: `docs/reference/roadmap.md`.
- Procedures: `.claude/skills/add-joint-kind/SKILL.md` and `.claude/skills/release/SKILL.md`.
- `tools/` holds the scripts named above, `tools/hot_path_lint.py` (run by its contract test) and
  `tools/docs_screenshots.mjs` (light and dark screenshots of the docs pages).
- `agent_notes/` (one folder per task: plan, state, notes) and `trash/` are ignored whole; nothing tracked cites
  them.
