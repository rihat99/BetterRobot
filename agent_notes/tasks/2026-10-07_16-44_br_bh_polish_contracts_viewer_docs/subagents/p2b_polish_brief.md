# P2b brief — polish the code to the contracts (BetterRobot, non-viewer)

Read `context.md`, then `p1_violations.md` in full (the ledger), `docs/contracts/00-index.md` (rule IDs), and the
pages for the rules you fix. Repo `/home/rikhat.akizhanov/better/dev/BetterRobot`, clean at commit `6ec3134`.
Scope: `src/better_robot/**` EXCEPT `src/better_robot/viewer/**` (another agent), and the matching tests
(`tests/**` except `tests/viewer`). Smallest diff that satisfies each rule; no refactors beyond the rule; no API
additions (an API-sugar list goes to the owner separately — do not implement any).

Work list, in order; after each group run the tests of the touched packages:
1. Ruff findings from the contract rules in `src` (not viewer): FBT001/FBT002 (make the boolean keyword-only or
   drop the positional default), BLE001 (catch the specific exception), ERA001 (delete commented-out code),
   ANN201 (add the return type), F401 (remove). Keep `# noqa` only for intended lazy imports (PLC0415) — do NOT
   add `noqa` to silence the contract rules.
2. Keyword-only (C-API-2) allowlist in `tests/contract/test_keyword_only.py`: fix every non-viewer entry by making
   the arguments after the second keyword-only; update all callers in src/tests/examples/docs (grep each name;
   docs snippets run in CI: `uv run pytest tests/docs -q`). Shrink the allowlist to the viewer entries only.
3. Frozen (C-STATE-1): freeze `optim.Phase` and `optim.Result` (ledger L-5); `Phase.on_start` mutation of
   `problem.data` → return a new data dict or document the exception in the contract page if freezing is
   impossible without redesign (say which).
4. `IRError` → subclass of `BetterRobotError`; remove never-raised `ConvergenceError`, `SingularityWarning`,
   `UnsupportedJointError` from `exceptions.py` (+ their docs mentions: grep docs/).
5. Sphinx roles (L-9): replace `:class:`/`:func:`/`:mod:` roles in `src/**` docstrings with plain backtick code
   spans (≈50 sites, `grep -rn ":class:\|:func:\|:mod:\|:meth:\|:attr:" src`). Viewer files included here ONLY
   for this mechanical docstring change if the other agent has not touched them — skip viewer if unsure.
6. `int | str` ids (L-2): widen `Model.frame_id`/`joint_id`/`body_id` (whichever exist) to accept an int
   (returned unchanged after a range check) and `KinematicState.frame_pose_of(frame)` to `int | str` via
   `model`… — only if `KinematicState` holds a reference to the model/structure; if not, leave L-2 and say why.
7. `kinematics._jacobian_columns` entry point used by `dynamics/_world.py` (L-3): expose it as a documented
   public-internal name (e.g. `kinematics.jacobian_columns` re-export with a docstring) and import that.
8. Everything else in the ledger that is ≤ 10 lines each; list what you leave.

Gates (exact tails): full BR suite, `uv run ruff check src tests` (expect the count to drop; report before/after
by rule), `uv run mkdocs build --strict`, `uv run pytest tests/docs -q`. BetterHuman call sites: if you changed a
signature BH uses (grep `/home/rikhat.akizhanov/better/dev/BetterHuman/src`), list them — do not edit BH.
Update `p1_violations.md`: mark each entry FIXED (commit pending) or LEFT (why). Final message ≤ 25 lines.
Do not spawn agents. Do not commit. If budget runs low, finish the current group and stop with the exact list.
