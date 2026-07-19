# Plan — optimization API v2, core truth, docs round

The polish phase (six orders, commits `c2236e8`…`b7f02e1`) delivered its
budgets: one optimizer stack, `optim/` at 3,742 lines, `src/` at 21,701, docs
rewritten for readers. Verification (2026-07-19, three independent reviewers +
first-hand checks) confirmed the work is real and the results files honest.

What it did not deliver is the API the owner actually wants to ship. The
current optimization surface still routes everything through a name-keyed
dict: variables are entries in `Values = dict[str, Tensor]`, residuals read
them as `ctx["q"]`, a 22-field `LMState` rides next to the values it
describes, and first-order optimization is a function taking an optimizer
factory. It works — the review found no correctness problems — but it is not
torch-like, and it is not how a user thinks.

This round rebuilds the optimization API around **objects with references**,
taking Theseus (`references/design/theseus.md`) as the sharpest reference
point without copying its mistakes; fixes the truth-and-LEGO gaps the
verification found in the core; and finishes the documentation style the
owner asked for (snippets that print and show their output, more examples).

## The three phases

| Order | What | Acceptance in one line |
|---|---|---|
| `for_agents/01_optim_v2.md` | Rebuild `optim/` around Variables, Residual classes, Weights, and Optimizer/Solver naming; rewire residual library and tasks | the two examples in `01_target_api.md` run as written; `optim/` ≤ 3,800 lines; full gate green |
| `for_agents/02_core_truth.md` | Delete surviving raise-only exports, dead wrappers, and the revolute copy-paste; fix the `ModelValues` cache footgun; add `frame_jacobian_raw` | grep-verifiable deletions; parity green; new raw pass tested |
| `for_agents/03_docs_examples.md` | Convert every snippet to print-plus-output style, add snippets to bare concept pages, add four examples | zero `assert` verification in doc snippets; snippet + doctest suites green |

Run them in order; each ends with the full gate green. `01` is by far the
largest and changes public API; `02` and `03` are bounded cleanups.

## Ground rules (unchanged from the polish phase unless marked new)

1. **No backward compatibility.** The library is unreleased. When a public
   symbol is removed or renamed, append a row to `MIGRATION.md` (repo root)
   and move on. Never keep an alias, a shim, or a deprecation path.
2. **The safety net stays green.** Pinocchio parity, contract tests, and the
   full non-bench/non-CUDA gate pass at the end of every order. Do not weaken
   parity or architecture tests to land a change; each order lists the
   contract files it is authorized to update.
3. **Deletion needs evidence.** Cite the zero-caller grep or the review
   finding before removing anything not named in an order.
4. **Size budgets are hard.** An order that cannot meet its budget stops and
   reports rather than redefining success.
5. **CUDA honesty (new).** This host has working GPUs; the previous
   implementing agent's sandbox did not, and a docs snippet that asserted
   device equality shipped broken because its test silently skipped without
   CUDA. Anything device-conditional must either behave identically on both
   paths or be validated on both. CUDA-marked pytest suites remain owner-run.
6. **Docs stay true per order.** An order that changes an API updates every
   page, docstring, and `CLAUDE.md` it falsifies in the same order. Order 03
   is the enhancement pass, not the cleanup pass.
7. **Honest results files.** Each order writes `for_agents/NN_results.md`:
   what was delivered, what deviated and why, exact test counts, line
   accounting. Deviations are findings, not failures.
8. **Subagents run on Opus.** Codex (read-only) is available for adversarial
   review; treat its taste for abstraction with suspicion and its factual
   findings with respect.
9. **Everything on `dev`.** No feature branches, no worktrees unless asked.
