# BetterRobot — Polish Plan

The redesign that rebuilt this library (seven milestones: correctness fixes, the
compute seam, the block optimizer, parametric models, consumer residuals, sparse
trajectories, GPU groundwork) is implemented and lives in git history. This
folder now holds the **next phase: polish**. The code works — 1,543 tests green,
pinocchio parity intact — but it grew to 26.9k source lines, carries a dead
legacy stack, hides its best building blocks behind facades, and reads in
places like it was written by a committee of cautious machines. This phase
makes it small, open, and elegant.

One sentence of intent: **delete everything without a caller, rebuild the
optimizer around a lean core, open every block to direct use, and rewrite the
docs for humans.**

## Documents

| File | What it is |
|------|------------|
| `01_vision.md` | What BetterRobot is and the principles that govern this phase. |
| `02_architecture.md` | The target state: what the library looks like when polish is done. |
| `03_roadmap.md` | The phases, in order, each with acceptance criteria. |
| `for_agents/` | Execution-ready work orders, one per phase. Start at `for_agents/README.md`. |
| `for_future.md` | Good ideas deliberately not in this phase. The owner picks from here later. |
| `migration_ledger.md` | Removed public symbols, for the owner's downstream migration. Agents append; the owner resolves. |

## Ground rules (bind every phase)

1. **No backward compatibility.** The project is unshipped. Nothing is kept
   "for compatibility"; removed public symbols get one row in
   `migration_ledger.md` and that is the entire obligation. Downstream
   repositories are the owner's business and out of scope for agents.
2. **The safety net stays green.** `tests/test_pinocchio/` (numerical parity)
   and `tests/contract/` (structural rules) are what make aggressive deletion
   safe. They pass after every phase; contract tests change only when a work
   order says so explicitly.
3. **Deletion needs evidence, not courage.** Every removal in these work orders
   is backed by a caller-graph check recorded in the order. If reality on the
   ground differs from the order — a caller appeared, a claim is stale — stop
   and report; do not improvise.
4. **Simplicity is a requirement, not a style.** Each work order carries a size
   budget. Growing past it means the design is wrong, not that the budget was
   optimistic.
