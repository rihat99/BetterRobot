# Legacy — historical record only

These files are **not authoritative** and are excluded from the Sphinx
site. They are kept as a forensic record of decisions taken on the way
to the current shape of BetterRobot. When a legacy doc disagrees with
anything in `docs/source/` or `docs/design/` or `docs/conventions/`,
the live doc wins.

## Contents

| Path | What it is | Why it's legacy |
|------|-----------|-----------------|
| `claude_plan/` | The 17 strategic proposals authored at the end of phase 1 plus their reconciliation notes. All accepted as of 2026-04-25 and folded into the canonical `docs/design/` and `docs/conventions/` specs. | Audit trail — content has been merged into the live specs. |
| `gpt_plan/` | A parallel strategic review by another reviewer. | Audit trail — reconciled into the live specs via `claude_plan/RECONCILIATION.md`. |
| `style/` | Two early coding-style drafts (`style_by_claude.md`, `style_by_gpt.md`). | Superseded by `docs/conventions/19_STYLE.md`. They disagree with the current codebase in several places (notably quaternion ordering). |
| `pypose_issues.md` | Catalogue of correctness bugs in PyPose plus the retirement plan. | PyPose has been retired (P10-D). The pure-PyTorch backend in `lie/_torch_native_backend.py` is the default. |

## Why keep them?

1. They explain *why* the code looks the way it does — useful when you
   open a file and wonder "why this convention and not the other one?"
2. The reconciliation notes (`claude_plan/RECONCILIATION.md`) document
   the trade-offs that were debated and resolved.
3. Removing them would lose the audit trail without making the live
   docs any cleaner.

If you are *using* BetterRobot, you should not need to open this
folder. If you are extending it, the live specs in `docs/design/` and
`docs/conventions/` are what matter.
