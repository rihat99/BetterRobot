# 0035 Shared test and docs code lives once, in `better_robot.testing`

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
Byte-identical in both repositories: `scripts/docs_screenshots.mjs`, `docs/stylesheets/extra.css`, `guidebook.css`,
`docs/javascripts/mathjax.js`; `mkdocs.yml` lines 1 to 95 match; `tests/docs/test_snippets.py` differs by 4 hunks. Two
fake viser servers exist, one `_cpu_name` is copied into both benchmark runners, and the two import walkers each miss
edges (`from pkg import subpackage` in BetterHuman, function-level imports in BetterRobot). The rule is that shared code lives once, in the lower library.

## Decision
We will put the stand-in viser server, the snippet runner, the import-edge walker and the citation resolver in one
public BetterRobot module, `better_robot.testing`, which BetterHuman's tests import; move every script to `tools/`
(BetterHuman's screenshot copy is trashed); and keep the three static theme files copied, as a written exception to
the shared-code rule.

## Example
```text
better_robot/testing/     public module; listed in the root __all__ (0019)
    FakeServer            records what a viewer sends (was two diverging copies)
    snippet runner        runs every fenced snippet of a docs tree and compares its output in full
    import walker         walks every import of a package against an edge table, submodule imports resolved
    citation resolver     resolves every path::symbol in tracked text
```
```python { .no-run }
from better_robot.testing import FakeServer      # BetterHuman tests and the studio smoke test (BetterHuman 0009)
```

## Consequences
One implementation of each tool, tested once. BetterHuman's tests depend on a public BetterRobot module, so its
interface follows the compatibility rules. The edge table gets a `testing` row listing only what these tools import.
The three theme files (no behaviour) stay duplicated.

## Alternatives
BetterRobot ships the theme files as package data and BetterHuman's mkdocs build reads them through a hook: a
build-time dependency on installed package data for three small files. A third repository for docs and test tooling:
another repository to pin. PyTorch, NumPy and pandas each ship a public `testing` module for their downstream
users, the precedent used here.

## Compatibility
Adds a public module (compatible). Removes BetterHuman's duplicates. File names inside the module are fixed when it is
written; its four jobs are fixed here.

## Enforcement
`tests/contract/test_public_surface.py` (the module is listed and documented; no second `FakeServer` class in either
repository, its BetterHuman twin `BetterHuman/tests/contract/test_public_surface.py`); the BetterRobot edge table gains
the `testing` row in the same change.

## Open questions
None.
