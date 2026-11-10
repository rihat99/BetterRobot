---
name: release
description: Prepare a BetterRobot release - release tier, public-API diff, changelog section, version. Use when a task prepares or checks a release; the tag and the downstream pins stay with the owner.
---

# Prepare a BetterRobot release

Each step names the check that holds it; run them on a clean checkout of the commit to release.

1. Run the release tier on a GPU nobody else uses: `bash tools/release_tier.sh --gpu <index>` (about 10 min). It
   stops at its first failure; `docs/guides/run-the-tests.md` lists its steps.
2. Diff the public API against the last release's snapshot:
   `uv run python tools/api_snapshot.py --diff <old public_api.json>` (the 0.2 line's is
   `tests/contract/public_api_before_0.3.0.json`). Every removed path has its migration-page row and every removal
   its replacement in the changelog (`tests/contract/test_changelog.py`).
3. In `docs/reference/changelog.md`, the `## Unreleased` section becomes `## vX.Y.Z (YYYY-MM-DD)`, under a new
   `## Unreleased` heading (`tests/contract/test_changelog.py`).
4. Set `version` in `pyproject.toml`, `0.MINOR.PATCH` before 1.0, where a minor release may break
   (`tests/contract/test_versioning.py`, rule C-EXT-6), then `uv lock`: the lockfile records the version, and the
   release tier's `uv lock --check` fails without it.
5. Each change is a commit of its own. The tag, the push and the revision BetterHuman pins are the owner's steps.
