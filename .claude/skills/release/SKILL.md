---
name: release
description: Prepare a BetterRobot release - public-API diff, version, release tier. Use when a task prepares or checks a release; the tag and the downstream pins stay with the owner.
---

# Prepare a BetterRobot release

A release is one commit and one tag. The commit raises the version, the release tier passes on it, the tag names
it, and BetterHuman pins it. Run the steps in order on a clean checkout of the commit to release; each step names the
check that holds it. The tag, the push and the revision BetterHuman pins are the owner's steps. No version has a tag
yet: the first release is 0.1.0, the version `pyproject.toml` already holds, and its tag will be `v0.1.0`.

## Review the public-API diff

`tools/api_snapshot.py --diff` prints the public paths added (`+`), removed (`-`) and changed (`~`) since an older
snapshot. The older snapshot is the last release tag's `tests/contract/public_api.json`; 0.1.0, the first release,
has none to compare with:

```bash
# for the release after 0.1.0
git show v0.1.0:tests/contract/public_api.json > /tmp/public_api_0.1.0.json
uv run python tools/api_snapshot.py --diff /tmp/public_api_0.1.0.json
```

Every `+` path has a docstring (`tests/contract/test_docstrings.py`), and every `-` or `~` path came with a commit
message that names its replacement (rule C-EXT-4 in `docs/contracts/08-extension-points.md`); a reviewer checks the
messages. A minor release may change or remove public names; a patch release changes no public signature (rule
C-EXT-6 in `docs/contracts/public-surface.md`).

## Raise the version

One commit, `Release: X.Y.Z`, holds two changes; 0.1.0 needs neither, since both files already hold it:

1. `version` in `pyproject.toml` becomes `0.MINOR.PATCH` (`tests/contract/test_versioning.py`).
2. `uv lock` writes the version into `uv.lock`. The release tier starts with `uv lock --check` and the hooks run
   through `uv run --locked`; both fail on a lockfile that holds the old version.

```bash
uv lock
uv run python -c "import better_robot; print(better_robot.__version__)"   # now X.Y.Z
```

## Run the release tier

On the release commit, with the `nvidia-smi` index of a GPU nobody else uses:

```bash
bash tools/release_tier.sh --gpu <index>   # about 10 min; ends with "== release tier passed"
```

It stops at its first failure; the script lists its steps. It tests the Python and torch of `uv.lock` only; no tier
tests the lower bounds in `pyproject.toml`. Its full tier holds the deprecation rule: this major keeps no deprecated
alias, and `tests/contract/test_versioning.py` fails on a deprecation warning anywhere in `src` (rule C-EXT-4 in
`docs/contracts/08-extension-points.md`). The first deprecation brings a table of deprecated names with their
removal version and a test that fails when that version is reached.

## Tag the release (the owner)

The owner tags the release commit `vX.Y.Z` and pushes the tag:

```bash
git tag -a v0.1.0 -m "BetterRobot 0.1.0"
git push origin v0.1.0
```

## Move the downstream pins (the owner)

BetterHuman depends on one BetterRobot commit, named under `[tool.uv.sources]` in its `pyproject.toml`:
`better-robot = { git = "https://github.com/rihat99/BetterRobot", rev = "<sha>" }`. After the tag is pushed, set
`rev` to the tagged commit, which `git rev-parse v0.1.0^{commit}` prints, then run `uv lock` and the full tier in
BetterHuman. A pin never names a branch.

## Once per clone

A clone does not install the hooks of `.pre-commit-config.yaml`. Install both stages once; the worktrees of a clone
share its hooks:

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```
