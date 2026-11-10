# Release a version

A release is one commit and one tag. The commit raises the version and dates the changelog section, the release
tier passes on it, the tag names it, and BetterHuman pins it. Run the steps in order on a clean checkout of the
commit to release. Version 0.3.0 is the first release with a tag: 0.2.0 (2026-04-11) has none.

## Review the public-API diff

`tools/api_snapshot.py --diff` prints the public paths added (`+`), removed (`-`) and changed (`~`) since an older
snapshot. The older snapshot is the last release's `tests/contract/public_api.json`. For 0.3.0 it is
`tests/contract/public_api_before_0.3.0.json`, the public tree of the last 0.2 commit, since 0.2.0 has no tag:

```bash
# for 0.3.0
uv run python tools/api_snapshot.py --diff tests/contract/public_api_before_0.3.0.json
# for the release after it
git show v0.3.0:tests/contract/public_api.json > /tmp/public_api_0.3.0.json
uv run python tools/api_snapshot.py --diff /tmp/public_api_0.3.0.json
```

Every `-` path is a Removed entry of the changelog that names its replacement, and a row of the migration page;
every `+` path has a docstring and an Added entry. `tests/contract/test_changelog.py` checks the Removed entries
and the 0.3.0 migration page, `tests/contract/test_docstrings.py` the docstrings; a reviewer checks the Added
entries. A minor release may change or remove public names; a patch release changes no public signature (rule
C-EXT-6 in [Public surface](../contracts/public-surface.md)).

## Raise the version and close the changelog section

One commit, `Release: X.Y.Z`, holds three changes:

1. `version` in `pyproject.toml` becomes `0.MINOR.PATCH` (`tests/contract/test_versioning.py`).
2. In `docs/reference/changelog.md`, the `## Unreleased` heading becomes `## vX.Y.Z (YYYY-MM-DD)` with the day of
   the release, and a new, empty `## Unreleased` heading opens the file (`tests/contract/test_changelog.py`).
3. `uv lock` writes the version into `uv.lock`. The release tier starts with `uv lock --check` and the hooks run
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

It stops at its first failure; [Run the tests](run-the-tests.md) lists its steps. It tests the Python and torch of
`uv.lock` only; no tier tests the lower bounds in `pyproject.toml`. Its full tier holds the deprecation rule: this
major keeps no deprecated alias, and `tests/contract/test_versioning.py` fails on a deprecation warning anywhere in
`src` (rule C-EXT-4 in [Extension points](../contracts/08-extension-points.md)). The first deprecation brings a
table of deprecated names with their removal version and a test that fails when that version is reached.

## Tag the release

The owner tags the release commit with the changelog heading's name and pushes the tag:

```bash
git tag -a v0.3.0 -m "BetterRobot 0.3.0"
git push origin v0.3.0
```

## Move the downstream pins

BetterHuman depends on one BetterRobot commit, named under `[tool.uv.sources]` in its `pyproject.toml`:
`better-robot = { git = "https://github.com/rihat99/BetterRobot", rev = "<sha>" }`. After the tag is pushed, set
`rev` to the tagged commit, which `git rev-parse v0.3.0^{commit}` prints, then run `uv lock` and the full tier in
BetterHuman. A pin never names a branch.

## Once per clone

A clone does not install the hooks of `.pre-commit-config.yaml`. Install both stages once; the worktrees of a clone
share its hooks:

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

## Where next

- [Changelog](../reference/changelog.md) and [Migrating to 0.3.0](../reference/migration-0.3.0.md): what each
  release changed, and the replacement of every renamed or removed name.
- [Run the tests](run-the-tests.md): the three tiers and what the full tier needs.
- [Public surface](../contracts/public-surface.md): what a release promises.
