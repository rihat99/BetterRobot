# BetterRobot — docs

This folder is the Sphinx source tree for the BetterRobot
documentation site. **The published site is the canonical view** —
the markdown files here are written for it, not for direct browsing
in GitHub: [rihat99.github.io/BetterRobot](https://rihat99.github.io/BetterRobot/).

## Build the site

```bash
# from the repo root
uv sync --extra dev
make -C docs html
# open docs/_build/html/index.html

uv run make -C docs serve
```

Or, with plain pip:

```bash
pip install -e .[dev]
sphinx-build -b html docs docs/_build/html
```

To serve it locally on `http://localhost:8000`:

```bash
make -C docs serve
```

`make -C docs strict` builds with warnings-as-errors.
`make -C docs linkcheck` validates external links.

## Publish GitHub Pages

Pages is a generated snapshot on the ``gh-pages`` branch. Publish from a clean,
committed ``dev`` source revision and inspect the generated commit before the
fast-forward push:

```bash
make -C docs strict
git fetch origin gh-pages
# First verify local gh-pages and origin/gh-pages name the expected old commit.
uv run ghp-import -n -m "docs: deploy $(git rev-parse --short HEAD)" \
  -b gh-pages docs/_build/html
git diff --stat origin/gh-pages..gh-pages
git push origin gh-pages:gh-pages
```

If the remote branch moved, fetch and regenerate instead of force-pushing.
The legacy ``make -C docs publish`` target wraps ``ghp-import -p -f`` and
therefore skips this review/fast-forward guard; use it only when that destructive
deployment behavior is explicitly intended.

## Folder map

| Folder | Purpose |
|--------|---------|
| `getting_started/` | Five-minute quickstarts (install → FK → IK → floating base). |
| `guides/` | Task-oriented how-to guides for extending and integrating BetterRobot. |
| `concepts/` | Explanation chapters that walk through every layer of the architecture. |
| `conventions/` | Cross-cutting normative specs (naming, performance, extension, testing, contracts, style, packaging). |
| `reference/` | Auto-generated API, changelog, roadmap, glossary. |
| `CHANGELOG.md` | Engineering changelog. Surfaced via `reference/changelog.md`. |

## Where to start reading

| If you are… | Read |
|-------------|------|
| **Using BetterRobot for the first time** | [`getting_started/`](getting_started/index.md) |
| **Adding a custom residual or provider** | [`guides/`](guides/index.md) |
| **Curious about the design** | [`concepts/`](concepts/index.md) |
| **Writing or reviewing code** | [`conventions/`](conventions/index.md) |
| **Looking up a symbol** | [`reference/`](reference/index.md) |

## Engineering changelog

The full release-by-release changelog lives at
[`CHANGELOG.md`](CHANGELOG.md). It is also rendered as
`reference/changelog` in the site.
