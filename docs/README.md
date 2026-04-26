# BetterRobot — docs

This folder is the Sphinx source tree for the BetterRobot
documentation site. **The published site is the canonical view** —
the markdown files here are written for it, not for direct browsing
in GitHub.

## Build the site

```bash
# from the repo root
uv sync --extra docs
make -C docs html
# open docs/_build/html/index.html

uv run make -C docs serve
```

Or, with plain pip:

```bash
pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
```

To serve it locally on `http://localhost:8000`:

```bash
make -C docs serve
```

`make -C docs strict` builds with warnings-as-errors (CI uses this).
`make -C docs linkcheck` validates external links.

## Folder map

| Folder | Purpose |
|--------|---------|
| `getting_started/` | Five-minute quickstarts (install → FK → IK → floating base). |
| `concepts/` | Explanation chapters that walk through every layer of the architecture. |
| `conventions/` | Cross-cutting normative specs (naming, performance, extension, testing, contracts, style, packaging). |
| `reference/` | Auto-generated API, changelog, roadmap, glossary. |
| `CHANGELOG.md` | Engineering changelog. Surfaced via `reference/changelog.md`. |

## Where to start reading

| If you are… | Read |
|-------------|------|
| **Using BetterRobot for the first time** | [`getting_started/`](getting_started/index.md) |
| **Curious about the design** | [`concepts/`](concepts/index.md) |
| **Writing or reviewing code** | [`conventions/`](conventions/index.md) |
| **Looking up a symbol** | [`reference/`](reference/index.md) |

## Engineering changelog

The full release-by-release changelog lives at
[`CHANGELOG.md`](CHANGELOG.md). It is also rendered as
`reference/changelog` in the site.
