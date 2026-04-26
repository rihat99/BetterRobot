# BetterRobot — docs

This folder is the Sphinx source tree for the BetterRobot
documentation site. **The published site is the canonical view** —
the markdown files here are written for it, not for direct browsing in
GitHub.

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

| Folder | Purpose | In the published site? |
|--------|---------|-----------------------|
| `getting_started/` | Five-minute quickstarts (install → FK → IK → floating base). | Yes |
| `concepts/` | Explanations of *why* the architecture is the way it is. | Yes |
| `design/` | Thirteen normative design specs (00–12). | Yes |
| `conventions/` | Cross-cutting normative specs (13–17, 19, 20). | Yes |
| `reference/` | Auto-generated API, changelog, roadmap, glossary. | Yes |
| `status/` | Roadmap source. Surfaced via `reference/roadmap.md`. | Sourced |
| `legacy/` | Audit-trail / historical material. | **No** |
| `CHANGELOG.md` | Engineering changelog. Surfaced via `reference/changelog.md`. | Sourced |
| `UPDATE_PHASES.md` | Implementation phase log P0–P11. Heavily referenced from `src/`. | **No** |

## Where to start reading

| If you are… | Read |
|-------------|------|
| **Using BetterRobot for the first time** | [`getting_started/`](getting_started/index.md) |
| **Curious about the design** | [`concepts/`](concepts/index.md), then [`design/`](design/index.md) |
| **Looking up a symbol** | [`reference/`](reference/index.md) |
| **Reconstructing a past decision** | [`legacy/`](legacy/README.md) |

## Engineering changelog

The full per-phase engineering changelog lives at
[`CHANGELOG.md`](CHANGELOG.md). It is also rendered as
`reference/changelog` in the site.
