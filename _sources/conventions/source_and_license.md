# Source and License Decision Memo

> **Decision status:** Apache-2.0 selected by the owner on 2026-07-17.
> BetterRobot uses the top-level `LICENSE`; the source ledger below remains
> mandatory for every external adaptation. This is an engineering provenance
> control, not legal advice.

BetterRobot plans to learn from MIT- and Apache-2.0-licensed projects. The
project license governs BetterRobot's original work; copied or modified
third-party material continues to carry its source license and notices. A
license choice therefore does not replace source-by-source provenance work.

## Options

| | Apache License 2.0 | MIT License |
|---|---|---|
| Text and administration | Longer license with explicit redistribution conditions. | Short permission notice; retain it in copies or substantial portions. |
| Patents | Express contributor patent grant plus defensive termination. | No express patent grant. |
| Attribution | Retain relevant copyright, patent, trademark, and attribution notices; mark modified files. If an upstream work ships a `NOTICE`, carry its relevant attributions in a permitted location. | Retain the copyright and permission notice. Third-party licenses still apply to their material. |
| Planned source fit | Matches JAXopt, MuJoCo Warp, and Newton, reducing project-level license mismatch when code is actually adapted. | Matches PyRoki and is operationally simple for BetterRobot-only code. |
| Main consequence | More release bookkeeping, especially for imported Apache material and notices. Choosing Apache-2.0 for new original work does **not by itself** require inventing a `NOTICE`; section 4(d) is triggered when a distributed upstream work includes one. | Apache-derived files cannot simply be relabeled MIT. A distribution may keep MIT for original BetterRobot files while retaining Apache-2.0 terms, change notices, attribution, and any required `NOTICE` material for Apache-derived files. |

Primary texts: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0),
[Apache guidance on applying it](https://www.apache.org/legal/apply-license),
and the [MIT License](https://opensource.org/license/mit).

## Candidate sources named by the roadmap

Repository licenses below were checked on 2026-07-17.

| Project | Verified repository license | Provenance rule before adaptation |
|---------|-----------------------------|----------------------------------------|
| [JAXopt](https://github.com/google/jaxopt) | Apache-2.0 | Open a ledger entry and retain applicable notices before porting. |
| [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) | Apache-2.0 | Open a ledger entry and retain applicable notices before porting. |
| [Newton](https://github.com/newton-physics/newton) | Apache-2.0 | Open a ledger entry and retain applicable notices before porting. |
| [PyRoki](https://github.com/chungmin99/pyroki) | MIT | Open a ledger entry and retain the MIT notice before porting. |

The table identifies planned references; it is not a finding that BetterRobot
currently contains their code. Verify the exact upstream commit and license
again when a ledger entry is opened.

## Required source ledger

Create an entry before adapting an external implementation. Each entry records:

| Field | Required content |
|-------|------------------|
| Source | Project, canonical URL, exact commit/tag, file/function. |
| License | SPDX identifier, link to the license at that revision, upstream copyright holder, and whether `NOTICE` exists. |
| Relationship | `concept-only`, `clean reimplementation`, `copied`, `modified`, or `vendored`; explain the evidence. |
| BetterRobot destination | Every affected file and symbol. |
| Compliance | Retained headers/notices, modified-file marker, bundled license/NOTICE location, and attribution text. |
| Review | Author, reviewer, and date. |

“Same algorithm” is not automatically copied code, but translating structure,
comments, constants, or tests can be a derivative adaptation. When in doubt,
record it as modified/copy-derived and preserve the upstream terms.

## Owner decision record

Recorded on 2026-07-17:

1. Project license: **Apache-2.0**.
2. Copyright: **2026 BetterRobot contributors**.
3. No voluntary project `NOTICE` is created initially. Required upstream
   notices or third-party attributions will be added when a ledger entry calls
   for them.
4. The repository owner reviews the existing tree for provenance before the
   first licensed release.

No candidate-project source was ported during M1. Future adaptation is blocked
until its ledger entry and required license/notice handling are reviewed.
