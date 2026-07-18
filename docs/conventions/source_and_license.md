# Source and license

BetterRobot is licensed under Apache-2.0. The top-level `LICENSE` file is
the authoritative license text. This page describes the provenance record
required when code or tests are adapted from another project. It is an
engineering procedure, not legal advice.

## What the project license does

Apache-2.0 grants broad copyright and patent permissions subject to its
conditions. It does not erase the license of material taken from elsewhere.
Copied or modified third-party work keeps the notices, attribution, and other
conditions required by its source license.

Read the primary texts when making a licensing decision:

- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [Apache guidance on applying the license](https://www.apache.org/legal/apply-license)
- [MIT License](https://opensource.org/license/mit)

Do not create a project `NOTICE` file merely because BetterRobot uses
Apache-2.0. Add or carry notice material when an included upstream work
requires it.

## Before adapting external material

Create a provenance record before copying, translating, or closely following
an external implementation. Record:

| Field | Required information |
|---|---|
| Source | project, canonical URL, exact revision, and source file or symbol |
| License | SPDX identifier, license text at that revision, copyright holder, and any notice file |
| Relationship | concept-only reference, independent implementation, copied, modified, or vendored |
| Destination | every BetterRobot file and symbol affected |
| Compliance | retained headers, modification notes, bundled license or notice location, and attribution |
| Review | author, reviewer, and review date |

Using the same published algorithm is not automatically copying code. Closely
translating structure, comments, constants, or tests may still be an
adaptation. When the relationship is uncertain, preserve the upstream terms
and ask for review before merging.

## Projects used as design references

The projects below have been discussed as possible references. This table does
not claim that BetterRobot contains their code.

| Project | Repository license | Requirement before adaptation |
|---|---|---|
| [JAXopt](https://github.com/google/jaxopt) | Apache-2.0 | verify the chosen revision and retain applicable notices |
| [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) | Apache-2.0 | verify the chosen revision and retain applicable notices |
| [Newton](https://github.com/newton-physics/newton) | Apache-2.0 | verify the chosen revision and retain applicable notices |
| [PyRoki](https://github.com/chungmin99/pyroki) | MIT | retain the MIT notice for adapted material |

Licenses can change between revisions. Verify the exact source again when the
record is created.

## Review rule

No external implementation is copied into BetterRobot until its provenance
record and required license handling have been reviewed. Keep that record
with the change so a future release can reproduce the attribution decision.
