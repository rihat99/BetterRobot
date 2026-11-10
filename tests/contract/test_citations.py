"""Every repository path cited in tracked text resolves, and no line number is cited (principle 7.8).

Citations are found by ``better_robot.testing.dead_citations``: a relative path with a directory and an extension must
be tracked (from the repository root; a Markdown link from its page, a ``mkdocs.yml`` path from ``docs/``), a
``path::symbol`` must name a symbol the Python file defines along its full dotted path, and a ``<path>:<line>``
citation is always dead. The dead citations that exist today sit in ``tests/contract/ratchets/dead_citations.json``
(``"<file>: <citation>"`` to the reason); the ratchet only shrinks, and the fix that resolves a citation removes its
entry. Permanent exceptions: the resolver's and the edge walker's test cases, this module's own exception list, and
citations in accepted ADRs, whose text never changes after acceptance (principle 8.2). A docs page outside the
changelog, the migration pages and the records tells no history: no plan code, no "no longer", no "was moved".
"""

from __future__ import annotations

import json
import re
from fnmatch import fnmatch
from pathlib import Path

from better_robot.testing import dead_citations

REPO = Path(__file__).resolve().parents[2]
CITATION_RATCHET: dict[str, str] = json.loads((Path(__file__).parent / "ratchets" / "dead_citations.json").read_text())
EXEMPT_FILES: dict[str, str] = {
    "tests/testing/test_citations.py": "its cases write dead citations into a tmp_path repository on purpose",
    "tests/testing/test_import_edges.py": "its cases write package files into tmp_path, named relative to the package",
    "tests/contract/test_citations.py": "its exception list below repeats the dead citations it exempts",
}
# Accepted records are never edited (ruling R-3), so their historical citations stay as written, each with the reason.
_PACKAGE_RELATIVE = "written relative to src/better_robot/, as records of that date did"
_BETTERHUMAN = "a BetterHuman path, written relative to BetterHuman/"
FROZEN_RECORD_CITATIONS: dict[str, str] = {
    "docs/contracts/decisions/0001-torch-only.md: tests/contract/test_import_dag.py": "now test_layer_dependencies.py",
    "docs/contracts/decisions/0003-terms-plain-functions.md: tests/contract/test_import_dag.py": (
        "now test_layer_dependencies.py"
    ),
    "docs/contracts/decisions/0004-viser-viewer.md: tests/contract/test_import_dag.py": "now test_layer_dependencies.py",
    "docs/contracts/decisions/0004-viser-viewer.md: tests/contract/test_optional_deps.py": "now test_optional_imports.py",
    "docs/contracts/decisions/0003-terms-plain-functions.md: tests/contract/test_protocols.py": (
        "trashed with the JointModel protocol (0013); C-EXT-1 is review-enforced"
    ),
    "docs/contracts/decisions/0040-model-value-memo.md: src/better_robot/kinematics/_fk_lean.py::placement_matrices": (
        "now src/better_robot/kinematics/_fk.py::placement_matrices"
    ),
    "docs/contracts/decisions/0035-shared-testing-module.md: scripts/docs_screenshots.mjs": (
        "now tools/docs_screenshots.mjs, the one copy both repositories use"
    ),
    "docs/contracts/decisions/0009-model-flat-frozen-record.md: data_model/model.py::Model": (
        "the package was data_model; now src/better_robot/model/model.py::Model"
    ),
    "docs/contracts/decisions/0009-model-flat-frozen-record.md: terms/robot.py": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0011-caches-and-plans.md: dynamics/_world.py::get_dynamics_plan": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0011-caches-and-plans.md: dynamics/state.py": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0011-caches-and-plans.md: kinematics/_fk_lean.py::placement_matrices": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0011-caches-and-plans.md: kinematics/_jacobian_columns.py::get_plan": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0012-one-joint-order.md: core/rig.py": _BETTERHUMAN,
    "docs/contracts/decisions/0012-one-joint-order.md: families/mhr_rig.py": _BETTERHUMAN,
    "docs/contracts/decisions/0012-one-joint-order.md: families/soma_rig.py": _BETTERHUMAN,
    "docs/contracts/decisions/0012-one-joint-order.md: io/build_model.py::build_model": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0013-joint-kinds.md: joint_models/revolute.py::_Revolute": (
        "the JointModel classes 0013 itself replaced"
    ),
    "docs/contracts/decisions/0013-joint-kinds.md: viewer/layers/skeleton.py::_MARKER": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0014-tensor-functions-for-lie-and-spatial.md: dynamics/_world.py": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0014-tensor-functions-for-lie-and-spatial.md: lie/types.py::SE3": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0014-tensor-functions-for-lie-and-spatial.md: spatial/ops.py": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0015-io-entry-points.md: families/mhr_rig.py": _BETTERHUMAN,
    "docs/contracts/decisions/0015-io-entry-points.md: io/build_model.py::build_model": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0016-geometry-out-of-model.md: io/build_model.py::build_model": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0016-geometry-out-of-model.md: viewer/robot.py::robot_meshes": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0021-validation-ladder.md: kinematics/state.py::_validate_q": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0022-viewer-boundary.md: viewer/_math.py::compose_poses": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0022-viewer-boundary.md: viewer/layer.py": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0022-viewer-boundary.md: viewer/robot.py": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0022-viewer-boundary.md: viz/studio/layers.py::_quat_matrix": _BETTERHUMAN,
    "docs/contracts/decisions/0024-surface-trim.md: spatial/ops.py": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0024-surface-trim.md: viewer/app.py": (
        "a BetterRetarget file, named as such in the sentence"
    ),
    "docs/contracts/decisions/0036-mjcf-compile-through-mujoco.md: io/parsers/mjcf.py::parse_mjcf": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0037-urdf-typed-adapter.md: io/parsers/urdf.py": _PACKAGE_RELATIVE,
    "docs/contracts/decisions/0038-remove-downloader-and-registry.md: io/assets.py::CachedDownloadResolver": (
        _PACKAGE_RELATIVE
    ),
}


def _dead() -> set[str]:
    return {
        entry
        for entry in dead_citations(REPO)
        if entry.split(": ", 1)[0] not in EXEMPT_FILES and entry not in FROZEN_RECORD_CITATIONS
    }


def test_cited_paths_resolve() -> None:
    """A new dead citation fails: cite a tracked path or ``path::symbol``, never a line number."""
    new = sorted(_dead() - CITATION_RATCHET.keys())
    assert not new, "cite a tracked path or path::symbol, never a line number:\n  " + "\n  ".join(new)


def test_citation_ratchet_has_no_stale_entries() -> None:
    """A citation that resolves now leaves the citation ratchet in the commit that fixes it."""
    stale = sorted(CITATION_RATCHET.keys() - _dead())
    assert not stale, "remove these entries from tests/contract/ratchets/dead_citations.json:\n  " + "\n  ".join(stale)


HISTORY = re.compile(r"\bP\d+-[A-Z]|\bruling R-\d|(?i:\bno\s+longer\b|\bwas\s+moved\b)")
HISTORY_PAGES = ("docs/reference/changelog.md", "docs/reference/migration-*.md", "docs/contracts/decisions/[0-9]*.md")


def test_docs_pages_tell_no_history() -> None:
    """A docs page says what holds for a reader with no history (principle 7.9); what changed, and the plan code
    that changed it, belong in the changelog, the migration pages and the records."""
    found = []
    for page in sorted((REPO / "docs").rglob("*.md")):
        relative = page.relative_to(REPO).as_posix()
        if not any(fnmatch(relative, pattern) for pattern in HISTORY_PAGES):
            found += [f"{relative}: {' '.join(match[0].split())}" for match in HISTORY.finditer(page.read_text())]
    assert not found, "state what holds now instead:\n  " + "\n  ".join(found)
