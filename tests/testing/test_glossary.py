"""``testing.glossary`` on a small page written to ``tmp_path``, and the build include that ships BetterRobot's page."""

from __future__ import annotations

import tomllib
from pathlib import Path, PurePosixPath

from better_robot.testing import glossary

PAGE = """# Glossary

| Term | Meaning |
|---|---|
| **SE(3), rigid motion** | A pose. |
| **free flyer** | A floating joint. |

## Everyday words

add, root,
skin
"""


def test_read_takes_terms_from_bold_cells_and_everyday_words_from_their_section(tmp_path: Path) -> None:
    page = tmp_path / "glossary.md"
    page.write_text(PAGE)
    assert glossary.read(page) == ({"se3", "rigidmotion", "freeflyer"}, ["add", "root", "skin"])


def test_a_name_is_spelled_with_entries_that_may_span_its_words() -> None:
    vocabulary = {"add", "freeflyer", "root", "joint"}
    assert glossary.words("add_free_flyer_root") == ["add", "free", "flyer", "root"]
    assert glossary.words("IRModel") == ["ir", "model"]
    assert glossary.spelling(glossary.words("add_free_flyer_root"), vocabulary) == ["add", "freeflyer", "root"]
    assert glossary.entry("joints", vocabulary) == "joint"
    assert glossary.spelling(glossary.words("add_free_root"), vocabulary) is None


def test_a_wheel_ships_the_page_where_read_looks() -> None:
    """The build include puts ``docs/reference/glossary.md`` beside the module, where ``read()`` opens it in a wheel
    (a source checkout reads ``docs/`` itself)."""
    pyproject = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    wheel_paths = {source: PurePosixPath(target).as_posix() for source, target in include.items()}
    assert wheel_paths == {"docs/reference/glossary.md": f"{glossary.__name__.replace('.', '/')}.md"}
    assert glossary.read() == glossary.read(Path(__file__).resolve().parents[2] / "docs" / "reference" / "glossary.md")
