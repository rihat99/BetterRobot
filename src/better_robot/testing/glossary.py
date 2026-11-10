"""A glossary page as data, and the words of a name spelled with its entries (principle 3.9, decision record 0027).

BetterRobot's page ``docs/reference/glossary.md`` is the one copy: a wheel ships it beside this module as
``glossary.md`` (the build includes it, ``pyproject.toml``), and a source checkout reads it from ``docs/``. So a
library built on BetterRobot checks its names against the shared words of an installed BetterRobot: ``read()``
parses that page, ``read(page)`` another page of the same form. A page's terms are the bold first cells of its
table rows (aliases split at commas, compared in lower case without punctuation, so ``SE(3)`` is ``se3`` and
``free flyer`` spells ``free_flyer``); its everyday words are the comma-separated lines under its
``## Everyday words`` heading. ``words`` splits a name, ``entry`` finds the entry a word is or is the plural of, and
``spelling`` spells a name's words with entries, where an entry may span several words.
"""

from __future__ import annotations

import re
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

__all__ = ["entry", "read", "spelling", "words"]

_EVERYDAY_SECTION = "## Everyday words"
_EVERYDAY_LINE = re.compile(r"[a-z\d]+(?:, [a-z\d]+)*,?")
# A word of an identifier: an acronym before a CapWord (the IR of IRModel), an acronym, a CapWord, a lower-case run.
_WORD = re.compile(r"[A-Z]+\d*(?=[A-Z][a-z])|[A-Z]+\d*(?![a-z])|[A-Z][a-z\d]*|[a-z\d]+")


def read(page: Path | None = None) -> tuple[set[str], list[str]]:
    """The terms and the everyday words of a glossary page, both lower case.

    Parameters
    ----------
    page
        A glossary page in Markdown; BetterRobot's own page when ``None``.

    Returns
    -------
    tuple[set[str], list[str]]
        The terms, without punctuation, and the everyday words in the order the page lists them.
    """
    source = _page() if page is None else page
    terms: set[str] = set()
    everyday: list[str] = []
    section = ""
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            section = line
        elif line.startswith("| **"):
            cell = line.split("|")[1].strip().removeprefix("**").removesuffix("**")
            terms |= {re.sub(r"[^a-z\d]", "", alias.lower()) for alias in cell.split(",")}
        elif section == _EVERYDAY_SECTION and _EVERYDAY_LINE.fullmatch(line):
            everyday += line.removesuffix(",").split(", ")
    return terms, everyday


def _page() -> Traversable:
    """BetterRobot's glossary page: the copy a wheel ships beside this module, else the page of the source checkout
    this module is imported from (an editable install or ``src`` on the path)."""
    packaged = resources.files(__package__).joinpath("glossary.md")
    return (
        packaged if packaged.is_file() else Path(__file__).resolve().parents[3] / "docs" / "reference" / "glossary.md"
    )


def words(name: str) -> list[str]:
    """An identifier's words, lower case: ``add_free_flyer_root`` is add free flyer root, ``IRModel`` is ir model."""
    return [word.lower() for word in _WORD.findall(name)]


def entry(word: str, vocabulary: set[str]) -> str | None:
    """The entry of ``vocabulary`` that ``word`` is, or is the plural of (``joints``, ``bodies``, ``poses``)."""
    forms = [word]
    for plural, singular in (("ies", "y"), ("es", ""), ("s", "")):
        if word.endswith(plural):
            forms.append(word.removesuffix(plural) + singular)
    return next((form for form in forms if form in vocabulary), None)


def spelling(name_words: list[str], vocabulary: set[str]) -> list[str] | None:
    """The entries of ``vocabulary`` that spell ``name_words`` left to right, longest entry first, or ``None`` when
    none do."""
    if not name_words:
        return []
    for end in range(len(name_words), 0, -1):
        found = entry("".join(name_words[:end]), vocabulary)
        rest = None if found is None else spelling(name_words[end:], vocabulary)
        if found is not None and rest is not None:
            return [found, *rest]
    return None
