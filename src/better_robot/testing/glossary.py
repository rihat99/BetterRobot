"""A glossary page as data, and the words of a name spelled with its entries (principle 3.9).

BetterRobot's page ``docs/reference/glossary.md`` is the one copy: a wheel ships it beside this module as
``glossary.md`` (the build includes it, ``pyproject.toml``), and a source checkout reads it from ``docs/``. So a
library built on BetterRobot checks its names against the shared words of an installed BetterRobot: ``read()``
parses that page, ``read(page)`` another page of the same form. A page's terms are the bold first cells of its
table rows (aliases split at commas, compared in lower case without punctuation, so ``SE(3)`` is ``se3`` and
``free flyer`` spells ``free_flyer``); its everyday words are the comma-separated lines under its
``## Everyday words`` heading. ``words`` splits a name, ``entry`` finds the entry a word is or is the plural of, and
``spelling`` spells a name's words with entries, where an entry may span several words. Over a package's
``__all__`` tree, ``unknown_words`` lists the public names (methods, fields and parameters included) whose words a
vocabulary cannot spell, and ``used_entries`` the entries that do spell them, so a page lists no word for nothing.
"""

from __future__ import annotations

import inspect
import re
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from .api import public_objects

__all__ = ["entry", "read", "spelling", "unknown_words", "used_entries", "words"]

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
    """An identifier's words, lower case: ``free_flyer_root`` is free flyer root, ``IRModel`` is ir model."""
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


def unknown_words(package: str, vocabulary: set[str]) -> list[str]:
    """``"name (path): word, ..."`` for each public name of ``package`` whose words ``vocabulary`` cannot spell, with
    the words no entry covers, sorted by name; a name is every listed name, method, field and parameter, shown with
    the first public path that has it."""
    missing = []
    for name, path in sorted(_public_names(package).items()):
        name_words = words(name)
        if spelling(name_words, vocabulary) is None:
            spans = [
                range(start, end) for start in range(len(name_words)) for end in range(start + 1, len(name_words) + 1)
            ]
            known = {
                i for span in spans if entry("".join(name_words[span.start : span.stop]), vocabulary) for i in span
            }
            missing.append(
                f"{name} ({path}): " + ", ".join(word for i, word in enumerate(name_words) if i not in known)
            )
    return missing


def used_entries(package: str, vocabulary: set[str]) -> set[str]:
    """The entries of ``vocabulary`` that spell the public names of ``package``; an everyday word outside them is
    listed for nothing."""
    return {found for name in _public_names(package) for found in spelling(words(name), vocabulary) or ()}


def _public_names(package: str) -> dict[str, str]:
    """Every public name, method, field and parameter of ``package``, mapped to the first public path that shows it."""
    names: dict[str, str] = {}
    for path, value in public_objects(package).items():
        names.setdefault(path.rsplit(".", 1)[-1], path)
        function = value.__func__ if isinstance(value, staticmethod | classmethod) else value
        # A class whose constructor is a builtin's (an exception, a Protocol) has no parameters of its own.
        if inspect.isfunction(function) or (inspect.isclass(function) and inspect.isfunction(function.__init__)):
            for parameter in inspect.signature(function).parameters:
                if parameter not in ("self", "cls") and not parameter.startswith("_"):
                    names.setdefault(parameter, f"{path}({parameter}=)")
    return names
