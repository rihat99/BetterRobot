"""Four vocabulary rules of chapter 3 over the ``__all__`` tree (C-NAME-2; principles 3.4, 3.5, 3.8, 3.9).

- 3.4: one name per thing. No public class or function is listed under two names, or bound by a listed module under
  a name other than its own (``kinematics.State = KinematicState``, listed or not), and no public class reuses a class
  name of a library ``src`` imports (``LIBRARIES``). ``SAME_ELEMENT`` holds the permanent, reasoned exceptions;
  ``tests/contract/ratchets/vocabulary.json`` the parked ones (``name`` to reason); it only shrinks.
- 3.5: an acronym is written in capitals inside CapWords (``IRModel``, ``LBFGS``), never ``IrModel``.
- 3.8: no module is named ``utils``, ``common``, ``misc`` or ``helpers``.
- 3.9: every word of every public name, method, field and parameter is in ``docs/reference/glossary.md``
  (decision record 0027): a term, the bold first cell of a table row (aliases split at commas, compared in lower
  case without punctuation, so ``SE(3)`` is ``se3`` and ``free flyer`` spells ``free_flyer``), or an everyday word of
  the page's last section. A term may span several words of a name, and a plural word matches its singular.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

from better_robot import kinematics
from better_robot.testing import public_objects

SRC = Path(__file__).resolve().parents[2] / "src" / "better_robot"
GLOSSARY = Path(__file__).resolve().parents[2] / "docs" / "reference" / "glossary.md"
EVERYDAY_SECTION = "## Everyday words"
EVERYDAY_LINE = re.compile(r"[a-z\d]+(?:, [a-z\d]+)*,?")
# A word of an identifier: an acronym before a CapWord (the IR of IRModel), an acronym, a CapWord, a lower-case run.
WORD = re.compile(r"[A-Z]+\d*(?=[A-Z][a-z])|[A-Z]+\d*(?![a-z])|[A-Z][a-z\d]*|[a-z\d]+")
RATCHET: dict[str, str] = json.loads((Path(__file__).parent / "ratchets" / "vocabulary.json").read_text())
# The third-party packages src imports, plus the torch namespaces a better_robot.optim user also meets.
LIBRARIES = ("torch", "torch.optim", "torch.distributions", "numpy", "viser", "trimesh", "mujoco", "yourdfpy", "PIL")
DESCRIPTION_ELEMENT = "the robot-description element of that name, kept as the file format gives it (principle 3.6)"
SAME_ELEMENT: dict[str, str] = {
    "Box": DESCRIPTION_ELEMENT,
    "Cylinder": DESCRIPTION_ELEMENT,
    "Geometry": DESCRIPTION_ELEMENT,
    "Mesh": DESCRIPTION_ELEMENT,
    "Sphere": DESCRIPTION_ELEMENT,
    "Capsule": "MJCF's capsule geom (principle 3.6); torch.Capsule is CPython's PyCapsule type, no robotics concept",
}
ACRONYMS = ("SE3", "SO3", "IR", "IK", "FK", "LM", "URDF", "MJCF", "SDF", "LBFGS", "RNEA", "CRBA", "ABA", "SMPL", "MANO")
BANNED_MODULES = {"utils", "common", "misc", "helpers"}


def _public_classes_and_functions() -> dict[str, object]:
    return {
        path: value
        for path, value in public_objects("better_robot").items()
        if inspect.isclass(value) or inspect.isfunction(value)
    }


def _library_class_names() -> dict[str, str]:
    """Class name to the first library in ``LIBRARIES`` whose namespace lists it."""
    names: dict[str, str] = {}
    for library in LIBRARIES:
        module = importlib.import_module(library)
        for name in dir(module):
            if not name.startswith("_") and inspect.isclass(getattr(module, name, None)):
                names.setdefault(name, library)
    return names


def _reused_names() -> dict[str, str]:
    taken = _library_class_names()
    ours = {value.__name__ for value in _public_classes_and_functions().values() if inspect.isclass(value)}
    return {name: taken[name] for name in ours & taken.keys() if name not in SAME_ELEMENT}


def test_no_public_object_is_listed_under_two_names() -> None:
    names: defaultdict[int, set[str]] = defaultdict(set)
    for path, value in _public_classes_and_functions().items():
        names[id(value)].add(path.rsplit(".", 1)[-1])
    assert sorted(sorted(found) for found in names.values() if len(found) > 1) == []


def _second_names() -> list[str]:
    """``module.name = Object`` for each library class or function a listed module binds under a public name that is
    not the object's own."""
    return sorted(
        f"{path}.{name} = {value.__name__}"
        for path, module in public_objects("better_robot").items()
        if inspect.ismodule(module)
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (inspect.isclass(value) or inspect.isfunction(value))
        and value.__module__.startswith("better_robot")
        and value.__name__ != name
    )


def test_no_module_binds_a_public_object_under_a_second_name() -> None:
    assert _second_names() == []


def test_an_unlisted_alias_in_a_subpackage_is_a_second_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The review's plant: only the root refused a name outside its ``__all__``."""
    monkeypatch.setattr(kinematics, "State", kinematics.KinematicState, raising=False)
    assert _second_names() == ["better_robot.kinematics.State = KinematicState"]


def test_no_public_class_reuses_a_library_class_name() -> None:
    new = sorted(f"{name} (also {library}.{name})" for name, library in _reused_names().items() if name not in RATCHET)
    assert not new, "rename these public classes (principle 3.4):\n  " + "\n  ".join(new)


def test_vocabulary_ratchet_has_no_stale_entries() -> None:
    stale = sorted(RATCHET.keys() - _reused_names().keys())
    assert not stale, "remove these entries from tests/contract/ratchets/vocabulary.json:\n  " + "\n  ".join(stale)


def test_acronyms_are_capitals_inside_class_names() -> None:
    wrong = []
    for value in _public_classes_and_functions().values():
        if not inspect.isclass(value):
            continue
        for acronym in ACRONYMS:
            # An acronym word: starts with a capital and ends at the next capital or the end of the name.
            for match in re.finditer(rf"(?=({acronym}))", value.__name__, flags=re.IGNORECASE):
                start, text = match.start(), match.group(1)
                end = start + len(text)
                at_word = text[0].isupper() and (end == len(value.__name__) or value.__name__[end].isupper())
                if at_word and text != acronym:
                    wrong.append(f"{value.__name__}: write {acronym}")
    assert wrong == []


def test_no_module_has_a_name_that_says_nothing() -> None:
    names = {path.relative_to(SRC).as_posix() for path in SRC.rglob("*") if path.suffix == ".py" or path.is_dir()}
    vague = sorted(path for path in names if Path(path).stem.removeprefix("_") in BANNED_MODULES)
    assert vague == [], "name the module after what it computes (principle 3.8)"


def _public_names() -> dict[str, str]:
    """Every public name, method, field and parameter, mapped to the first public path that shows it."""
    names: dict[str, str] = {}
    for path, value in public_objects("better_robot").items():
        names.setdefault(path.rsplit(".", 1)[-1], path)
        function = value.__func__ if isinstance(value, staticmethod | classmethod) else value
        # A class whose constructor is a builtin's (an exception, a Protocol) has no parameters of its own.
        if inspect.isfunction(function) or (inspect.isclass(function) and inspect.isfunction(function.__init__)):
            for parameter in inspect.signature(function).parameters:
                if parameter not in ("self", "cls") and not parameter.startswith("_"):
                    names.setdefault(parameter, f"{path}({parameter}=)")
    return names


def _glossary() -> tuple[set[str], list[str]]:
    """The page's terms and its everyday words, in the order the page lists them; both lower case."""
    terms: set[str] = set()
    everyday: list[str] = []
    section = ""
    for line in GLOSSARY.read_text().splitlines():
        if line.startswith("#"):
            section = line
        elif line.startswith("| **"):
            cell = line.split("|")[1].strip().removeprefix("**").removesuffix("**")
            terms |= {re.sub(r"[^a-z\d]", "", alias.lower()) for alias in cell.split(",")}
        elif section == EVERYDAY_SECTION and EVERYDAY_LINE.fullmatch(line):
            everyday += line.removesuffix(",").split(", ")
    return terms, everyday


def _words(name: str) -> list[str]:
    """An identifier's words, lower case: ``add_free_flyer_root`` is add free flyer root, ``IRModel`` is ir model."""
    return [word.lower() for word in WORD.findall(name)]


def _entry(word: str, vocabulary: set[str]) -> str | None:
    """The glossary entry that ``word`` is, or is the plural of (``joints``, ``bodies``, ``poses``)."""
    forms = [word]
    for plural, singular in (("ies", "y"), ("es", ""), ("s", "")):
        if word.endswith(plural):
            forms.append(word.removesuffix(plural) + singular)
    return next((form for form in forms if form in vocabulary), None)


def _spelling(words: list[str], vocabulary: set[str]) -> list[str] | None:
    """The glossary entries that spell ``words`` left to right, longest entry first, or ``None`` when none do."""
    if not words:
        return []
    for end in range(len(words), 0, -1):
        entry = _entry("".join(words[:end]), vocabulary)
        rest = _spelling(words[end:], vocabulary) if entry else None
        if rest is not None:
            return [entry, *rest]
    return None


def test_glossary_words() -> None:
    terms, everyday = _glossary()
    vocabulary = terms | set(everyday)
    missing = []
    for name, path in sorted(_public_names().items()):
        words = _words(name)
        if _spelling(words, vocabulary) is None:
            spans = [range(start, end) for start in range(len(words)) for end in range(start + 1, len(words) + 1)]
            known = {i for span in spans if _entry("".join(words[span.start : span.stop]), vocabulary) for i in span}
            missing.append(f"{name} ({path}): " + ", ".join(word for i, word in enumerate(words) if i not in known))
    assert not missing, (
        "define these words in docs/reference/glossary.md, or rename (principle 3.9):\n  " + "\n  ".join(missing)
    )


def test_glossary_lists_each_everyday_word_once_and_only_while_a_name_uses_it() -> None:
    terms, everyday = _glossary()
    vocabulary = terms | set(everyday)
    used = {entry for name in _public_names() for entry in _spelling(_words(name), vocabulary) or ()}
    assert everyday == sorted(set(everyday)), "keep the everyday words sorted, each once"
    assert sorted(set(everyday) & terms) == [], "a word is a term or an everyday word, not both"
    assert sorted(set(everyday) - used) == [], "no public name uses these everyday words; remove them"
