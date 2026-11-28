"""Four vocabulary rules of chapter 3 over the ``__all__`` tree (C-NAME-2; principles 3.4, 3.5, 3.8, 3.9).

- 3.4: one name per thing. No public class or function is listed under two names, or bound by a listed module under
  a name other than its own (``kinematics.State = KinematicState``, listed or not), and no public class reuses a class
  name of a library ``src`` imports (``LIBRARIES``). ``SAME_ELEMENT`` holds the permanent, reasoned exceptions;
  ``tests/contract/ratchets/vocabulary.json`` the parked ones (``name`` to reason); it only shrinks.
- 3.5: an acronym is written in capitals inside CapWords (``IRModel``, ``LBFGS``), never ``IrModel``.
- 3.8: no module is named ``utils``, ``common``, ``misc`` or ``helpers``.
- 3.9: every word of every public name, method, field and parameter is in ``docs/reference/glossary.md``:
  a term or an everyday word as ``better_robot.testing.glossary`` reads the page, which BetterHuman's test reads
  too. A term may span several words of a name, and a plural word matches its singular.

``better_robot.testing.naming_problems`` checks 3.4's two names, 3.5 and 3.8, and ``glossary.unknown_words`` 3.9,
for BetterHuman's test too; a class name reused from a library stays here, with BetterRobot's libraries.
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import pytest

from better_robot import kinematics
from better_robot.testing import glossary, naming_problems, public_objects

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


def _reused_names() -> dict[str, str]:
    """Each public class name a library of ``LIBRARIES`` also lists, mapped to the first such library."""
    taken: dict[str, str] = {}
    for library in LIBRARIES:
        module = importlib.import_module(library)
        for name in dir(module):
            if not name.startswith("_") and inspect.isclass(getattr(module, name, None)):
                taken.setdefault(name, library)
    ours = {value.__name__ for value in public_objects("better_robot").values() if inspect.isclass(value)}
    return {name: taken[name] for name in ours & taken.keys() if name not in SAME_ELEMENT}


def test_naming_rules() -> None:
    """One name per public object, acronyms in capitals inside class names, no module named utils, common, misc or
    helpers."""
    problems = naming_problems("better_robot", acronyms=ACRONYMS)
    assert not problems, "\n".join(problems)


def test_an_unlisted_alias_in_a_subpackage_is_a_second_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A subpackage, like the root, refuses a public name outside its ``__all__``."""
    monkeypatch.setattr(kinematics, "State", kinematics.KinematicState, raising=False)
    assert naming_problems("better_robot", acronyms=ACRONYMS) == [
        "better_robot.kinematics.State = KinematicState: a second name of a public object; remove it (principle 3.4)"
    ]


def test_no_public_class_reuses_a_library_class_name() -> None:
    new = sorted(f"{name} (also {library}.{name})" for name, library in _reused_names().items() if name not in RATCHET)
    assert not new, "rename these public classes (principle 3.4):\n  " + "\n  ".join(new)


def test_vocabulary_ratchet_has_no_stale_entries() -> None:
    stale = sorted(RATCHET.keys() - _reused_names().keys())
    assert not stale, "remove these entries from tests/contract/ratchets/vocabulary.json:\n  " + "\n  ".join(stale)


def test_glossary_words() -> None:
    terms, everyday = glossary.read()
    missing = glossary.unknown_words("better_robot", terms | set(everyday))
    assert not missing, (
        "define these words in docs/reference/glossary.md, or rename (principle 3.9):\n  " + "\n  ".join(missing)
    )


def test_glossary_lists_each_everyday_word_once_and_only_while_a_name_uses_it() -> None:
    terms, everyday = glossary.read()
    used = glossary.used_entries("better_robot", terms | set(everyday))
    assert everyday == sorted(set(everyday)), "keep the everyday words sorted, each once"
    assert sorted(set(everyday) & terms) == [], "a word is a term or an everyday word, not both"
    assert sorted(set(everyday) - used) == [], "no public name uses these everyday words; remove them"
