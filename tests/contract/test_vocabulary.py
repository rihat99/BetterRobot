"""Three vocabulary rules of chapter 3 over the ``__all__`` tree (C-NAME-2; principles 3.4, 3.5, 3.8).

- 3.4: one name per thing. No public class or function is listed under two names, and no public class reuses a class
  name of a library ``src`` imports (``LIBRARIES``). ``SAME_ELEMENT`` holds the permanent, reasoned exceptions;
  ``tests/contract/ratchets/vocabulary.json`` the parked ones (``name`` to reason); it only shrinks.
- 3.5: an acronym is written in capitals inside CapWords (``IRModel``, ``LBFGS``), never ``IrModel``.
- 3.8: no module is named ``utils``, ``common``, ``misc`` or ``helpers``.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
from collections import defaultdict
from pathlib import Path

from better_robot.testing import public_objects

SRC = Path(__file__).resolve().parents[2] / "src" / "better_robot"
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
