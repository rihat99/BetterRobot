"""The public API of a package as data: every path its ``__all__`` tree lists, one line describing each, the snapshot
those lines are kept in, and the naming rules the tree follows.

The tree is the package, each name its ``__all__`` lists, the listed modules recursively, and the public members of
each listed class (its own attributes without a leading underscore). A name listed at two paths
(``better_robot.Model`` and ``better_robot.model.Model``) appears under both. The contract sweeps of both libraries
walk ``public_objects``; their snapshot tests compare ``public_api`` with ``tests/contract/public_api.json`` through
``snapshot_differences``, and ``python -m better_robot.testing.api_snapshot`` writes, checks and diffs that file.

``naming_problems`` checks three naming rules of chapter 3 over the tree (3.4, 3.5, 3.8; its docstring lists them).
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
from collections import defaultdict
from collections.abc import Collection
from pathlib import Path
from types import ModuleType

_BANNED_MODULES = {"utils", "common", "misc", "helpers"}


def public_objects(package: str) -> dict[str, object]:
    """Map every dotted path of ``package``'s ``__all__`` tree to the object it names, the package included.

    Parameters
    ----------
    package
        An importable package, ``"better_robot"``. Every listed module needs an ``__all__``.

    Returns
    -------
    dict[str, object]
        Dotted path to object, in walk order: a module before what it lists, a class before its members.
    """
    module = importlib.import_module(package)
    found: dict[str, object] = {package: module}
    _walk(module, package, found)
    return found


def public_api(package: str) -> dict[str, str]:
    """Describe every path of ``package``'s ``__all__`` tree in one line: the snapshot the API tools compare.

    Parameters
    ----------
    package
        An importable package, ``"better_robot"``.

    Returns
    -------
    dict[str, str]
        Dotted path to ``"module"``, ``"class(<bases>)<constructor signature>"`` (no signature when the
        constructor is a builtin's), ``"function(<signature>)"`` (a method's signature starts at ``self``),
        ``"staticmethod(...)"``, ``"classmethod(...)"``, ``"property"`` or ``"value: <type name>"``.
    """
    return {path: _describe(value) for path, value in public_objects(package).items()}


def snapshot_differences(package: str, snapshot: Path) -> list[str]:
    """Lines ``- path: old``, ``+ path: new`` and ``~ path: old -> new``, sorted by path, from the ``snapshot`` file
    (a ``public_api`` written as JSON) to ``package`` as it is now; empty when they match."""
    old, new = json.loads(snapshot.read_text()), public_api(package)
    lines = [f"- {path}: {old[path]}" for path in old.keys() - new.keys()]
    lines += [f"+ {path}: {new[path]}" for path in new.keys() - old.keys()]
    lines += [f"~ {path}: {old[path]} -> {new[path]}" for path in old.keys() & new.keys() if old[path] != new[path]]
    return sorted(lines, key=lambda line: line[2:])


def naming_problems(package: str, *, acronyms: Collection[str], ignore: Collection[str] = ()) -> list[str]:
    """One line per break of three naming rules in ``package``'s ``__all__`` tree, each with its fix.

    3.4, one name per thing: no public class or function is listed under two names, or bound by a listed module
    under a name other than its own (``kinematics.State = KinematicState``, listed or not). 3.5: each of
    ``acronyms`` is written in capitals inside a class name (``IRModel``, never ``IrModel``). 3.8: no module of the
    package is named ``utils``, ``common``, ``misc`` or ``helpers``. A path whose last name is in ``ignore`` is not a
    second name: a field whose value is a class, such as a format's ``classic_type``.
    """
    tree = public_objects(package)
    listed = {
        path: value
        for path, value in tree.items()
        if (inspect.isclass(value) or inspect.isfunction(value)) and path.rsplit(".", 1)[-1] not in ignore
    }
    names: defaultdict[int, set[str]] = defaultdict(set)
    for path, value in listed.items():
        names[id(value)].add(path.rsplit(".", 1)[-1])
    problems = [
        f"{' and '.join(sorted(found))}: one object, two public names; keep one (principle 3.4)"
        for found in names.values()
        if len(found) > 1
    ]
    problems += sorted(
        f"{path}.{name} = {value.__name__}: a second name of a public object; remove it (principle 3.4)"
        for path, module in tree.items()
        if inspect.ismodule(module)
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (inspect.isclass(value) or inspect.isfunction(value))
        and value.__module__.startswith(package)
        and value.__name__ != name
    )
    for value in {id(value): value for value in listed.values()}.values():
        for acronym in acronyms if inspect.isclass(value) else ():
            # An acronym word: starts with a capital and ends at the next capital or the end of the name.
            for match in re.finditer(rf"(?=({acronym}))", value.__name__, flags=re.IGNORECASE):
                end, text = match.start() + len(match[1]), match[1]
                if (
                    text[0].isupper()
                    and (end == len(value.__name__) or value.__name__[end].isupper())
                    and text != acronym
                ):
                    problems.append(f"{value.__name__}: write {acronym} (principle 3.5)")
    source = Path(inspect.getfile(importlib.import_module(package))).parent
    problems += sorted(
        f"{path.relative_to(source).as_posix()}: name the module after what it computes (principle 3.8)"
        for path in source.rglob("*")
        if (path.suffix == ".py" or path.is_dir()) and path.stem.removeprefix("_") in _BANNED_MODULES
    )
    return problems


def _walk(module: ModuleType, prefix: str, found: dict[str, object]) -> None:
    for name in module.__all__:
        path, value = f"{prefix}.{name}", getattr(module, name)
        found[path] = value
        if inspect.ismodule(value):
            _walk(value, path, found)
        elif inspect.isclass(value):
            found.update({f"{path}.{member}": item for member, item in vars(value).items() if member[0] != "_"})


def _describe(value: object) -> str:
    if inspect.ismodule(value):
        return "module"
    if isinstance(value, property):
        return "property"
    if isinstance(value, staticmethod | classmethod):
        return f"{type(value).__name__}{inspect.signature(value.__func__)}"
    if inspect.isclass(value):
        bases = ", ".join(base.__qualname__ for base in value.__bases__)
        # A class whose constructor is a builtin's (an exception, a Protocol) has no Python signature to show.
        constructor = inspect.signature(value) if inspect.isfunction(value.__init__) else ""
        return f"class({bases}){constructor}"
    if inspect.isfunction(value):
        return f"function{inspect.signature(value)}"
    return f"value: {type(value).__name__}"
