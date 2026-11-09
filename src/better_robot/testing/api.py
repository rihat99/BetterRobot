"""The public API of a package as data: every path its ``__all__`` tree lists, and one line describing each.

The tree is the package, each name its ``__all__`` lists, the listed modules recursively, and the public members of
each listed class (its own attributes without a leading underscore). A name listed at two paths
(``better_robot.Model`` and ``better_robot.model.Model``) appears under both. The contract sweeps of both libraries
walk ``public_objects``; ``tools/api_snapshot.py`` writes and compares ``public_api``.
"""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType


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
