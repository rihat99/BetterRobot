"""Every public object has a docstring, and every tensor parameter's shape is in it (principle 4.6).

The sweep walks the ``__all__`` tree (``better_robot.testing.public_objects``): modules, classes, functions,
methods and properties, each once, under the first path that lists it. A tensor parameter is one whose annotation
has ``torch.Tensor`` as a member; its shape is present when the parameter's name (or ``name_*`` for a family) is
followed within a few words by a parenthesised shape, ``q: (B..., nq)`` or ``origin (7,)``, or when a shape
signature ``(..., 7), (..., 7) -> (..., 7)`` lists it by position. A method may document a parameter its class
docstring documents for every method.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable

from better_robot.testing import public_objects

SHAPE = re.compile(r"\([^()\n]*\)")
SIGNATURE = re.compile(r"``([^`]*?)(?:→|->)[^`]*``")
NEAR = 80  # characters after a parameter's name in which its shape must start


def _documented() -> dict[str, object]:
    """Each public module, class, function, method and property once, under the first path that lists it."""
    found: dict[int, tuple[str, object]] = {}
    for path, listed in public_objects("better_robot").items():
        value = listed.__func__ if isinstance(listed, staticmethod | classmethod) else listed
        if (
            inspect.ismodule(value)
            or inspect.isclass(value)
            or inspect.isfunction(value)
            or isinstance(value, property)
        ):
            found.setdefault(id(value), (path, value))
    return dict(found.values())


def _is_tensor(annotation: object) -> bool:
    return any(part.strip() in {"torch.Tensor", "Tensor"} for part in str(annotation).split("|"))


def _names_shape(text: str, name: str) -> bool:
    patterns = [name, f"{name.rsplit('_', 1)[0]}_*"] if "_" in name else [name]
    for pattern in patterns:
        for match in re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(pattern)}(?![A-Za-z0-9_])", text):
            if SHAPE.search(text[match.end() : match.end() + NEAR]):
                return True
    return False


def _missing_shapes(path: str, function: Callable[..., object], owner: object | None) -> list[str]:
    text = inspect.getdoc(function) or ""
    owner_text = (inspect.getdoc(owner) or "") if owner is not None else ""
    signature = SIGNATURE.search(text)
    listed = len(SHAPE.findall(signature.group(1))) if signature else 0
    tensors = [
        parameter for parameter in inspect.signature(function).parameters.values() if _is_tensor(parameter.annotation)
    ]
    return [
        f"{path}::{parameter.name}"
        for position, parameter in enumerate(tensors)
        if position >= listed
        and not _names_shape(text, parameter.name)
        and not _names_shape(owner_text, parameter.name)
    ]


def _violations() -> dict[str, str]:
    objects = _documented()
    classes = {path: value for path, value in public_objects("better_robot").items() if inspect.isclass(value)}
    found: dict[str, str] = {}
    for path, value in objects.items():
        if not (inspect.getdoc(value) or "").strip():
            found[path] = "no docstring"
        elif inspect.isfunction(value):
            owner = classes.get(path.rsplit(".", 1)[0])
            found.update({site: "tensor shape missing" for site in _missing_shapes(path, value, owner)})
    return found


def test_every_public_object_has_a_docstring_with_its_tensor_shapes() -> None:
    """A new public object without a docstring, or a tensor parameter whose shape the docstring omits, fails."""
    found = sorted(f"{site}: {problem}" for site, problem in _violations().items())
    assert not found, "document these (principle 4.6):\n  " + "\n  ".join(found)
