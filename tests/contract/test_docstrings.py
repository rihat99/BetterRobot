"""Every public object has a docstring, and every tensor parameter's shape is in it (principle 4.6); no docstring in
``src`` uses a Sphinx role (C-DOC-4).

The sweep walks the ``__all__`` tree (``better_robot.testing.public_objects``): modules, classes, functions,
methods and properties, each once, under the first path that lists it. A shape is a parenthesised list of axes with
a comma or ``...`` in it, ``(B..., nq)``, ``(7,)``, ``(B or 1, N)``, or the scalar shape ``()``; prose in parentheses
is not one. A tensor parameter is one whose annotation has ``torch.Tensor`` as a member; its shape is present when the
parameter's name (or ``name_*`` for a family) is followed within a few words of the same sentence by a shape,
``q: (B..., nq)`` or ``origin (7,)`` (a call's parentheses, ``broadcast(B..., I...)``, do not count), or when a shape
signature ``(..., 7), (..., 7) -> (..., 7)`` lists it by position. A method may document a parameter its class
docstring documents for every method. A function or property whose return annotation
holds ``torch.Tensor`` (alone, in a tuple or in a mapping) names a shape that no tensor parameter's name directly
precedes: the result's.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from pathlib import Path

import torch

from better_robot.kinematics import forward_kinematics
from better_robot.testing import public_objects

AXIS = r"[A-Za-z0-9_*.]+(?:\s*(?:[-+*/]|or)\s*[A-Za-z0-9_.]+)*"
SHAPE = re.compile(rf"\((?=[^()\n]*(?:,|\.\.\.)|\s*\))\s*(?:{AXIS}(?:\s*,\s*{AXIS})*\s*,?)?\s*\)")
NAMED_SHAPE = re.compile(rf"(?<![\w\])]){SHAPE.pattern}")  # a shape after a name: not a call's argument list
SENTENCE_END = re.compile(r"[.;]\s")
SIGNATURE = re.compile(r"``([^`]*?)(?:→|->)[^`]*``")
NEAR = 80  # characters after a parameter's name in which its shape must start
SRC = Path(__file__).resolve().parents[2] / "src" / "better_robot"
SPHINX_ROLE = re.compile(r":(?:py:)?(?:class|func|meth|attr|mod|obj|data|exc|const|math):`")


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
            after = text[match.end() : match.end() + NEAR]
            shape = NAMED_SHAPE.search(after)
            if shape and not SENTENCE_END.search(after[: shape.start()]):
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


def _result_shape_missing(value: object) -> bool:
    """Whether ``value`` returns a tensor but every shape in its docstring sits right after a tensor parameter."""
    function = value.fget if isinstance(value, property) else value
    signature = inspect.signature(function)
    if "torch.Tensor" not in str(signature.return_annotation):
        return False
    text = inspect.getdoc(value) or ""
    parameters = [p.name for p in signature.parameters.values() if _is_tensor(p.annotation)]
    after_parameter = [rf"(?<![A-Za-z0-9_]){re.escape(name)}(?:_\*)?(?::\s*|\s+|=)$" for name in parameters]
    return not any(
        not any(re.search(pattern, text[max(0, shape.start() - NEAR) : shape.start()]) for pattern in after_parameter)
        for shape in SHAPE.finditer(text)
    )


def _violations() -> dict[str, str]:
    objects = _documented()
    classes = {path: value for path, value in public_objects("better_robot").items() if inspect.isclass(value)}
    found: dict[str, str] = {}
    for path, value in objects.items():
        if not (inspect.getdoc(value) or "").strip():
            found[path] = "no docstring"
            continue
        if inspect.isfunction(value):
            owner = classes.get(path.rsplit(".", 1)[0])
            found.update({site: "tensor shape missing" for site in _missing_shapes(path, value, owner)})
        if (inspect.isfunction(value) or isinstance(value, property)) and _result_shape_missing(value):
            found[f"{path}::return"] = "result shape missing"
    return found


def test_every_public_object_has_a_docstring_with_its_tensor_shapes() -> None:
    """A new public object without a docstring, or a tensor parameter whose shape the docstring omits, fails."""
    found = sorted(f"{site}: {problem}" for site, problem in _violations().items())
    assert not found, "document these (principle 4.6):\n  " + "\n  ".join(found)


def test_no_docstring_uses_a_sphinx_role() -> None:
    """MkDocs prints ``:class:`Model``` as literal text: write ``Model`` or an mkdocstrings cross-reference."""
    files = sorted(
        path.relative_to(SRC).as_posix() for path in SRC.rglob("*.py") if SPHINX_ROLE.search(path.read_text())
    )
    assert files == [], f"Sphinx roles in: {files}"


def test_a_shape_near_a_parameter_documents_it_only_in_its_own_sentence_and_not_as_a_call() -> None:
    """The review's plant: with ``q: (B..., nq)`` deleted from ``forward_kinematics``, the call
    ``broadcast(B..., I...)`` within the same few words documented ``q``."""

    def planted(model: object, q: torch.Tensor) -> None: ...

    planted.__doc__ = inspect.getdoc(forward_kinematics).replace("``q: (B..., nq)`` ", "")
    assert planted.__doc__ != inspect.getdoc(forward_kinematics)
    assert _missing_shapes("forward_kinematics", forward_kinematics, None) == []
    assert _missing_shapes("planted", planted, None) == ["planted::q"]
