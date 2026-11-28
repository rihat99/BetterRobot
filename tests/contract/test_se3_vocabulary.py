"""C-NAME-5: public names use the SE(3) vocabulary (placement / pose / transform).

A *placement* is a model constant in the parent frame, a *pose* a computed world
7-vector, a *transform* the same world quantity as a ``(..., 3, 4)`` matrix. The
regex below rejects the synonyms and mixed forms the contract rules out; it scans
every name of the ``__all__`` tree (``better_robot.testing.public_objects``), the parameter
names of exported functions and constructors, and the annotated fields, members (methods,
properties) and method parameters of exported classes and their ``better_robot`` bases.
"""

from __future__ import annotations

import inspect
import re

import pytest

from better_robot.testing import public_objects

# Synonyms for an SE(3) quantity, and mixed forms: a matrix is a transform,
# a world quantity is a pose, and a placement is never "world".
_FORBIDDEN = re.compile(
    r"(^|_)(tf|tfs|tfm|xform|xforms|trafo|homogeneous|se3_matrix)(_|$)"
    r"|(^|_)(pose|poses|placement|placements)_(matrix|matrices|mat)(_|$)"
    r"|(^|_)world_placements?(_|$)|(^|_)placements?_world(_|$)"
)


def _parameters(value: object) -> list[str]:
    """Parameter names of a function, a property's getter, a static or class method, or a class's Python
    constructor; a builtin constructor (an exception's) has none."""
    if isinstance(value, property):
        value = value.fget
    elif isinstance(value, staticmethod | classmethod):
        value = value.__func__
    elif inspect.isclass(value):
        value = value.__init__
    return list(inspect.signature(value).parameters) if inspect.isfunction(value) else []


def _public_names() -> set[tuple[str, str]]:
    names: set[tuple[str, str]] = set()
    for path, value in public_objects("better_robot").items():
        owner, _, name = path.rpartition(".")
        names.add((owner, name))
        names.update((path, parameter) for parameter in _parameters(value))
        if inspect.isclass(value):
            for klass in value.__mro__[:-1]:
                names.update((path, field) for field in inspect.get_annotations(klass) if not field.startswith("_"))
                if klass.__module__.startswith("better_robot"):
                    for member, item in vars(klass).items():
                        if not member.startswith("_"):
                            names.add((path, member))
                            names.update((f"{path}.{member}", parameter) for parameter in _parameters(item))
    return names


def test_public_names_follow_se3_vocabulary() -> None:
    offenders = sorted(f"{owner}: {name}" for owner, name in _public_names() if _FORBIDDEN.search(name))
    assert not offenders, "names break C-NAME-5 (SE(3) vocabulary):\n" + "\n".join(offenders)


@pytest.mark.parametrize("bad", ["frame_tf", "pose_matrix", "world_placement", "joint_xform", "placement_matrices"])
def test_regex_rejects_known_bad_names(bad: str) -> None:
    assert _FORBIDDEN.search(bad)


@pytest.mark.parametrize("good", ["joint_placements", "frame_pose", "frame_transform", "joint_pose_world"])
def test_regex_accepts_vocabulary_names(good: str) -> None:
    assert not _FORBIDDEN.search(good)
