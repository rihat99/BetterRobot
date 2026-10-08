"""C-NAME-5: public names use the SE(3) vocabulary (placement / pose / transform).

A *placement* is a model constant in the parent frame, a *pose* a computed world
7-vector, a *transform* the same world quantity as a ``(..., 3, 4)`` matrix. The
regex below rejects the synonyms and mixed forms the contract rules out; it scans
``__all__`` of every public package, the parameter names of exported callables,
and the annotated fields, members (methods, properties) and method parameters of
exported classes.
"""

from __future__ import annotations

import importlib
import inspect
import re

import pytest

_PACKAGES = (
    "better_robot",
    "better_robot.lie",
    "better_robot.spatial",
    "better_robot.data_model",
    "better_robot.kinematics",
    "better_robot.dynamics",
    "better_robot.collision",
    "better_robot.terms",
    "better_robot.io",
    "better_robot.optim",
    "better_robot.viewer",
)

# Synonyms for an SE(3) quantity, and mixed forms: a matrix is a transform,
# a world quantity is a pose, and a placement is never "world".
_FORBIDDEN = re.compile(
    r"(^|_)(tf|tfs|tfm|xform|xforms|trafo|homogeneous|se3_matrix)(_|$)"
    r"|(^|_)(pose|poses|placement|placements)_(matrix|matrices|mat)(_|$)"
    r"|(^|_)world_placements?(_|$)|(^|_)placements?_world(_|$)"
)


def _member_names(owner: str, obj: type, klass: type) -> set[tuple[str, str]]:
    """Public members of ``klass`` (methods, properties, attributes) and their parameter names."""
    names: set[tuple[str, str]] = set()
    for member_name, member in vars(klass).items():
        if member_name.startswith("_"):
            continue
        names.add((owner, member_name))
        function = member.fget if isinstance(member, property) else getattr(obj, member_name)
        if not callable(function):
            continue
        try:
            parameters = inspect.signature(function).parameters
        except (TypeError, ValueError):
            continue
        names.update((f"{owner}.{member_name}", parameter) for parameter in parameters)
    return names


def _public_names() -> set[tuple[str, str]]:
    names: set[tuple[str, str]] = set()
    for package_name in _PACKAGES:
        package = importlib.import_module(package_name)
        for exported in getattr(package, "__all__", ()):
            names.add((package_name, exported))
            obj = getattr(package, exported, None)
            if inspect.isclass(obj):
                for klass in obj.__mro__[:-1]:
                    for field in getattr(klass, "__annotations__", {}):
                        if not field.startswith("_"):
                            names.add((f"{package_name}.{exported}", field))
                    if klass.__module__.startswith("better_robot"):
                        names |= _member_names(f"{package_name}.{exported}", obj, klass)
            if callable(obj):
                try:
                    signature = inspect.signature(obj)
                except (TypeError, ValueError):
                    continue
                for parameter in signature.parameters:
                    names.add((f"{package_name}.{exported}", parameter))
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
