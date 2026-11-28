"""C-STATE-1 / C-STATE-2: public dataclasses are frozen.

Every dataclass of the ``__all__`` tree (``better_robot.testing.public_objects``) must be declared
``frozen=True``. The exceptions are listed in ``MUTABLE_DATACLASSES`` with
their reason, and each entry must still name a public mutable dataclass.
"""

from __future__ import annotations

import dataclasses
import inspect

from better_robot.testing import public_objects
from tests.support.cases import failures

MUTABLE_DATACLASSES: dict[str, str] = {
    "better_robot.io.IRBody": "parser IR, filled incrementally while parsing",
    "better_robot.io.IRFrame": "parser IR, filled incrementally while parsing",
    "better_robot.io.IRJoint": "parser IR, filled incrementally while parsing",
    "better_robot.io.IRModel": "parser IR, filled incrementally while parsing",
}


DATACLASSES = [
    (path, value)
    for path, value in public_objects("better_robot").items()
    if inspect.isclass(value) and dataclasses.is_dataclass(value)
]
FROZEN = [(qualname, cls) for qualname, cls in DATACLASSES if qualname not in MUTABLE_DATACLASSES]


def test_public_dataclasses_are_frozen() -> None:
    classes = dict(FROZEN)

    def check(qualname: str) -> None:
        assert classes[qualname].__dataclass_params__.frozen, "must be @dataclass(frozen=True)"

    failed = failures(check, classes)
    assert not failed, "\n".join(failed)


def test_mutable_list_names_public_mutable_dataclasses() -> None:
    """An entry whose class was frozen, renamed or removed leaves ``MUTABLE_DATACLASSES``."""
    found = dict(DATACLASSES)
    stale = [name for name in MUTABLE_DATACLASSES if name not in found or found[name].__dataclass_params__.frozen]
    assert not stale, f"remove from MUTABLE_DATACLASSES: {stale}"
