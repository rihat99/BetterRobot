"""One error tree under ``BetterRobotError`` (decision record 0020).

Every error the library raises derives from ``BetterRobotError`` and from the builtin a caller already knows:
``ValueError`` for a wrong structure or argument, ``KeyError`` for an unknown name, ``ImportError`` for a missing
optional package. This module imports nothing, so every package, ``optim`` included, may import it.
"""

from __future__ import annotations

from collections.abc import Sequence


class BetterRobotError(Exception):
    """Root of every error raised by ``better_robot``."""


class ShapeError(BetterRobotError, ValueError):
    """A tensor has the wrong trailing shape, or batch shapes do not broadcast."""


class DtypeMismatchError(BetterRobotError, ValueError):
    """A tensor's dtype differs from the model's."""


class DeviceMismatchError(BetterRobotError, ValueError):
    """A tensor's device differs from the model's."""


class ArgumentError(BetterRobotError, ValueError):
    """An argument is out of its accepted set: an id out of range, an unsupported reference frame, a tensor whose
    values an operation cannot use (a point set with non-finite entries, a mesh that is not closed)."""


class ModelError(BetterRobotError, ValueError):
    """A robot description cannot become a model, or a model lacks what an operation needs."""


class QuaternionNormError(BetterRobotError, ValueError):
    """A quaternion in ``q`` has a norm outside ``[0.9, 1.1]`` (checked only on request)."""


class UnknownNameError(BetterRobotError, KeyError):
    """A name is not among the valid ones: a model's joints, frames or bodies, a problem's terms, a viewer's groups.

    Every lookup writes one sentence shape, ``unknown <kind> '<name>' (valid names: [...])``; past 20 valid names it
    gives their count instead, ``unknown <kind> '<name>' (<n> valid names)``.
    """

    def __init__(self, kind: str, name: str, *, valid: Sequence[str]) -> None:
        names = f"valid names: {list(valid)}" if len(valid) <= 20 else f"{len(valid)} valid names"
        super().__init__(f"unknown {kind} {name!r} ({names})")

    def __str__(self) -> str:
        # KeyError.__str__ would wrap the message in quotes.
        return self.args[0]


class OptionalDependencyError(BetterRobotError, ImportError):
    """An optional package is not installed; the message carries the install line."""

    def __init__(self, package: str, *, install: str) -> None:
        super().__init__(f"{package} is not installed; run: {install}")


__all__ = [
    "BetterRobotError",
    "ShapeError",
    "DtypeMismatchError",
    "DeviceMismatchError",
    "ArgumentError",
    "ModelError",
    "QuaternionNormError",
    "UnknownNameError",
    "OptionalDependencyError",
]
