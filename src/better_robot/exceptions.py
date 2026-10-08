"""Typed exceptions raised by ``better_robot``.

See ``docs/conventions/contracts.md §2`` for the normative taxonomy.

All library-raised errors inherit from `BetterRobotError`. Where
possible, they also inherit from a standard-library exception class
(``ValueError``, ``RuntimeError``, ``ImportError``) so existing
``except ValueError`` / ``except RuntimeError`` blocks keep catching the
common cases.

Usage::

    from better_robot.exceptions import ShapeError

    if q.shape[-1] != model.nq:
        raise ShapeError(f"q has trailing size {q.shape[-1]}, expected {model.nq}")
"""

from __future__ import annotations


class BetterRobotError(Exception):
    """Root of every exception raised by ``better_robot``.

    Catch this to distinguish library errors from arbitrary exceptions
    bubbling up from user code or from ``torch`` internals.
    """


class ModelInconsistencyError(BetterRobotError, ValueError):
    """Parsed ``Model`` violates one of the topology invariants.

    Invariants (see ``docs/conventions/contracts.md §1.5``):

    * ``parents[0] == -1`` — joint 0 is the universe.
    * ``parents[i] < i`` for ``i > 0`` — topologically sorted.
    * ``sum(nqs) == nq`` and ``sum(nvs) == nv``.
    * ``idx_qs[i] + nqs[i] == idx_qs[i+1]`` — contiguous slicing.

    Raised at *build* time (``io.build_model``), never at query time.
    """


class DeviceMismatchError(BetterRobotError, ValueError):
    """A ``q`` / ``v`` / ``tau`` tensor lives on a different device than
    the ``Model``.

    Remediation: ``model.to(q.device)`` or ``q.to(model.device)``.
    """


class DtypeMismatchError(BetterRobotError, ValueError):
    """A query tensor has a dtype incompatible with the model's dtype.

    ``float16`` is unsupported anywhere in the library. Cast to
    ``float32`` or ``float64`` before the call.
    """


class QuaternionNormError(BetterRobotError, ValueError):
    """Input quaternion's norm is outside ``[0.9, 1.1]``.

    Norms inside that window are re-normalised silently (floating-point
    drift tolerance). Outside it, the input is clearly not meant to be a
    unit quaternion and the library refuses to guess.
    """


class ShapeError(BetterRobotError, ValueError):
    """A public input has the wrong trailing-axis size.

    The message names the offending argument and the expected shape.
    """


class BackendNotAvailableError(BetterRobotError, ImportError):
    """An optional integration dependency is not importable.

    Example::

        load("robot.urdf")  # raises if the optional `yourdfpy` parser is absent

    The name is reserved for a future kernel brick (an optional compiled
    backend behind the same public signatures); today it only reports a missing
    optional integration dependency. It is not evidence of a runtime
    compute-backend registry.
    """


class UnknownNameError(BetterRobotError, KeyError):
    """A joint, frame or body name is not in the model.

    Example::

        model.joint_id("no_such_joint")  # raises; ids may be ``int`` or ``str`` everywhere
    """


__all__ = [
    "BetterRobotError",
    "ModelInconsistencyError",
    "DeviceMismatchError",
    "DtypeMismatchError",
    "QuaternionNormError",
    "ShapeError",
    "UnknownNameError",
    "BackendNotAvailableError",
]
