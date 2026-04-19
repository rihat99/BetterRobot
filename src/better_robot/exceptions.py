"""Typed exceptions raised by ``better_robot``.

See ``docs/17_CONTRACTS.md §2`` for the normative taxonomy.

All library-raised errors inherit from :class:`BetterRobotError`. Where
possible, they also inherit from a standard-library exception class
(``ValueError``, ``RuntimeError``, ``ImportError``) so existing
``except ValueError`` / ``except RuntimeError`` blocks keep catching the
common cases.

Usage::

    from better_robot.exceptions import ShapeError

    if q.shape[-1] != model.nq:
        raise ShapeError(f"q has trailing size {q.shape[-1]}, expected {model.nq}")

``SingularityWarning`` is a warning, not an exception — it is filtered
by ``warnings.filterwarnings`` rather than caught with ``try/except``.
"""

from __future__ import annotations


class BetterRobotError(Exception):
    """Root of every exception raised by ``better_robot``.

    Catch this to distinguish library errors from arbitrary exceptions
    bubbling up from user code or from ``torch`` internals.
    """


class ModelInconsistencyError(BetterRobotError, ValueError):
    """Parsed ``Model`` violates one of the topology invariants.

    Invariants (see ``docs/17_CONTRACTS.md §1.5``):

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


class ConvergenceError(BetterRobotError, RuntimeError):
    """Optimizer exited without reaching the requested tolerance.

    Not always a bug: inspect the returned ``SolverState`` for the final
    residual norm and gain ratio. ``solve_ik`` returns a non-converged
    ``IKResult`` instead of raising; only
    ``optim.solve(..., raise_on_nonconvergence=True)`` promotes this
    into an exception.
    """


class BackendNotAvailableError(BetterRobotError, ImportError):
    """Requested backend's dependency is not importable.

    Example::

        from better_robot.backends import set_backend
        set_backend("warp")   # raises if `warp` not installed
    """


class UnsupportedJointError(BetterRobotError, ValueError):
    """URDF/MJCF declared a joint kind without a built-in ``JointModel``.

    Register a custom ``JointModel`` — see ``docs/15_EXTENSION.md §2``.
    """


class SingularityWarning(UserWarning):
    """Emitted when a Jacobian's condition number exceeds ``1e12``.

    This is a warning, not an error: kinematics continues with the
    best-effort result and ``optim`` falls back to damped solves.
    """


__all__ = [
    "BetterRobotError",
    "ModelInconsistencyError",
    "DeviceMismatchError",
    "DtypeMismatchError",
    "QuaternionNormError",
    "ShapeError",
    "ConvergenceError",
    "BackendNotAvailableError",
    "UnsupportedJointError",
    "SingularityWarning",
]
