"""Typed value classes for SE(3) / SO(3).

These are thin frozen dataclasses around the storage tensor — they exist
so user-facing code can spell out *what kind* of pose a value is, without
losing the bare-tensor fast paths used inside hot loops.

Storage convention is the library-wide one:

* ``SE3.tensor``: ``(..., 7)`` ``[tx, ty, tz, qx, qy, qz, qw]``
* ``SO3.tensor``: ``(..., 4)`` ``[qx, qy, qz, qw]``

The only operator overloaded is ``@`` (composition for both groups; SE3
applied to a ``(..., 3)`` point on the right). Scalar multiplication is
deliberately not defined — see ``docs/design/03_LIE_AND_SPATIAL.md §7``.

``Pose`` is an alias for ``SE3``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..backends.protocol import Backend


def _is_point(other: object) -> bool:
    return isinstance(other, torch.Tensor) and other.shape[-1:] == (3,)


@dataclass(frozen=True)
class SO3:
    """Frozen SO(3) value type wrapping a unit quaternion.

    Storage: ``(..., 4)`` tensor ``[qx, qy, qz, qw]`` (scalar last).
    """

    tensor: torch.Tensor  # (..., 4)

    # ── factories ────────────────────────────────────────────────────

    @classmethod
    def identity(
        cls,
        *,
        batch_shape: tuple[int, ...] = (),
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "SO3":
        from . import so3 as _so3
        return cls(_so3.identity(batch_shape=batch_shape, device=device, dtype=dtype))

    @classmethod
    def exp(
        cls, w: torch.Tensor, *, backend: "Backend | None" = None,
    ) -> "SO3":
        """``so3 → SO3``. ``w: (..., 3)``."""
        from . import so3 as _so3
        return cls(_so3.exp(w, backend=backend))

    @classmethod
    def from_matrix(
        cls, R: torch.Tensor, *, backend: "Backend | None" = None,
    ) -> "SO3":
        from . import so3 as _so3
        return cls(_so3.from_matrix(R, backend=backend))

    # ── methods ──────────────────────────────────────────────────────

    def inverse(self, *, backend: "Backend | None" = None) -> "SO3":
        from . import so3 as _so3
        return SO3(_so3.inverse(self.tensor, backend=backend))

    def log(self, *, backend: "Backend | None" = None) -> torch.Tensor:
        from . import so3 as _so3
        return _so3.log(self.tensor, backend=backend)

    def to_matrix(self, *, backend: "Backend | None" = None) -> torch.Tensor:
        from . import so3 as _so3
        return _so3.to_matrix(self.tensor, backend=backend)

    def normalize(self, *, backend: "Backend | None" = None) -> "SO3":
        from . import so3 as _so3
        return SO3(_so3.normalize(self.tensor, backend=backend))

    def compose(
        self, other: "SO3", *, backend: "Backend | None" = None,
    ) -> "SO3":
        from . import so3 as _so3
        return SO3(_so3.compose(self.tensor, other.tensor, backend=backend))

    def act(
        self, p: torch.Tensor, *, backend: "Backend | None" = None,
    ) -> torch.Tensor:
        from . import so3 as _so3
        return _so3.act(self.tensor, p, backend=backend)

    # ── operators ────────────────────────────────────────────────────

    def __matmul__(self, other):
        if isinstance(other, SO3):
            return self.compose(other)
        if _is_point(other):
            return self.act(other)
        return NotImplemented

    def __mul__(self, other):
        raise TypeError(
            "SO3 does not support `*`. Use `@` for composition / point action; "
            "scalar scaling is not a meaningful Lie-group operation."
        )

    __rmul__ = __mul__


@dataclass(frozen=True)
class SE3:
    """Frozen SE(3) value type wrapping a 7-vector pose.

    Storage: ``(..., 7)`` tensor ``[tx, ty, tz, qx, qy, qz, qw]``.
    """

    tensor: torch.Tensor  # (..., 7)

    # ── factories ────────────────────────────────────────────────────

    @classmethod
    def identity(
        cls,
        *,
        batch_shape: tuple[int, ...] = (),
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "SE3":
        from . import se3 as _se3
        return cls(_se3.identity(batch_shape=batch_shape, device=device, dtype=dtype))

    @classmethod
    def exp(
        cls, xi: torch.Tensor, *, backend: "Backend | None" = None,
    ) -> "SE3":
        """``se3 → SE3``. ``xi: (..., 6) [vx, vy, vz, wx, wy, wz]``."""
        from . import se3 as _se3
        return cls(_se3.exp(xi, backend=backend))

    # ── accessors ────────────────────────────────────────────────────

    @property
    def translation(self) -> torch.Tensor:
        """``(..., 3)`` translation component."""
        return self.tensor[..., :3]

    @property
    def rotation(self) -> SO3:
        """SO(3) part as an :class:`SO3` value."""
        return SO3(self.tensor[..., 3:7])

    # ── methods ──────────────────────────────────────────────────────

    def inverse(self, *, backend: "Backend | None" = None) -> "SE3":
        from . import se3 as _se3
        return SE3(_se3.inverse(self.tensor, backend=backend))

    def log(self, *, backend: "Backend | None" = None) -> torch.Tensor:
        from . import se3 as _se3
        return _se3.log(self.tensor, backend=backend)

    def adjoint(self, *, backend: "Backend | None" = None) -> torch.Tensor:
        from . import se3 as _se3
        return _se3.adjoint(self.tensor, backend=backend)

    def adjoint_inv(self, *, backend: "Backend | None" = None) -> torch.Tensor:
        from . import se3 as _se3
        return _se3.adjoint_inv(self.tensor, backend=backend)

    def normalize(self, *, backend: "Backend | None" = None) -> "SE3":
        from . import se3 as _se3
        return SE3(_se3.normalize(self.tensor, backend=backend))

    def compose(
        self, other: "SE3", *, backend: "Backend | None" = None,
    ) -> "SE3":
        from . import se3 as _se3
        return SE3(_se3.compose(self.tensor, other.tensor, backend=backend))

    def act(
        self, p: torch.Tensor, *, backend: "Backend | None" = None,
    ) -> torch.Tensor:
        from . import se3 as _se3
        return _se3.act(self.tensor, p, backend=backend)

    # ── operators ────────────────────────────────────────────────────

    def __matmul__(self, other):
        if isinstance(other, SE3):
            return self.compose(other)
        if _is_point(other):
            return self.act(other)
        return NotImplemented

    def __mul__(self, other):
        raise TypeError(
            "SE3 does not support `*`. Use `@` for composition / point action; "
            "scalar scaling is not a meaningful Lie-group operation."
        )

    __rmul__ = __mul__


# ``Pose`` is the user-facing alias — same type as SE3 but conveys intent.
Pose = SE3


__all__ = ["SE3", "SO3", "Pose"]
