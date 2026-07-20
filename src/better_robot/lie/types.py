"""Typed value classes for SE(3) / SO(3).

These are thin frozen dataclasses around the storage tensor — they exist
so user-facing code can spell out *what kind* of pose a value is, without
losing the bare-tensor fast paths used inside hot loops.

Storage convention is the library-wide one:

* ``SE3.tensor``: ``(..., 7)`` ``[tx, ty, tz, qx, qy, qz, qw]``
* ``SO3.tensor``: ``(..., 4)`` ``[qx, qy, qz, qw]``

The only operator overloaded is ``@`` (composition for both groups; SE3
applied to a ``(..., 3)`` point on the right). Scalar multiplication is
deliberately not defined — see ``docs/concepts/lie_and_spatial.md §7``.

``Pose`` is an alias for ``SE3``.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from . import se3, so3


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
        return cls(so3.identity(batch_shape=batch_shape, device=device, dtype=dtype))

    @classmethod
    def exp(cls, w: torch.Tensor) -> "SO3":
        """``so3 → SO3``. ``w: (..., 3)``."""
        return cls(so3.exp(w))

    @classmethod
    def from_matrix(cls, R: torch.Tensor) -> "SO3":
        return cls(so3.from_matrix(R))

    # ── methods ──────────────────────────────────────────────────────

    def inverse(self) -> "SO3":
        return SO3(so3.inverse(self.tensor))

    def log(self) -> torch.Tensor:
        return so3.log(self.tensor)

    def to_matrix(self) -> torch.Tensor:
        return so3.to_matrix(self.tensor)

    def normalize(self) -> "SO3":
        return SO3(so3.normalize(self.tensor))

    def compose(self, other: "SO3") -> "SO3":
        return SO3(so3.compose(self.tensor, other.tensor))

    def act(self, p: torch.Tensor) -> torch.Tensor:
        return so3.act(self.tensor, p)

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
        return cls(se3.identity(batch_shape=batch_shape, device=device, dtype=dtype))

    @classmethod
    def exp(cls, xi: torch.Tensor) -> "SE3":
        """``se3 → SE3``. ``xi: (..., 6) [vx, vy, vz, wx, wy, wz]``."""
        return cls(se3.exp(xi))

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

    def inverse(self) -> "SE3":
        return SE3(se3.inverse(self.tensor))

    def log(self) -> torch.Tensor:
        return se3.log(self.tensor)

    def adjoint(self) -> torch.Tensor:
        return se3.adjoint(self.tensor)

    def adjoint_inv(self) -> torch.Tensor:
        return se3.adjoint_inv(self.tensor)

    def normalize(self) -> "SE3":
        return SE3(se3.normalize(self.tensor))

    def compose(self, other: "SE3") -> "SE3":
        return SE3(se3.compose(self.tensor, other.tensor))

    def act(self, p: torch.Tensor) -> torch.Tensor:
        return se3.act(self.tensor, p)

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
