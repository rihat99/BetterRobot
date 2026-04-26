"""``Backend`` Protocols — ``LieOps`` / ``KinematicsOps`` / ``DynamicsOps``.

A :class:`Backend` is a bundle of three orthogonal sub-Protocols. Library
hot paths route through one of them; user code that wants to change
backends does so by passing a different :class:`Backend` instance via the
``backend=`` kwarg of the public functions, or by switching the global
default with :func:`better_robot.backends.set_backend`.

Today the only concrete backend is :mod:`better_robot.backends.torch_native`;
the Warp backend lands in P11 of ``docs/UPDATE_PHASES.md``.

See ``docs/design/10_BATCHING_AND_BACKENDS.md §7``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

    from ..data_model.data import Data
    from ..data_model.model import Model


@runtime_checkable
class LieOps(Protocol):
    """SE(3) / SO(3) primitive operations.

    Storage convention is the library-wide one — SE(3) as
    ``(..., 7)`` ``[tx, ty, tz, qx, qy, qz, qw]``, SO(3) as ``(..., 4)``
    scalar-last quaternion. See ``docs/conventions/13_NAMING.md``.
    """

    def se3_compose(self, a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor": ...
    def se3_inverse(self, t: "torch.Tensor") -> "torch.Tensor": ...
    def se3_log(self, t: "torch.Tensor") -> "torch.Tensor": ...
    def se3_exp(self, v: "torch.Tensor") -> "torch.Tensor": ...
    def se3_act(self, t: "torch.Tensor", p: "torch.Tensor") -> "torch.Tensor": ...
    def se3_adjoint(self, t: "torch.Tensor") -> "torch.Tensor": ...
    def se3_adjoint_inv(self, t: "torch.Tensor") -> "torch.Tensor": ...
    def se3_normalize(self, t: "torch.Tensor") -> "torch.Tensor": ...

    def so3_compose(self, a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor": ...
    def so3_inverse(self, q: "torch.Tensor") -> "torch.Tensor": ...
    def so3_log(self, q: "torch.Tensor") -> "torch.Tensor": ...
    def so3_exp(self, w: "torch.Tensor") -> "torch.Tensor": ...
    def so3_act(self, q: "torch.Tensor", p: "torch.Tensor") -> "torch.Tensor": ...
    def so3_to_matrix(self, q: "torch.Tensor") -> "torch.Tensor": ...
    def so3_from_matrix(self, R: "torch.Tensor") -> "torch.Tensor": ...
    def so3_normalize(self, q: "torch.Tensor") -> "torch.Tensor": ...


@runtime_checkable
class KinematicsOps(Protocol):
    """Forward kinematics and joint-Jacobian primitives."""

    def forward_kinematics(
        self, model: "Model", q: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return ``(joint_pose_world, joint_pose_local)`` for every joint."""
        ...

    def compute_joint_jacobians(
        self, model: "Model", data: "Data",
    ) -> "torch.Tensor":
        """Return the world-frame joint Jacobian stack ``(B..., njoints, 6, nv)``."""
        ...


@runtime_checkable
class DynamicsOps(Protocol):
    """Rigid-body dynamics primitives.

    The torch-native backend currently delegates to the dynamics skeletons
    in :mod:`better_robot.dynamics`; most paths still raise
    ``NotImplementedError`` until P11 lands.
    """

    def rnea(
        self,
        model: "Model",
        data: "Data",
        q: "torch.Tensor",
        v: "torch.Tensor",
        a: "torch.Tensor",
    ) -> "torch.Tensor": ...

    def aba(
        self,
        model: "Model",
        data: "Data",
        q: "torch.Tensor",
        v: "torch.Tensor",
        tau: "torch.Tensor",
    ) -> "torch.Tensor": ...

    def crba(
        self, model: "Model", data: "Data", q: "torch.Tensor",
    ) -> "torch.Tensor": ...

    def center_of_mass(
        self,
        model: "Model",
        data: "Data",
        q: "torch.Tensor",
        v: "torch.Tensor | None" = None,
        a: "torch.Tensor | None" = None,
    ) -> "torch.Tensor": ...


@runtime_checkable
class Backend(Protocol):
    """A bundle of ops that together implement the math layer of the library."""

    name: str
    lie: LieOps
    kinematics: KinematicsOps
    dynamics: DynamicsOps


__all__ = ["Backend", "LieOps", "KinematicsOps", "DynamicsOps"]
