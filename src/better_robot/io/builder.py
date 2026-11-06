"""``ModelBuilder``: a robot written in Python, body by body and joint by joint, as the ``IRModel`` that
``build_model`` turns into a ``Model`` (record 0015).

Each ``add_<kind>`` helper takes the data of its kind as keywords and builds the ``JointModel`` there, so a bad axis
fails at the call, not at the build. ``add_joint`` takes any ``JointModel`` instance (record 0025), which is how a
kind written outside the package enters without a registry. A description that cannot be a robot raises
``ModelError``; a body name not added before raises ``UnknownNameError``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import torch

from ..exceptions import ArgumentError, ModelError, UnknownNameError
from ..lie import se3
from ..model.geometry import Shape
from ..model.joints import (
    JointFixed,
    JointFreeFlyer,
    JointHelical,
    JointModel,
    JointPlanar,
    JointPrismatic,
    JointRevolute,
    JointRevoluteUnbounded,
    JointSpherical,
)
from .ir import IRBody, IRFrame, IRJoint, IRModel, IRShape


def _axis(axis: Sequence[float] | torch.Tensor) -> tuple[float, ...]:
    """``axis`` as the tuple of floats a joint kind stores; the kind checks that it is a unit 3-vector."""
    return tuple(float(value) for value in axis)


@final
class ModelBuilder:
    """Collects bodies, joints, frames and shapes by name and emits one ``IRModel`` from ``finalize``.

    A joint connects ``parent`` to ``child``, two bodies added before; ``parent="world"`` makes it the root joint of
    the model. ``origin (7,)`` is the joint's placement in the parent body's frame, identity when omitted. The scalar
    limits of a joint apply to every one of its coordinates.

    Example::

        builder = ModelBuilder("arm")
        base = builder.add_body("base")
        link = builder.add_body("link", mass=1.2)
        builder.add_revolute("shoulder", parent=base, child=link, axis=(0.0, 0.0, 1.0), lower=-3.0, upper=3.0)
        model = build_model(builder.finalize())
    """

    def __init__(self, name: str, *, gravity: Sequence[float] | torch.Tensor | None = None) -> None:
        """``gravity (3,)`` is the world's linear acceleration, ``(0, 0, -9.81)`` when omitted."""
        self._name = name
        self._gravity = None if gravity is None else torch.as_tensor(gravity, dtype=torch.float64).reshape(3)
        self._bodies: list[IRBody] = []
        self._body_names: set[str] = set()
        self._joints: list[IRJoint] = []
        self._joint_names: set[str] = set()
        self._frames: list[IRFrame] = []
        self._shapes: list[IRShape] = []

    def add_body(
        self,
        name: str,
        *,
        mass: float = 0.0,
        com: torch.Tensor | None = None,
        inertia: torch.Tensor | None = None,
    ) -> str:
        """Add a body and return its name.

        ``mass`` is in kg; ``com (3,)`` and ``inertia (3, 3)`` (about the centre of mass) are in the body frame and
        zero when omitted.
        """
        if name in self._body_names:
            raise ModelError(f"body {name!r} already exists")
        self._body_names.add(name)
        self._bodies.append(
            IRBody(
                name=name,
                mass=mass,
                com=com if com is not None else torch.zeros(3),
                inertia=inertia if inertia is not None else torch.zeros(3, 3),
            )
        )
        return name

    def add_frame(self, name: str, *, parent: str, placement: torch.Tensor) -> str:
        """Add a frame placed by ``placement (7,)`` on the body ``parent`` and return its name."""
        if parent not in self._body_names:
            raise UnknownNameError("body", parent, sorted(self._body_names))
        self._frames.append(IRFrame(name=name, parent_body=parent, placement=placement))
        return name

    def add_shape(self, body: str, shape: Shape, *, collision: bool = True) -> None:
        """Attach ``shape`` to ``body`` as collision geometry or, with ``collision=False``, for display only."""
        if body not in self._body_names:
            raise UnknownNameError("body", body, sorted(self._body_names))
        self._shapes.append(IRShape(body=body, shape=shape, collision=collision))

    def add_joint(
        self,
        name: str,
        *,
        kind: JointModel,
        parent: str,
        child: str,
        origin: torch.Tensor | None = None,
        lower: float | None = None,
        upper: float | None = None,
        velocity_limit: float | None = None,
        effort_limit: float | None = None,
        mimic_source: str | None = None,
        mimic_multiplier: float = 1.0,
        mimic_offset: float = 0.0,
    ) -> str:
        """Add a joint of any kind, ``kind`` a ``JointModel`` instance the model keeps as is, and return its name.

        A joint with ``mimic_source`` follows that joint as ``mimic_multiplier * q_source + mimic_offset`` and has
        no coordinates of its own.
        """
        if not isinstance(kind, JointModel):
            raise ArgumentError(f"kind must be a JointModel instance, got {type(kind).__name__}")
        if name in self._joint_names:
            raise ModelError(f"joint {name!r} already exists")
        self._joint_names.add(name)
        self._joints.append(
            IRJoint(
                name=name,
                parent_body=parent,
                child_body=child,
                joint=kind,
                origin=se3.identity() if origin is None else origin,
                lower=lower,
                upper=upper,
                velocity_limit=velocity_limit,
                effort_limit=effort_limit,
                mimic_source=mimic_source,
                mimic_multiplier=mimic_multiplier,
                mimic_offset=mimic_offset,
            )
        )
        return name

    def add_revolute(
        self,
        name: str,
        *,
        parent: str,
        child: str,
        axis: Sequence[float] | torch.Tensor,
        origin: torch.Tensor | None = None,
        lower: float | None = None,
        upper: float | None = None,
        velocity_limit: float | None = None,
        effort_limit: float | None = None,
        unbounded: bool = False,
        mimic_source: str | None = None,
        mimic_multiplier: float = 1.0,
        mimic_offset: float = 0.0,
    ) -> str:
        """Add a revolute joint about the unit ``axis (3,)`` of the joint frame.

        ``unbounded=True`` makes it a continuous joint, which turns without position limits.
        """
        kind = JointRevoluteUnbounded(axis=_axis(axis)) if unbounded else JointRevolute(axis=_axis(axis))
        return self.add_joint(
            name,
            kind=kind,
            parent=parent,
            child=child,
            origin=origin,
            lower=lower,
            upper=upper,
            velocity_limit=velocity_limit,
            effort_limit=effort_limit,
            mimic_source=mimic_source,
            mimic_multiplier=mimic_multiplier,
            mimic_offset=mimic_offset,
        )

    def add_prismatic(
        self,
        name: str,
        *,
        parent: str,
        child: str,
        axis: Sequence[float] | torch.Tensor,
        origin: torch.Tensor | None = None,
        lower: float | None = None,
        upper: float | None = None,
        velocity_limit: float | None = None,
        effort_limit: float | None = None,
    ) -> str:
        """Add a prismatic joint along the unit ``axis (3,)`` of the joint frame."""
        return self.add_joint(
            name,
            kind=JointPrismatic(axis=_axis(axis)),
            parent=parent,
            child=child,
            origin=origin,
            lower=lower,
            upper=upper,
            velocity_limit=velocity_limit,
            effort_limit=effort_limit,
        )

    def add_helical(
        self,
        name: str,
        *,
        parent: str,
        child: str,
        axis: Sequence[float] | torch.Tensor,
        pitch: float,
        origin: torch.Tensor | None = None,
        lower: float | None = None,
        upper: float | None = None,
    ) -> str:
        """Add a helical joint about and along the unit ``axis (3,)``: ``pitch`` metres of travel per radian."""
        return self.add_joint(
            name,
            kind=JointHelical(axis=_axis(axis), pitch=pitch),
            parent=parent,
            child=child,
            origin=origin,
            lower=lower,
            upper=upper,
        )

    def add_spherical(self, name: str, *, parent: str, child: str, origin: torch.Tensor | None = None) -> str:
        """Add a spherical joint: three rotations, ``q`` a unit quaternion."""
        return self.add_joint(name, kind=JointSpherical(), parent=parent, child=child, origin=origin)

    def add_planar(self, name: str, *, parent: str, child: str, origin: torch.Tensor | None = None) -> str:
        """Add a planar joint: two translations in the joint's xy plane and a rotation about its z axis."""
        return self.add_joint(name, kind=JointPlanar(), parent=parent, child=child, origin=origin)

    def add_fixed(self, name: str, *, parent: str, child: str, origin: torch.Tensor | None = None) -> str:
        """Add a fixed joint: ``child`` rigidly placed at ``origin`` in ``parent``."""
        return self.add_joint(name, kind=JointFixed(), parent=parent, child=child, origin=origin)

    def add_free_flyer_root(self, name: str = "free_flyer", *, child: str, origin: torch.Tensor | None = None) -> str:
        """Add the free-flyer root joint, world to ``child``: the robot's base floats with six degrees of freedom."""
        return self.add_joint(name, kind=JointFreeFlyer(), parent="world", child=child, origin=origin)

    def finalize(self) -> IRModel:
        """The ``IRModel`` of everything added; its root body is the one no joint moves (exactly one, or ``ModelError``)."""
        # A joint from "world" does not make its child a moved body: that body is the root.
        children = {joint.child_body for joint in self._joints if joint.parent_body != "world"}
        roots = [body.name for body in self._bodies if body.name not in children]
        if len(roots) != 1:
            raise ModelError(f"expected exactly one root body (a body no joint moves), found {len(roots)}: {roots}")
        ir = IRModel(
            name=self._name,
            bodies=list(self._bodies),
            joints=list(self._joints),
            frames=list(self._frames),
            shapes=list(self._shapes),
            root_body=roots[0],
        )
        if self._gravity is not None:
            ir.gravity = torch.cat((self._gravity, torch.zeros(3, dtype=torch.float64)))
        return ir
