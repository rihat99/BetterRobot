"""``ModelBuilder``: a robot written in Python, body by body and joint by joint, as the ``IRModel`` that
``build_model`` turns into a ``Model``.

``add_joint`` takes any ``JointModel`` instance, built in or written outside the package, so a kind enters without
a registry, and a bad axis fails where the caller builds the kind, not at the build. A description that cannot be a
robot raises ``ModelError``; a body name not added before raises ``UnknownNameError``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final, override

import torch

from ..exceptions import ArgumentError, ModelError, UnknownNameError
from ..lie import se3
from ..model.geometry import Shape
from ..model.joints import JointModel
from .ir import IRBody, IRFrame, IRJoint, IRModel, IRShape


@final
class ModelBuilder:
    """Collects bodies, joints, frames and shapes by name and emits one ``IRModel`` from ``finalize``.

    A joint connects ``parent`` to ``child``, two bodies added before; ``parent="world"`` makes it the root joint of
    the model. ``origin (7,)`` is the joint's placement in the parent body's frame, identity when omitted; its
    quaternion, like a frame's, has unit norm (``build_model`` refuses another). The scalar limits of a joint apply to
    every one of its coordinates.

    Example::

        builder = ModelBuilder("arm")
        base = builder.add_body("base")
        link = builder.add_body("link", mass=1.2)
        builder.add_joint("shoulder", kind=JointRevolute(axis=(0.0, 0.0, 1.0)), parent=base, child=link, lower=-3.0)
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

    @override
    def __repr__(self) -> str:
        sizes = f"{len(self._bodies)} bodies, {len(self._joints)} joints, {len(self._frames)} frames"
        return f"ModelBuilder({self._name!r}, {sizes}, {len(self._shapes)} shapes)"

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
            raise UnknownNameError("body", parent, valid=sorted(self._body_names))
        self._frames.append(IRFrame(name=name, parent_body=parent, placement=placement))
        return name

    def add_shape(self, body: str, shape: Shape, *, collision: bool = True) -> None:
        """Attach ``shape`` to ``body`` as collision geometry or, with ``collision=False``, for display only."""
        if body not in self._body_names:
            raise UnknownNameError("body", body, valid=sorted(self._body_names))
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
        for body in (child,) if parent == "world" else (parent, child):
            if body not in self._body_names:
                raise UnknownNameError("body", body, valid=sorted(self._body_names))
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

    def finalize(self) -> IRModel:
        """The ``IRModel`` of everything added; ``build_model`` checks that it is one tree."""
        ir = IRModel(
            name=self._name,
            bodies=list(self._bodies),
            joints=list(self._joints),
            frames=list(self._frames),
            shapes=list(self._shapes),
        )
        if self._gravity is not None:
            ir.gravity = torch.cat((self._gravity, torch.zeros(3, dtype=torch.float64)))
        return ir
