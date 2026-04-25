"""``overlays/force_vectors.py`` — draw force-vector arrows at named frames.

Intended for visualising ground-reaction forces or other per-contact
linear force estimates over an animation. The overlay owns one arrow per
contact joint; the owning animation loop calls
:meth:`ForceVectorsOverlay.update_frame` each frame with the current
world-frame anchor positions and force 3-vectors. Arrows with
near-zero magnitude are hidden automatically.

Example::

    overlay = ForceVectorsOverlay(contact_joint_names=["left_foot", ...])
    scene.add_mode(overlay)
    for k in range(T):
        overlay.update_frame(anchors_k, forces_k)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from ..render_modes.base import RenderContext
from ..render_modes.skeleton import _align_z_to_vec

if TYPE_CHECKING:
    from ...data_model.data import Data
    from ...data_model.model import Model


_ARROW_RGBA: tuple[float, float, float, float] = (1.0, 0.25, 0.2, 1.0)
_MIN_LENGTH: float = 5e-3  # 5 mm — below this the arrow is hidden


class ForceVectorsOverlay:
    """Render one arrow per contact joint, length ∝ force magnitude.

    ``scale`` is the metres-per-newton conversion factor used when
    mapping a force magnitude to arrow length. The default (5e-4 m / N)
    puts a 500 N force at ~0.25 m, which reads well for a human-scale
    body. Shaft / head dimensions are given in metres.
    """

    name = "Force vectors"
    description = "Linear force arrows at contact joints"

    def __init__(
        self,
        contact_joint_names: Sequence[str],
        *,
        scale: float = 5e-4,
        shaft_radius: float = 0.006,
        head_length: float = 0.03,
        head_radius: float = 0.015,
        rgba: tuple[float, float, float, float] = _ARROW_RGBA,
        visible: bool = True,
    ) -> None:
        self._names = tuple(contact_joint_names)
        self._scale = float(scale)
        self._shaft_radius = float(shaft_radius)
        self._head_length = float(head_length)
        self._head_radius = float(head_radius)
        self._rgba = tuple(rgba)
        self._initial_visible = bool(visible)
        self._ctx: RenderContext | None = None
        self._visible: bool = visible
        # Per-arrow state: name-in-backend, last-length (for avoiding
        # redundant mesh rebuilds on identical frames).
        self._last_length: list[float] = []

    @classmethod
    def is_available(cls, model: "Model", data: "Data") -> bool:
        return True

    def attach(self, context: RenderContext, model: "Model", data: "Data") -> None:
        self._ctx = context
        backend = context.backend
        ns = context.namespace

        self._last_length = [0.0] * len(self._names)
        for i, _n in enumerate(self._names):
            name = f"{ns}/force_{i}"
            # Seed with a tiny invisible arrow; update_frame fills in real geometry.
            backend.add_arrow(
                name,
                length=_MIN_LENGTH,
                shaft_radius=self._shaft_radius,
                head_length=self._head_length,
                head_radius=self._head_radius,
                rgba=self._rgba,
            )
            backend.set_visible(name, False)

        if self._initial_visible:
            self.set_visible(True)

    def update(self, data: "Data") -> None:
        """No-op — this overlay is driven externally via ``update_frame``."""
        return

    def update_frame(
        self,
        anchor_positions: torch.Tensor,
        forces: torch.Tensor,
    ) -> None:
        """Refresh arrows from per-contact anchor positions and force vectors.

        Parameters
        ----------
        anchor_positions
            ``(N, 3)`` world-frame origins, one per contact joint.
        forces
            ``(N, 3)`` world-frame linear force vectors. Arrows with
            magnitude below ``_MIN_LENGTH / scale`` are hidden.
        """
        if self._ctx is None or not self._visible:
            return
        backend = self._ctx.backend
        ns = self._ctx.namespace

        anchor_cpu = anchor_positions.detach().cpu()
        forces_cpu = forces.detach().cpu()

        for i in range(len(self._names)):
            name = f"{ns}/force_{i}"
            f = forces_cpu[i]
            mag = float(f.norm())
            length = mag * self._scale
            if length < _MIN_LENGTH:
                backend.set_visible(name, False)
                continue

            # Only rebuild the mesh when the length changes meaningfully —
            # repeated add_arrow with the same length would waste a websocket
            # roundtrip per frame per arrow.
            prev = self._last_length[i]
            if abs(length - prev) > 1e-4 * max(length, prev, 1e-6):
                backend.add_arrow(
                    name,
                    length=length,
                    shaft_radius=self._shaft_radius,
                    head_length=min(self._head_length, length * 0.4),
                    head_radius=self._head_radius,
                    rgba=self._rgba,
                )
                self._last_length[i] = length

            # Orient: rotate +Z to align with force direction.
            quat_xyzw = _align_z_to_vec(f.to(dtype=torch.float32))
            pose = torch.cat([anchor_cpu[i].to(dtype=torch.float32), quat_xyzw])
            backend.set_transform(name, pose)
            backend.set_visible(name, True)

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        if self._ctx is None:
            return
        backend = self._ctx.backend
        ns = self._ctx.namespace
        # When hiding, hide all arrows immediately; when showing, leave them
        # hidden until the next update_frame fills in real geometry.
        if not visible:
            for i in range(len(self._names)):
                backend.set_visible(f"{ns}/force_{i}", False)

    def detach(self) -> None:
        if self._ctx is None:
            return
        backend = self._ctx.backend
        ns = self._ctx.namespace
        for i in range(len(self._names)):
            backend.remove(f"{ns}/force_{i}")
        self._ctx = None
        self._last_length = []
