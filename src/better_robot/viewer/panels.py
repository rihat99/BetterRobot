"""``panels.py`` — GUI panel builders for the viser interactive session.

See ``docs/concepts/viewer.md §10.2``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .scene import Scene


def build_joint_panel(
    server: object,
    scene: "Scene",
    *,
    group: str = "Joints",
    show_free_flyer: bool = False,
) -> None:
    """Attach a per-joint slider panel to the viser server.

    One slider per actuated (``nv > 0``) joint with limits taken from
    ``model.lower_pos_limit`` / ``model.upper_pos_limit``.  Dragging a
    slider retracts the current ``q`` on the corresponding joint's tangent
    and calls ``scene.update_from_q``.  Free-flyer joints are hidden by
    default (they are controlled by the base gizmo instead).

    Parameters
    ----------
    server:
        A ``ViserServer`` instance (passed as ``Any`` to keep the module
        importable without viser installed).
    scene:
        The ``Scene`` to update on slider change.
    group:
        GUI folder label.
    show_free_flyer:
        If True, expose the 6 free-flyer DOF as sliders too.
    """
    model = scene._model
    folder = server.gui.add_folder(group)  # type: ignore[attr-defined]

    # Keep the current q in a mutable container so closures can read/write it
    state = {"q": model.q_neutral.clone().clamp(
        model.lower_pos_limit, model.upper_pos_limit
    )}

    for j in range(model.njoints):
        jm = model.joint_models[j]
        if jm.nv == 0:
            continue  # fixed / universe joint

        # Skip free-flyer unless requested
        from ..data_model.joint_models.free_flyer import JointFreeFlyer
        if isinstance(jm, JointFreeFlyer) and not show_free_flyer:
            continue

        # One slider per scalar DOF in this joint
        iv = model.idx_vs[j]
        iq = model.idx_qs[j]
        for dof in range(jm.nv):
            v_idx = iv + dof
            q_idx = iq + dof

            lo = float(model.lower_pos_limit[v_idx])
            hi = float(model.upper_pos_limit[v_idx])
            if lo >= hi:
                lo, hi = -3.14159, 3.14159
            init_val = float(state["q"][q_idx].clamp(lo, hi))

            label = f"{model.joint_names[j]}" if jm.nv == 1 else \
                    f"{model.joint_names[j]}[{dof}]"

            def _make_cb(qi: int, vi: int, lo: float, hi: float):
                def _cb(event) -> None:
                    val = float(event.target.value)
                    q = state["q"].clone()
                    q[qi] = max(lo, min(hi, val))
                    state["q"] = q
                    scene.update_from_q(q)
                return _cb

            with folder:
                slider = server.gui.add_slider(  # type: ignore[attr-defined]
                    label,
                    min=lo, max=hi, step=(hi - lo) / 200.0,
                    initial_value=init_val,
                )
                slider.on_update(_make_cb(q_idx, v_idx, lo, hi))
