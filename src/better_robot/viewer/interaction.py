"""``interaction.py`` — GizmoHandle and FramePicker (stubs).

See ``docs/12_VIEWER.md §10``.
"""

from __future__ import annotations


class GizmoHandle:
    """Draggable SE(3) gizmo handle (stub).

    See ``docs/12_VIEWER.md §10.1``.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("see docs/12_VIEWER.md §10.1")


class FramePicker:
    """Click-to-select frame picker (stub).

    Fires a callback with the frame name whenever the user clicks.

    See ``docs/12_VIEWER.md §10.3``.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("see docs/12_VIEWER.md §10.3")
