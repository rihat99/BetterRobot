"""Name → id lookup helpers for joints, bodies, and frames.

Thin utility module used by ``Model`` to build ``joint_name_to_id``,
``body_name_to_id``, ``frame_name_to_id``.

See ``docs/concepts/model_and_data.md §2``.
"""

from __future__ import annotations


def build_name_to_id(names: tuple[str, ...]) -> dict[str, int]:
    """Return a dict mapping each name to its index in ``names``.

    Raises ``ValueError`` on duplicate names. See docs/concepts/model_and_data.md §2.
    """
    raise NotImplementedError("see docs/concepts/model_and_data.md §2")
