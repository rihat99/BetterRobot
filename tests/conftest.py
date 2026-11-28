"""Settings shared by every test folder: the internal checks are on and the plugin ``better_robot.testing.no_skips``
fails every skip or xfail."""

from __future__ import annotations

import os

# Every test tier runs the internal invariants. Set before ``pytest_plugins`` imports better_robot,
# because ``_internal_checks`` reads the switch once, at import; an explicit value (``0`` for timing) wins.
os.environ.setdefault("BETTER_ROBOT_INTERNAL_CHECKS", "1")

pytest_plugins = ["better_robot.testing.no_skips"]
