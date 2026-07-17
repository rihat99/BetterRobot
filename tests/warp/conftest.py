"""Shared configuration for the optional Warp prototype tests."""

from __future__ import annotations

import os


_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "main")
os.environ["WARP_CACHE_PATH"] = f"/tmp/betterrobot-warp-tests-{_WORKER}"
