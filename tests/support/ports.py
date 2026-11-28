"""A free loopback port, for the tests that start a real viser server."""

from __future__ import annotations

import socket


def free_port() -> int:
    """A TCP port on ``127.0.0.1`` that no socket holds when the call returns; another process may take it before the
    caller binds it."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
