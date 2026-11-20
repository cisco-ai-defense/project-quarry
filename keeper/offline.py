"""Offline lock: the Keeper never reaches the network.

Two fences, both fail closed:

1. ``require_offline_config`` refuses any Quarry configuration that names an AI model,
   a model endpoint, or any web address. Phase 1 runs local rules only.
2. ``network_lock`` blocks the network libraries Quarry's optional model add-on would use,
   and makes every outbound socket connection raise, for as long as a check runs.
   Local-only plumbing (Unix sockets that Python's own event loop uses) still works.
"""

from __future__ import annotations

import contextlib
import importlib.abc
import socket
import sys
from collections.abc import Generator, Iterator, Mapping
from typing import Any

# Libraries that exist only to talk to the network. The Quarry core never imports them;
# its optional model add-on does. If anything tries while a check runs, it fails.
BLOCKED_MODULES = frozenset(
    {
        "aiohttp",
        "boto3",
        "botocore",
        "httpx",
        "litellm",
        "requests",
        "urllib3",
    }
)

_ADDRESS_MARKERS = ("http://", "https://", "ftp://", "ws://", "wss://", "ssh://")


class OfflineViolation(RuntimeError):
    """Something tried to use the network, or a config would let it."""


def require_offline_config(document: Mapping[str, Any]) -> None:
    """Refuse a Quarry config mapping that could make a network call."""

    if document.get("models"):
        raise OfflineViolation(
            "The config lists AI models. The Keeper runs local rules only, so it refuses to load it."
        )
    if document.get("allow_stub_probers"):
        raise OfflineViolation(
            "The config turns on test-only stub checks. The Keeper refuses them outside tests."
        )
    for path, value in _walk(document):
        if isinstance(value, str) and any(marker in value.lower() for marker in _ADDRESS_MARKERS):
            raise OfflineViolation(f"The config names a web address at '{path}'. Refused.")


def _walk(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")
    else:
        yield path, value


class _BlockNetworkImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.split(".", 1)[0] in BLOCKED_MODULES:
            raise OfflineViolation(f"Blocked import of network library '{fullname}'.")
        return None


def _is_local_family(sock: socket.socket) -> bool:
    return getattr(socket, "AF_UNIX", None) is not None and sock.family == socket.AF_UNIX


@contextlib.contextmanager
def network_lock() -> Generator[None]:
    """Block network imports and outbound connections while the body runs."""

    already_loaded = sorted(name for name in BLOCKED_MODULES if name in sys.modules)
    if already_loaded:
        raise OfflineViolation(
            "Network libraries are already loaded in this program: "
            + ", ".join(already_loaded)
            + ". The Keeper will not run a check next to them."
        )

    finder = _BlockNetworkImports()
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        if _is_local_family(self):
            return original_connect(self, address)
        raise OfflineViolation(f"Blocked an outbound connection to {address!r}.")

    def guarded_connect_ex(self: socket.socket, address: Any) -> Any:
        if _is_local_family(self):
            return original_connect_ex(self, address)
        raise OfflineViolation(f"Blocked an outbound connection to {address!r}.")

    def blocked_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        raise OfflineViolation(f"Blocked an outbound connection to {address!r}.")

    def blocked_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        raise OfflineViolation(f"Blocked a network name lookup for {host!r}.")

    sys.meta_path.insert(0, finder)
    setattr(socket.socket, "connect", guarded_connect)  # noqa: B010 - typed stubs reject a patched function
    setattr(socket.socket, "connect_ex", guarded_connect_ex)  # noqa: B010 - typed stubs reject a patched function
    setattr(socket, "create_connection", blocked_create_connection)  # noqa: B010 - typed stubs reject a patched function
    setattr(socket, "getaddrinfo", blocked_getaddrinfo)  # noqa: B010 - typed stubs reject a patched function
    try:
        yield
    finally:
        setattr(socket, "getaddrinfo", original_getaddrinfo)  # noqa: B010 - typed stubs reject a patched function
        setattr(socket, "create_connection", original_create_connection)  # noqa: B010 - typed stubs reject a patched function
        setattr(socket.socket, "connect_ex", original_connect_ex)  # noqa: B010 - typed stubs reject a patched function
        setattr(socket.socket, "connect", original_connect)  # noqa: B010 - typed stubs reject a patched function
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(finder)
