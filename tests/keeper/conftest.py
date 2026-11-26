"""Run Keeper tests with a fresh network import state."""

import sys

import pytest

from keeper.offline import BLOCKED_MODULES


@pytest.fixture(autouse=True)
def isolate_network_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    # Other suites import network clients during collection. Restore these
    # modules after each Keeper test so the rest of the suite can still use them.
    for module_name in tuple(sys.modules):
        if module_name.split(".", 1)[0] in BLOCKED_MODULES:
            monkeypatch.delitem(sys.modules, module_name)
