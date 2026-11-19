"""Shared test fixtures."""
import os
import sys
from pathlib import Path

import pytest

# Child processes started by the tests (python -c / child.py) must find pfb_app without an install.
_SRC = str(Path(__file__).resolve().parents[1] / "src")
os.environ["PYTHONPATH"] = _SRC + os.pathsep + os.environ.get("PYTHONPATH", "")


@pytest.fixture(autouse=True)
def keep_main_module():
    """AppTest installs its script as sys.modules["__main__"] and leaves it there; a later
    multiprocessing "spawn" child (test_worker) would then re-run that script and die. Every
    test gets the original __main__ back afterwards."""
    main = sys.modules.get("__main__")
    yield
    if main is not None:
        sys.modules["__main__"] = main


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_auth: run ui/app.py's sign-in gate for real (no synthetic admin)")


@pytest.fixture(autouse=True)
def quick_password_hashes(monkeypatch):
    """argon2 with tiny costs in tests (the product keeps the library's defaults)."""
    from argon2 import PasswordHasher
    from pfb_app.services import auth
    monkeypatch.setattr(auth, "HASHER", PasswordHasher(time_cost=1, memory_cost=8,
                                                       parallelism=1))
    monkeypatch.setattr(auth, "_DUMMY", [])


@pytest.fixture(scope="session", autouse=True)
def signed_in_as_admin_everywhere():
    """Every AppTest on ui/app.py runs as a synthetic administrator (id 1, no DB row), so the
    screens' own tests need no sign-in (session scope: module-scoped screen fixtures too);
    tests marked `real_auth` meet the real gate (`signed_in_as_admin`). The product has no
    such bypass: this patches the gate's lookup inside the test process only."""
    from pfb_app.services import auth
    from pfb_app.ui import auth_ui
    user = auth.User(id=1, login="tester", display_name="テスト 管理", role="admin")
    patch = pytest.MonkeyPatch()
    REAL_RESOLVE.append(auth_ui.resolve)
    patch.setattr(auth_ui, "resolve", lambda settings: auth.Session(0, user, 4e9))
    yield
    patch.undo()


REAL_RESOLVE: list = []


@pytest.fixture(autouse=True)
def signed_in_as_admin(request, monkeypatch):
    if request.node.get_closest_marker("real_auth"):
        from pfb_app.ui import auth_ui
        monkeypatch.setattr(auth_ui, "resolve", REAL_RESOLVE[0])


@pytest.fixture(autouse=True)
def plenty_of_disk(monkeypatch):
    """The low-disk guard reads the real free space; on a nearly full machine every screen would
    warn and the screen tests would fail for a reason that has nothing to do with them. Tests
    that exercise the guard pass their own `usage=` and are not affected."""
    from collections import namedtuple
    from pfb_app.ops import disk
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(disk, "_usage", lambda path: usage(2 * 1024**4, 1024**3, 1024**4))
