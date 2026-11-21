"""How pyKLIP's per-reduction worker pools are started.

``klip_parallelized`` builds a new ``multiprocessing.Pool`` on every call.  Under ``spawn``
(macOS's default) each worker is a new interpreter importing numpy, scipy and pyklip, which
doubled or tripled a MIRI evaluation, and under ``fork`` (Linux's) a worker forked beside the
run's threads can start with a lock taken for good and hang the pool (FLIP note, 2026-11-20).
The backend hands pyKLIP a forkserver context on Linux and macOS, ``KLIP_TPE_PYKLIP_START``
overrides, and the pools' workers never import the run's main script, so an unguarded script
works too.
"""
from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
import textwrap
import types

import pytest

from klip_tpe.backends import pyklip as B


def _par():
    return types.SimpleNamespace(mp=multiprocessing)


@pytest.mark.parametrize("platform,name", [("linux", "forkserver"), ("win32", "spawn")])
def test_linux_gets_a_forkserver_too_and_windows_keeps_spawn(monkeypatch, platform, name):
    monkeypatch.delenv("KLIP_TPE_PYKLIP_START", raising=False)
    monkeypatch.setattr(B.sys, "platform", platform)
    par = _par()
    B._pyklip_start_method(par)
    assert par.mp.get_start_method() == name and par._klip_tpe_start == name


def test_the_main_module_is_hidden_only_while_a_pool_starts():
    m = sys.modules["__main__"]
    path, spec = getattr(m, "__file__", None), getattr(m, "__spec__", None)
    with B._main_hidden():
        assert "__file__" not in vars(m) and m.__spec__ is None
    assert getattr(m, "__file__", None) == path and getattr(m, "__spec__", None) is spec


@pytest.mark.skipif(sys.platform == "win32", reason="forkserver")
def test_an_unguarded_script_gets_its_pool(tmp_path):
    """A forkserver worker would import the main script first and run it again, and a script
    without a main guard never got its pool (the FLIP note's test)."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(B.__file__)))
    script = tmp_path / "unguarded.py"
    script.write_text(textwrap.dedent(f"""
        import math, sys, types
        sys.path.insert(0, {os.path.dirname(repo)!r})
        from klip_tpe.backends import pyklip as B
        par = types.SimpleNamespace()
        B._pyklip_start_method(par)
        pool = par.mp.Pool(2, None, (), 50)
        print(sum(pool.map(math.sqrt, range(10))) > 0, par._klip_tpe_start)
        pool.close(); pool.join()
        print("DONE")
    """))
    env = dict(os.environ)
    env.pop("KLIP_TPE_PYKLIP_START", None)
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0 and "True forkserver" in r.stdout and "DONE" in r.stdout, r.stdout + r.stderr


def test_macos_gets_a_forkserver(monkeypatch):
    monkeypatch.delenv("KLIP_TPE_PYKLIP_START", raising=False)
    monkeypatch.setattr(B.sys, "platform", "darwin")
    par = _par()
    B._pyklip_start_method(par)
    assert par.mp.get_start_method() == "forkserver"
    assert all(hasattr(par.mp, a) for a in ("Array", "Pool", "cpu_count")), "what pyklip.parallelized uses"


@pytest.mark.parametrize("name", ["spawn", "fork"])
def test_the_variable_overrides(monkeypatch, name):
    monkeypatch.setenv("KLIP_TPE_PYKLIP_START", name)
    monkeypatch.setattr(B.sys, "platform", "darwin")
    par = _par()
    B._pyklip_start_method(par)
    assert par.mp.get_start_method() == name
