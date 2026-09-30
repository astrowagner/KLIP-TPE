"""How pyKLIP's per-reduction worker pools are started.

``klip_parallelized`` builds a new ``multiprocessing.Pool`` on every call; under ``spawn``
(macOS's default) each worker is a fresh interpreter importing numpy, scipy and pyklip, which
doubled or tripled a MIRI evaluation.  On macOS the backend now hands pyKLIP a forkserver
context; ``KLIP_TPE_PYKLIP_START`` overrides; elsewhere nothing changes.
"""
from __future__ import annotations

import multiprocessing
import types

import pytest

from klip_tpe.backends import pyklip as B


def _par():
    return types.SimpleNamespace(mp=multiprocessing)


def test_nothing_changes_off_macos_without_the_variable(monkeypatch):
    monkeypatch.delenv("KLIP_TPE_PYKLIP_START", raising=False)
    monkeypatch.setattr(B.sys, "platform", "linux")
    par = _par()
    B._pyklip_start_method(par)
    assert par.mp is multiprocessing


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
