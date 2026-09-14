"""Shared fixtures for the klip_tpe test suite.

Everything here is deliberately tiny so the whole suite stays well under a few
minutes on two cores: 2 synthetic partitions, 100x100 images, budgets of ~15
evaluations per annulus.
"""
from __future__ import annotations

import numpy as np
import pytest

from klip_tpe import (CalibrationConfig, MawetPeakSNR, Objective, Param, PartitionedReducer,
                      PositionSampler, RunConfig, SearchSpace, ValidationConfig)
from klip_tpe.space import kgrid
from klip_tpe.synthetic import make_synthetic_partitions

PARTS = ["n1", "n2"]


def build_synthetic_run(**overrides):
    """Reducer / space / objective / sampler / config for a tiny two-partition run.

    Mirrors the pattern in /tmp/resume_test.py: per-partition ``bin`` and ``k_klip``
    blocks plus a two-slot partition selection, clean-subtracted Mawet objective,
    two annuli with 16/14 evaluations.  ``overrides`` go into :class:`RunConfig`.
    """
    reds = make_synthetic_partitions(PARTS, k_opts=[6, 12])
    red = PartitionedReducer(reds)
    block = [Param("bin", 5, 30, "int", default=12),
             Param("k_klip", 1, 30, "int", grid=kgrid(30), default=5)]
    space = SearchSpace().replicate(block, PARTS).with_selection("two_slot", partitions=PARTS)
    obj = Objective(MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm), clean_subtract=True)
    samp = PositionSampler(fwhm_as=red.fwhm * red.pxscale)
    kw = dict(ann_edges=[8, 30, 45], n_iter=[16, 14], n_init=6, seed=11, defaults={"k_klip": 5},
              validation=ValidationConfig(n_top=2, n_valid=2),
              calibration=CalibrationConfig(recal_check=6))
    kw.update(overrides)
    cfg = RunConfig(**kw)
    return red, space, obj, samp, cfg


@pytest.fixture
def synthetic_run():
    return build_synthetic_run


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def simple_space():
    """Two global floats + one int with a grid + one categorical."""
    return SearchSpace([
        Param("a", -1.0, 1.0, default=0.0),
        Param("b", 0.0, 10.0, default=5.0),
        Param("k", 1, 30, "int", grid=kgrid(30), default=5),
        Param("mode", 0, 2, "categorical", choices=["x", "y", "z"], default="y"),
    ])


@pytest.fixture
def partition_space():
    """Replicated block over three partitions with two-slot selection."""
    block = [Param("bin", 1, 20, "int", default=4),
             Param("k_klip", 1, 30, "int", grid=kgrid(30), default=5),
             Param("angsep", 0.0, 3.0, default=0.5)]
    sp = SearchSpace([Param("filter", 0, 20, "int", default=10)])
    return sp.replicate(block, ["p0", "p1", "p2"]).with_selection("two_slot", partitions=["p0", "p1", "p2"])


@pytest.fixture(autouse=True, scope="session")
def _never_skip_panels():
    """Give the render thread all the time it needs, for every test but the gate's own.

    In production a panel is skipped when the render thread is still busy, so the live
    window's lag stays the cost of one panel instead of growing with the run (see
    ``LiveDisplay._render_room``).  The rest of the suite is asserting that every requested
    panel is written -- the progress movie and the resumed history both depend on it -- and
    a synthetic run evaluates far faster than it draws, so without this the gate would fire
    and those assertions would be measuring thread scheduling.  ``tests/test_display_lag.py``
    sets its own grace per instance, which still wins over this.

    Session-scoped on purpose: some of those runs are built by *module*-scoped fixtures,
    which are set up before any function-scoped fixture gets a chance to patch anything.
    """
    from klip_tpe.display import LiveDisplay
    mp = pytest.MonkeyPatch()
    mp.setattr(LiveDisplay, "RENDER_GRACE", 600.0)
    yield
    mp.undo()
