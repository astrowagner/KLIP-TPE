"""The count line an annulus logs claims only the rings the probe at the sources saw.

b7c903c gave an annulus the cap leaves alone a line of its own (NEAR2 s6 note, section 4):
``annulus 1: 1 source a draw, as asked, and each separation's Mawet ring keeps at least 6
clean apertures``.  It was printed whenever the count was not lowered, and a line later the
probe at the sources could say the opposite: ``annulus 1: one source at 0.275" keeps 5 clean
apertures in its Mawet ring, fewer than 6, ...``.  The cap never asks a single source
(``Runner._ring_survives`` passes n <= 1), and its blank probe can differ from the sources' own
rings close to the star.  What must hold:

* where the probe at the sources finds a starved ring, the count line gives the count alone and
  the starved-ring line says what the ring keeps;
* where every ring keeps ``min_ring``, the count line says so, with and without the sidelobe rule;
* where the rings cannot be measured, the count line makes no claim;
* no annulus both claims full rings and reports a starved one.
"""
from __future__ import annotations

import re

import pytest

from klip_tpe import (GaussianPSF, KLIPReducer, MawetPeakSNR, Objective, Param, PartitionedReducer,
                      RunConfig, Runner, SearchSpace)
from klip_tpe.positions import PositionSampler
from klip_tpe.synthetic import synthetic_klip_dataset

from conftest import build_synthetic_run

CLAIM = re.compile(r"annulus (\d+): \d+ sources? a draw, .*keeps at least")
STARVED = re.compile(r'annulus (\d+): (?:one source|with \d+ sources, the one) at [0-9.]+" keeps')


def _contradictions(logs):
    """The annuli whose log both claims full rings and reports a starved one."""
    claimed = {m.group(1) for m in map(CLAIM.search, logs) if m}
    starved = {m.group(1) for m in map(STARVED.search, logs) if m}
    return sorted(claimed & starved)


def _klip_runner(tmp_path, edges, rmin, n_sources=1):
    px = 0.05
    ds = synthetic_klip_dataset(nframes=20, size=64, pa_span=60, seed=1)
    kr = KLIPReducer(ds, px, 3.89 * px / 206265 * 8.2, 8.2, outrad_cap=28,
                     injection_model=GaussianPSF(4.0, star_flux=1e4))
    red = PartitionedReducer({"n1": kr})
    space = SearchSpace()
    space.add(Param("k_klip", 1, 10, "int", default=5))
    obj = Objective(MawetPeakSNR(pxscale=red.pxscale, fwhm=red.fwhm))
    logs = []
    r = Runner(red, space, obj, PositionSampler(fwhm_as=red.fwhm * red.pxscale),
               RunConfig(ann_edges=edges, n_sources=n_sources, min_r_fwhm=rmin, seed=1), str(tmp_path),
               log=logs.append)
    return r, logs


def test_a_starved_single_source_gets_the_count_alone(tmp_path):
    r, logs = _klip_runner(tmp_path, [2, 9], 0.3)        # one source at 1.4 FWHM keeps 5 of 6
    assert r._nsrc(0) == 1
    assert "  annulus 1: 1 source a draw, as asked" in logs, logs
    assert not [l for l in logs if "keeps at least" in l], logs
    assert len([l for l in logs if STARVED.search(l)]) == 1, logs
    assert not _contradictions(logs), logs


@pytest.mark.parametrize("edges,rmin,n", [([2, 9], 0.3, 1), ([2, 9], 0.3, 2), ([8, 30], 1.0, 1),
                                          ([8, 30], 1.0, 3)])
def test_no_annulus_claims_full_rings_and_a_starved_one(tmp_path, edges, rmin, n):
    r, logs = _klip_runner(tmp_path, edges, rmin, n_sources=n)
    r._nsrc(0)
    assert not _contradictions(logs), logs
    # and the annulus gives its count, one way or the other
    assert [l for l in logs if "annulus 1: " in l and (" a draw" in l or " used" in l)], logs


@pytest.mark.parametrize("rule", ["off", "farthest"])
@pytest.mark.parametrize("probe,claims,starves", [((9, 0.5, 9), True, False),
                                                   ((2, 0.5, 2), False, True),
                                                   ((2, 0.5, 9), False, True),
                                                   (None, False, False)])
def test_the_count_line_follows_the_probe_at_the_sources(tmp_path, monkeypatch, rule, probe, claims,
                                                         starves):
    """Each answer the probe can give, with and without the sidelobe rule: rings that all keep
    enough, a ring starved at every azimuth, one starved at some, and rings it cannot measure."""
    monkeypatch.setattr(Runner, "_ring_clean", lambda self, ia, n, ndraw=2, at_sources=False: probe)
    logs = []
    red, space, obj, samp, cfg = build_synthetic_run(sidelobes=rule, n_sources=1, ann_edges=[8, 30],
                                                     n_iter=[4], n_init=3)
    r = Runner(red, space, obj, samp, cfg, str(tmp_path), log=logs.append)
    assert r._nsrc(0) == 1
    count = [l for l in logs if "annulus 1: 1 source a draw, as asked" in l]
    assert len(count) == 1, logs
    assert ("keeps at least" in count[0]) == claims, logs
    assert bool([l for l in logs if STARVED.search(l)]) == starves, logs
    assert not _contradictions(logs), logs
