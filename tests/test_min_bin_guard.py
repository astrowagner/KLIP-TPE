"""The temporal-bin floor: a cost guard, and the one that can rescue a run in flight.

Measured on the four-night LMIRCam RX J0534 search (484 evaluations, 4 partitions of
~1500 frames each):

    bin      1   n=  3   median 8300 s   max 12371 s
    bin    2-9   n= 19   median   35 s   max   170 s
    bin  10-29   n= 61   median   26 s   max    36 s
    bin    30+   n=401   median   23 s   max    34 s

Three evaluations out of 484 had taken two thirds of the run's wall clock, and eval 445 ran
3 h 26 m between neighbours of 32 s and 22 s.  ``bin`` is how many consecutive frames are
mean-combined before KLIP, so the reference library and its covariance scale with
``nframes / bin``; at ``bin = 1`` nothing is combined and each partition carries its full
~1500 frames.  The generic space spans ``bin`` from 1 to ``nframes/8`` -- correct for the
61-frame beta Pic cube it was written for, wrong for a 1500-frame night.  The NEAR space has
always used the IDL production range of 5-30 and shows none of this.

The floor is a *projection* as well as a range because a checkpoint restores the recorded
parameter bounds (the history was produced under them) but takes its projection hook from
the driver on every resume -- so this rescues a running search without discarding its
evaluations.
"""
from __future__ import annotations

import numpy as np
import pytest

from klip_tpe import Param, SearchSpace
from klip_tpe.feasibility import MinBinGuard, compose


def _space(lo=1, hi=200, partitions=("n1", "n2")):
    sp = SearchSpace()
    for pid in partitions:
        sp.add(Param(f"bin_{pid}", lo, hi, "int", default=50, base="bin", partition=pid))
        sp.add(Param(f"k_klip_{pid}", 1, 20, "int", default=10, base="k_klip", partition=pid))
    return sp


def _bins(sp, x):
    return [x[i] for i, p in enumerate(sp.params) if p.base == "bin"]


def test_a_bin_below_the_floor_is_raised_to_it():
    sp = _space()
    x = np.array([1.0, 10.0, 1.0, 10.0])
    out = MinBinGuard(5)(x, sp)
    assert _bins(sp, out) == [5.0, 5.0]


def test_a_bin_at_or_above_the_floor_is_untouched():
    sp = _space()
    x = np.array([5.0, 10.0, 137.0, 10.0])
    out = MinBinGuard(5)(x, sp)
    assert _bins(sp, out) == [5.0, 137.0]
    assert out.tolist() == x.tolist(), "nothing but bin may be touched"


def test_every_partition_block_is_floored():
    """``bin`` is usually tied across partitions, but the guard must not assume it."""
    sp = _space(partitions=("n1", "n2", "n3", "n4"))
    x = np.array([1.0, 10.0, 2.0, 10.0, 60.0, 10.0, 3.0, 10.0])
    out = MinBinGuard(5)(x, sp)
    assert _bins(sp, out) == [5.0, 5.0, 60.0, 5.0]


def test_the_floor_never_exceeds_the_parameter_range():
    """A short sequence whose whole range is below the floor must stay inside its bounds."""
    sp = _space(lo=1, hi=3)
    out = MinBinGuard(5)(np.array([1.0, 10.0, 1.0, 10.0]), sp)
    assert _bins(sp, out) == [3.0, 3.0]


def test_it_reports_how_many_it_snapped():
    sp = _space()
    g = MinBinGuard(5)
    g(np.array([1.0, 10.0, 99.0, 10.0]), sp)
    assert g.last_snapped == 1
    g(np.array([99.0, 10.0, 99.0, 10.0]), sp)
    assert g.last_snapped == 0


def test_a_floor_of_one_is_a_no_op():
    """The default: short cubes must keep the whole range."""
    sp = _space()
    x = np.array([1.0, 10.0, 1.0, 10.0])
    assert MinBinGuard(1)(x, sp).tolist() == x.tolist()


def test_a_space_without_a_bin_dim_is_left_alone():
    sp = SearchSpace()
    sp.add(Param("k_klip", 1, 20, "int", default=10, base="k_klip"))
    x = np.array([3.0])
    assert MinBinGuard(5)(x, sp).tolist() == x.tolist()


def test_the_input_vector_is_not_mutated():
    sp = _space()
    x = np.array([1.0, 10.0, 1.0, 10.0])
    MinBinGuard(5)(x, sp)
    assert x.tolist() == [1.0, 10.0, 1.0, 10.0]


def test_it_composes_ahead_of_the_reference_census():
    """Order matters: the reference-count guard counts references in the *executed* frame
    set, so it has to see the floored bin, not the proposed one."""
    seen = {}

    def census(x, space, **kw):
        seen["bins"] = _bins(space, x)
        return x

    sp = _space()
    compose(MinBinGuard(5), census)(np.array([1.0, 10.0, 1.0, 10.0]), sp)
    assert seen["bins"] == [5.0, 5.0]


# ------------------------------------------------------------------- wired into the driver

def test_make_guard_defaults_to_no_floor_and_honours_bin_min():
    """Short-sequence instruments must be unaffected; the floor is opt-in."""
    import klip_tpe.instruments.generic as G

    def describe_of(g):
        d = g.describe()
        return [s.get("name") for s in d.get("steps", [])] if d.get("name") == "compose" else [d.get("name")]

    class _R:
        fwhm = 4.0
        lam_over_d_px = 4.0

        def frame_tags(self):
            return None

        @property
        def data(self):
            class D:
                nframes = 100
            return D()

    class _P:
        reducers = {"n1": _R()}

        def frame_angles(self, pid=None):
            return np.linspace(0, 30, 100)

        def frame_tags(self, pid=None):
            return None

    red = _P()
    assert "min_bin_guard" not in describe_of(G.make_guard(red))
    assert "min_bin_guard" in describe_of(G.make_guard(red, bin_min=5))


def test_the_rxj_driver_floors_the_bin_both_ways():
    """Range for a fresh run, projection for a resumed one -- a resume cannot change the
    range, so only the projection can rescue the search that is already going."""
    import os
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "scripts", "run_rxj0534.py")).read()
    assert "bin_range=(a.bin_min, b_hi)" in src
    assert "bin_min=a.bin_min" in src
    assert '"--bin-min"' in src
