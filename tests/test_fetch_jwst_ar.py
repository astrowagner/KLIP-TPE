"""``scripts/fetch_jwst_ar.py``: the pure helpers, offline.

MAST is unreachable from this session, so the queries cannot be tested here.  What can be
is everything that decides what the queries MEAN: which rows are real data, which pointing
is which, and the sort order.  Each of these has already been wrong once.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

import fetch_jwst_ar as F                                            # noqa: E402


class _Row(dict):
    pass


def _table(rows):
    return [_Row(proposal_id=p, target_name=t, filters=f, instrument_name=i)
            for p, t, f, i in rows]


def test_planned_observations_are_told_from_delivered_ones():
    """GO 11225 listed sixteen public observations and had no products behind any of them.

    MAST publishes approved-but-undelivered observations in CAOM with dataRights already
    set from the planned release date.  A download of one succeeds, downloads nothing and
    leaves an empty directory.
    """
    # the placeholder visit group, and the bare instrument instead of a detector
    assert F._planned("jw11225001001_xx101_00001_miri")
    assert F._planned("jw11225003001_xx10a_00009_miri")
    # delivered data, from programmes whose downloads work
    for oid in ("jw01386001001_04101_00001_nrcalong",       # HIP 65426 ERS
                "jw04014001001_03106_00001_mirimage",       # MWC 758, the control
                "jw01193002001_02101_00002_mirimage",       # Fomalhaut GTO
                "jw06122009001_03102_00001_nrcalong"):      # EV Lac
        assert not F._planned(oid), oid


def test_either_placeholder_alone_is_enough():
    """Both signals appear together on GO 11225, but neither is documented, so neither is
    relied on alone."""
    assert F._planned("jw11225001001_xx101_00001_mirimage")   # placeholder visit only
    assert F._planned("jw11225001001_04101_00001_miri")       # bare instrument only


def test_role_separates_science_from_its_supporting_pointings():
    assert F._role("AU_Mic") == "sci"
    assert F._role("AU_Mic_psf_reference") == "ref"          # NOT science, despite the prefix
    assert F._role("BKG-AU_Mic") == "bkg"
    assert F._role("BKG-AU_Mic_psf_reference") == "bkg"      # background wins over reference
    assert F._role("L-98-59-BACKGROUND") == "bkg"
    assert F._role("HD-4907-reference") == "ref"
    assert F._role("REFSTAR") == "sci"                       # a name, not a role


def test_programmes_sort_numerically():
    """proposal_id is a string, so the default sort put 10758 between 1046 and 1193 and a
    programme's rows were not contiguous."""
    rows = _table([("10758", "a", "F1140C", "MIRI/CORON"), ("1046", "b", "F1065C", "MIRI/CORON"),
                   ("1193", "c", "F1550C", "MIRI/CORON"), ("11225", "d", "F1140C", "MIRI/CORON")])
    assert [r[0] for r in F._rows(rows)] == ["1046", "1193", "10758", "11225"]


def test_rows_are_deduplicated():
    rows = _table([("1193", "VEGA", "F1550C", "MIRI/CORON")] * 3)
    assert len(F._rows(rows)) == 1
