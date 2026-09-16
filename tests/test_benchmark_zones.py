"""Every benchmark must search a zone that contains its known companion.

The point of running the same protocol on beta Pic, HD 95086 and HIP 65426 is that each
field has a real planet in it, so each run carries an end-to-end check that the injections
and the recovery agree with something nobody injected.  G2 did not: it took run C's FIRST
annulus, [20, 45] px = 0.245-0.551", while HD 95086 b sits at 0.620" = 50.6 px -- 5.6 px,
1.26 FWHM, beyond the outer edge.  The companion marker was drawn clipped by the annulus
boundary in every slot, which is how it was noticed, and the ``known=[HD]`` exclusion G2
passed to the noise ring was a no-op because the planet was never in the zone.

E2/F2 (beta Pic b at 16.6 px in [8, 22]) and H2 (HIP 65426 b at 13.2 px in [6, 20]) were
both fine, so this was G2 alone.  It now searches C's full range, [20, 45, 75].
"""
from __future__ import annotations

import os
import re

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_demos() -> str:
    for p in (os.path.join(HERE, "paper_runs", "run_demos.py"),
              os.path.join(os.path.dirname(HERE), "paper_runs", "run_demos.py")):
        if os.path.exists(p):
            return open(p).read()
    pytest.skip("run_demos.py not next to the package")


# (name, mas/px, companion separation ["], annulus edges [px])
ZONES = [
    ("E2/F2  beta Pic  NACO L'",   27.19, 0.452, [8, 22]),
    ("G2     HD 95086  IRDIS K1",  12.25, 0.620, [20, 45, 75]),
    ("H2     HIP 65426 NIRCam",    62.60, 0.826, [6, 20]),
]


@pytest.mark.parametrize("name,mas,rho,edges", ZONES)
def test_the_known_companion_falls_inside_a_searched_annulus(name, mas, rho, edges):
    p = rho / (mas / 1000.0)
    ann = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    hit = [i for i, (lo, hi) in enumerate(ann) if lo <= p <= hi]
    assert hit, (f"{name}: companion at {p:.1f} px is outside every annulus {ann}; "
                 f"the run has no real-companion check")


def test_hd95086_b_is_in_the_second_annulus_not_the_first():
    """The specific geometry, so the edges cannot drift back."""
    p = 0.620 / 0.01225
    assert p == pytest.approx(50.6, abs=0.2)
    assert not (20 <= p <= 45), "this is what was wrong"
    assert 45 <= p <= 75


def test_nircam_lw_scale_is_63_not_28_mas():
    """62.60 mas/px comes from PIXAR_A2 on the ERS 1386 calints (NIRCam long-wavelength).
    Using ~28 mas/px instead makes HIP 65426 b land at 29.6 px, outside H2's [6, 20], and
    would have sent this whole check the wrong way."""
    assert 0.826 / 0.0626 == pytest.approx(13.2, abs=0.1)
    assert 6 <= 0.826 / 0.0626 <= 20


def test_run_G2_searches_both_annuli_with_one_contrast_each():
    src = _run_demos()
    m = re.search(r'_bench_hi\(\s*"G2".*?\)\n', src, re.S)
    assert m, "run_G2's _bench_hi call not found"
    call = m.group(0)
    assert "[20, 45, 75]" in call, "G2 must search C's full range, not just its first annulus"
    assert "7.946e-6" in call and "6.201e-6" in call, \
        "one calibrated contrast per annulus, both on the absolute axis"
    assert "5.899e-9" not in call, "the pre-ff20c4b contrast must not come back"


def test_forced_contrasts_are_per_annulus():
    """A scalar still means 'this one everywhere' for the single-annulus benches, but a
    mismatched count is an error rather than a silent reuse of annulus 1's contrast."""
    import importlib.util
    p = next((q for q in (os.path.join(HERE, "paper_runs", "run_demos.py"),
                          os.path.join(os.path.dirname(HERE), "paper_runs", "run_demos.py"))
              if os.path.exists(q)), None)
    if p is None:
        pytest.skip("run_demos.py not next to the package")
    spec = importlib.util.spec_from_file_location("rd_zones", p)
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    assert rd._forced_list(2e-4, [6, 20]) == [2e-4]
    assert rd._forced_list(2e-4, [20, 45, 75]) == [2e-4, 2e-4]
    assert rd._forced_list((7.946e-6, 6.201e-6), [20, 45, 75]) == [7.946e-6, 6.201e-6]
    with pytest.raises(ValueError):
        rd._forced_list((1.0, 2.0, 3.0), [20, 45, 75])
