"""Every benchmark must search a zone that contains its known companion.

The point of running the same protocol on beta Pic, HD 95086 and HIP 65426 is that each
field has a real planet in it, so each run carries an end-to-end check that the injections
and the recovery agree with something nobody injected.  G2 did not: it took run C's FIRST
annulus, [20, 45] px = 0.245-0.551", while HD 95086 b sits at 0.620" = 50.6 px -- 5.6 px,
1.26 FWHM, beyond the outer edge.  The companion marker was drawn clipped by the annulus
boundary in every slot, which is how it was noticed, and the ``known=[HD]`` exclusion G2
passed to the noise ring was a no-op because the planet was never in the zone.

E2/F2 (beta Pic b at 16.6 px in [8, 22]) and H2 (HIP 65426 b at 13.2 px in [6, 20]) were
both fine, so this was G2 alone.  Containment alone was not enough either: C's own split,
[20, 45, 75], contains the planet but only 1.26 FWHM inside the inner edge.  G2 now uses
[20, 36, 66], which centres it -- 3.3 FWHM from the inner edge, 3.5 from the outer -- with
contrasts measured for those zones rather than inherited from C's.
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
    ("G2     HD 95086  IRDIS K1",  12.25, 0.620, [20, 36, 66]),
    ("H2     HIP 65426 NIRCam",    62.60, 0.826, [6, 20]),
]


@pytest.mark.parametrize("name,mas,rho,edges", ZONES)
def test_the_known_companion_falls_inside_a_searched_annulus(name, mas, rho, edges):
    p = rho / (mas / 1000.0)
    ann = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    hit = [i for i, (lo, hi) in enumerate(ann) if lo <= p <= hi]
    assert hit, (f"{name}: companion at {p:.1f} px is outside every annulus {ann}; "
                 f"the run has no real-companion check")


def test_hd95086_b_sits_well_inside_annulus_2_not_on_its_edge():
    """Containment is not enough -- it has to be clear of both edges.  C's own split,
    [20, 45, 75], does contain the planet but leaves it 1.26 FWHM inside the inner edge,
    which is still on the boundary.  [20, 36, 66] centres it."""
    p = 0.620 / 0.01225
    fwhm = 4.44
    assert p == pytest.approx(50.6, abs=0.2)
    assert not (20 <= p <= 45), "the original [20, 45] did not contain it at all"
    for lo, hi, ok in ((45, 75, False), (36, 66, True)):
        margin = min(p - lo, hi - p) / fwhm
        assert (margin >= 3.0) is ok, f"[{lo}, {hi}] margin {margin:.2f} FWHM"
    assert min(p - 36, 66 - p) / fwhm == pytest.approx(3.29, abs=0.1)


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
    assert "[20, 36, 66]" in call, "G2 must search the zones HD 95086 b is centred in"
    assert "1.112e-5" in call and "4.773e-6" in call, \
        "the contrasts MEASURED for these zones by scripts/calibrate_bench_contrast.py"
    assert "5.899e-9" not in call, "the pre-ff20c4b contrast must not come back"
    assert "7.946e-6" not in call, \
        "C's contrast is a calibration of C's zone; these edges are not C's edges"


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


def test_every_forced_bench_contrast_is_on_the_current_flux_axis():
    """A forced contrast is a calibration on a flux axis.  ff20c4b moved the beta Pic axis
    from the template's own sum (4.3491) to VIP's published 3.3268e6 and HD 95086's by
    1347.0246; every forced constant that was not re-measured afterwards injects something
    else entirely.  E2's 1.31e-3 and F2's 2.087e-3 were "run A's / run B's calibration" on
    the OLD axis -- on the current one they put a 105-168 count peak into a cube whose
    beta Pic b peaks at ~50, so the benchmark was on a source brighter than the planet.

    These are the values scripts/calibrate_bench_contrast.py measures through
    Runner.calibrate on the current axis.  If one changes, re-measure; do not convert."""
    src = _run_demos()
    want = {"E2": ("3.0e-4", "[8, 22]"), "F2": ("3.0e-4", "[8, 22]"),
            "G2": ("(1.112e-5, 4.773e-6)", "[20, 36, 66]"), "H2": ("2.324e-04", "[6, 20]")}
    for tag, (contrast, edges) in want.items():
        m = re.search(r'_bench_hi\(\s*"%s".*?\)\n' % tag, src, re.S)
        assert m, f"run_{tag}'s _bench_hi call not found"
        call = m.group(0)
        assert contrast in call, f"{tag}: forced contrast must be the measured {contrast}"
        assert edges in call, f"{tag}: edges {edges}"
    # the old constants may stay in the comments that explain what replaced them, but not
    # as an ARGUMENT: check the call lines only (the same trap once caught H2's 5.270e1 in
    # a comment and let the real one through)
    calls = "\n".join(l for l in src.splitlines() if "_bench_hi(" in l or "forced=[" in l)
    for stale in ("1.31e-3", "2.087e-3", "5.899e-9", "5.270e1", "7.946e-6, 6.201e-6"):
        assert stale not in calls, f"pre-fix constant {stale} is still being passed"
    # the superseded single-annulus E and F use the same measured value
    assert src.count("forced=[3.0e-4]") == 2, "run_E and run_F carry the measured contrast too"
    assert "forced=[1.31e-3]" not in src and "forced=[2.087e-3]" not in src


def test_beta_pic_b_is_brighter_than_the_calibrated_injection():
    """Sanity of the measured number: 3.0e-4 gives median S/N ~4.2-4.5 at the default config,
    and beta Pic b (published 6.25e-4, Absil et al. 2013) is 2.1x brighter -- so the real
    planet should stand out at roughly twice that, which is what the runs see."""
    assert 6.25e-4 / 3.0e-4 == pytest.approx(2.08, abs=0.05)
    # and the OLD constants, on the CURRENT axis, are brighter than the planet itself
    for old in (1.31e-3, 2.087e-3):
        assert old > 6.25e-4, "this is why the old E2/F2 would not have been a benchmark"
