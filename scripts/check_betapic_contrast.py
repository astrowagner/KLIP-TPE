#!/usr/bin/env python
"""Verify the beta Pictoris contrast axis against beta Pic b's published brightness.

    python scripts/check_betapic_contrast.py          # ~4 min, downloads the VIP data once

The NACO L' set is the one case in the tutorials where the star's flux has to be *imported*:
the distributed ``naco_betapic_psf.fits`` is a normalised template (flux 1.000000000 inside
r = 2.000 px) and the science frames are AGPM coronagraphic, so neither file says how bright
beta Pictoris is.  ``datasets.PHOTOMETRY`` carries VIP's published ``starphot`` for this very
cube, measured off the non-coronagraphic PSF and rescaled to the coronagraphic integration
time.  This script is the check that it is the right number, and that everything between the
injection and the contrast axis preserves it.

Method -- throughput-matched, with no model of the throughput anywhere:

  1. reduce clean with the DEFAULT parameters;
  2. at each test azimuth, reduce again with ONE injected source of known contrast at beta
     Pic b's own separation, and measure it in ``inj - clean``, where the speckle field has
     cancelled and only the fake is left;
  3. measure the planet in the clean image with the same matched filter;
  4. contrast(beta Pic b) = c_inj * peak_planet / peak_inj.

Planet and fake are point sources at the same radius reduced with the same parameters, so
the KLIP throughput divides out exactly and only the flux ratio is left -- PROVIDED they are
also of the same brightness.  KLIP is not linear in the source: a companion contributes to
the principal components that then subtract it, so a fake four times brighter than the planet
loses a quarter more of itself (the sweep at the end prints this).  So the contrast is solved
for as a fixed point instead of read off one injection: inject at the current estimate,
measure, re-inject at the answer, until the fake and the planet are equally bright and their
throughputs are therefore identical.

Expected: dL' = 7.78 against the 8.01 +/- 0.16 that Absil et al. (2013, A&A 559, L12)
published from these same data -- agreement at ~1.2 sigma.
"""
from __future__ import annotations

import sys

import numpy as np

from klip_tpe import datasets
from klip_tpe.instruments import generic
from klip_tpe.metrics import Source, mawet_peak_snr, radprof
from klip_tpe.reducer import ReductionRequest

PUB_DMAG, PUB_ERR = 8.01, 0.16
PUB_REF = "Absil et al. 2013, A&A 559, L12 -- these same data"
PUB = 10.0 ** (-0.4 * PUB_DMAG)

RHO, PA = 0.452, 211.9               # beta Pic b, 2013-01-31 (Absil: 452 +/- 10 mas, PA 211.2 +/- 1.3)
INRAD, OUTRAD = 8, 26                # annulus bracketing 0.452" = 16.6 px
#: azimuths for the fakes: clear of beta Pic b AND of both debris-disc ansae (PA 29 / 209)
TEST_PA = [70.0, 110.0, 150.0, 260.0, 300.0, 340.0]
#: multiples of the fixed point, to show how far the recovery is from linear
SWEEP = [0.25, 0.5, 2.0, 4.0]
TOL_MAG = 0.45                       # fail above this; the expected offset is 0.24 mag


def measure(img, red, rho, theta, *, known=(), pixel_mask=None):
    """(matched-filter peak, S/N, details) at one position."""
    snr, det = mawet_peak_snr(radprof(img), [rho], [theta], red.pxscale, red.fwhm,
                              kernel=red.matched_filter_kernel(rho), known=known,
                              pixel_mask=pixel_mask, return_details=True)
    return float(det[0]["peak"]), float(snr[0]), det[0]


def main(verbose=True):
    def log(m):
        if verbose:
            print(m, flush=True)

    files = datasets.fetch("naco_betapic", quiet=not verbose)
    inst = datasets.INSTRUMENT["naco_betapic"]
    phot = datasets.PHOTOMETRY["naco_betapic"]
    ds = generic.load_cube(files["cube"], files["angles"], psf=files["psf"], name="betapic")
    tpl = ds.meta["psf"]

    a = generic.aperture_sum(tpl, (tpl.shape[1] - 1) / 2.0, (tpl.shape[0] - 1) / 2.0,
                             phot["aperture_px"])
    sf = generic.star_flux_from_aperture_photometry(tpl, phot["starphot"], phot["aperture_px"])
    log(f"template {tpl.shape}: sum {tpl.sum():.6g}, flux inside r={phot['aperture_px']} px "
        f"= {a:.9f}  <- normalised, so its own counts are not beta Pic")
    log(f"starphot {phot['starphot']:.1f} in that aperture ({phot['ref']})")
    log(f"  -> star_flux = {sf:.6g} in the template's total-flux normalisation")

    red = generic.make_reducer({"betapic": ds}, star_flux=sf, max_workers="auto",
                               log=lambda s: None, **inst)
    space = generic.make_space(red, k_klip_max=30)
    space.project = generic.make_guard(red, k_max=30)
    from klip_tpe.metrics import pa_wedge_mask
    pmask = pa_wedge_mask(ds.cube.shape[-2:], [(29.0, 20.0), (209.0, 20.0)])   # disc ansae

    p0 = dict(space.decode(space.default_vector()).params, inrad=INRAD, outrad=OUTRAD)
    clean = red.reduce(ReductionRequest(params=p0)).image
    pk_p, snr_p, det_p = measure(clean, red, RHO, PA, pixel_mask=pmask)
    log(f"\nbeta Pic b: matched-filter peak {pk_p:.5g}, S/N {snr_p:.2f} "
        f"(at x={det_p['x']}, y={det_p['y']}, r={det_p['r']:.2f} px)")

    def throughput(c):
        """(median peak_inj / c over the test azimuths, per-azimuth values) at contrast c."""
        t = []
        for th in TEST_PA:
            img = red.reduce(ReductionRequest(params=p0,
                                              injections=[Source(RHO, th, c)])).image
            pk_i, _, _ = measure(img - clean, red, RHO, th, pixel_mask=pmask)
            t.append(pk_i / c)
        t = np.array(t, float)
        return float(np.median(t)), t

    log(f"\nsolving for the contrast at which the fake is as bright as the planet:")
    log(f"{'iter':>5} {'c_inj':>12} {'peak_inj/c':>12} {'-> contrast':>13} {'dLprime':>9}")
    c, per_az = PUB, None
    for it in range(6):
        tp, per_az = throughput(c)
        got = pk_p / tp
        log(f"{it:5d} {c:12.4e} {tp:12.5g} {got:13.4e} {-2.5 * np.log10(got):8.3f}")
        if abs(np.log(got / c)) < 0.02:
            break
        c = got

    # the same measurement's azimuth-to-azimuth spread, propagated to the contrast
    az = pk_p / per_az
    err = float(np.std(az) / np.median(az) / np.sqrt(len(az)))    # throughput scatter
    err = float(np.hypot(err, 1.0 / snr_p)) * 2.5 / np.log(10)    # + the planet's own S/N
    dmag = -2.5 * np.log10(got)

    log(f"\nhow far the recovery is from linear (peak_inj / c; constant = linear):")
    for k in SWEEP:
        tp, _ = throughput(k * got)
        log(f"   c = {k:4.2f} x the planet ({k * got:.3e}): {tp:10.5g}  "
            f"{100 * (tp / (pk_p / got) - 1):+6.1f}% vs at the planet's own brightness")

    log(f"\nbeta Pic b, fixed point over {len(az)} azimuths "
        f"({', '.join(f'{v:.3e}' for v in np.sort(az))}):")
    log(f"   contrast {got:.4e}  ->  dL' = {dmag:.3f} +/- {err:.3f}")
    log(f"   published {PUB:.4e}  ->  dL' = {PUB_DMAG} +/- {PUB_ERR}   ({PUB_REF})")
    d = dmag - PUB_DMAG
    log(f"   difference {d:+.3f} mag = {abs(d) / np.hypot(err, PUB_ERR):.1f} sigma "
        f"(ratio {got / PUB:.3f})")
    ok = abs(d) <= TOL_MAG
    log(f"\n{'PASS' if ok else 'FAIL'}: the contrast axis is "
        f"{'on beta Pic b' if ok else f'off by more than {TOL_MAG} mag'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
